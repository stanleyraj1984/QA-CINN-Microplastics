"""One-command reproduction of every CI-MoE number in the manuscript.

  python run_all.py --manifest data/manifest.csv          # real data (writes ../results/*.tex, results.json)
  python run_all.py --synthetic                           # software test only; output goes to results_synthetic/

The LaTeX tables written to results/ are \\input by the manuscript, so re-running fills the paper automatically."""
import argparse, json, os
import numpy as np
from qacinn.config import SEEDS
from qacinn.data import load_manifest, Modality
from qacinn.simulate import make_dataset
from qacinn.protocol import run_cv, agg
from qacinn.features import expert_names
from qacinn import fusion, train as T

SK = ["SVM", "RF", "KNN", "LR", "MLP"]
pm = lambda ms: f"{ms[0]:.3f} $\\pm$ {ms[1]:.3f}"

def table(res, mod, names):
    rows = []
    for n in names:
        a = {k: agg(res[n], k) for k in ("acc", "f1", "brier", "ece", "unc_err_sigma", "unc_err_Im")}
        rows.append(f"{n} & {pm(a['acc'])} & {pm(a['f1'])} & {a['brier'][0]:.3f} & {a['ece'][0]:.3f} & "
                    f"{a['unc_err_sigma'][0]:.2f} & {a['unc_err_Im'][0]:.2f} \\\\")
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEEDS)); ap.add_argument("--out", default=None)
    a = ap.parse_args(); seeds = SEEDS[:a.seeds]
    out = a.out or ("../results_synthetic" if a.synthetic else "../results"); os.makedirs(out, exist_ok=True)
    if a.synthetic:
        rng = np.random.default_rng(0)
        data = {"ftir": Modality("ftir", *make_dataset("ftir", ["PE", "PET", "HDPE"], [8, 4, 2], rng, lo_map={"PET": 650})),
                "raman": Modality("raman", *make_dataset("raman", ["PE", "CA", "PP"], [4, 3, 3], rng))}
    else:
        data = load_manifest(a.manifest)
    summary, tex, res_store = {}, [], {}
    for mod, ds in data.items():
        cfgs = list(T.CONFIGS)
        res = run_cv(ds, cfgs, SK, seeds=seeds); res_store[mod] = res
        names = SK + cfgs
        summary[mod] = {n: {k: agg(res[n], k) for k in ("acc", "f1", "bacc", "brier", "ece")} for n in names}
        for n in cfgs:
            summary[mod][n]["stress"] = {k: float(np.mean(v)) for k, v in res[n]["stress"].items()}
        tex.append(f"% ---- {mod} ----")
        head = ("\\begin{table}[t]\\centering\\caption{" + mod.upper() + " specimen-level CV, mean $\\pm$ std over folds and "
                f"{len(seeds)} seeds. Unc.--err.: Pearson correlation of $\\sigma$ (dense) or $\\sigma$ / $I_m$ (CI-MoE) with error.}}"
                "\\label{tab:cimoe_" + mod + "}\\small\\begin{tabular}{lcccccc}\\toprule Model & Acc. & Macro-F1 & Brier & ECE & "
                "$r_\\sigma$ & $r_{I}$ \\\\ \\midrule")
        tex += [head] + table(res, mod, names) + ["\\bottomrule\\end{tabular}\\end{table}"]
        # stress tests
        tex.append("\\begin{table}[t]\\centering\\caption{" + mod.upper() + " stress tests (accuracy): region masking "
                   "(1 or 2 regions removed at test time) and added noise (fraction of signal std).}\\label{tab:stress_" + mod +
                   "}\\small\\begin{tabular}{lcccc}\\toprule Model & mask-1 & mask-2 & noise 0.5 & noise 1.0 \\\\ \\midrule")
        for n in cfgs:
            s = summary[mod][n]["stress"]
            tex.append(f"{n} & {s['mask1']:.3f} & {s['mask2']:.3f} & {s['noise0.5']:.3f} & {s['noise1.0']:.3f} \\\\")
        tex.append("\\bottomrule\\end{tabular}\\end{table}")
        # gate profile per class (H3)
        g = res["CI-MoE (soft gate)"]["gate"]
        en = expert_names(mod)
        tex.append("\\begin{table}[t]\\centering\\caption{" + mod.upper() + " mean gate mass per class (out-of-fold, seed "
                   f"{seeds[0]}).}}\\label{{tab:gate_{mod}}}\\scriptsize\\begin{{tabular}}{{l{'c'*len(en)}}}\\toprule Class & "
                   + " & ".join(e.replace("&", "\\&") for e in en) + " \\\\ \\midrule")
        for ci, c in enumerate(ds.classes):
            tex.append(c + " & " + " & ".join(f"{v:.2f}" for v in g[ds.y == ci].mean(0)) + " \\\\")
        tex.append("\\bottomrule\\end{tabular}\\end{table}")
    # PE fusion on out-of-fold probabilities, in the union label space (PE is the only common polymer)
    if "ftir" in data and "raman" in data and "PE" in data["ftir"].classes and "PE" in data["raman"].classes:
        f, r = data["ftir"], data["raman"]; U = sorted(set(f.classes) | set(r.classes))
        def lift(P, cls):
            L = np.zeros((len(P), len(U)))
            for j, c in enumerate(cls): L[:, U.index(c)] = P[:, j]
            return L
        fi = np.where(f.y == f.classes.index("PE"))[0]; ri = np.where(r.y == r.classes.index("PE"))[0]
        tex.append("\\begin{table}[t]\\centering\\caption{PE identification, FTIR only vs. fused (out-of-fold probabilities; "
                   "Raman replicates averaged).}\\label{tab:fusion_cimoe}\\small\\begin{tabular}{llcc}\\toprule Backbone & Fusion & FTIR only & Fused \\\\ \\midrule")
        summary["fusion"] = {}
        for cfg in ("Dense CINN", "CI-MoE (soft gate)"):
            PF = lift(res_store["ftir"][cfg]["P"][fi], f.classes)
            PR = lift(res_store["raman"][cfg]["P"][ri], r.classes).mean(0)
            for mode in ("equal", "quality", "confidence", "quality_conf"):
                ok_f = np.mean(PF.argmax(1) == U.index("PE")); ok = []
                for k, i in enumerate(fi):
                    Pf, _ = fusion.fuse(np.stack([PF[k], PR]), [f.Q[i], r.Q[ri].mean()], mode)
                    ok.append(Pf.argmax() == U.index("PE"))
                summary["fusion"][f"{cfg}/{mode}"] = float(np.mean(ok))
                tex.append(f"{cfg} & {mode.replace('_', '$\\times$')} & {ok_f:.3f} & {np.mean(ok):.3f} \\\\")
        tex.append("\\bottomrule\\end{tabular}\\end{table}")
    json.dump(summary, open(os.path.join(out, "results.json"), "w"), indent=1)
    open(os.path.join(out, "cimoe_results.tex"), "w").write("\n".join(tex))
    print("wrote", out)

if __name__ == "__main__":
    main()
