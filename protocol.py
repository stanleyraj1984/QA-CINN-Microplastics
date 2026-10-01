"""Cross-validation protocol (Sec. 4.9) with identical folds/seeds for all models; stress tests; gate profiles."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from .config import SEEDS
from .features import partition, REGIONS
from . import metrics as M
from . import train as T


def make_splits(y, groups, mod, seed=42):
    if mod == "ftir":
        n_min = np.bincount(y).min()
        k = int(min(5, max(2, len(set(groups)))))
        return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups))


def run_cv(ds, config_names, sk_names, seeds=SEEDS, stress=True, log=print):
    """Returns dict[name] -> dict(metrics per fold, pooled OOF probabilities, gate & Im diagnostics)."""
    C = len(ds.classes); y = ds.y; groups = ds.specimens
    res = {n: dict(folds=[], P=np.zeros((len(y), C)), sigma=np.zeros(len(y)), Im=np.zeros(len(y)),
                   gate=np.zeros((len(y), 0)), stress={}) for n in list(config_names) + list(sk_names)}
    for seed in seeds:
        splits = make_splits(y, groups, ds.mod, seed=seed if ds.mod == "ftir" else 42)
        for fi, (tr, te) in enumerate(splits):
            sc = StandardScaler().fit(ds.X[tr])                       # fit on training fold only
            Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
            unseen = int(sum(1 for c in set(y[te]) if c not in set(y[tr])))
            for n in sk_names:
                P = T.fit_predict_sk(n, Xtr, y[tr], Xte, seed, C)
                s = M.summary(P, y[te], C); s["unseen_classes"] = unseen; res[n]["folds"].append(s)
                if seed == seeds[0]: res[n]["P"][te] = P
            for n in config_names:
                mdl = T.fit_torch(n, ds.mod, Xtr, ds.q[tr], ds.m[tr], y[tr], C, seed)
                out = T.predict_torch(mdl, Xte, ds.q[te], ds.m[te])
                s = M.summary(out["P"], y[te], C); s["unseen_classes"] = unseen
                s["unc_err_sigma"] = M.unc_error_corr(out["sigma"], out["P"], y[te])
                if "Im" in out:
                    s["unc_err_Im"] = M.unc_error_corr(out["Im"], out["P"], y[te])
                res[n]["folds"].append(s)
                if seed == seeds[0]:
                    res[n]["P"][te] = out["P"]; res[n]["sigma"][te] = out["sigma"]
                    if "g" in out:
                        if res[n]["gate"].shape[1] == 0: res[n]["gate"] = np.zeros((len(y), out["g"].shape[1]))
                        res[n]["gate"][te] = out["g"]; res[n]["Im"][te] = out["Im"]
                if stress:
                    rng = np.random.default_rng(seed * 1000 + fi)
                    st = res[n]["stress"]
                    for r in (1, 2):                                  # region masking at test time
                        K = len(REGIONS[ds.mod]); sets = partition(ds.mod)
                        Xm, mm = Xte.copy(), ds.m[te].copy()
                        for i in range(len(te)):
                            for e in rng.choice(K, size=r, replace=False):
                                Xm[i, sets[e]] = 0.0; mm[i, e] = 0.0
                        o = T.predict_torch(mdl, Xm, ds.q[te], mm)
                        st.setdefault(f"mask{r}", []).append(M.summary(o["P"], y[te], C)["acc"])
                    for nf in (0.5, 1.0):                             # quality degradation (lower SNR)
                        Xn, qn, mn = ds.perturbed(te, nf, rng)
                        o = T.predict_torch(mdl, sc.transform(Xn), qn, mn)
                        st.setdefault(f"noise{nf}", []).append(M.summary(o["P"], y[te], C)["acc"])
        log(f"[{ds.mod}] seed {seed} done")
    return res


def agg(res_n, key):
    v = np.array([f[key] for f in res_n["folds"] if not np.isnan(f.get(key, np.nan))])
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float("nan"), float("nan"))
