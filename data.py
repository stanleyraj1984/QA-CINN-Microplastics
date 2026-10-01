"""Dataset assembly. A manifest CSV (columns: file,modality,label,specimen) points to two-column
(wavenumber, intensity) text/CSV files; replicates of one physical specimen share the `specimen` id."""
import os
import numpy as np
import pandas as pd
from .preprocess import preprocess
from .features import extract, coverage
from .quality import quality_metrics, quality_score


class Modality:
    def __init__(self, mod, raw, labels, specimens):
        self.mod, self.raw, self.specimens = mod, raw, np.asarray(specimens)
        self.classes = sorted(set(labels)); self.y = np.array([self.classes.index(l) for l in labels])
        self.build()

    def _one(self, wn, I):
        grid, spec, covered = preprocess(wn, I, self.mod)
        x = extract(grid, spec, self.mod)
        q = quality_metrics(wn, I, grid, spec, covered, self.mod)
        m = coverage(self.mod, float(np.min(wn)), float(np.max(wn)))
        return x, q, m

    def build(self):
        res = [self._one(wn, I) for wn, I in self.raw]
        self.X = np.stack([r[0] for r in res]); self.q = np.stack([r[1] for r in res])
        self.m = np.stack([r[2] for r in res]); self.Q = np.array([quality_score(v) for v in self.q])

    def perturbed(self, idx, noise_frac, rng):
        """Recompute (X,q,m) after adding Gaussian noise (fraction of signal std) to the raw spectra idx."""
        res = []
        for i in idx:
            wn, I = self.raw[i]
            res.append(self._one(wn, I + rng.normal(0, noise_frac * np.std(I), size=len(I))))
        return np.stack([r[0] for r in res]), np.stack([r[1] for r in res]), np.stack([r[2] for r in res])


def load_manifest(path, root=None):
    df = pd.read_csv(path); root = root or os.path.dirname(path)
    out = {}
    for mod, g in df.groupby("modality"):
        raw = []
        for f in g["file"]:
            a = np.loadtxt(os.path.join(root, f), delimiter=",", comments="#", ndmin=2)
            raw.append((a[:, 0], a[:, 1]))
        out[mod] = Modality(mod, raw, list(g["label"]), list(g["specimen"]))
    return out
