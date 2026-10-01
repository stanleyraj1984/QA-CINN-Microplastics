"""Torch-free smoke tests of preprocessing, features, quality, coverage and fusion."""
import numpy as np
from qacinn.simulate import make_dataset
from qacinn.data import Modality
from qacinn.features import feature_schema, partition, coverage
from qacinn import fusion, metrics
rng = np.random.default_rng(0)
for mod, polys in (("ftir", ["PE", "PET", "HDPE"]), ("raman", ["PE", "CA", "PP"])):
    raw, lab, spec = make_dataset(mod, polys, [3, 3, 2], rng, lo_map={"PET": 650} if mod == "ftir" else {})
    ds = Modality(mod, raw, lab, spec)
    sets = partition(mod)
    print(mod, "features:", ds.X.shape[1], "experts:", len(sets), "sizes:", [len(s) for s in sets])
    print("  Q by class:", {c: round(float(ds.Q[ds.y == i].mean()), 2) for i, c in enumerate(ds.classes)})
    assert np.isfinite(ds.X).all() and ((ds.q >= 0) & (ds.q <= 1)).all()
    assert all(len(s) > 0 for s in sets[:-1]) or print("  empty expert sets:", [i for i, s in enumerate(sets) if not s])
print("coverage PET-like (650-4000) on FTIR:", coverage("ftir", 650, 4000).round(2))
P, om = fusion.fuse(np.array([[.15, .75, .10], [.80, .10, .10]]), [0.90, 0.70])
print("toy fusion:", P.round(4), "weights:", om.round(3))
assert abs(P.sum() - 1) < 1e-9
# Proposition 1 numerical check (Brier & log-loss convexity) on random simplex points
ok = True
for _ in range(2000):
    Pm = rng.dirichlet(np.ones(4), size=2); w = rng.dirichlet(np.ones(2)); y = rng.integers(4)
    Pf = w @ Pm
    ok &= fusion.brier(Pf, y) <= sum(w[i] * fusion.brier(Pm[i], y) for i in range(2)) + 1e-12
    ok &= -np.log(Pf[y]) <= -sum(w[i] * np.log(Pm[i][y]) for i in range(2)) + 1e-12
print("Proposition 1 holds on 2000 random draws:", ok)
# entropy decomposition check (Eq. 6): I >= 0
g = rng.dirichlet(np.ones(5), size=500); pis = rng.dirichlet(np.ones(4), size=(500, 5))
Pm = (g[:, :, None] * pis).sum(1)
I = metrics.entropy(Pm) - (g * metrics.entropy(pis)).sum(1)
print("min expert disagreement I over 500 draws:", float(I.min()) >= -1e-12)
print("Clopper-Pearson 12/13:", metrics.clopper_pearson(12, 13))
