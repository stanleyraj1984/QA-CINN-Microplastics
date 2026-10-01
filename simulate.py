"""Synthetic vibrational spectra (Gaussian bands at reference positions + drifting baseline + noise).
Used ONLY for software tests (`--synthetic`); such data must never be reported as experimental results."""
import numpy as np
from .config import PEAKS


def synth(mod, poly, rng, lo=None, hi=None, noise=0.01, step=1.0):
    lo = lo or (650 if mod == "ftir" else 100); hi = hi or (4000 if mod == "ftir" else 3200)
    wn = np.arange(lo, hi, step)
    I = np.zeros_like(wn)
    for p in PEAKS[mod][poly]:
        p = p + rng.normal(0, 2.0)
        I += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((wn - p) / rng.uniform(6, 12)) ** 2)
    base = rng.uniform(0.05, 0.3) * np.polyval(rng.normal(0, 1, 3), (wn - wn.mean()) / wn.std())
    return wn, I + base + rng.normal(0, noise, len(wn))


def make_dataset(mod, polys, n_per, rng, **kw):
    raw, labels, spec = [], [], []
    for pi, poly in enumerate(polys):
        for k in range(n_per[pi]):
            lo = kw.get("lo_map", {}).get(poly)
            raw.append(synth(mod, poly, rng, lo=lo, noise=kw.get("noise", 0.01)))
            labels.append(poly); spec.append(f"{poly}_{k:02d}")
    return raw, labels, spec
