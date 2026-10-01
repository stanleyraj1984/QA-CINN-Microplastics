"""Five-metric spectral quality score Q in [0,1] (Sec. 4.6)."""
import numpy as np
from scipy.signal import find_peaks
from .config import QUALITY_WEIGHTS

SIG = {"ftir": (2800, 3000), "raman": (2800, 3100)}
EXPECTED = {"ftir": (650, 4000), "raman": (100, 3200)}


def _sel(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]


def quality_metrics(wn_raw, I_raw, grid, spec, covered, mod):
    """Return q in [0,1]^5: SNR, completeness, baseline stability, intensity consistency, peak detectability."""
    lo, hi = SIG[mod]
    _, s = _sel(grid, spec, lo, hi)
    _, noise_reg = _sel(grid, spec, 1800, 2800)
    _, raw_sig = _sel(np.sort(wn_raw), np.asarray(I_raw)[np.argsort(wn_raw)], lo, hi)
    # noise estimated from high-frequency residual of raw signal in the silent region (robust to slope)
    order = np.argsort(wn_raw); x = np.asarray(wn_raw)[order]; y = np.asarray(I_raw)[order]
    xs, ys = _sel(x, y, 1800, 2800)
    if len(ys) > 8:
        noise = np.std(np.diff(ys)) / np.sqrt(2)
        snr_raw = (np.max(raw_sig) - np.median(raw_sig)) / (noise + 1e-12) if len(raw_sig) else 0.0
    else:
        snr_raw = 0.0
    snr = float(np.clip(np.log10(1 + max(snr_raw, 0)) / 2.0, 0, 1))          # 100:1 -> 1
    e_lo, e_hi = EXPECTED[mod]
    comp = float(np.clip((min(x.max(), e_hi) - max(x.min(), e_lo)) / (e_hi - e_lo), 0, 1))
    xg = grid[covered]; yg = spec[covered]
    if len(xg) > 10 and yg.std() > 0:
        t = (xg - xg.mean()) / (xg.std() + 1e-12)
        res = yg - np.polyval(np.polyfit(t, yg, 3), t)
        base = 1.0 / (1.0 + res.std() / yg.std())
    else:
        base = 0.0
    xf, yf = _sel(x, y, 1800, 2800)
    cons = 1.0 / (1.0 + abs(np.std(yf) / (abs(np.mean(yf)) + 1e-12))) if len(yf) > 3 else 0.0
    pk, _ = find_peaks(yg, prominence=1.0) if len(yg) else ([], None)
    peaks = min(len(pk) / 20.0, 1.0)
    return np.clip(np.array([snr, comp, base, cons, peaks], float), 0, 1)


def quality_score(q, w=QUALITY_WEIGHTS):
    return float(np.clip(np.dot(w, q), 0, 1))
