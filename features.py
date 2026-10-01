"""Chemistry-informed feature extraction with a per-feature wavenumber tag used to build the CI-MoE partition."""
import numpy as _np
_trapz = getattr(_np, "trapezoid", None) or _np.trapz
import numpy as np
from scipy.stats import skew, kurtosis
from .config import REGIONS, PEAKS, PEAK_HALF_WIDTH, peak_list


def _fwhm(x, y, i):
    half = y[i] / 2.0
    l = i
    while l > 0 and y[l] > half: l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half: r += 1
    return float(x[r] - x[l])


def _band(grid, spec, lo, hi):
    m = (grid >= lo) & (grid <= hi)
    return grid[m], spec[m]


def feature_schema(mod):
    """Return list of (name, group) where group is a region index, 'ratio', for the expert partition."""
    regs = REGIONS[mod]
    names = []
    for p in peak_list(mod):
        g = next((k for k, (_, lo, hi) in enumerate(regs) if lo <= p <= hi), None)
        for s in ("H", "A", "W", "pos"):
            names.append((f"peak{p}_{s}", g if g is not None else "ratio"))
    for k, (n, lo, hi) in enumerate(regs):
        for s in ("mean", "std", "max", "min", "area", "median", "skew", "kurt"):
            names.append((f"reg[{n}]_{s}", k))
        for s in ("d1mean", "d1absmax", "d2mean", "d2zc"):
            names.append((f"reg[{n}]_{s}", k))
    for s in ("ratio1", "ratio2"):
        names.append((f"{mod}_{s}", "ratio"))
    return names


def extract(grid, spec, mod):
    """Feature vector aligned with feature_schema(mod). Peak windows outside the measured range give 0."""
    regs = REGIONS[mod]
    d1 = np.gradient(spec, grid)
    d2 = np.gradient(d1, grid)
    f = []
    for p in peak_list(mod):
        x, y = _band(grid, spec, p - PEAK_HALF_WIDTH, p + PEAK_HALF_WIDTH)
        if len(x) < 3 or not np.any(y != 0):
            f += [0.0, 0.0, 0.0, float(p)]; continue
        i = int(np.argmax(y))
        f += [float(y[i]), float(_trapz(y, x)), _fwhm(x, y, i) if y[i] > 0 else 0.0, float(x[i])]
    for (n, lo, hi) in regs:
        x, y = _band(grid, spec, lo, hi)
        if len(x) < 4 or not np.any(y != 0):
            f += [0.0] * 12; continue
        _, a = _band(grid, d1, lo, hi); _, b = _band(grid, d2, lo, hi)
        zc = float(np.sum(np.diff(np.sign(b)) != 0))
        f += [y.mean(), y.std(), y.max(), y.min(), float(_trapz(y, x)), np.median(y),
              float(skew(y)), float(kurtosis(y)), a.mean(), np.abs(a).max(), b.mean(), zc]
    if mod == "ftir":
        def h(c):
            x, y = _band(grid, spec, c - 15, c + 15); return float(y.max()) if len(y) else 0.0
        eps = 1e-6
        f += [h(2914) / (abs(h(2847)) + eps), h(1715) / (abs(h(2914)) + eps)]
    else:
        x1, y1 = _band(grid, spec, 1000, 1150); x2, y2 = _band(grid, spec, 2800, 3100)
        a1 = float(_trapz(np.abs(y1), x1)) if len(x1) else 0.0
        a2 = float(_trapz(np.abs(y2), x2)) if len(x2) else 0.0
        f += [a1 / (a2 + 1e-6), a2 / (a1 + 1e-6)]
    return np.nan_to_num(np.asarray(f, float), nan=0.0, posinf=0.0, neginf=0.0)


def partition(mod):
    """CI-MoE index sets: S_1..S_K (regions), S_rho (ratios & unassigned), S_0 (generalist, all)."""
    schema = feature_schema(mod)
    K = len(REGIONS[mod])
    sets = [[i for i, (_, g) in enumerate(schema) if g == k] for k in range(K)]
    sets.append([i for i, (_, g) in enumerate(schema) if g == "ratio"])
    sets.append(list(range(len(schema))))
    return sets


def expert_names(mod):
    return [n for n, _, _ in REGIONS[mod]] + ["ratio/deriv", "generalist"]


def coverage(mod, wn_min, wn_max):
    """m_e: fraction of each expert's wavenumber window covered by the measurement (ratio & generalist = 1)."""
    m = []
    for _, lo, hi in REGIONS[mod]:
        c = max(0.0, min(hi, wn_max) - max(lo, wn_min)) / (hi - lo)
        m.append(c)
    return np.array(m + [1.0, 1.0])
