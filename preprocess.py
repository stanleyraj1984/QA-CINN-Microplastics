"""Chemistry-informed preprocessing: ALS baseline, SNV, Savitzky-Golay, interpolation to a common grid."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from .config import ALS, SG, GRID


def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    H = lam * D.dot(D.T)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + H).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def snv(y):
    s = y.std()
    return (y - y.mean()) / (s if s > 0 else 1.0)


def make_grid(mod):
    lo, hi, step = GRID[mod]
    return np.arange(lo, hi + 1e-9, step)


def preprocess(wn, I, mod):
    """Return (grid, processed spectrum, coverage mask on grid). Points outside the measured range are
    filled with the edge value *only for numerical continuity* and are flagged in the mask."""
    order = np.argsort(wn)
    wn, I = np.asarray(wn, float)[order], np.asarray(I, float)[order]
    grid = make_grid(mod)
    covered = (grid >= wn.min()) & (grid <= wn.max())
    y = np.interp(grid, wn, I)
    yc = y[covered]
    base = als_baseline(yc, **ALS)
    yc = yc - base
    yc = savgol_filter(yc, SG["window"] if len(yc) > SG["window"] else 5, SG["polyorder"])
    yc = snv(yc)
    out = np.zeros_like(y)
    out[covered] = yc
    return grid, out, covered
