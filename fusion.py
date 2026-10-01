"""Decision-level quality-aware fusion (Eqs. 7-10) and ablation variants."""
import numpy as np

def weights(Q, P, mode="quality_conf"):
    """Q: (M,), P: (M, C). Returns raw weights w_m for the chosen variant."""
    conf = P.max(axis=1)
    return {"equal": np.ones_like(Q), "quality": Q, "confidence": conf, "sqrt_quality": np.sqrt(Q),
            "quality_conf": Q * conf}[mode]

def fuse(P, Q, mode="quality_conf"):
    """P: (M, C) modality distributions (rows sum to one); returns fused (C,) and normalised weights."""
    w = weights(np.asarray(Q, float), np.asarray(P, float), mode)
    if w.sum() <= 0:
        w = np.ones_like(w)
    om = w / w.sum()
    return om @ P, om

def brier(P, y):
    e = np.zeros_like(P); e[..., y] = 1.0
    return float(((P - e) ** 2).sum(-1))
