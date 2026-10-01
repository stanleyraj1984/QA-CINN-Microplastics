import numpy as np
from scipy.stats import beta, pearsonr
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

def brier_score(P, y, C):
    Y = np.eye(C)[y]
    return float(((P - Y) ** 2).sum(1).mean())

def ece(P, y, bins=10):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(-1)

def unc_error_corr(u, P, y):
    err = (P.argmax(1) != y).astype(float)
    if np.std(u) < 1e-12 or np.std(err) < 1e-12:
        return float("nan")
    return float(pearsonr(u, err)[0])

def summary(P, y, C):
    pred = P.argmax(1)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro", zero_division=0),
                bacc=balanced_accuracy_score(y, pred), brier=brier_score(P, y, C), ece=ece(P, y))

def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)
