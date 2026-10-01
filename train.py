"""Training / prediction helpers for CINN, CI-MoE and scikit-learn baselines (deterministic seeds)."""
import copy, warnings
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from .config import CINN
from .models import CINN as CINNet, CIMoE
from .features import partition


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)


CONFIGS = {   # name -> kwargs for CIMoE (None => dense CINN)
    "Dense CINN": None,
    "CI-MoE (soft gate)": dict(gate="soft"),
    "CI-MoE (top-2 gate)": dict(gate="topk"),
    "CI-MoE w/o quality/coverage in gate": dict(gate="soft", use_qm=False),
    "CI-MoE w/o coverage mask": dict(gate="soft", use_mask=False),
    "CI-MoE w/o load/prior term": dict(gate="soft", lam_bal=0.0),
    "CI-MoE feature-mixing": dict(gate="soft", mixing="feature"),
}


def build(name, d, C, mod):
    kw = CONFIGS[name]
    return CINNet(d, C) if kw is None else CIMoE(d, C, partition(mod), **kw)


def fit_torch(name, mod, X, q, m, y, C, seed):
    set_seed(seed)
    model = build(name, X.shape[1], C, mod)
    opt = torch.optim.Adam(model.parameters(), lr=CINN["lr"], weight_decay=CINN["wd"])
    Xt, qt, mt, yt = map(lambda a, t: torch.as_tensor(a, dtype=t), (X, q, m, y),
                         (torch.float, torch.float, torch.float, torch.long))
    n = len(yt); best, bad, best_state = 1e18, 0, None
    for ep in range(CINN["max_epochs"]):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, CINN["batch"]):
            b = perm[i:i + CINN["batch"]]
            if len(b) < 2:                       # BatchNorm needs >=2 samples
                continue
            out = model(Xt[b], qt[b], mt[b]); loss = model.loss(out, yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        tot /= n
        if tot < best - 1e-4:
            best, bad, best_state = tot, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= CINN["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, X, q, m):
    model.eval()
    out = model(torch.as_tensor(X, dtype=torch.float), torch.as_tensor(q, dtype=torch.float),
                torch.as_tensor(m, dtype=torch.float))
    return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in out.items()}


def sk_baseline(name, seed):
    base = {"SVM": SVC(kernel="rbf", C=1.0, probability=False, random_state=seed),
            "RF": RandomForestClassifier(n_estimators=200, random_state=seed),
            "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "LR": LogisticRegression(C=1.0, max_iter=2000),
            "MLP": MLPClassifier((64, 32), max_iter=500, random_state=seed)}[name]
    return base


def fit_predict_sk(name, Xtr, ytr, Xte, seed, C):
    est = sk_baseline(name, seed)
    counts = np.bincount(ytr, minlength=C); k = int(min(3, counts[counts > 0].min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if k >= 2:
            est = CalibratedClassifierCV(est, method="isotonic", cv=k)
        elif name == "SVM":
            est.set_params(probability=True)
        est.fit(Xtr, ytr)
        P = est.predict_proba(Xte) if hasattr(est, "predict_proba") else np.eye(C)[est.predict(Xte)]
    full = np.zeros((len(Xte), C)); full[:, est.classes_ if hasattr(est, "classes_") else np.arange(P.shape[1])] = P
    return full
