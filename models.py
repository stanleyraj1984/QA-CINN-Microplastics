"""PyTorch models: dense CINN (Sec. 4.4) and chemistry-informed mixture of experts CI-MoE (Sec. 4.5)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CINN, MOE


class FeatureAttention(nn.Module):
    """a = sigmoid(W2 ReLU(W1 x)); x_tilde = a * x."""
    def __init__(self, d, r=4):
        super().__init__()
        self.w1 = nn.Linear(d, max(d // r, 4)); self.w2 = nn.Linear(max(d // r, 4), d)
    def forward(self, x):
        a = torch.sigmoid(self.w2(F.relu(self.w1(x))))
        return a * x, a


class UncertaintyHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus())
    def forward(self, h):
        return self.net(h).squeeze(-1)


def unc_loss(p_true, sigma):
    """L_unc = (p_y - exp(-sigma))^2 : confident-and-correct <-> small sigma."""
    return ((p_true - torch.exp(-sigma)) ** 2).mean()


class CINN(nn.Module):
    def __init__(self, d, n_classes, hidden=CINN["hidden"], dropout=CINN["dropout"]):
        super().__init__()
        self.att = FeatureAttention(d)
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_classes))
        self.unc = UncertaintyHead(prev)
        self.lam_unc = CINN["lam_unc"]

    def forward(self, x, q=None, m=None):
        xt, a = self.att(x)
        h = self.trunk(xt)
        logits = self.head(h)
        return dict(P=F.softmax(logits, -1), logits=logits, sigma=self.unc(h), attn=a)

    def loss(self, out, y):
        p_true = out["P"].gather(1, y[:, None]).squeeze(1)
        return F.cross_entropy(out["logits"], y) + self.lam_unc * unc_loss(p_true.detach(), out["sigma"])


class Expert(nn.Module):
    def __init__(self, d_in, n_classes, hidden=MOE["hidden"], out=MOE["out"], dropout=MOE["dropout"]):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(hidden, out))
        self.head = nn.Linear(out, n_classes)
    def forward(self, x):
        h = self.f(x)
        return h, self.head(h)


class CIMoE(nn.Module):
    """Chemistry-informed mixture of experts.

    sets  : list of index lists S_1..S_K, S_rho, S_0 (see features.partition)
    gate  : 'soft' (default) or 'topk' (noisy top-k, Shazeer et al. 2017)
    use_qm: feed [q; m] to the gate; use_mask: add log(m+eps) to the gate logits
    mixing: 'output' (mixture of expert distributions, Eq. 4) or 'feature' (sparse-MoE-style ablation)
    """
    def __init__(self, d, n_classes, sets, gate="soft", use_qm=True, use_mask=True, mixing="output",
                 rho=None, lam_bal=MOE["lam_bal"], lam_z=MOE["lam_z"], lam_unc=MOE["lam_unc"]):
        super().__init__()
        self.E = len(sets); self.C = n_classes
        self.sets = [torch.as_tensor(s, dtype=torch.long) for s in sets]
        for i, s in enumerate(self.sets):
            self.register_buffer(f"idx{i}", s)
        self.experts = nn.ModuleList([Expert(len(s), n_classes) for s in sets])
        self.gate_mode, self.use_qm, self.use_mask, self.mixing = gate, use_qm, use_mask, mixing
        gin = 2 * self.E + (5 + self.E if use_qm else 0)
        self.gate = nn.Linear(gin, self.E)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)     # start from uniform routing
        rho = torch.full((self.E,), 1.0 / self.E) if rho is None else torch.as_tensor(rho, dtype=torch.float)
        self.register_buffer("rho", rho / rho.sum())
        if mixing == "feature":
            self.shared_head = nn.Linear(MOE["out"], n_classes)
        self.unc = UncertaintyHead(MOE["out"])
        self.lam_bal, self.lam_z, self.lam_unc = lam_bal, lam_z, lam_unc

    def _idx(self, i):
        return getattr(self, f"idx{i}")

    def forward(self, x, q, m):
        hs, ls = [], []
        for i, e in enumerate(self.experts):
            h, l = e(x[:, self._idx(i)]); hs.append(h); ls.append(l)
        H = torch.stack(hs, 1)                                    # B,E,h
        pis = F.softmax(torch.stack(ls, 1), -1)                   # B,E,C
        s = torch.cat([torch.stack([x[:, self._idx(i)].mean(1) for i in range(self.E)], 1),
                       torch.stack([x[:, self._idx(i)].max(1).values for i in range(self.E)], 1)], 1)
        gin = torch.cat([s, q, m], 1) if self.use_qm else s
        logit = self.gate(gin)
        if self.use_mask:
            logit = logit + torch.log(m + MOE["eps"])
        if self.training and MOE["expert_drop"] > 0:
            keep = (torch.rand_like(logit) > MOE["expert_drop"]).float()
            keep[:, -1] = torch.clamp(keep[:, -1] + (keep.sum(1) == 0).float(), max=1.0)   # generalist survives if all dropped
            logit = logit.masked_fill(keep == 0, -1e9)
        if self.gate_mode == "topk":
            noisy = logit + (torch.randn_like(logit) * MOE["noise"] if self.training else 0.0)
            k = min(MOE["topk"], self.E)
            top = noisy.topk(k, -1)
            masked = torch.full_like(noisy, -1e9).scatter(1, top.indices, top.values)
            g = F.softmax(masked / MOE["tau"], -1)
        else:
            g = F.softmax(logit / MOE["tau"], -1)
        if self.mixing == "output":
            P = (g[:, :, None] * pis).sum(1)
        else:
            hbar = (g[:, :, None] * H).sum(1)
            P = F.softmax(self.shared_head(hbar), -1)
        hmix = (g[:, :, None] * H).sum(1)
        sigma = self.unc(hmix)
        ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        Im = ent(P) - (g * ent(pis)).sum(1)                       # expert disagreement (>=0 for output mixing)
        return dict(P=P, pis=pis, g=g, sigma=sigma, Im=Im, logit=logit)

    def loss(self, out, y):
        P, g = out["P"], out["g"]
        p_true = P.gather(1, y[:, None]).squeeze(1)
        nll = -torch.log(p_true.clamp_min(1e-12)).mean()
        gbar = g.mean(0).clamp_min(1e-12)
        kl = (gbar * (gbar.log() - self.rho.log())).sum()
        z = (torch.logsumexp(out["logit"], 1) ** 2).mean()
        return nll + self.lam_unc * unc_loss(p_true.detach(), out["sigma"]) + self.lam_bal * kl + self.lam_z * z
