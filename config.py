"""Fixed a-priori configuration (Tables 3, 4, 17, 18 of the manuscript)."""
import numpy as np

GRID = {"ftir": (650.0, 4000.0, 1.0), "raman": (100.0, 3200.0, 1.5)}
ALS = dict(lam=1e5, p=0.01, n_iter=10)
SG = dict(window=11, polyorder=3)
PEAK_HALF_WIDTH = 15.0
QUALITY_WEIGHTS = np.array([0.25, 0.15, 0.20, 0.15, 0.25])  # SNR, complete, baseline, consist, peaks

REGIONS = {
    "ftir": [("OH/NH stretch", 3200, 3600), ("CH stretch", 2800, 3000), ("C=O stretch", 1680, 1750),
             ("Aromatic CC", 1450, 1600), ("CH2/CH3 bend", 1350, 1480), ("C-O stretch", 1000, 1300),
             ("Fingerprint", 650, 900)],
    "raman": [("CH stretch", 2800, 3100), ("C=C stretch", 1600, 1680), ("Aromatic ring", 1000, 1600),
              ("C-C stretch", 800, 1150), ("Skeletal", 400, 800)],
}

# Table 3 reference peaks (cm^-1); they inform features, never labels.
PEAKS = {
    "ftir": {"PE": [2914, 2847, 1470, 718], "PET": [1715, 1240, 1100, 725], "HDPE": [2914, 2847, 1470, 730, 720],
             "LDPE": [2914, 2847, 1470, 730, 720], "PP": [2950, 2916, 1455, 1376, 1167],
             "PS": [3025, 3060, 1600, 1493, 1452, 756, 699],
             "PVC": [2916, 2848, 1426, 1330, 1250, 1094, 960, 695, 615],
             "PA": [3290, 3070, 2930, 2850, 1640, 1540, 1460, 1370, 1260, 1150],
             "PC": [2970, 2930, 1770, 1500, 1460, 1370, 1240, 1170, 1080, 1010],
             "PU": [3330, 1725, 1530, 1220, 1100], "PMMA": [2950, 1730, 1450, 1380, 1270, 1160, 1070, 990],
             "PTFE": [2920, 1380, 1240, 1150, 1020, 970, 880, 770, 720, 650],
             "CA": [3349, 1728, 1252, 1031], "EVA": [2915, 2850, 1735, 1470, 1370, 1240, 1020, 950]},
    "raman": {"PE": [1064, 1128, 1295, 1440, 2885], "PET": [1615, 1728, 2920],
              "HDPE": [1064, 1128, 1295, 1440, 2885], "PP": [809, 841, 973, 1152, 1452, 2885],
              "PS": [621, 1001, 1032, 1155, 1207, 1583, 1602, 2850, 3055],
              "PVC": [638, 694, 964, 1095, 1220, 1325, 1430, 2910, 2965],
              "PA": [935, 1080, 1130, 1240, 1300, 1450, 1660, 2880, 2935],
              "PC": [972, 1170, 1200, 1340, 1585, 1600, 2850, 2920, 2970],
              "CA": [1095, 1375, 2900]},
}

# Peaks are drawn from the polymers present in the local data (keeps D ~ 120 / 115 as in the manuscript).
FEATURE_POLYMERS = {"ftir": ["PE", "PET", "HDPE"], "raman": ["PE", "CA", "PP"]}

def peak_list(mod):
    s = set()
    for p in FEATURE_POLYMERS[mod]:
        s.update(PEAKS[mod][p])
    return sorted(s)

# CI-MoE hyper-parameters (Table 18) -- fixed a priori, never tuned on held-out specimens.
MOE = dict(hidden=32, out=16, dropout=0.3, tau=1.0, eps=1e-6, topk=2, noise=0.1, expert_drop=0.1,
           lam_unc=0.1, lam_bal=0.01, lam_z=1e-3)
CINN = dict(hidden=(256, 128, 64), dropout=0.3, lr=1e-3, wd=1e-4, batch=16, patience=20, max_epochs=300,
            lam_unc=0.1)
SEEDS = [42, 43, 44, 45, 46]
