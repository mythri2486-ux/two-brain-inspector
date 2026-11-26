"""Module 2: logical brain + pose check + calibration + fused verdict.
Usage:
    python module2.py fit2  --product cable
    python module2.py eval2 --product cable
(fit2 holds out 20% of train/good for calibration, so it refits its own banks.)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

import baseline_patchcore as pc

ROOT = "data/mvtec_ad"
BANKS = "banks"
OUT = "outputs"

# ---- constants to tune using your eval table ----
G = 4                 # grid is G x G cells
MARGIN = 2            # patches of tolerance around each cell (absorbs small shifts)
LFRAC = 0.20          # fraction of patches kept per cell
HOLDOUT = 0.20        # share of train/good held out for calibration
POSE_RATIO_MAX = 1.10 # logical brain ON only if held-out normals look no worse position-aware
REVIEW_LO = 0.85     # fused score in [REVIEW_LO, 1.0) -> NEEDS_REVIEW
FORCE_OFF = set()     # e.g. {"tile", "leather", "grid", "metal_nut"} to force-disable manually


def _up(d):
    d = F.interpolate(d[:, None], size=(pc.IMG, pc.IMG), mode="bilinear",
                      align_corners=False)[:, 0].cpu().numpy()
    return np.stack([gaussian_filter(m, sigma=pc.SIGMA) for m in d])


def build_lbanks(ext, paths, batch=16, seed=0):
    """One memory bank per grid cell (position-aware), good images only."""
    g = torch.Generator().manual_seed(seed)
    cells = {(a, b): [] for a in range(G) for b in range(G)}
    for i in range(0, len(paths), batch):
        x = torch.stack([pc.to_tensor(pc.load_rgb(p)) for p in paths[i:i + batch]])
        f = ext(x).cpu()
        _, C, h, w = f.shape
        for gy in range(G):
            for gx in range(G):
                y0, y1 = max(0, gy * h // G - MARGIN), min(h, (gy + 1) * h // G + MARGIN)
                x0, x1 = max(0, gx * w // G - MARGIN), min(w, (gx + 1) * w // G + MARGIN)
                p = f[:, :, y0:y1, x0:x1].permute(0, 2, 3, 1).reshape(-1, C)
                n = max(1, int(len(p) * LFRAC))
                cells[(gy, gx)].append(p[torch.randperm(len(p), generator=g)[:n]])
        print(f"  logical fit {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return {k: torch.cat(v) for k, v in cells.items()}


@torch.no_grad()
def both_maps(ext, bank, lbanks, rgbs):
    """Physical map (search whole bank) and logical map (search own cell only)."""
    f = ext(torch.stack([pc.to_tensor(r) for r in rgbs]))
    B, C, h, w = f.shape
    dp = pc.min_dist(pc.patches(f), bank).reshape(B, h, w)
    dl = torch.zeros(B, h, w, device=f.device)
    for gy in range(G):
        for gx in range(G):
            ys = slice(gy * h // G, (gy + 1) * h // G)
            xs = slice(gx * w // G, (gx + 1) * w // G)
            cell = f[:, :, ys, xs]
            ch, cw = cell.shape[2], cell.shape[3]
            d = pc.min_dist(pc.patches(cell), lbanks[(gy, gx)])
            dl[:, ys, xs] = d.reshape(B, ch, cw)
    return _up(dp), _up(dl)


def score_paths(ext, bank, lbanks, paths, bs=8):
    sp, sl = [], []
    for i in range(0, len(paths), bs):
        mp, ml = both_maps(ext, bank, lbanks, [pc.load_rgb(p) for p in paths[i:i + bs]])
        sp += [float(m.max()) for m in mp]
        sl += [float(m.max()) for m in ml]
        print(f"  scored {min(i + bs, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.array(sp), np.array(sl)


def fuse(sp, sl, tp, tl, pose_ok):
    a = sp / tp
    return np.maximum(a, sl / tl) if pose_ok else a


def cmd_fit2(a):
    paths = pc.list_train(a.root, a.product)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(paths))
    n_hold = max(10, int(HOLDOUT * len(paths)))
    hold = [paths[i] for i in perm[:n_hold]]
    train = [paths[i] for i in perm[n_hold:]]
    print(f"{a.product}: train={len(train)} held-out={len(hold)} device={pc.DEVICE}")

    ext = pc.Extractor()
    bank = pc.build_bank(ext, train)
    lbanks = {k: v for k, v in build_lbanks(ext, train).items()}
    bank_d = bank.to(pc.DEVICE)
    lb_d = {k: v.to(pc.DEVICE) for k, v in lbanks.items()}

    # pose check: held-out normals must look at least as normal when restricted to position
    mp_means, ml_means = [], []
    sp, sl = [], []
    for i in range(0, len(hold), 8):
        mp, ml = both_maps(ext, bank_d, lb_d, [pc.load_rgb(p) for p in hold[i:i + 8]])
        mp_means += [float(m.mean()) for m in mp]
        ml_means += [float(m.mean()) for m in ml]
        sp += [float(m.max()) for m in mp]
        sl += [float(m.max()) for m in ml]
    sp, sl = np.array(sp), np.array(sl)
    ratio = float(np.mean(ml_means) / np.mean(mp_means))
    pose_ok = bool(ratio <= POSE_RATIO_MAX and a.product not in FORCE_OFF)
    tp, tl = float(sp.max()), float(sl.max())
    hold_fused = fuse(sp, sl, tp, tl, pose_ok)

    print(f"pose ratio (logical/physical on held-out normals) = {ratio:.3f}"
          f"  -> logical brain {'ON' if pose_ok else 'OFF'}")
    print(f"thresholds: physical={tp:.3f} logical={tl:.3f}")
    os.makedirs(BANKS, exist_ok=True)
    torch.save({"bank": bank, "lbanks": lbanks, "tp": tp, "tl": tl, "pose_ok": pose_ok,
                "pose_ratio": ratio, "hold_fused": hold_fused}, os.path.join(BANKS, f"{a.product}_m2.pt"))
    print("saved banks/%s_m2.pt" % a.product)


def verdict(f):
    return "FAIL" if f >= 1.0 else ("NEEDS_REVIEW" if f >= REVIEW_LO else "PASS")


def cmd_eval2(a):
    d = torch.load(os.path.join(BANKS, f"{a.product}_m2.pt"), weights_only=False)
    ext = pc.Extractor()
    bank = d["bank"].to(pc.DEVICE)
    lb = {k: v.to(pc.DEVICE) for k, v in d["lbanks"].items()}
    items = pc.list_test(a.root, a.product)
    types = np.array([t for _, t in items])
    sp, sl = score_paths(ext, bank, lb, [p for p, _ in items])
    tp, tl, pose_ok = d["tp"], d["tl"], d["pose_ok"]
    phys = sp / tp
    logi = sl / tl
    fused = fuse(sp, sl, tp, tl, pose_ok)
    y = (types != "good").astype(int)

    print(f"\n{a.product}  (logical brain {'ON' if pose_ok else 'OFF'}, pose ratio {d['pose_ratio']:.3f})")
    print(f"{'defect type':25s} {'physical':>9s} {'logical':>9s} {'fused':>9s}")
    rows = {}
    for t in ["ALL"] + sorted(set(types) - {"good"}):
        sel = np.ones(len(types), bool) if t == "ALL" else (types == "good") | (types == t)
        yy = y[sel]
        r = [float(roc_auc_score(yy, s[sel])) for s in (phys, logi, fused)]
        rows[t] = dict(physical=r[0], logical=r[1], fused=r[2])
        print(f"{t:25s} {r[0]:9.3f} {r[1]:9.3f} {r[2]:9.3f}")

    # decision layer + conformal p-values
    hf = np.sort(d["hold_fused"])
    pvals = np.array([(1 + np.sum(hf >= s)) / (len(hf) + 1) for s in fused])
    vd = np.array([verdict(s) for s in fused])
    print("\nverdicts   PASS  REVIEW  FAIL")
    for name, m in (("good", types == "good"), ("defective", types != "good")):
        print(f"{name:10s} {np.sum(vd[m] == 'PASS'):4d} {np.sum(vd[m] == 'NEEDS_REVIEW'):7d} {np.sum(vd[m] == 'FAIL'):5d}")
    print(f"conformal p-value resolution is only 1/{len(hf) + 1} (small held-out set)")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{a.product}_module2.json"), "w") as f:
        json.dump({"pose_ok": pose_ok, "pose_ratio": d["pose_ratio"], "auroc": rows}, f, indent=2)
    np.savez(os.path.join(OUT, f"{a.product}_m2_scores.npz"), phys=phys, logi=logi, fused=fused,
             types=types, pvals=pvals)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit2", "eval2"])
    ap.add_argument("--product", required=True)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    {"fit2": cmd_fit2, "eval2": cmd_eval2}[a.cmd](a)