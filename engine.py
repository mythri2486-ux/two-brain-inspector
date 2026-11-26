"""Module 1 engine (physical brain only).
    python engine.py fit  --product cable
    python engine.py eval --product cable
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import baseline_patchcore as pc

ROOT = "data/mvtec_ad"
BANKS = "banks"
OUT = "outputs"


def cmd_fit(a):
    paths = pc.list_train(a.root, a.product)
    assert paths, f"No train/good images in {a.root}/{a.product}"
    print(f"{a.product}: {len(paths)} good train images, device={pc.DEVICE}")
    t = time.time()
    ext = pc.Extractor()
    bank = pc.build_bank(ext, paths)
    os.makedirs(BANKS, exist_ok=True)
    torch.save(bank, os.path.join(BANKS, f"{a.product}_bank.pt"))
    print(f"bank {tuple(bank.shape)} saved in {time.time() - t:.0f}s")


def overlay(rgb, m, lo, hi):
    """Returns original and heatmap overlay side by side (RGB uint8)."""
    n = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    heat = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    ov = (0.5 * rgb + 0.5 * heat).astype(np.uint8)
    return np.concatenate([rgb, ov], 1)


def cmd_eval(a):
    bank = torch.load(os.path.join(BANKS, f"{a.product}_bank.pt"), weights_only=False)
    ext = pc.Extractor()
    items = pc.list_test(a.root, a.product)
    print(f"{a.product}: {len(items)} test images")
    maps, scores = [], []
    for i in range(0, len(items), 8):
        rgbs = [pc.load_rgb(p) for p, _ in items[i:i + 8]]
        m = pc.score_maps(ext, bank, rgbs)
        maps.append(m)
        scores += [float(x.max()) for x in m]
        print(f"  scored {min(i + 8, len(items))}/{len(items)}", end="\r")
    print()
    maps = np.concatenate(maps)
    types = np.array([t for _, t in items])
    scores = np.array(scores)
    y = (types != "good").astype(int)

    res = {"product": a.product, "image_auroc": float(roc_auc_score(y, scores)), "per_type": {}}
    print(f"\n{a.product} IMAGE AUROC = {res['image_auroc']:.4f}")
    for t in sorted(set(types) - {"good"}):
        sel = (types == "good") | (types == t)
        auc = float(roc_auc_score((types[sel] != "good").astype(int), scores[sel]))
        res["per_type"][t] = auc
        print(f"  {t:25s} AUROC = {auc:.4f}   (n={int((types == t).sum())})")

    os.makedirs(os.path.join(OUT, "heatmaps"), exist_ok=True)
    with open(os.path.join(OUT, f"{a.product}_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.save(os.path.join(OUT, f"{a.product}_scores.npy"), scores)

    rng = np.random.RandomState(0)
    good_idx = np.where(types == "good")[0][:5]
    bad_idx = rng.choice(np.where(types != "good")[0], 5, replace=False)
    lo, hi = float(maps.min()), float(maps.max())
    for k in list(good_idx) + list(bad_idx):
        rgb = pc.load_rgb(items[k][0])
        img = overlay(rgb, maps[k], lo, hi)
        name = f"{a.product}_{types[k]}_{k}_score{scores[k]:.2f}.png"
        cv2.imwrite(os.path.join(OUT, "heatmaps", name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"\nheatmaps saved in {OUT}/heatmaps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--product", required=True)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    {"fit": cmd_fit, "eval": cmd_eval}[a.cmd](a)