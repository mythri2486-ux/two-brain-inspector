import os, glob, argparse
import numpy as np, torch
from PIL import Image
import baseline_patchcore as pc, module2 as m2
import guard

ROOT = r"data\mvtec_ad"


def load(prod):
    d = torch.load(f"banks/{prod}_m2.pt", weights_only=False)
    d["bank"] = d["bank"].to(pc.DEVICE)
    d["lbanks"] = {k: v.to(pc.DEVICE) for k, v in d["lbanks"].items()}
    return d


def score(ext, d, rgb):
    mp, ml = m2.both_maps(ext, d["bank"], d["lbanks"], [rgb])
    ph, lg = mp[0] / d["tp"], ml[0] / d["tl"]
    fmap = np.maximum(ph, lg) if d["pose_ok"] else ph
    return float(fmap.max())


def run(ext, prod, nmax):
    d = load(prod)
    tdir = os.path.join(ROOT, prod, "test")
    good, bad = [], []
    for fo in sorted(os.listdir(tdir)):
        files = sorted(glob.glob(os.path.join(tdir, fo, "*.png")))[:nmax]
        for f in files:
            img = Image.open(f).convert("RGB")
            rgb = np.array(img.resize((pc.IMG, pc.IMG), Image.BILINEAR))
            (good if fo == "good" else bad).append(score(ext, d, rgb))
    g, b = np.array(good), np.array(bad)
    print(f"\n== {prod}: {len(g)} good, {len(b)} defect images")
    print(f"good scored FAIL (false alarms): {int((g >= 1.0).sum())}")
    print(f"defects scored below FAIL: {int((b < 1.0).sum())}")
    print(" REVIEW_LO | good in REVIEW | defects missed (PASS)")
    best = None
    for lo in np.arange(0.50, 0.951, 0.05):
        gr = int(((g >= lo) & (g < 1.0)).sum())
        bp = int((b < lo).sum())
        miss = bp / max(len(b), 1)
        print(f"   {lo:.2f}    |  {gr:3d} ({100*gr/max(len(g),1):4.0f}%)  |  {bp:3d} ({100*miss:4.0f}%)")
        if miss <= 0.05:
            best = lo
    if best is None:
        best = 0.50
    guard.save_lo(prod, best)
    print(f"-> saved REVIEW_LO = {best:.2f} (largest value with 5% or fewer defects missed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--product")
    ap.add_argument("--max", type=int, default=60)
    a = ap.parse_args()
    if a.product:
        prods = [a.product]
    else:
        prods = sorted(f.split("_m2")[0] for f in os.listdir("banks")
                       if f.endswith("_m2.pt"))
    ext = pc.Extractor()
    for p in prods:
        run(ext, p, a.max)