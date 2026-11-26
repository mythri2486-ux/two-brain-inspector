import os, sys, glob, json, argparse
import numpy as np
from PIL import Image
import baseline_patchcore as pc

ROOT = r"data\mvtec_ad"
NAMES = ["brightness", "contrast", "red", "green", "blue", "sharpness"]
_cache = {}


def stats(rgb):
    x = rgb.astype(np.float64)
    g = x.mean(axis=2)
    gy, gx = np.gradient(g)
    sharp = np.sqrt(gx ** 2 + gy ** 2).mean()
    return np.array([g.mean(), g.std(), x[..., 0].mean(),
                     x[..., 1].mean(), x[..., 2].mean(), sharp])


def read(path):
    img = Image.open(path).convert("RGB")
    return np.array(img.resize((pc.IMG, pc.IMG), Image.BILINEAR))


def fit(prod):
    files = sorted(glob.glob(os.path.join(ROOT, prod, "train", "good", "*.png")))
    if not files:
        print("no training images for", prod)
        return
    S = np.array([stats(read(f)) for f in files])
    prof = {"mean": S.mean(0).tolist(), "std": S.std(0).tolist(), "n": len(files)}
    os.makedirs("banks", exist_ok=True)
    with open(os.path.join("banks", f"{prod}_shift.json"), "w") as f:
        json.dump(prof, f)
    print(prod, "profile saved from", len(files), "images")


def load_profile(prod):
    if prod in _cache:
        return _cache[prod]
    p = os.path.join("banks", f"{prod}_shift.json")
    prof = None
    if os.path.exists(p):
        with open(p) as f:
            prof = json.load(f)
    _cache[prod] = prof
    return prof


def check(rgb, prod, z_thr=4.0):
    prof = load_profile(prod)
    if prof is None:
        return dict(ok=False, flag=False, zmax=0.0, worst="", z=None)
    mean = np.array(prof["mean"])
    std = np.maximum(np.array(prof["std"]), 0.03 * np.abs(mean) + 1e-6)
    z = (stats(rgb) - mean) / std
    i = int(np.abs(z).argmax())
    zmax = float(abs(z[i]))
    return dict(ok=True, flag=zmax > z_thr, zmax=zmax, worst=NAMES[i], z=z)


def test(prod, n=30):
    gfiles = sorted(glob.glob(os.path.join(ROOT, prod, "test", "good", "*.png")))[:n]
    bfiles = []
    tdir = os.path.join(ROOT, prod, "test")
    for fo in sorted(os.listdir(tdir)):
        if fo != "good":
            bfiles += sorted(glob.glob(os.path.join(tdir, fo, "*.png")))[:5]

    def rate(files, f=1.0):
        c = 0
        for p in files:
            rgb = np.clip(read(p).astype(np.float32) * f, 0, 255).astype(np.uint8)
            c += int(check(rgb, prod)["flag"])
        return 100.0 * c / max(len(files), 1)

    print(f"== {prod}")
    print(f"  good images flagged (normal light): {rate(gfiles):.0f}%  (want near 0)")
    print(f"  defect images flagged (normal light): {rate(bfiles):.0f}%  (want low)")
    print(f"  good images flagged at brightness x0.6: {rate(gfiles, 0.6):.0f}%  (want high)")
    print(f"  good images flagged at brightness x1.4: {rate(gfiles, 1.4):.0f}%  (want high)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "test"])
    ap.add_argument("--product")
    a = ap.parse_args()
    if a.product:
        prods = [a.product]
    else:
        prods = sorted(f for f in os.listdir(ROOT)
                       if os.path.isdir(os.path.join(ROOT, f, "train", "good")))
    for p in prods:
        fit(p) if a.cmd == "fit" else test(p)