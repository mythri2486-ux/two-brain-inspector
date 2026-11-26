"""Physical brain: PatchCore-style memory bank (trained on GOOD images only)."""
import glob
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

IMG = 256                      # input size (CPU friendly)
BACKBONE = "resnet18"          # try "wide_resnet50_2" later if time allows
CORESET_FRAC = 0.10            # random 10% of train patches kept as memory bank
SIGMA = 4                      # heatmap smoothing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------- data helpers ----------
def _imgs(folder):
    files = []
    for ext in ("png", "jpg", "jpeg", "bmp"):
        files += glob.glob(os.path.join(folder, f"*.{ext}"))
    return sorted(files)


def list_train(root, product):
    return _imgs(os.path.join(root, product, "train", "good"))


def list_test(root, product):
    """Returns [(path, defect_type)]; defect_type == 'good' for normal images."""
    out = []
    test_dir = os.path.join(root, product, "test")
    for d in sorted(os.listdir(test_dir)):
        for p in _imgs(os.path.join(test_dir, d)):
            out.append((p, d))
    return out


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR))


def to_tensor(rgb_uint8):
    x = (rgb_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)


# ---------- feature extractor ----------
class Extractor:
    def __init__(self, name=BACKBONE):
        self.m = timm.create_model(
            name, pretrained=True, features_only=True, out_indices=(2, 3)
        ).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, x):
        f2, f3 = self.m(x.to(DEVICE))            # layer2 (stride 8), layer3 (stride 16)
        f2 = F.avg_pool2d(f2, 3, 1, 1)           # local neighbourhood aggregation
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)            # B, C, h, w


def patches(feat):
    B, C, h, w = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(B * h * w, C)


@torch.no_grad()
def global_feat(ext, rgbs, bs=16):
    """One mean-pooled feature vector per image -> numpy (N, C). Used by router and shift detector."""
    out = []
    for i in range(0, len(rgbs), bs):
        x = torch.stack([to_tensor(r) for r in rgbs[i:i + bs]])
        out.append(ext(x).mean((2, 3)).cpu())
    return torch.cat(out).numpy()


# ---------- fit ----------
def build_bank(ext, paths, batch=16, frac=CORESET_FRAC, seed=0):
    g = torch.Generator().manual_seed(seed)
    kept = []
    for i in range(0, len(paths), batch):
        x = torch.stack([to_tensor(load_rgb(p)) for p in paths[i:i + batch]])
        p = patches(ext(x)).cpu()
        n = max(1, int(len(p) * frac))
        idx = torch.randperm(len(p), generator=g)[:n]
        kept.append(p[idx])
        print(f"  fit {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return torch.cat(kept)


# ---------- score ----------
def min_dist(p, bank, chunk=2048):
    out = []
    for i in range(0, len(p), chunk):
        out.append(torch.cdist(p[i:i + chunk], bank).min(1).values)
    return torch.cat(out)


@torch.no_grad()
def score_maps(ext, bank, rgb_list):
    """rgb_list: list of HxWx3 uint8 arrays -> (N, IMG, IMG) anomaly maps."""
    x = torch.stack([to_tensor(r) for r in rgb_list])
    f = ext(x)
    B, C, h, w = f.shape
    d = min_dist(patches(f), bank.to(DEVICE)).reshape(B, 1, h, w)
    d = F.interpolate(d, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
    maps = d.cpu().numpy()
    return np.stack([gaussian_filter(m, sigma=SIGMA) for m in maps])