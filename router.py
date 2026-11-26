"""Product router: which product is this? UNKNOWN_PRODUCT if never onboarded.
Usage:
    python router.py onboard --products cable transistor pill
    python router.py test
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.covariance import LedoitWolf

import baseline_patchcore as pc

ROOT = "data/mvtec_ad"
BANKS = "banks"
MARGIN = 1.5   # hand-tuned: threshold = MARGIN x worst held-out good distance


def feats(ext, paths):
    return pc.global_feat(ext, [pc.load_rgb(p) for p in paths])


def maha(x, mu, prec):
    d = x - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))


def route(models, f):
    names = list(models)
    sc = np.stack([maha(f, m["mu"], m["prec"]) / m["thr"] for m in models.values()], 1)
    best = sc.argmin(1)
    return [names[b] if sc[i, b] <= 1.0 else "UNKNOWN_PRODUCT" for i, b in enumerate(best)], sc


def cmd_onboard(a):
    ext = pc.Extractor()
    models = {}
    for prod in a.products:
        paths = pc.list_train(a.root, prod)
        perm = np.random.RandomState(0).permutation(len(paths))
        n_hold = max(10, int(0.2 * len(paths)))
        hold = [paths[i] for i in perm[:n_hold]]
        tr = [paths[i] for i in perm[n_hold:]]
        lw = LedoitWolf().fit(feats(ext, tr))
        mu, prec = lw.location_, lw.precision_
        thr = float(maha(feats(ext, hold), mu, prec).max()) * MARGIN
        models[prod] = dict(mu=mu, prec=prec, thr=thr)
        print(f"onboarded {prod}: threshold {thr:.2f}")
    os.makedirs(BANKS, exist_ok=True)
    pickle.dump(models, open(os.path.join(BANKS, "router.pkl"), "wb"))
    print("saved banks/router.pkl")


def cmd_test(a):
    models = pickle.load(open(os.path.join(BANKS, "router.pkl"), "rb"))
    ext = pc.Extractor()
    print(f"onboarded: {list(models)}")
    ok = tot = 0
    for prod in sorted(os.listdir(a.root)):
        if prod.startswith(".") or not os.path.isdir(os.path.join(a.root, prod, "test")):
            continue
        items = pc.list_test(a.root, prod)
        step = max(1, len(items) // 40)
        paths = [p for p, _ in items][::step]
        pred, _ = route(models, feats(ext, paths))
        want = prod if prod in models else "UNKNOWN_PRODUCT"
        n_ok = sum(x == want for x in pred)
        ok += n_ok
        tot += len(pred)
        print(f"{prod:12s} should be {want:16s} correct {n_ok}/{len(pred)}")
    print(f"\noverall routing accuracy: {ok}/{tot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["onboard", "test"])
    ap.add_argument("--products", nargs="+", default=[])
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    {"onboard": cmd_onboard, "test": cmd_test}[a.cmd](a)