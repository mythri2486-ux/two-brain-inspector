import os

ROOT = "data/mvtec_ad"

for name in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, name)
    if name.startswith(".") or not os.path.isdir(os.path.join(p, "train", "good")):
        continue
    ntrain = len(os.listdir(os.path.join(p, "train", "good")))
    test_dir = os.path.join(p, "test")
    print(f"{name}: train/good = {ntrain}")
    for d in sorted(os.listdir(test_dir)):
        n = len(os.listdir(os.path.join(test_dir, d)))
        print(f"   test/{d}: {n}")