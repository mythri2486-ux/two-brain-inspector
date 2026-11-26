import glob, json, os

rows = []
for p in sorted(glob.glob("outputs/*_module2.json")):
    d = json.load(open(p))
    name = os.path.basename(p).replace("_module2.json", "")
    a = d["auroc"]["ALL"]
    rows.append((name, "ON" if d["pose_ok"] else "OFF", d["pose_ratio"], a["physical"], a["logical"], a["fused"]))

print(f"{'product':12s} {'logical':8s} {'pose':>6s} {'physical':>9s} {'logical':>8s} {'fused':>7s}")
for r in rows:
    print(f"{r[0]:12s} {r[1]:8s} {r[2]:6.3f} {r[3]:9.3f} {r[4]:8.3f} {r[5]:7.3f}")