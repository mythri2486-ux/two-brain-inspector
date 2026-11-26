import numpy as np

d = np.load("outputs/cable_m2_scores.npz")
f, t = d["fused"], d["types"]
g = t == "good"
print("good fused scores: median %.2f  90th pct %.2f  max %.2f" % (np.median(f[g]), np.percentile(f[g], 90), f[g].max()))
print("lo     good (PASS, REVIEW, FAIL)     defective (PASS, REVIEW, FAIL)")
for lo in [0.80, 0.85, 0.90, 0.95, 1.00]:
    def c(m):
        return (int(np.sum(f[m] < lo)), int(np.sum((f[m] >= lo) & (f[m] < 1))), int(np.sum(f[m] >= 1)))
    print(f"{lo:.2f}   {c(g)}     {c(~g)}")