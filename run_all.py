import subprocess, sys

for p in ["transistor", "pill", "screw", "tile"]:
    for c in ["fit2", "eval2"]:
        print(f"\n===== {c} {p} =====")
        subprocess.run([sys.executable, "module2.py", c, "--product", p])