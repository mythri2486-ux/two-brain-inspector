"""Rule-based knowledge base: defect kind -> likely process stage + repair.
NOT learned. A hand-written table. Say so honestly in the demo."""
G = 4
ROWS = ["top", "upper-middle", "lower-middle", "bottom"]
COLS = ["left", "centre-left", "centre-right", "right"]

RULES = {
    "cable": {
        "physical": ("Insulation stripping / extrusion / handling",
                     "Inspect the flagged region, re-strip or replace the damaged segment, check the stripping blade."),
        "logical": ("Wire assembly / colour-order routing",
                    "Check wire order and positions against the reference, re-route wires, verify the cable count."),
    },
    "transistor": {
        "physical": ("Lead forming / handling",
                     "Inspect leads and body for damage, replace the part, check the handling fixture."),
        "logical": ("Pick-and-place / mounting",
                    "Check the part is present and seated, recalibrate the placement head, re-mount."),
    },
    "metal_nut": {
        "physical": ("Stamping / surface finishing",
                     "Inspect for scratches or contamination, check the die and cleaning stage."),
        "logical": ("Feeding / orientation",
                    "Check the nut is not flipped and the feeder is aligned."),
    },
}

def cell_name(y, x, size):
    return f"{ROWS[min(G-1, y * G // size)]}, {COLS[min(G-1, x * G // size)]}"

def advise(product, kind, y, x, size):
    stage, fix = RULES.get(product, RULES["cable"])[kind]
    return f"location: {cell_name(y, x, size)}\nlikely stage: {stage}\nsuggested action: {fix}"