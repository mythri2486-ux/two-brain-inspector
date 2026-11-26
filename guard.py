import os, json

LO_FILE = os.path.join("banks", "review_lo.json")


def _read():
    try:
        with open(LO_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def load_lo(prod, default=0.90):
    return float(_read().get(prod, default))


def save_lo(prod, lo):
    data = _read()
    data[prod] = float(lo)
    os.makedirs("banks", exist_ok=True)
    with open(LO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def decide(score, lo, shifted):
    """Returns (verdict, note). Shift downgrades FAIL to REVIEW."""
    if score >= 1.0:
        v = "FAIL"
    elif score >= lo:
        v = "REVIEW"
    else:
        v = "PASS"
    note = ""
    if shifted and v == "FAIL":
        v = "REVIEW"
        note = ("FAIL downgraded to REVIEW: the image looks different from "
                "training (lighting or camera shift), so a person should check.")
    elif shifted and v == "PASS":
        note = "PASS may be less reliable: the image is outside normal conditions."
    return v, note