# Two-Brain Inspector

**Live demo:** PASTE-YOUR-ONRENDER-LINK-HERE

A factory defect-detection app that learns from **good parts only**. It never sees a defect during training, yet it can flag scratches, cracks, contamination and misplaced parts. Built with PyTorch and Streamlit, and tested on the public [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

> Author: Mythri JM, Department of AI & ML, PES Institute of Technology & Management, Shivamogga

---

## Why this project

Real factories have very few defect samples, so a normal classifier is impractical and misses defect types it has never seen. Two-Brain Inspector instead learns what a *normal* part looks like and flags anything that does not match (unsupervised anomaly detection).

## Key features

- **Two brains**
  - *Physical brain*: a PatchCore-style memory bank of patch features. Finds local damage such as scratches, dents, cracks and contamination.
  - *Logical brain*: a separate memory bank for each cell of a 4x4 grid, so it knows what should appear where. Finds missing or misplaced parts.
  - The fused score is the larger of the two maps.
- **Self-disabling logical brain**: a pose-ratio check on held-out good images switches it off for products where it does not help.
- **Calibrated threshold**: 20% of the good images are held out. A fused score of 1.00 (FAIL) means the part is worse than every held-out good part. No defect data is needed.
- **Conformal p-value**: for example, "only 2.4% of good parts score this high".
- **Shift guard**: if lighting or camera conditions drift away from the training images, a FAIL is downgraded to REVIEW instead of raising a false alarm.
- **Three-level verdict**: PASS, REVIEW, FAIL, with an adjustable REVIEW level.
- **Operator tools**: heatmap, defect zone, severity, defect zoom, localization check (HIT or MISS and IoU), repair advice, downloadable report and heatmap, session log and alert center.
- **Conveyor scan**: simulates parts rolling past the scanner, with live counts of defects caught and good parts flagged.
- **AI assistant**: explains the live result ("why this verdict?", "what should I do next?"). Works offline with built-in rules, or with Claude if you enter an API key.
- **Interactive 3D part viewer** with a grid, and a dark digital-twin inspection-bay background.

## How it works

1. **Data**: only `train/good` images of each MVTec AD product are used for training.
2. **Split**: 80% of the good images build the memory. 20% (at least 10 images) are held out for calibration.
3. **Features**: each image goes through a frozen pretrained network, and patch features are collected.
4. **Physical bank**: patch features from the training images.
5. **Logical banks**: 16 smaller banks, one per grid cell, keeping 20% of the patches in each cell.
6. **Pose check**: on the held-out images the logical and physical scores are compared. The logical brain stays ON only if the ratio is 1.10 or lower.
7. **Calibration**: the highest held-out scores set the thresholds, and the held-out scores are saved for the p-value.
8. **Save**: everything goes to `banks/<product>_m2.pt`.

There is no gradient training, so fitting takes minutes rather than hours. Test images (including defects) are used only for evaluation.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app (Inspect one part, Conveyor scan, AI assistant) |
| `baseline_patchcore.py` | Feature extractor and physical brain |
| `module2.py` | Logical brain, pose check, calibration, `fit2` and `eval2` commands |
| `guard.py` | PASS, REVIEW and FAIL decision and saved REVIEW level |
| `shift_detector.py` | Lighting and camera shift check |
| `kb.py` | Defect knowledge base and repair advice |
| `run_all.py` | Fits and evaluates several products in a row |
| `style.css`, `script.js` | Extra styling and UI effects loaded by `app.py` |
| `banks/` | Fitted memory banks (not in the repo) |
| `data/mvtec_ad/` | Dataset (not in the repo) |

## Setup

**1. Clone and create an environment**

```powershell
git clone https://github.com/mythri2486-ux/two-brain-inspector.git
cd two-brain-inspector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get the dataset**

Download MVTec AD from the link above and extract it so the layout is:

```
data/mvtec_ad/<product>/train/good/*.png
data/mvtec_ad/<product>/test/<defect_type>/*.png
data/mvtec_ad/<product>/ground_truth/<defect_type>/*_mask.png
```

**3. Fit a product** (for example `bottle`)

```powershell
python module2.py fit2 --product bottle
python shift_detector.py fit
```

Optional: `python module2.py eval2 --product bottle` prints AUROC per defect type.

**4. Run the app**

```powershell
python -m streamlit run app.py --server.port 8510
```

Open http://localhost:8510. Every `banks/<product>_m2.pt` file appears in the Product dropdown.

## Using the app

1. Choose a **Product** in the sidebar.
2. Choose **Dataset test image** (pick a folder such as `good`, `broken_large` or `contamination`) or **Upload image**.
3. Read the verdict, heatmap, defect zone and report. Drag the 3D viewer to turn the part.
4. Try the **Conveyor scan** view to see defects caught against false alarms.
5. Open the **AI assistant** view and ask about the result.

### Optional: Claude for the AI assistant

Without a key, the assistant answers from built-in rules. To use Claude, enter your key in the sidebar, or set it before starting:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
```

Never commit an API key to the repository.

## Limitations

- **Product-specific**: each product needs its own bank. A random photo from the internet looks different from the training images and will be scored as a defect.
- **Fixed camera setup**: new photos should match the training angle, background and lighting.
- **Evaluated on MVTec AD**, not on a live production line.
- **Memory**: loading banks needs several GB of free RAM. Close other apps on a small laptop.

## Tech stack

PyTorch, NumPy, SciPy, scikit-learn, OpenCV, Pillow, Matplotlib, Pandas, Streamlit, and the optional Anthropic API. A pretrained backbone is downloaded on first use (see `baseline_patchcore.py`).

## Acknowledgements

- MVTec AD dataset by MVTec Software GmbH.
- PatchCore: *Towards Total Recall in Industrial Anomaly Detection* (Roth et al., 2022), which inspired the physical brain.

## License

Add a license of your choice here (for example MIT). Note that the MVTec AD dataset has its own license (CC BY-NC-SA 4.0).

