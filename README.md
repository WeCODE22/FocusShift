# FocusShift

**Adaptive local focus for efficient vision on Clock Drawing Test (CDT) sketches.**

FocusShift builds on LSNet’s “see large, focus small” design and adds a **learned soft focus map** so the expensive dynamic mix is applied where the ink is—not on empty paper.

| | |
|---|---|
| **Author** | Solomon Panganai Mupona |
| **Affiliation** | Zhejiang University of Science and Technology |
| **Task** | 3-class CDT picture classification `{0, 4, 5}` (Shulman-style) |
| **Prior work** | LSNet — Wang et al., CVPR 2025 (“See Large, Focus Small”) |

> Paper / Overleaf package is maintained separately (not in this repo).

---

## Why FocusShift?

CDT pages are **sparse**: most pixels are blank paper; score-relevant evidence is the circle, digits, and hands. Fixed LS convolution still runs the same dynamic schedule everywhere. FocusShiftConv predicts a per-location gate \(f\) and blends:

\[
y = f \odot d + (1-f) \odot s + x
\]

- \(d\) — dynamic LS path (LKP + SKA)  
- \(s\) — cheap static depthwise \(3\times3\)  
- \(x\) — residual  

Also included: **Micro** models (~3.7M) for fair tiny-size comparisons, and a **batch-1 / batch-8 latency** protocol on a named device.

---

## Models

| Model | Adaptive focus | Params (approx.) | Role |
|-------|----------------|------------------|------|
| `lsnet_t` | No | 11.03M | Fixed-focus baseline |
| `focusshift_t` | Yes | 11.11M | Proposed (matched size) |
| `lsnet_micro` | No | 3.67M | Fair small baseline |
| `focusshift_micro` | Yes | 3.69M | Focus + fair size |

Core implementation: [`model/focusshift.py`](model/focusshift.py) (`FocusShiftConv`).

---

## Repository layout

```
focusshift/
├── main.py                 # Train / evaluate
├── bench_latency.py        # Batch-1 / batch-8 latency
├── prepare_cdt_excel.py    # Build labels from image folders
├── transfer_utils.py       # ImageNet weight transfer helpers
├── model/
│   ├── focusshift.py       # FocusShiftConv + T / Micro constructors
│   ├── lsnet.py            # LSNet blocks
│   └── ska.py              # Small-kernel aggregation
├── custom_data/
│   └── cdt_labels.xlsx     # Labels + train/val split (portable paths)
├── concetto_CDT/           # Local images only (gitignored)
│   ├── 0/  4/  5/
└── pretrain/               # Place lsnet_t.pth here (gitignored)
```

---

## Setup

**Requirements:** Python 3.10+ (3.11 recommended on Windows), PyTorch.

```bash
git clone https://github.com/WeCODE22/FocusShift.git
cd FocusShift

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements-win.txt
# or: pip install -r requirements.txt
```

### Dataset (not shipped in this repo)

1. Place CDT PNGs under:
   - `concetto_CDT/0/`
   - `concetto_CDT/4/`
   - `concetto_CDT/5/`
2. Build the Excel split:

```bash
python prepare_cdt_excel.py
```

This writes `custom_data/cdt_labels.xlsx` with **repo-relative** paths.

### Pretrained weights (optional)

For `--pretrained` fine-tuning of T-models, put ImageNet LSNet-T weights at:

```
pretrain/lsnet_t.pth
```

---

## Training

### Full-split FocusShift-T (sketch-aware aug)

```bash
python -u main.py \
  --models focusshift_t \
  --epochs 40 \
  --batch-size 8 \
  --lr 8e-5 \
  --pretrained \
  --label-smoothing 0.1 \
  --patience 15 \
  --aug sketch \
  --out-dir custom_data/gap_results_focusshift_sketch40
```

`--aug sketch` uses mild crop + small affine and **does not flip** (clock digit order / hands are orientation-sensitive).

### Matched subset (fair head-to-head)

Use a stratified subset and the shorter schedule described in the paper; compare `lsnet_t` vs `focusshift_t` with shared ImageNet init, and Micro models from scratch.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--models` | One or more of `lsnet_t`, `focusshift_t`, `lsnet_micro`, `focusshift_micro` |
| `--aug sketch\|default` | Sketch-aware vs photo-style augmentation |
| `--patience N` | Early stopping; `0` disables |
| `--pretrained` | Load ImageNet LSNet-T-compatible weights (T models) |

---

## Latency benchmark

```bash
python -u bench_latency.py \
  --models lsnet_t focusshift_t lsnet_micro focusshift_micro \
  --batch-sizes 1 8 \
  --warmup 5 \
  --iters 20 \
  --out custom_data/latency_results.json
```

Reports wall-clock ms and images/s after warm-up (batch sizes 1 and 8).

---

## Quick sanity check

```bash
python -c "from timm.models import create_model; import model.build; import torch; m=create_model('focusshift_t', num_classes=3).eval(); y=m(torch.randn(1,3,224,224)); print('OK', tuple(y.shape))"
```

Expected: `OK (1, 3)`

---

## What is not in git

| Path | Reason |
|------|--------|
| `concetto_CDT/` | Clinical / dataset images — keep local |
| `pretrain/` | Large weight files |
| `.venv/` | Virtual environment |
| `custom_data/gap_results*/` | Training runs & checkpoints |

---

## Citation / stance

This codebase implements **FocusShift** as an improvement over fixed-focus LS convolution for sparse CDT classification. Models are research artifacts for image classification—they are **not** diagnostic devices for Alzheimer’s disease.

---

## License

Use consistent with your institution’s rules and the license of any upstream LSNet / third-party weights you download.
