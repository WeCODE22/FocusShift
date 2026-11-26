# FocusShift

**Adaptive local focus for efficient vision on Clock Drawing Test (CDT) sketches.**

FocusShift builds on [LSNet](https://github.com/THU-MIG/lsnet) (“see large, focus small”, Wang et al., CVPR 2025) and adds a **learned soft focus map** so the expensive dynamic mix runs where the ink is—not on empty paper.

| | |
|---|---|
| **Author** | Solomon Panganai Mupona |
| **Affiliation** | Zhejiang University of Science and Technology |
| **Task** | 3-class CDT picture classification `{0, 4, 5}` (Shulman-style) |
| **Code** | https://github.com/WeCODE22/FocusShift |
| **Paper** | Overleaf / local TeX package (not in this repo) |

---

## Headline results (paper)

| Protocol | Model | Params | Val. acc. |
|----------|-------|--------|-----------|
| Matched subset (640 / 160, 6 epochs) | LSNet-T | 11.03M | 39.38% |
| Matched subset (640 / 160, 6 epochs) | **FocusShift-T** | 11.11M | **45.62%** (+6.2 pp) |
| Full split | FocusShift-T | 11.11M | **77.65%** (best @ epoch 9) |

Micro models (~3.7M) support fair tiny-size comparisons. Latency is measured with a **batch-1 / batch-8** protocol on a named CPU device (`bench_latency.py`).

> These models are research artifacts for image classification—**not** diagnostic devices.

---

## Why FocusShift?

CDT pages are **sparse**: most pixels are blank paper; score-relevant evidence is the circle, digits, and hands. Fixed LS convolution still runs the same dynamic schedule everywhere. FocusShiftConv predicts a per-location gate \(f\) and blends:

\[
y = f \odot d + (1-f) \odot s + x
\]

| Symbol | Meaning |
|--------|---------|
| \(d\) | Dynamic LS path (LKP + SKA) |
| \(s\) | Cheap static depthwise \(3\times3\) |
| \(x\) | Residual |
| \(f\) | Soft spatial focus map in \([0,1]\) |

Core implementation: [`model/focusshift.py`](model/focusshift.py) (`FocusShiftConv`).

---

## Models

| Model | Adaptive focus | Params (approx.) | Role |
|-------|----------------|------------------|------|
| `lsnet_t` | No | 11.03M | Fixed-focus baseline |
| `focusshift_t` | Yes | 11.11M | Proposed (matched size) |
| `lsnet_micro` | No | 3.67M | Fair small baseline |
| `focusshift_micro` | Yes | 3.69M | Focus + fair size |

---

## Repository layout

```
FocusShift/
├── main.py                 # Train / evaluate
├── bench_latency.py        # Batch-1 / batch-8 latency
├── prepare_cdt_excel.py    # Build labels from image folders
├── transfer_utils.py       # ImageNet weight transfer helpers
├── requirements.txt
├── requirements-win.txt    # Windows-friendly pins
├── model/
│   ├── focusshift.py       # FocusShiftConv + T / Micro constructors
│   ├── lsnet.py            # LSNet blocks
│   ├── ska.py              # Small-kernel aggregation (Triton optional)
│   └── build.py            # timm registration
├── custom_data/
│   └── cdt_labels.xlsx     # Labels + train/val split (portable paths)
├── concetto_CDT/           # Local images only (gitignored)
│   ├── 0/  4/  5/
└── pretrain/               # Place lsnet_t.pth here (gitignored)
```

---

## Setup

**Requirements:** Python 3.10+ (3.11 recommended on Windows), PyTorch, `timm`.

```bash
git clone https://github.com/WeCODE22/FocusShift.git
cd FocusShift

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -U pip
pip install -r requirements-win.txt   # Windows
# pip install -r requirements.txt     # Linux / macOS
```

**Windows notes:** `model/ska.py` falls back to a PyTorch path when Triton is unavailable. Training and latency in the paper protocols are CPU-friendly; install a CUDA PyTorch build yourself if you want GPU.

### Dataset (not shipped)

1. Place CDT PNGs under `concetto_CDT/0/`, `concetto_CDT/4/`, `concetto_CDT/5/`.
2. Build the Excel split (writes **repo-relative** paths):

```bash
python prepare_cdt_excel.py
```

### Pretrained weights (optional)

For `--pretrained` fine-tuning of T-models, put ImageNet LSNet-T weights at:

```
pretrain/lsnet_t.pth
```

---

## Training

Training always reports validation accuracy each epoch and saves `{model}_best.pth` under `--out-dir`.

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

Paper headline **77.65%** used a related full-split schedule with early stopping (best at epoch 9); keep `--patience` enabled if you want that behaviour.

### Matched subset (fair T head-to-head)

Stratified **640 train / 160 val**, short schedule (6 epochs), shared ImageNet init for T models; Micro from scratch. Use `--aug default` for the photo-style matched protocol described in the paper, and compare:

```bash
python -u main.py \
  --models lsnet_t focusshift_t \
  --epochs 6 \
  --batch-size 8 \
  --lr 8e-5 \
  --pretrained \
  --label-smoothing 0.1 \
  --patience 0 \
  --aug default \
  --out-dir custom_data/gap_results_matched_t
```

(Use your stratified subset labels file via `--labels` if it is not the full `cdt_labels.xlsx`.)

### Useful flags

| Flag | Meaning |
|------|---------|
| `--models` | One or more of `lsnet_t`, `focusshift_t`, `lsnet_micro`, `focusshift_micro` |
| `--aug sketch\|default` | Sketch-aware (no flip) vs photo-style augmentation |
| `--patience N` | Early stopping; `0` disables |
| `--pretrained` | Load ImageNet LSNet-T-compatible weights (T models) |
| `--class-weight` | Inverse-frequency class weights |
| `--labels` | Path to labels Excel (default: `custom_data/cdt_labels.xlsx`) |
| `--out-dir` | Checkpoints + `summary.json` |

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
| `*.pth` | Model weights |

---

## Citation / stance

This codebase implements **FocusShift** as an improvement over fixed-focus LS convolution for sparse CDT classification. Please also cite LSNet (Wang et al., CVPR 2025) when comparing against the baseline.

---

## License

Use consistent with your institution’s rules and the license of any upstream LSNet / third-party weights you download.
