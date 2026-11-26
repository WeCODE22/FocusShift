# FocusShift (CDT solution)

**FocusShift: Adaptive Local Focus for Efficient Vision**

Movable local focus (`FocusShiftConv`), micro models, and batch-1 latency reporting.
Baseline comparison uses fixed-focus **LSNet**.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Train / evaluate FocusShift on CDT |
| `concetto_CDT/` | Dataset images (classes `0`, `4`, `5`) |
| `custom_data/cdt_labels.xlsx` | Image paths + labels + train/val split |
| `model/focusshift.py` | `focusshift_*` + `lsnet_micro` |
| `model/lsnet.py` | Baseline LSNet |
| `pretrain/lsnet_t.pth` | ImageNet LSNet-T weights (warm-start) |
| `bench_latency.py` | Batch-1 / small-batch latency |

## Setup

```cmd
cd /d D:\Projects\Repo\focusshift
.\.venv\Scripts\activate.bat
```

## Best training command (sketch aug)

```cmd
python -u main.py --models focusshift_t --epochs 40 --batch-size 8 --lr 8e-5 --pretrained --label-smoothing 0.1 --patience 8 --aug sketch --out-dir custom_data/gap_results_focusshift_sketch40
```

Current best checkpoint: `custom_data/gap_results_focusshift_sketch40/focusshift_t_best.pth` (~79.1% val).

## Rebuild labels from folders (optional)

```cmd
python prepare_cdt_excel.py
```

## Latency

```cmd
python -u bench_latency.py --models focusshift_t --batch-sizes 1 8 --out custom_data/latency_results.json
```
