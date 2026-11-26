# FocusShift

**FocusShift: Adaptive Local Focus for Efficient Vision**

Improves LSNet-style “see large, focus small” convolution with a learned soft focus map (`FocusShiftConv`) for sparse Clock Drawing Test (CDT) picture classification.

## Features

- `FocusShiftConv`: blend dynamic LS path with a cheap static depthwise path via \(f=\sigma(\mathrm{FocusHead}(x))\)
- Micro models (~3.7M) for fair lightweight comparison with LSNet-T (~11M)
- Named-device batch-1 / batch-8 latency protocol
- Sketch-aware augmentation (no horizontal flip)

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Train / evaluate FocusShift vs LSNet on CDT |
| `bench_latency.py` | Batch-1 / batch-8 latency |
| `model/focusshift.py` | `FocusShiftConv`, `focusshift_t`, `focusshift_micro`, `lsnet_micro` |
| `model/lsnet.py` | LSNet baseline blocks |
| `custom_data/cdt_labels.xlsx` | Labels + train/val split (regenerate if paths differ) |
| `prepare_cdt_excel.py` | Rebuild Excel labels from `concetto_CDT/{0,4,5}/` |
| `requirements.txt` / `requirements-win.txt` | Dependencies |
| `pretrain/` | Place ImageNet `lsnet_t.pth` here (not committed) |
| `concetto_CDT/` | Local CDT images (not committed by default) |

Paper / Overleaf package: `D:\Projects\Repo\Masters_Project\FocusShift\`

## Setup (Windows)

```cmd
cd /d D:\Projects\Repo\focusshift
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements-win.txt
```

Place CDT images under `concetto_CDT\0`, `concetto_CDT\4`, `concetto_CDT\5`, then:

```cmd
python prepare_cdt_excel.py
```

Place pretrained LSNet-T weights at `pretrain\lsnet_t.pth` if using `--pretrained`.

## Train (sketch aug)

```cmd
python -u main.py --models focusshift_t --epochs 40 --batch-size 8 --lr 8e-5 --pretrained --label-smoothing 0.1 --patience 15 --aug sketch --out-dir custom_data/gap_results_focusshift_sketch40
```

## Latency

```cmd
python -u bench_latency.py --models lsnet_t focusshift_t lsnet_micro focusshift_micro --batch-sizes 1 8 --out custom_data/latency_results.json
```

## GitHub notes

- `.gitignore` excludes `.venv/`, `concetto_CDT/`, `pretrain/`, and training checkpoints.
- Prefer a **private** repository if unpublished results or clinical materials are involved.
- Do not commit absolute machine-specific paths; regenerate `cdt_labels.xlsx` on a new machine.
