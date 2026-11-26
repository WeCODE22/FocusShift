"""Load ImageNet LSNet-T weights into lsnet_t / focusshift_t (partial for FocusShift)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pretrain" / "lsnet_t.pth"
WEIGHTS_URL = "https://huggingface.co/jameslahm/lsnet/resolve/main/lsnet_t.pth"


def ensure_lsnet_t_weights() -> Path:
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        print(f"Downloading {WEIGHTS.name}")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    return WEIGHTS


def load_compatible(model: torch.nn.Module, num_classes: int, verbose: bool = True):
    """Load pretrained backbone; reshape / skip classifier and new focus/static keys."""
    path = ensure_lsnet_t_weights()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}

    model_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in state.items():
        if k not in model_state:
            skipped.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped.append(k)
            continue
        model_state[k] = v
        loaded.append(k)
    model.load_state_dict(model_state)
    if verbose:
        print(f"Loaded {len(loaded)} tensors from {path.name}; skipped {len(skipped)}")
    return model
