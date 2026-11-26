# LSNet Windows copy

This folder is a Windows-ready copy of LSNet.  
The original project is still at `D:\Projects\Repo\lsnet-master` and was not changed for this setup.

## What was changed for Windows

1. `model/ska.py` (and the copies under `detection/` and `segmentation/`): Triton is optional. On Windows/CPU the code uses a PyTorch fallback.
2. `main.py`: Weights & Biases is off unless you pass `--wandb`.
3. `infer.py`: classify one image without needing ImageNet.
4. `requirements-win.txt`: installable pins for Python 3.11 on Windows.

## Setup (once)

```powershell
cd D:\Projects\Repo\focusshift
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-win.txt
```

## Quick demo

```powershell
cd D:\Projects\Repo\focusshift
.\.venv\Scripts\Activate.ps1
python infer.py --image images\56527110-funny-cat-and-dog-picture.jpg
```

`infer.py` downloads the LSNet-T weights the first time.

## PyCharm

1. Open `D:\Projects\Repo\focusshift`
2. Interpreter: `D:\Projects\Repo\focusshift\.venv\Scripts\python.exe`
3. Run `infer.py` with parameters:  
   `--image images/56527110-funny-cat-and-dog-picture.jpg`

## Notes

- This copy uses CPU PyTorch unless you install a CUDA build yourself.
- Full ImageNet training/eval still needs the ImageNet folder and `--data-path`.
- For paper-matching GPU Triton speed, use Linux. This Windows copy is for running and testing locally.
