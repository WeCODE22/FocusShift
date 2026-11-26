"""Train/eval LSNet vs FocusShift on CDT.

Compares:
  - lsnet_t           (baseline, fixed focus)
  - focusshift_t      (same size, movable focus)
  - lsnet_micro       (compact plain LSNet)
  - focusshift_micro  (compact + movable focus)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from openpyxl import load_workbook
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm.models import create_model

import model.build  # noqa: F401
from transfer_utils import load_compatible

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "concetto_CDT"  # CDT images colocated with this solution
LABELS = ROOT / "custom_data" / "cdt_labels.xlsx"  # paths + labels + train/val split
OUT_DIR = ROOT / "custom_data" / "gap_results"


class ExcelImageDataset(Dataset):
    def __init__(self, rows, class_to_idx, train=True, aug="default"):
        self.rows = rows
        self.class_to_idx = class_to_idx
        # CDT sketches: avoid flip/strong crop (digits/hands are orientation-sensitive).
        if train and aug == "sketch":
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif train:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        img = Image.open(path).convert("RGB")
        return self.tf(img), self.class_to_idx[label]


def load_rows(path: Path):
    """Expect columns: image_name, class_id, class_name, image_path, split."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        if not row or row[1] is None or row[3] is None:
            continue
        raw = Path(str(row[3]))
        image_path = raw if raw.is_absolute() else (ROOT / raw)
        class_id = int(row[1])
        split = str(row[4]).lower() if row[4] is not None else "train"
        rows.append((str(image_path), class_id, split))
    wb.close()
    return rows


def split_rows(rows, seed=42, val_ratio=0.2):
    # Prefer the Excel split when present; otherwise stratified random split.
    has_split = any(r[2] in ("train", "val") for r in rows)
    if has_split:
        train = [(p, c) for p, c, s in rows if s == "train"]
        val = [(p, c) for p, c, s in rows if s == "val"]
        if train and val:
            return train, val

    plain = [(p, c) for p, c, _ in rows]
    rng = random.Random(seed)
    by_cls = {}
    for r in plain:
        by_cls.setdefault(r[1], []).append(r)
    train, val = [], []
    for items in by_cls.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def train_one(model_name, train_loader, val_loader, device, epochs, lr, num_classes, out_dir,
              pretrained=False, label_smoothing=0.05, class_weights=None, patience=5):
    model = create_model(model_name, num_classes=num_classes, distillation=False)
    model = model.to(device)
    if pretrained and model_name in ("lsnet_t", "focusshift_t"):
        load_compatible(model, num_classes=num_classes)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    history = []
    best_acc = -1.0
    best_path = out_dir / f"{model_name}_best.pth"
    bad_epochs = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = n = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * labels.size(0)
            n += labels.size(0)
        sched.step()
        train_loss = loss_sum / max(n, 1)
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"[{model_name}] epoch {epoch}/{epochs}  loss={train_loss:.4f}  val={val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                stopped_epoch = epoch
                print(f"[{model_name}] early stop at epoch {epoch} (patience={patience})")
                break

    return {
        "model": model_name,
        "params_m": round(params_m, 3),
        "best_val_acc": round(best_acc, 2),
        "epochs": epochs,
        "stopped_epoch": stopped_epoch,
        "label_smoothing": label_smoothing,
        "patience": patience,
        "history": history,
        "checkpoint": str(best_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lsnet_t", "focusshift_t", "lsnet_micro", "focusshift_micro"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", type=str, default=str(LABELS))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet LSNet-T weights into lsnet_t/focusshift_t")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="CE label smoothing (0.0 allows lower train loss; 0.05 recommended)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stop if val acc does not improve for N epochs (0 disables)")
    parser.add_argument("--class-weight", action="store_true",
                        help="Use inverse-frequency class weights (helps rare/hard classes)")
    parser.add_argument(
        "--aug",
        type=str,
        default="default",
        choices=["default", "sketch"],
        help="Training augmentation: sketch = gentler CDT-friendly transforms (no flip)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.labels))
    classes = sorted({c for _, c, _ in rows})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_rows, val_rows = split_rows(rows, seed=args.seed)
    print(f"CDT: {len(rows)} images, classes={classes}, train={len(train_rows)}, val={len(val_rows)}")
    print(f"Augmentation: {args.aug}")

    train_ds = ExcelImageDataset(train_rows, class_to_idx, train=True, aug=args.aug)
    val_ds = ExcelImageDataset(val_rows, class_to_idx, train=False, aug=args.aug)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_weights = None
    if args.class_weight:
        counts = [0] * len(classes)
        for _, c in train_rows:
            counts[class_to_idx[c]] += 1
        # inverse frequency, normalized to mean 1
        inv = [1.0 / max(n, 1) for n in counts]
        mean_inv = sum(inv) / len(inv)
        class_weights = [w / mean_inv for w in inv]
        print(f"Class weights: {dict(zip(classes, [round(w, 3) for w in class_weights]))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    for name in args.models:
        print("=" * 60)
        print(f"Training {name}")
        result = train_one(
            name, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            num_classes=len(classes), out_dir=out_dir,
            pretrained=args.pretrained,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
            patience=args.patience,
        )
        results.append(result)
        print(f"Best {name}: {result['best_val_acc']}%  ({result['params_m']}M params)")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=" * 60)
    print(f"Wrote {summary_path}")
    for r in results:
        print(f"  {r['model']:<14} {r['params_m']:>7.2f}M  best_val={r['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()
