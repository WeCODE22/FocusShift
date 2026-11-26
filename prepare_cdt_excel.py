"""Build an Excel label sheet from the concetto_CDT image folders."""
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "concetto_CDT"  # colocated with FocusShift solution
OUT = ROOT / "custom_data" / "cdt_labels.xlsx"

# Folders present on disk. Names are placeholders until you rename them.
CLASS_NAMES = {
    "0": "CDT_class_0",
    "4": "CDT_class_4",
    "5": "CDT_class_5",
}


def main():
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        folder = IMAGE_ROOT / class_id
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder: {folder}")
        for path in sorted(folder.glob("*.png")):
            # Store repo-relative paths so the Excel sheet is portable on GitHub clones.
            rel = path.relative_to(ROOT).as_posix()
            rows.append((path.name, class_id, class_name, rel, "train"))

    # Simple split: every 5th image goes to val
    for i, row in enumerate(rows):
        if i % 5 == 0:
            rows[i] = (row[0], row[1], row[2], row[3], "val")

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    ws.append(["image_name", "class_id", "class_name", "image_path", "split"])
    for row in rows:
        ws.append(list(row))

    summary = wb.create_sheet("summary")
    summary.append(["class_id", "class_name", "count"])
    for class_id, class_name in CLASS_NAMES.items():
        count = sum(1 for row in rows if row[1] == class_id)
        summary.append([class_id, class_name, count])
    summary.append(["total", "", len(rows)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Images: {len(rows)}")
    for class_id, class_name in CLASS_NAMES.items():
        count = sum(1 for row in rows if row[1] == class_id)
        print(f"  {class_id} ({class_name}): {count}")


if __name__ == "__main__":
    main()
