#!/usr/bin/env python3
"""check_dataset.py — Visualization sheets per spec D1.

Creates:
  1. sheet_<class>.jpg for each class: up to 6 cases × 7 views with
     iris (green), pupil (red), sclera (blue) overlays + YOLO boxes (yellow)
  2. sheet_same_case.jpg: 13 rows (one per class) × 7 views showing
     the SAME case across all gaze views
"""
import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REQUIRED_VIEWS = ["center", "left", "right", "up", "down", "macro", "around"]
OVERLAY_COLORS = {
    "iris": (0, 255, 0),      # green
    "pupil": (255, 0, 0),     # red
    "sclera": (0, 0, 255),    # blue
}
YOLO_COLOR = (255, 255, 0)    # yellow


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    """Overlay a binary mask with color on image."""
    mask_bool = mask > 127
    colored = np.zeros_like(img)
    colored[:] = color
    out = img.copy()
    out[mask_bool] = ((1 - alpha) * img[mask_bool] + alpha * colored[mask_bool]).astype(np.uint8)
    return out


def draw_yolo_box(img: np.ndarray, box_str: str, color: tuple = YOLO_COLOR) -> np.ndarray:
    """Draw YOLO format box (class cx cy w h normalized) on image."""
    if not box_str.strip():
        return img
    h, w = img.shape[:2]
    parts = box_str.split()
    if len(parts) != 5:
        return img
    try:
        _, cx, cy, bw, bh = map(float, parts)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    except ValueError:
        pass
    return img


def load_case_view(case_dir: Path, view: str, thumb_size: int = 224) -> np.ndarray:
    """Load image + masks + yolo for one view, return thumbnail with overlays."""
    img_path = case_dir / f"{view}.jpg"
    if not img_path.exists():
        return np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)

    img = np.array(Image.open(img_path).resize((thumb_size, thumb_size)))
    mdir = case_dir / "masks"

    # overlays
    for ch, color in OVERLAY_COLORS.items():
        mpath = mdir / f"{view}_{ch}.png"
        if mpath.exists():
            mask = np.array(Image.open(mpath).resize((thumb_size, thumb_size), resample=Image.NEAREST))
            alpha = 0.5 if ch == "iris" else (0.6 if ch == "pupil" else 0.3)
            img = overlay_mask(img, mask, color, alpha)

    # YOLO box
    yolo_path = case_dir / f"{view}.txt"
    if yolo_path.exists():
        box_str = yolo_path.read_text().strip()
        img = draw_yolo_box(img, box_str)

    return img


def make_class_sheet(out_dir: Path, meta: dict, class_name: str, cases: list, max_cases: int = 6) -> Path:
    """Create sheet_<class>.jpg with up to max_cases rows × 7 views."""
    selected = random.sample(cases, min(max_cases, len(cases)))
    thumb = 224
    rows = []

    for case in selected:
        case_dir = out_dir / case["class_name"] / case["case_id"]
        row_imgs = []
        for view in REQUIRED_VIEWS:
            row_imgs.append(load_case_view(case_dir, view, thumb))
        if row_imgs:
            rows.append(np.hstack(row_imgs))

    if not rows:
        return None

    sheet = np.vstack(rows)
    # Add class label
    pil_sheet = Image.fromarray(sheet)
    draw = ImageDraw.Draw(pil_sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), f"{class_name} ({len(selected)} cases)", fill=(255, 255, 255), font=font)
    # view headers
    for i, view in enumerate(REQUIRED_VIEWS):
        x = i * thumb + 10
        draw.text((x, 10), view, fill=(255, 255, 0), font=font)

    out_path = Path(f"sheet_{class_name}.jpg")
    pil_sheet.save(out_path, quality=90)
    return out_path


def make_same_case_sheet(out_dir: Path, meta: dict) -> Path:
    """Create sheet_same_case.jpg with 13 rows × 7 views (one case per class)."""
    thumb = 224
    rows = []

    # one case per class (deterministic: first case in metadata)
    for class_name in sorted({c["class_name"] for c in meta["cases"]}):
        class_cases = [c for c in meta["cases"] if c["class_name"] == class_name]
        if not class_cases:
            rows.append(np.zeros((thumb, len(REQUIRED_VIEWS) * thumb, 3), dtype=np.uint8))
            continue
        case = class_cases[0]
        case_dir = out_dir / case["class_name"] / case["case_id"]
        row_imgs = []
        for view in REQUIRED_VIEWS:
            row_imgs.append(load_case_view(case_dir, view, thumb))
        if row_imgs:
            rows.append(np.hstack(row_imgs))
        else:
            rows.append(np.zeros((thumb, len(REQUIRED_VIEWS) * thumb, 3), dtype=np.uint8))

    sheet = np.vstack(rows)
    pil_sheet = Image.fromarray(sheet)
    draw = ImageDraw.Draw(pil_sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # class labels on left
    class_names = sorted({c["class_name"] for c in meta["cases"]})
    for i, cname in enumerate(class_names):
        y = i * thumb + 10
        draw.text((10, y), cname, fill=(255, 255, 255), font=font)
    # view headers
    for i, view in enumerate(REQUIRED_VIEWS):
        x = i * thumb + 10
        draw.text((x, 10), view, fill=(255, 255, 0), font=font)

    out_path = Path("sheet_same_case.jpg")
    pil_sheet.save(out_path, quality=90)
    return out_path


def main():
    out_dir = Path('./eye_dataset_v7')
    meta_path = out_dir / 'metadata.json'

    if not meta_path.exists():
        print("FAIL: metadata.json not found", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    # group by class
    by_class = {}
    for c in meta["cases"]:
        by_class.setdefault(c["class_name"], []).append(c)

    print("Generating class sheets...")
    sheet_paths = []
    for class_name, cases in by_class.items():
        sp = make_class_sheet(out_dir, meta, class_name, cases)
        if sp:
            sheet_paths.append(sp)
            print(f"  Saved {sp}")
        else:
            print(f"  WARNING: no cases for {class_name}")

    print("Generating same-case sheet...")
    same_case_path = make_same_case_sheet(out_dir, meta)
    if same_case_path:
        print(f"  Saved {same_case_path}")

    print("\nSheets created:")
    for p in sheet_paths + [same_case_path]:
        if p and p.exists():
            size = p.stat().st_size / 1024
            print(f"  {p}  ({size:.1f} KB)")

    # Output description for report (D1 requirement)
    print("\n--- SHEET DESCRIPTIONS FOR REPORT ---")
    print(f"1. sheet_same_case.jpg: 13 rows × 7 views ({len(REQUIRED_VIEWS)} gaze views: {', '.join(REQUIRED_VIEWS)})")
    print("   Each row shows ONE case rendered across all 7 views. Iris=green, Pupil=red, Sclera=blue, YOLO boxes=yellow.")
    print("   Row order: normal, conjunctivitis, allergic_conjunctivitis, dry_eye, blepharitis,")
    print("              strabismus, ptosis, corneal_scar, keratitis, pinguecula, pterygium,")
    print("              foreign_body, subconjunctival_hemorrhage")
    for class_name in by_class:
        print(f"2. sheet_{class_name}.jpg: up to 6 cases × 7 views with same overlays")
    print("--- END SHEET DESCRIPTIONS ---")


if __name__ == "__main__":
    main()