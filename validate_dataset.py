#!/usr/bin/env python3
"""validate_dataset.py — Full validation per spec D2.

Checks:
  1. metadata.json exists and is valid JSON
  2. Every case has 7 views (center, left, right, up, down, macro, around)
  3. All required files exist (image, 5 masks, yolo label, meta.json)
  4. Split distribution (70/15/15 ±2%)
  5. Quality monotonic sharpness (q0<q1<q2<q3 mean sharpness)
  6. Gaze geometry: left/right centroid shifts match view names
  7. Strabismus class (5) center view has visible offset
  8. Ptosis class (6) upper lid occludes iris
  9. YOLO boxes present for lesion classes, valid format, in bounds
 10. Segmentation masks: pupil⊆iris, iris⊆open, disjoint iris/sclera
 11. Per-case meta.json present with 7 views
 12. images.csv present and matches metadata
 13. dataset_manifest.json present with rejection rates
"""
import sys
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image

REQUIRED_VIEWS = ["center", "left", "right", "up", "down", "macro", "around"]
REQUIRED_MASKS = ["iris", "sclera", "pupil", "upper_lid", "lower_lid"]
LESION_CLASSES_IDX = {4, 8, 9, 10, 11, 12}  # classes requiring YOLO boxes
SPLIT_TOL = 0.02
MIN_TEST_CASES = 8  # for classes 5,6,11 (strabismus, ptosis, foreign_body)
MIN_TEST_CASES_SPECIAL = 8  # for classes 5,6,11

CLASSES = {0:"normal",1:"conjunctivitis",2:"allergic_conjunctivitis",3:"dry_eye",
           4:"blepharitis",5:"strabismus",6:"ptosis",7:"corneal_scar",8:"keratitis",
           9:"pinguecula",10:"pterygium",11:"foreign_body",12:"subconjunctival_hemorrhage"}


def compute_sharpness(img_arr):
    import cv2
    arr = np.asarray(img_arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def check_file(p: Path, desc: str) -> bool:
    if not p.exists():
        print(f"  FAIL: {desc} not found: {p}")
        return False
    return True


def main():
    out_dir = Path('./eye_dataset_v7')
    meta_path = out_dir / 'metadata.json'
    images_csv = out_dir / 'images.csv'
    manifest_path = out_dir / 'dataset_manifest.json'

    results = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        results.append((name, status, detail))
        icon = "✓" if condition else "✗"
        print(f"  {icon} {name}: {status} {detail}")
        return condition

    # 1. metadata.json
    if not check_file(meta_path, "metadata.json"):
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    cases = meta.get("cases", [])
    check("metadata.json valid", len(cases) > 0, f"{len(cases)} cases")

    # 2. Every case has 7 views
    all_7 = True
    for c in cases:
        views = [v["view"] for v in c.get("views", [])]
        missing = [v for v in REQUIRED_VIEWS if v not in views]
        if missing:
            check(f"case {c['case_id']} 7 views", False, f"missing {missing}")
            all_7 = False
    check("all cases 7 views", all_7)

    # 3. Required files per case per view
    files_ok = True
    for c in cases:
        cdir = out_dir / c["class_name"] / c["case_id"]
        for v in c.get("views", []):
            view = v["view"]
            if not check_file(cdir / f"{view}.jpg", f"image {c['case_id']}/{view}"):
                files_ok = False
            for m in REQUIRED_MASKS:
                if not check_file(cdir / "masks" / f"{view}_{m}.png",
                                  f"mask {c['case_id']}/{view}/{m}"):
                    files_ok = False
            # yolo label
            if not check_file(cdir / f"{view}.txt", f"yolo {c['case_id']}/{view}"):
                files_ok = False
        # per-case meta.json
        if not check_file(cdir / "meta.json", f"per-case meta {c['case_id']}"):
            files_ok = False
    check("all required files exist", files_ok)

    # 4. Split distribution
    splits = {"train": 0, "val": 0, "test": 0}
    for c in cases:
        splits[c["split"]] = splits.get(c["split"], 0) + 1
    # 4. Per-class split distribution (70/15/15 ±2% per class, with MIN_TEST_CASES floor for classes 5,6,11)
    from collections import Counter
    per_class_splits = {}
    for c in cases:
        cid = c["class_id"]
        if cid not in per_class_splits:
            per_class_splits[cid] = Counter()
        per_class_splits[cid][c["split"]] += 1

    split_ok = True
    for cid, counts in per_class_splits.items():
        n_total = sum(counts.values())
        expected_train = round(0.70 * n_total)
        expected_val = round(0.15 * n_total)
        expected_test = n_total - expected_train - expected_val
        # Apply MIN_TEST_CASES floor for classes 5, 6, 11
        if cid in (5, 6, 11) and expected_test < MIN_TEST_CASES:
            # floor applies if test+val <= 50%
            if expected_test + expected_val <= n_total / 2:
                needed = MIN_TEST_CASES - expected_test
                expected_test = MIN_TEST_CASES
                expected_train = max(0, expected_train - needed)
        actual_train = counts.get("train", 0)
        actual_val = counts.get("val", 0)
        actual_test = counts.get("test", 0)
        if abs(actual_train - expected_train) > 1 or abs(actual_val - expected_val) > 1 or abs(actual_test - expected_test) > 1:
            split_ok = False
            print(f"  Class {CLASSES[cid]} (cid={cid}): expected train={expected_train}, val={expected_val}, test={expected_test} got train={actual_train}, val={actual_val}, test={actual_test}")

    check("per-class split distribution 70/15/15", split_ok)

    # 5. Quality monotonic sharpness
    sharpness_by_q = {0: [], 1: [], 2: [], 3: []}
    for c in cases:
        q = c.get("quality_label", 3)
        for v in c.get("views", []):
            img_path = out_dir / c["class_name"] / c["case_id"] / f"{v['view']}.jpg"
            if img_path.exists():
                sharpness_by_q[q].append(compute_sharpness(Image.open(img_path)))
    means = {q: np.mean(vals) if vals else 0 for q, vals in sharpness_by_q.items()}
    mono = means[0] < means[1] < means[2] < means[3]
    check("quality monotonic sharpness (q0<q1<q2<q3)", mono, f"means={means}")

    # 6. Gaze geometry: left/right centroid shifts
    gaze_ok = True
    for c in cases:
        views_dict = {v["view"]: v for v in c.get("views", [])}
        if "center" in views_dict and "left" in views_dict:
            cx_c, cy_c = views_dict["center"]["iris_px"]
            cx_l, cy_l = views_dict["left"]["iris_px"]
            # left view: iris should shift LEFT (negative x) relative to center
            if cx_l >= cx_c - 5:  # allow small tolerance
                gaze_ok = False
                break
        if "center" in views_dict and "right" in views_dict:
            cx_c, cy_c = views_dict["center"]["iris_px"]
            cx_r, cy_r = views_dict["right"]["iris_px"]
            # right view: iris should shift RIGHT (positive x)
            if cx_r <= cx_c + 5:
                gaze_ok = False
                break
    check("gaze geometry left/right shifts", gaze_ok)

    # 7. Strabismus (class 5) center view offset
    strab_ok = True
    for c in cases:
        if c["class_id"] == 5:
            v = next(v for v in c["views"] if v["view"] == "center")
            ix, iy = v["iris_px"]
            # convert to axis units
            cx = (ix / 512 - 0.5) * 2.24
            cy = (0.5 - iy / 512) * 2.24
            if abs(cx) < 0.08 and abs(cy) < 0.03:
                strab_ok = False
                break
    check("strabismus class center offset visible", strab_ok)

    # 8. Ptosis (class 6) upper lid occludes iris
    ptosis_ok = True
    for c in cases:
        if c["class_id"] == 6:
            v = next(v for v in c["views"] if v["view"] == "center")
            mask_path = out_dir / c["class_name"] / c["case_id"] / "masks" / "center_upper_lid.png"
            iris_path = out_dir / c["class_name"] / c["case_id"] / "masks" / "center_iris.png"
            if mask_path.exists() and iris_path.exists():
                upper = np.array(Image.open(mask_path)) > 0
                iris = np.array(Image.open(iris_path)) > 0
                if upper.any() and iris.any():
                    lid_bottom = np.max(np.where(upper)[0])
                    iris_top = np.min(np.where(iris)[0])
                    if lid_bottom < iris_top - 2:
                        ptosis_ok = False
                        break
    check("ptosis class upper lid occludes iris", ptosis_ok)

    # 9. YOLO boxes for lesion classes
    yolo_ok = True
    for c in cases:
        if c["class_id"] in LESION_CLASSES_IDX:
            for v in c.get("views", []):
                yolo_path = out_dir / c["class_name"] / c["case_id"] / f"{v['view']}.txt"
                if yolo_path.exists():
                    content = yolo_path.read_text().strip()
                    if content:
                        parts = content.split()
                        if len(parts) != 5:
                            yolo_ok = False
                            break
                        try:
                            cls_id, cx, cy, bw, bh = map(float, parts)
                            if not (0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0 and bh > 0):
                                yolo_ok = False
                                break
                            if cls_id not in {1, 2, 3, 4, 5, 6}:
                                yolo_ok = False
                                break
                        except ValueError:
                            yolo_ok = False
                            break
    check("YOLO boxes valid for lesion classes", yolo_ok)

    # 10. Segmentation masks anatomy
    mask_ok = True
    for c in cases:
        for v in c.get("views", []):
            cdir = out_dir / c["class_name"] / c["case_id"]
            mdir = cdir / "masks"
            iris = np.array(Image.open(mdir / f"{v['view']}_iris.png")) > 0
            pupil = np.array(Image.open(mdir / f"{v['view']}_pupil.png")) > 0
            sclera = np.array(Image.open(mdir / f"{v['view']}_sclera.png")) > 0
            upper = np.array(Image.open(mdir / f"{v['view']}_upper_lid.png")) > 0
            lower = np.array(Image.open(mdir / f"{v['view']}_lower_lid.png")) > 0

            open_region = ~upper & ~lower
            # pupil ⊆ iris
            if pupil.any() and not np.all(pupil[~iris] == False):
                mask_ok = False
                break
            # iris ⊆ open
            if iris.any() and not np.all(iris[~open_region] == False):
                mask_ok = False
                break
            # iris and sclera disjoint
            if np.any(iris & sclera):
                mask_ok = False
                break
    check("segmentation masks anatomy (pupil⊆iris⊆open, iris∩sclera=∅)", mask_ok)

    # 11. per-case meta.json with 7 views
    per_meta_ok = True
    for c in cases:
        cdir = out_dir / c["class_name"] / c["case_id"]
        mpath = cdir / "meta.json"
        if mpath.exists():
            try:
                with open(mpath) as f:
                    m = json.load(f)
                if len(m.get("views", [])) != 7:
                    per_meta_ok = False
                    break
            except Exception:
                per_meta_ok = False
                break
    check("per-case meta.json with 7 views", per_meta_ok)

    # 12. images.csv
    if check_file(images_csv, "images.csv"):
        with open(images_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        csv_ok = len(rows) == sum(len(c["views"]) for c in cases)
        check("images.csv row count matches metadata", csv_ok, f"{len(rows)} rows")
    else:
        check("images.csv exists", False)

    # 13. dataset_manifest.json
    if check_file(manifest_path, "dataset_manifest.json"):
        with open(manifest_path) as f:
            man = json.load(f)
        man_ok = "rejection_rate" in man and "total_cases" in man
        check("dataset_manifest.json has required fields", man_ok)
    else:
        check("dataset_manifest.json exists", False)

    # Summary table
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, status, detail in results:
        print(f"  {status:4s}  {name}")
        if detail and status == "FAIL":
            print(f"         -> {detail}")

    failed = sum(1 for _, s, _ in results if s == "FAIL")
    if failed:
        print(f"\n  FAILED: {failed} check(s)")
        sys.exit(1)
    else:
        print(f"\n  ALL {len(results)} CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()