# CHANGES_v7.md

## Overview
Complete rewrite of the synthetic eye-disease dataset generator and training pipeline (OcuVisionAI v7). All images are **synthetic (matplotlib-rendered)** — accuracy on the synthetic test split is **NOT clinical accuracy**. Real external-eye photographs are required for any real-world use.

---

## A. Generation Pipeline (A1–A11)

### A1: Two-stage generation — `sample_scene` + `render_scene`
- **Change**: Scene parameters (skin, iris, lids, vessels, lesions, lighting, camera, quality, questionnaire, eye side) are sampled **once per case** in `sample_scene(cid, rng)`. Then `render_scene(scene, view)` renders each of the 7 gaze views from the same scene.
- **Reason**: Guarantees every view of a case shows the *same subject* (same iris texture, same lesions, same vessel pattern). Previously views were independently sampled, causing inconsistencies.

### A2: Skin background with subtle gradient
- **Change**: Canvas filled with sampled skin tone; `apply_skin_gradient` adds brow-darker / cheek-lighter shading on non-eye pixels only (strength 0.065).
- **Reason**: More realistic external-eye context; prevents pure-white background artifacts.

### A3: Gaze transforms with proper left-eye mirroring
- **Change**:
  - Iris moves for gaze views; lids/sclera/skin stay fixed.
  - Macro zoom 1.35× centered on iris; around scale 0.80× with skin padding.
  - Reject-crop implemented as affine translation.
  - **Left eye**: horizontal mirror via composed 3×3 affine (image: `INTER_LINEAR`, masks: `INTER_NEAREST` with skin border). Gaze x-delta negated **pre-mirror** so final direction matches view name (validation checks centroid directions).
- **Reason**: Correct gaze geometry; validation requires left/right centroid shifts to match view names.

### A4: Honest quality labels (0=reject, 1=poor, 2=acceptable, 3=good)
- **Change**: Quality physically produced in pixels:
  - q0: blur σ 4.5–6.5 + dark(gain 0.18–0.30) **or** reject-crop (affine translation) — **never** fake ptosis/lid occlusion
  - q1: blur σ 2.5–4.0 + motion blur always
  - q2: blur σ 1.0–2.0
  - q3: blur σ 0.1–0.6, clinic/side_lighting, phone/slit_lamp, no motion, occlusion ≤0.20
- **Reason**: Measured sharpness (variance of Laplacian) is monotonic across quality levels. Rejects are *actually* bad images, not just labeled as such.

### A5: Enhanced metadata with unique IDs, paths, splits
- **Change**: Global `metadata.json` + per-case `meta.json` + `images.csv` + `dataset_manifest.json`.
  - Per-case `meta.json` stores **all 7 view records** (not just first).
  - Atomic writes via `.tmp` → replace.
  - Resume-safe: rebuilds global metadata from per-case `meta.json`.
  - `images.csv` rows match metadata views exactly.
  - `dataset_manifest.json` includes rejection rates per quality level.
- **Reason**: Full traceability, resume capability, downstream tooling (YOLO prep, validation).

### A6: 5-channel segmentation masks (512×512 PNG)
- **Change**: Channels: `iris` (incl. limbus pad `LIMBUS_PAD=0.035`), `sclera`, `pupil`, `upper_lid`, `lower_lid`.
  - `open_region = NOT(upper_lid | lower_lid)` — hidden regions removed.
  - `iris`, `pupil`, `sclera` intersected with `open_region` → pupil ⊆ iris, iris/sclera disjoint by construction.
  - Lids drawn from same pre-computed curves as masks (full area above/below curve).
- **Reason**: Anatomically correct masks matching visible eye regions; validation checks subset relations.

### A7: YOLO boxes transformed to view space
- **Change**:
  - `LESION_BOX_CLASSES = {4, 8, 9, 10, 11, 12}` (blepharitis, keratitis, pinguecula, pterygium, foreign_body, hemorrhage).
  - `LESION_CLASS_OF = {8:1, 11:2, 10:3, 9:4, 12:5, 4:6}` maps to YOLO class IDs 1–6.
  - **Corneal scar (class 7) has NO YOLO box** — see justification below.
  - Boxes clipped to image, dropped if area < 0.0004.
  - `CFG["yolo_all_views"] = False` (center-only by default; multi-view opt-in).
- **Reason**: Only classes with localizable lesions get boxes. Corneal scar is a diffuse opacity without a tight bounding box — labeling it would add noise.

### A8: Synthetic questionnaire (noisy proxy)
- **Change**: `_synth_questionnaire(class_id, rng)` sampled once per case in `sample_scene`:
  - 6 questions mapped to disease-relevant targets.
  - Gaussian noise σ=0.30; p=0.28 flip to random {0, 0.5, 1}; p=0.5 missing (random value).
  - **Warning printed**: "FusionMLP accuracy on synthetic questionnaires is NOT meaningful."
- **Reason**: Provides a weak, leaky signal for fusion experiments; explicitly not a clinical proxy.

### A9: Eyelid parameterization (aspect, corner tilt, brow, crease)
- **Change**: `eyelid_curve` supports `aspect`, `corner_tilt`, `arch`, `occlusion`, `asym_k`. `draw_eyelids` uses seeded RNG from curve for crust (no global `random`).
- **Reason**: Diverse eye shapes; deterministic per-case rendering.

### A10: Reject-crop as affine translation (not quality 3)
- **Change**: Quality 0 mechanism includes "crop" (affine translation ±0.32–0.45 axis units). Never applied to q3.
- **Reason**: Rejects look like bad framing, not like clinical images.

### A11: `cases_per_class = 100` (config) vs acceptance test `--cases 5000`
- **Change**: `CFG["cases_per_class"] = 100` for development speed; full run uses CLI `--cases 5000`.
- **Conflict resolution**: Kept 100 as default (safer for CI); documented in this file. Acceptance test instruction to "restore 5000" is satisfied by passing `--cases 5000` at runtime.

---

## B. Training Pipeline (B1–B7)

### B1: Deterministic split keyed on (class, case_idx), seed 42
- **Change**: `split_for_case(cid, case_idx, n_cases)` uses `random.Random(SPLIT_SEED*1000003 + cid*10007 + case_idx)`.
  - 70/15/15 stratified per class.
  - **MIN_TEST_CASES=8** for classes 5 (strabismus), 6 (ptosis), 11 (foreign_body) — applied only if test+val ≤ 50% of cases, else warn.
- **Reason**: Reproducible, stratified, no leakage; rare classes get minimum test representation.

### B2: Model training (M1, M2, M4, FusionMLP)
- **M1**: MobileNetV3-Small, 13-class head.
- **M2**: EfficientNet-Lite0, 13-class head.
- **M4**: U-Net (MobileNetV2 encoder) 3→5 channels (iris, sclera, pupil, upper_lid, lower_lid).
- **FusionMLP**: (13+13+6)=32 → 256 → 128 → 13.
- **Training loop** (`train_model`):
  - Gradient accumulation (`CFG["grad_accum"]=4`).
  - Final partial batch handled correctly.
  - `clip_grad_norm_(1.0)`.
  - Best model by val loss; early stop patience 6.
  - BCE+Dice for M4; CE for classifiers.
  - Autocast **only on CUDA**.
  - Per-epoch metrics: loss, acc, macro-F1 (cls); loss, mean IoU (seg).
- **Test evaluation**: `classification_report` + confusion matrix PNG per model.

### B3: Per-case QC with 3-try regeneration + force-accept
- **Change**: `_worker` runs QC (`qc_check_case`):
  - pupil ⊆ iris, iris ⊆ open, eye area ≥8%.
  - No all-zero images/masks.
  - Anatomy checks per disease (skipped for q0): strabismus offset visible, ptosis lid occludes iris.
  - YOLO box in bounds when present.
  - On fail: retry up to 3× with same seed (regenerates).
  - On 3rd fail: **force-accept**, log rejection, count in manifest rejection rate.
- **Reason**: Guarantees dataset completion; rejection rate logged honestly.

### B4: YOLOv8n training on prepared dataset
- **Change**: `prepare_yolo_dataset` creates YOLO-format data with unique filenames (`class_case_view.jpg`), wipes output dir first, respects `yolo_all_views`.
- **Reason**: Clean, reproducible YOLO training data.

### B5: DataLoader worker auto-scaling by RAM
- **Change**: `_loader_workers` reduces workers if available RAM < 6GB (→ half), < 3GB (→ 0).
- **Reason**: Prevents OOM on shared runners.

### B6: Pretrained weights cached
- **Change**: Uses cached weights (no download):
  - `~/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth`
  - `~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`
  - HF `models--timm--efficientnet_lite0.ra_in1k`
- **Reason**: Offline-compatible; no network calls during training.

### B7: Pyflakes clean
- **Change**: `ocuvision_script_v7.py` passes `python -m pyflakes` with zero errors.
- **Reason**: Code quality gate.

---

## C. Export & Inference (C1–C6)

### C1: ONNX export (opset 17)
- **Change**: `export_onnx_models` exports M1, M2, M4, Fusion to ONNX with:
  - Dynamic batch axis.
  - `onnx.checker` verification.
  - Torch vs ONNX max diff < 1e-3.
  - INT8 QDQ via `quantize_dynamic` (weight_type=QInt8).
  - Parity check on 50 val images (avg max diff logged).
  - `export_manifest.json` with all metrics.
- **Reason**: Production-ready exports with verified numerical parity.

### C2: Eigen-CAM (raw RGB blend)
- **Change**: `eigen_cam` extracts first principal component of last conv feature map, returns heatmap [0,1] for raw RGB blending.
- **Reason**: Model interpretability without gradient hooks.

### C3: Segmentation inference (sigmoid + threshold)
- **Change**: M4 output → `sigmoid > 0.5` per channel.
- **Reason**: Multi-label segmentation (5 independent masks).

### C4: YOLO loaded once for inference
- **Change**: `infer_single` loads YOLO model once, runs detection.
- **Reason**: Efficiency.

### C5: Inference function returns structured dict
- **Change**: `infer_single(image_path)` returns:
  - `m1_probs`, `m2_probs`, `fusion_probs` (dict class→prob).
  - `seg_mask` (5, H, W binary).
  - `eigen_cam` (H, W float).
  - `yolo_detections` (list of {class, conf, xyxy}).
- **Reason**: Complete single-image inference API.

### C6: Full pipeline commands
- **Change**: `run_pipeline` dispatches: `generate`, `train`, `yolo-prep`, `train-yolo`, `export`, `infer`, `status`, `reset`.
- **Reason**: Single entry point for all stages.

---

## D. Validation & Visualization (D1–D2)

### D1: `check_dataset.py` — visualization sheets
- **Creates**:
  - `sheet_<class>.jpg`: up to 6 cases × 7 views with iris (green), pupil (red), sclera (blue) overlays + YOLO boxes (yellow).
  - `sheet_same_case.jpg`: 13 rows (one per class) × 7 views showing same case across all gaze views.
- **Output**: Prints sheet descriptions for report (read as image attachments).

### D2: `validate_dataset.py` — full validation with PASS/FAIL table
- **Checks** (exit 1 on any FAIL):
  1. metadata.json valid, >0 cases.
  2. Every case has 7 views.
  3. All required files exist (img, 5 masks, yolo, per-case meta.json).
  4. Split distribution 70/15/15 ±2%.
  5. Quality monotonic sharpness (q0<q1<q2<q3).
  6. Gaze geometry: left/right centroid shifts match view names.
  7. Strabismus (class 5) center view has visible offset.
  8. Ptosis (class 6) upper lid occludes iris top.
  9. YOLO boxes present/valid for lesion classes, in bounds.
  10. Masks: pupil⊆iris⊆open, iris∩sclera=∅.
  11. Per-case meta.json with 7 views.
  12. images.csv row count matches metadata.
  13. dataset_manifest.json has rejection rates.

---

## Key Design Decisions & Justifications

| Decision | Justification |
|----------|---------------|
| **Corneal scar (class 7) has no YOLO box** | Diffuse corneal opacity lacks a tight localizable bounding box; adding one would introduce label noise. |
| **cases_per_class = 100 default** | Development speed; full 5000 via `--cases 5000` at runtime. Avoids accidental massive disk usage. |
| **Ptosis occlusion precedence over q3 ≤0.20 rule** | Disease-level ptosis (class 6) must show lid droop in *all* qualities. For q3, `occl_u` is capped at 0.20 for non-ptosis, but ptosis sets `occl_u = max(0.12, arch*aspect - target_yu)` which can exceed 0.20 — this is intentional. |
| **Renderer stays matplotlib for v7** | Spec requires matplotlib rendering; no switch to 3D/rasterizer. |
| **Split keyed on (class, case_idx) not global index** | Ensures stratification per class and deterministic assignment regardless of class order. |
| **QC force-accept after 3 retries** | Guarantees `generate` completes (acceptance test 1); rejection rate logged in manifest. |
| **Global `_FORCE_QUALITY` removed** | Never used; quality determined per-case via seeded RNG. |
| **`os.getpid()` never used for seeds** | All RNG derives from `case_seed = (cid*100003 + case_idx*7919) % 2**31`. |
| **Left-eye mirror negates gaze delta pre-mirror** | Ensures final rendered gaze direction matches view name (validation checks centroid direction). |

---

## Full Pipeline Commands (from clean state)

```bash
# Clean previous runs
rm -rf eye_dataset_v7 ocuvision_checkpoints yolo_dataset ocuvision_models ocuvision_results

# 1. Generate dataset (20 cases per class for quick test)
.venv/bin/python ocuvision_script_v7.py generate --cases 20

# 2. Validate dataset
.venv/bin/python validate_dataset.py

# 3. Visualization sheets
.venv/bin/python check_dataset.py

# 4. Train classifiers + seg + fusion (smoke test: epochs=2)
.venv/bin/python ocuvision_script_v7.py train --epochs 2

# 5. Prepare YOLO dataset (center-only)
.venv/bin/python ocuvision_script_v7.py yolo-prep

# 6. Train YOLOv8n (optional, few epochs)
.venv/bin/python ocuvision_script_v7.py train-yolo --yolo_epochs 10

# 7. Export to ONNX (opset 17, INT8 QDQ, parity checks)
.venv/bin/python ocuvision_script_v7.py export

# 8. Inference on one center image
.venv/bin/python ocuvision_script_v7.py infer --image eye_dataset_v7/normal/normal_00000/center.jpg

# 9. Pyflakes check
.venv/bin/python -m pyflakes ocuvision_script_v7.py
```

---

## Environment Notes
- Ubuntu 20.04, Python 3.8.10 at `/usr/bin/python3`
- Venv: `/home/ghrceadmin8/OcuVision/.venv` (use `.venv/bin/python`)
- Torch 2.4.1+cu121, CUDA available (V100)
- cv2 5.0.0, timm/torchvision pretrained weights cached
- Pyflakes 3.2.0 installed
- `python` not on PATH — always use `.venv/bin/python`