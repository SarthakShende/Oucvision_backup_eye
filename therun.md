# OcuVisionAI v7 — Run & Test Guide

## Overview
This document describes the complete OcuVisionAI v7 synthetic eye-disease dataset generator and training pipeline. All images are **synthetic (matplotlib-rendered)** — accuracy on the synthetic test split is **NOT clinical accuracy**. Real external-eye photographs are required for any real-world use.

---

## What Was Implemented

### 1. Dataset Generation (`ocuvision_script_v7.py generate`)
- **Two-stage generation**: `sample_scene(cid, rng)` → `render_scene(scene, view)`
- **Deterministic per-case seed**: `case_seed = (cid*100003 + case_idx*7919) % 2**31`
- **7 gaze views**: center, left, right, up, down, macro, around
- **Left-eye mirroring**: horizontal affine + pre-mirror gaze delta negation
- **Quality labels (0-3)**: physically produced blur/darkness/crop with monotonic sharpness
- **5-channel masks**: iris (incl. limbus), sclera, pupil, upper_lid, lower_lid at 512×512
- **YOLO boxes**: 6 lesion classes (keratitis, foreign_body, pterygium, pinguecula, hemorrhage, blepharitis); corneal scar excluded
- **Per-case meta.json** + global metadata.json + images.csv + dataset_manifest.json
- **QC with 3 retries + force-accept** (logs rejection rate)
- **Stratified split**: 70/15/15 per class, MIN_TEST_CASES=8 for rare classes (5,6,11)

### 2. Training (`ocuvision_script_v7.py train`)
- **M1**: MobileNetV3-Small, 13-class head
- **M2**: EfficientNet-Lite0, 13-class head
- **M4**: U-Net (MobileNetV2 encoder), 3→5 channel segmentation
- **FusionMLP**: (13+13+6)=32 → 256 → 128 → 13
- Gradient accumulation, clip_grad=1.0, early stopping (patience=6)
- BCE+Dice for segmentation, CE for classification
- Autocast only on CUDA
- Per-epoch metrics: loss, acc, macro-F1 (cls); loss, mean IoU (seg)
- Test evaluation: classification_report + confusion matrix PNG

### 3. YOLO Preparation (`ocuvision_script_v7.py yolo-prep`)
- Unique filenames: `class_case_view.jpg`
- Center-only by default (`yolo_all_views=False`)
- Wipes output directory first

### 4. ONNX Export (`ocuvision_script_v7.py export`)
- Opset 17, dynamic batch axis
- `onnx.checker` verification
- Torch vs ONNX max diff < 1e-3
- INT8 QDQ via `quantize_dynamic` (where supported)
- Val-set parity check (50 images)
- `export_manifest.json` with all metrics

### 5. Inference (`ocuvision_script_v7.py infer`)
- Returns: M1/M2/Fusion probs, seg mask (5×H×W binary), Eigen-CAM, YOLO detections

### 6. Validation (`validate_dataset.py`)
13 checks — all must PASS:
1. metadata.json valid
2. All cases have 7 views
3. All required files exist
4. Per-class split distribution 70/15/15
5. Quality monotonic sharpness (q0<q1<q2<q3)
6. Gaze geometry left/right centroid shifts
7. Strabismus class center offset visible
8. Ptosis class upper lid occludes iris
9. YOLO boxes valid for lesion classes
10. Mask anatomy (pupil⊆iris⊆open, iris∩sclera=∅)
11. Per-case meta.json with 7 views
12. images.csv row count matches metadata
13. dataset_manifest.json has rejection rates

### 7. Visualization (`check_dataset.py`)
- `sheet_<class>.jpg`: up to 6 cases × 7 views with overlays
- `sheet_same_case.jpg`: 13 rows × 7 views (same case across views)

---

## Environment Setup

```bash
# Ubuntu 20.04, Python 3.8.10
# Venv at /home/ghrceadmin8/OcuVision/.venv
# Use .venv/bin/python for all commands

cd /home/ghrceadmin8/OcuVision

# Verify environment
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "import cv2; print(cv2.__version__)"
.venv/bin/python -m pyflakes ocuvision_script_v7.py validate_dataset.py check_dataset.py
```

---

## Full Pipeline Commands

### Clean Start
```bash
rm -rf eye_dataset_v7 ocuvision_checkpoints yolo_dataset ocuvision_models ocuvision_results
```

### 1. Generate Dataset (20 cases/class for quick test)
```bash
.venv/bin/python ocuvision_script_v7.py generate --cases 20
```
- Output: `eye_dataset_v7/` with 260 cases, 1820 images
- Time: ~2-3 minutes on V100

### 2. Validate Dataset
```bash
.venv/bin/python validate_dataset.py
```
- Must show "ALL 13 CHECKS PASSED"

### 3. Create Visualization Sheets
```bash
.venv/bin/python check_dataset.py
```
- Creates 14 JPG files in current directory

### 4. Train Models (smoke test: 2 epochs)
```bash
.venv/bin/python ocuvision_script_v7.py train --epochs 2
```
- Output: `ocuvision_models/m1_best.pt`, `m2_best.pt`, `m4_best.pt`, `fusion_best.pt`
- Output: `ocuvision_results/*_confusion_test.png`

### 5. Prepare YOLO Dataset
```bash
.venv/bin/python ocuvision_script_v7.py yolo-prep
```
- Output: `yolo_dataset/` with data.yaml

### 6. Train YOLOv8n (optional)
```bash
.venv/bin/python ocuvision_script_v7.py train-yolo --yolo_epochs 10
```
- Output: `ocuvision_models/yolo_best.pt`

### 7. Export to ONNX
```bash
.venv/bin/python ocuvision_script_v7.py export
```
- Output: `ocuvision_models/*.onnx`, `export_manifest.json`

### 8. Run Inference
```bash
.venv/bin/python ocuvision_script_v7.py infer --image eye_dataset_v7/normal/normal_00000/center.jpg
```
- Returns JSON with probs, masks, Eigen-CAM, YOLO detections

### 9. Pyflakes Check
```bash
.venv/bin/python -m pyflakes ocuvision_script_v7.py validate_dataset.py check_dataset.py
```
- Must show zero errors

---

## Full Production Run (5000 cases/class)
```bash
rm -rf eye_dataset_v7 ocuvision_checkpoints yolo_dataset ocuvision_models ocuvision_results
.venv/bin/python ocuvision_script_v7.py generate --cases 5000
.venv/bin/python validate_dataset.py
.venv/bin/python check_dataset.py
.venv/bin/python ocuvision_script_v7.py train --epochs 20
.venv/bin/python ocuvision_script_v7.py yolo-prep
.venv/bin/python ocuvision_script_v7.py train-yolo --yolo_epochs 50
.venv/bin/python ocuvision_script_v7.py export
.venv/bin/python ocuvision_script_v7.py infer --image eye_dataset_v7/normal/normal_00000/center.jpg
.venv/bin/python -m pyflakes ocuvision_script_v7.py
```

---

## Testing Efficiency / Performance

### 1. Training Speed Benchmark
```bash
# Time training for 1 epoch
time .venv/bin/python ocuvision_script_v7.py train --epochs 1

# Check GPU utilization during training
watch -n 1 nvidia-smi
```

### 2. Inference Latency
```bash
# Single image inference time
time .venv/bin/python ocuvision_script_v7.py infer --image eye_dataset_v7/normal/normal_00000/center.jpg

# Batch inference (modify script to loop)
.venv/bin/python -c "
import time
from ocuvision_script_v7 import infer_single
from pathlib import Path
models_dir = Path('./ocuvision_models')
meta_path = Path('./eye_dataset_v7/metadata.json')
img = list(Path('./eye_dataset_v7/normal/normal_00000').glob('center.jpg'))[0]
times = []
for _ in range(10):
    t0 = time.time()
    infer_single(str(img), models_dir, meta_path)
    times.append(time.time() - t0)
print(f'Avg: {sum(times)/len(times)*1000:.1f} ms')
"
```

### 3. ONNX vs PyTorch Parity
```bash
# Check export_manifest.json for diffs
cat ocuvision_models/export_manifest.json | jq '.models[] | {name: .fp32, torch_onnx_diff: .torch_onnx_max_diff, int8_diff: .int8_parity_max_diff}'
```

### 4. Dataset Statistics
```bash
# Dataset manifest
cat eye_dataset_v7/dataset_manifest.json | jq '.'

# Split distribution
cat eye_dataset_v7/metadata.json | jq '.cases | group_by(.split) | map({split: .[0].split, count: length})'
```

### 5. Model Size
```bash
ls -lh ocuvision_models/*.pt ocuvision_models/*.onnx
```

### 6. Memory Usage
```bash
# Peak GPU memory during training (check logs or nvidia-smi)
# DataLoader worker scaling (auto-reduces below 6GB RAM)
```

---

## Key Configuration (CFG in ocuvision_script_v7.py)

```python
CFG = {
    "img_size": 224,           # Classification input
    "seg_size": 256,           # Segmentation input
    "gen_px": 512,             # Generation resolution
    "dpi": 100,
    "num_classes": 13,
    "gaze_views": 7,
    "cases_per_class": 100,    # Default (override with --cases)
    "out_dir": "./eye_dataset_v7",
    "models_dir": "./ocuvision_models",
    "results_dir": "./ocuvision_results",
    "batch": 128,              # Classification batch
    "batch_seg": 32,           # Segmentation batch
    "epochs": 20,
    "lr": 1e-3,
    "shuffle": 256,
    "num_workers": 8,
    "mixed_precision": True,
    "test_split": 0.15,
    "val_split": 0.15,
    "jpeg_quality": 90,
    "save_full_res": False,
    "use_questionnaire_in_fusion": True,
    # v7 additions:
    "gen_workers": 24,         # Auto: min(24, cpu-2)
    "grad_accum": 4,
    "early_stop_patience": 6,
    "yolo_all_views": False,   # Center-only YOLO
}
```

---

## Expected Outputs (Quick Test: --cases 20)

| Stage | Output | Expected |
|-------|--------|----------|
| Generate | `eye_dataset_v7/` | 260 cases, 1820 images |
| Validate | Console | ALL 13 CHECKS PASSED |
| Check | `sheet_*.jpg`, `sheet_same_case.jpg` | 14 files |
| Train | `ocuvision_models/*.pt` | 4 checkpoints |
| Train | `ocuvision_results/*_confusion_test.png` | 4 PNGs |
| YOLO-prep | `yolo_dataset/` | data.yaml + images/labels |
| Export | `ocuvision_models/*.onnx` | 4 ONNX + manifest |
| Infer | JSON stdout | probs, masks, cam, yolo |
| Pyflakes | Console | (no output = clean) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` not found | Use `.venv/bin/python` |
| CUDA OOM | Reduce `batch`/`batch_seg`, workers auto-scale |
| INT8 quantization fails | Some architectures unsupported; FP32 ONNX always works |
| Split distribution fails | Rare with small N; uses MIN_TEST_CASES floor |
| QC force-accept high | Expected for synthetic data; logged in manifest |
| Low accuracy | Synthetic data ≠ clinical data; warning printed |

---

## File Structure After Run

```
OcuVision/
├── ocuvision_script_v7.py      # Main pipeline
├── validate_dataset.py         # Validation (13 checks)
├── check_dataset.py            # Visualization sheets
├── CHANGES_v7.md               # Change log
├── therun.md                   # This file
├── eye_dataset_v7/             # Generated dataset
│   ├── metadata.json
│   ├── images.csv
│   ├── dataset_manifest.json
│   ├── normal/
│   │   └── normal_00000/
│   │       ├── center.jpg, left.jpg, ...
│   │       ├── masks/ (5 PNGs per view)
│   │       ├── *.txt (YOLO labels)
│   │       └── meta.json
│   └── ... (13 classes)
├── ocuvision_checkpoints/      # Resume checkpoints
├── ocuvision_models/           # Trained models
│   ├── m1_best.pt, m2_best.pt, m4_best.pt, fusion_best.pt
│   ├── yolo_best.pt (optional)
│   ├── *.onnx, *_int8.onnx
│   └── export_manifest.json
├── ocuvision_results/          # Confusion matrices
└── yolo_dataset/               # YOLO format
    ├── data.yaml
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

---

## Important Notes

⚠️ **ALL IMAGES ARE SYNTHETIC** — matplotlib-rendered, not real photographs  
⚠️ **SYNTHETIC ACCURACY ≠ CLINICAL ACCURACY** — do not use for medical decisions  
⚠️ **Questionnaire is noisy proxy** — FusionMLP accuracy on it is NOT meaningful  
⚠️ **Corneal scar (class 7) has no YOLO box** — diffuse opacity, no tight bbox  
⚠️ **Default cases_per_class=100** — use `--cases 5000` for full dataset