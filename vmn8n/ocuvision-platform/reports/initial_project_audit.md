# Stage 1 — Initial Project Audit

## 1) Location & Git
- Working dir: `/home/ghrceadmin8/OcuVision/vmn8n` (empty)
- Repo root: `/home/ghrceadmin8/OcuVision`
- Branch: main, clean, remote: github.com/SarthakShende/Oucvision_backup_eye.git
- Tracked files: ~14,030; `.gitignore` excludes `eye_dataset_v6`, `.venv`, `ocuvision_models` etc.

## 2) Existing Architecture
Monolithic `ocuvision_script_v7.py` (2830 lines): generation (two-stage), training (M1/M2/M4/Fusion + YOLO), ONNX export, inference, CLI. Scripts: `validate_dataset.py`, `check_dataset.py`. Docs: `CHANGES_v7.md`, `therun.md`.

## 3) Existing Models
- M1: MobileNetV3-Small (m1_best.pt + onnx)
- M2: EfficientNet-Lite0 (m2_best.pt + onnx)
- M4: U-Net MobileNetV2 (m4_best.pt + onnx)
- Fusion: MLP 32->256->128->13 (fusion_best.pt + fp32/int8 onnx)
- YOLO weights present. No ConvNeXt "Model 5".

## 4) Dataset v7 (synthetic)
260 cases, 1820 images (512x512), 7 views/case, 5-channel masks. Classes match 13-ID spec. gitignored `eye_dataset_v6` (15GB) removed to free space.

## 5) Class IDs (preserved)
0 normal,1 conjunctivitis,2 allergic_conjunctivitis,3 dry_eye,4 blepharitis,5 strabismus,6 ptosis,7 corneal_scar,8 keratitis,9 pinguecula,10 pterygium,11 foreign_body,12 subconjunctival_hemorrhage

## 6) Environment
Ubuntu 20.04, Python 3.8.10. New venv at `vmn8n/.venv` (py3.8). GPU: Tesla V100, CUDA 12.2, driver 535.261.03, torch 2.3.1+cu121 working. Some newer wheels adjusted for py3.8 compatibility.

## 7) Hardware
Disk: 57GB free now (was 42GB). RAM 22GB avail. 32 vCPUs. Disk reserve policy 25GB enforced conceptually.

## 8) Issues
- Disk tight (57GB) — still must gate downloads
- No API/job queue/systemd yet
- Requirements adjusted for py3.8 compatibility (onnxruntime CPU stable)

## 9) Plan
New platform at `vmn8n/ocuvision-platform/` built. Stages 2-10 as defined. No changes to existing repo files.
