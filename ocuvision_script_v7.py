#!/usr/bin/env python3
"""OcuVisionAI v7 — synthetic external-eye dataset generator + training pipeline.

IMPORTANT: all images in this dataset are SYNTHETIC (matplotlib-rendered).
Accuracy measured on the synthetic test split is NOT clinical accuracy.
Real external-eye photographs are required for any real-world use.

Architecture (v7):
  * Two-stage generation: sample_scene(cid, rng) -> scene dict sampled ONCE per
    case, then render_scene(scene, view) -> image/masks/yolo for each gaze view.
    Every view of a case shows the SAME subject (skin, iris, lesions, vessels).
  * Per-case deterministic seeds: case_seed = (cid*100003 + case_idx*7919) % 2**31
    (never os.getpid()), so generation is reproducible and resume-safe.
  * Honest quality labels (0 reject / 1 poor / 2 acceptable / 3 good) that are
    physically produced in the pixels (blur / darkness / crop / low-res),
    with measured sharpness logged per image.
"""
import os
import cv2
import json
import math
import time
import random
import signal as _signal
import threading as _threading
import warnings
import multiprocessing
import shutil
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from PIL import Image
import scipy.ndimage as ndi

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*call to the `.*` method.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from torch.utils.data import Dataset, DataLoader

try:
    import psutil as _psutil
    _HAVE_PSUTIL = True
except ImportError:
    _psutil = None
    _HAVE_PSUTIL = False
    print("  [note] psutil not installed — RAM-based worker scaling disabled")


GENERATOR_VERSION = "v7"
SEED = 42
AXIS_SPAN = 2.24          # axis extent of the render canvas (-1.12 .. 1.12)
EYE_RX = 0.90             # sclera ellipse semi-axis (axis units)
EYE_RY = 0.45
LIMBUS_PAD = 0.035        # iris mask pad (matches the drawn limbal ring)
_GAZE_DELTAS = {"center": (0.0, 0.0), "left": (-0.22, 0.0), "right": (0.22, 0.0),
                "up": (0.0, 0.10), "down": (0.0, -0.10),
                "macro": (0.0, 0.0), "around": (0.0, 0.0)}
MACRO_ZOOM = 1.35
AROUND_SCALE = 0.80
M4_VIEWS = ["center", "left", "right", "up", "down", "macro"]
LESION_BOX_CLASSES = {4, 8, 9, 10, 11, 12}   # classes that must have a YOLO box
LESION_CLASS_OF = {8: 1, 11: 2, 10: 3, 9: 4, 12: 5, 4: 6}   # class_id -> LESION_CLASSES id


class NpEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


_stop_requested = _threading.Event()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(16)
try:
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass

CFG = {
    "img_size": 224, "seg_size": 256, "gen_px": 512, "dpi": 100,
    "num_classes": 13, "gaze_views": 7, "cases_per_class": 100,
    "out_dir": "./eye_dataset_v7", "models_dir": "./ocuvision_models",
    "results_dir": "./ocuvision_results", "batch": 128, "batch_seg": 32,
    "epochs": 20, "lr": 1e-3, "shuffle": 256, "num_workers": 8,
    "mixed_precision": True, "test_split": 0.15, "val_split": 0.15,
    "jpeg_quality": 90, "save_full_res": False, "use_questionnaire_in_fusion": True,
    # --- v7 additions ---
    "gen_workers": min(24, max(4, (os.cpu_count() or 8) - 2)),
    "grad_accum": 4, "early_stop_patience": 6,
    "yolo_all_views": False,       # yolo-prep keeps center-only by default (A7)
}
CLASSES = {0:"normal",1:"conjunctivitis",2:"allergic_conjunctivitis",3:"dry_eye",
           4:"blepharitis",5:"strabismus",6:"ptosis",7:"corneal_scar",8:"keratitis",
           9:"pinguecula",10:"pterygium",11:"foreign_body",12:"subconjunctival_hemorrhage"}
LESION_CLASSES = {0:"corneal_abrasion",1:"corneal_ulcer",2:"foreign_body_visible",
                  3:"pterygium_lesion",4:"pinguecula_lesion",5:"hemorrhage_spot",6:"lid_lesion"}
_QUESTION_TARGETS = {0:{1,2,3,4},1:{3,4},2:{8,3,9,10},3:{8,11,3},4:{2,1,8},5:{8,11,7}}
QUALITY_LABELS = {0:"reject",1:"poor",2:"acceptable",3:"good"}
GAZE_VIEWS = ["center","left","right","up","down","macro","around"]
LIGHTING_TYPES = ["clinic","low_light","overexposed","side_lighting","glare"]
CAMERA_TYPES = ["phone","phone","phone","slit_lamp","low_res"]
SKIN_TONES = ["#f5e6d0","#f0d0a8","#e8c49a","#d4a67a","#c8956c","#b07850","#8d5524","#6b3d1e"]
MASK_CHANNELS = ["iris", "sclera", "pupil", "upper_lid", "lower_lid"]
GEN_SIZE = CFG["gen_px"]
FIG_IN = GEN_SIZE / CFG["dpi"]
NC = CFG["num_classes"]
SPLIT_SEED = 42
MIN_TEST_IMAGES = 50          # special test floor for classes 5, 6, 11 (B1)
MIN_TEST_CASES = math.ceil(MIN_TEST_IMAGES / len(GAZE_VIEWS))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_col: str) -> np.ndarray:
    """'#b07850' -> float RGB array in [0,1]."""
    return np.array([int(hex_col[1:3], 16) / 255.0,
                     int(hex_col[3:5], 16) / 255.0,
                     int(hex_col[5:7], 16) / 255.0], dtype=np.float64)


def rgb_to_hex(rgb) -> str:
    """float RGB in [0,1] -> '#rrggbb'."""
    arr = np.clip(np.asarray(rgb, dtype=np.float64) * 255.0, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(arr)


def axis_to_px(x: float, y: float, size: int) -> tuple:
    """Axis units -> pixel coordinates (y flipped: image row 0 at top)."""
    px = (x / AXIS_SPAN + 0.5) * size
    py = (1.0 - (y / AXIS_SPAN + 0.5)) * size
    return px, py


def compute_sharpness(img_arr: np.ndarray) -> float:
    """Variance of the Laplacian — the sharpness metric logged per image."""
    arr = np.asarray(img_arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def resolve_path(path_str: str) -> Path:
    """Resolve a metadata path (stored relative to out_dir) to an absolute Path."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(CFG["out_dir"]) / p


# ---------------------------------------------------------------------------
# Checkpoint / pause-resume (Ctrl+C) infrastructure
# ---------------------------------------------------------------------------
def _setup_signal_handler() -> None:
    def _handler(sig, frame):
        print("\n\n  [PAUSE] Ctrl+C — finishing current item then stopping ...")
        print("  Re-run the same command to resume from this point.")
        _stop_requested.set()
    _signal.signal(_signal.SIGINT, _handler)
    _signal.signal(_signal.SIGTERM, _handler)


_CKPT_DIR = Path("./ocuvision_checkpoints")


def _ckpt_path(stage: str) -> Path:
    _CKPT_DIR.mkdir(exist_ok=True)
    return _CKPT_DIR / f"{stage}.ckpt.json"


def ckpt_save(stage: str, data: dict) -> None:
    data["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = _ckpt_path(stage)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(p)


def ckpt_load(stage: str) -> dict:
    p = _ckpt_path(stage)
    if p.exists():
        try:
            with open(p) as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            print(f"  Warning: corrupt checkpoint {p.name} — ignoring")
            return {}
    return {}


def ckpt_clear(stage: str) -> None:
    p = _ckpt_path(stage)
    if p.exists():
        p.unlink()


def ckpt_status() -> None:
    _CKPT_DIR.mkdir(exist_ok=True)
    files = sorted(_CKPT_DIR.glob("*.ckpt.json"))
    if not files:
        print("\n  No checkpoints found.\n")
        return
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  CHECKPOINT STATUS                                          │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    for fp in files:
        try:
            with open(fp) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
        stage = fp.stem.replace(".ckpt", "")
        updated = data.get("_updated", "?")
        status = data.get("status", "in_progress")
        icon = "done     " if status == "done" else "paused   "
        detail = ""
        if "completed_classes" in data and "total_classes" in data:
            done = len(data["completed_classes"])
            total = data["total_classes"]
            pct = int(done / total * 100) if total else 0
            detail = f"{done}/{total} classes ({pct}%)"
        print(f"  │  {icon}  {stage:<22} {updated}   {detail:<30}│")
    print("  └─────────────────────────────────────────────────────────────┘\n")


def ckpt_reset(stage: str = None) -> None:
    _CKPT_DIR.mkdir(exist_ok=True)
    if stage:
        ckpt_clear(stage)
        print(f"  Checkpoint cleared: {stage}")
    else:
        cleared = 0
        for fp in _CKPT_DIR.glob("*.ckpt.json"):
            fp.unlink()
            cleared += 1
        print(f"  Cleared {cleared} checkpoint(s).")


def _clear_gpu() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def gpu_usage() -> None:
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        out = result.stdout.decode().strip()
        if out:
            used, total = out.split(", ")
            print(f"  GPU Memory: {used} MB / {total} MB")
    except Exception:
        pass


def _available_ram_gb() -> float:
    if _HAVE_PSUTIL:
        return _psutil.virtual_memory().available / 1024 ** 3
    return 999.0


def _loader_workers(preferred: int = None) -> int:
    """DataLoader workers: CFG num_workers, reduced automatically below 6 GB RAM (B5)."""
    workers = CFG["num_workers"] if preferred is None else preferred
    avail = _available_ram_gb()
    if avail < 6.0:
        workers = max(1, workers // 2)
        if avail < 3.0:
            workers = 0
        print(f"  [RAM] {avail:.1f}GB available — DataLoader workers -> {workers}")
    return workers


# ---------------------------------------------------------------------------
# Matplotlib rendering primitives
# ---------------------------------------------------------------------------
def make_fig_with_skin(skin_rgb):
    """White-frame fix (A2): the whole canvas gets a skin-coloured rectangle."""
    fig = plt.figure(figsize=(FIG_IN, FIG_IN), dpi=CFG["dpi"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-1.12, -1.12), 2.24, 2.24,
                               facecolor=skin_rgb, edgecolor="none", zorder=0))
    return fig, ax


def fig_to_arr(fig, size: int = GEN_SIZE) -> np.ndarray:
    """Rasterise a figure to an RGB uint8 array of `size`x`size`."""
    fig.canvas.draw()
    img_arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    if img_arr.shape[0] != size or img_arr.shape[1] != size:
        img_arr = cv2.resize(img_arr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    plt.close(fig)
    return img_arr


def iris_color(rng) -> np.ndarray:
    """Continuous iris colour: base hue jittered in HSV (A9) — not 7 fixed colours."""
    base_hues = [0.06, 0.08, 0.10, 0.22, 0.30, 0.52, 0.58]
    h = float(rng.choice(base_hues)) + float(rng.uniform(-0.035, 0.035))
    if h < 0.0:
        h += 1.0
    s = float(np.clip(rng.uniform(0.35, 0.80) + rng.uniform(-0.08, 0.08), 0.15, 0.95))
    v = float(np.clip(rng.uniform(0.30, 0.70) + rng.uniform(-0.06, 0.06), 0.18, 0.85))
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return np.array([r, g, b])


def draw_vessels(ax, layout, color, alpha) -> None:
    """Draw a pre-sampled vessel layout (identical across all views of a case)."""
    for seg in layout:
        for (x, y), (x2, y2), lw in seg:
            ax.plot([x, x2], [y, y2], color=color, alpha=alpha, linewidth=lw,
                    zorder=3, solid_capstyle="round")


def draw_sclera(ax, tint=(0.96, 0.95, 0.93), layout=None, va=0.30) -> None:
    ax.add_patch(Ellipse((0, 0), EYE_RX * 2, EYE_RY * 2, facecolor=tint,
                         edgecolor="#c8b8a8", linewidth=0.5, zorder=2))
    if layout:
        draw_vessels(ax, layout, (0.65, 0.25, 0.25), va)


def draw_iris(ax, cx=0.0, cy=0.0, r=0.30, color=(0.3, 0.5, 0.4)) -> None:
    limbal = np.clip(np.asarray(color) * 0.28, 0, 1)
    ax.add_patch(Ellipse((cx, cy), (r + LIMBUS_PAD) * 2, (r + LIMBUS_PAD) * 2,
                         facecolor=limbal, zorder=5))
    ax.add_patch(Ellipse((cx, cy), r * 2, r * 2, facecolor=color, zorder=6))
    if hasattr(ax, "_ov_striations"):
        for sx, sy, ex, ey, a in ax._ov_striations:
            ax.plot([cx + sx, cx + ex], [cy + sy, cy + ey], color=limbal,
                    alpha=a, linewidth=0.6, zorder=7)


def draw_pupil(ax, cx=0.0, cy=0.0, r=0.13, hi_off=(0.0, 0.0), hi_r=0.03) -> None:
    """Pupil + specular highlight — highlight is tied to the iris, so it moves with it."""
    ax.add_patch(Ellipse((cx, cy), r * 2, r * 2, facecolor=(0.04, 0.04, 0.06), zorder=8))
    hx, hy = cx + hi_off[0], cy + hi_off[1]
    ax.add_patch(Ellipse((hx, hy), hi_r * 2, hi_r * 1.35,
                         facecolor=(0.97, 0.98, 1.0), alpha=0.90, zorder=9))


def eyelid_curve(n=240, upper=True, arch=0.38, asym_k=0.05, occlusion=0.0,
                 corner_tilt=0.0, aspect=1.0, rng=None):
    """Pure lid-margin curve (no RNG when all shape params are given).

    The same curve feeds both the drawing and the segmentation masks, so they
    can never disagree.
    """
    if arch is None:
        rs = rng if rng is not None else random
        arch = rs.uniform(0.28, 0.48) if upper else rs.uniform(0.10, 0.22)
    if asym_k is None:
        rs = rng if rng is not None else random
        asym_k = rs.uniform(0.02, 0.10)
    t = np.linspace(-0.90, 0.90, n)
    y = arch * aspect * (1.0 - (t / 0.90) ** 2) + asym_k * t + corner_tilt * (t / 0.90) ** 2
    if upper:
        y = y - occlusion
    else:
        y = -y + occlusion
    return t, y


def draw_eyelids(ax, skin_rgb, t_u, y_u, t_l, y_l, lash_shadow=False,
                 lash_len=0.045, lash_alpha=0.20, crust=False, lash_count=0) -> None:
    """Skin-coloured lids above/below the (pre-computed) lid curves."""
    xu = np.concatenate([t_u, t_u[::-1]])
    yu = np.concatenate([y_u, np.full_like(t_u, 1.30)])
    ax.fill(xu, yu, color=skin_rgb, zorder=11, linewidth=0)
    xl = np.concatenate([t_l, t_l[::-1]])
    yl = np.concatenate([y_l, np.full_like(t_l, -1.30)])
    ax.fill(xl, yl, color=skin_rgb, zorder=11, linewidth=0)
    if lash_shadow:
        ax.fill(np.concatenate([t_u, t_u[::-1]]),
                np.concatenate([y_u - lash_len, y_u[::-1]]),
                color=(0, 0, 0), alpha=lash_alpha, zorder=11)
    if lash_count > 0:
        rs = random.Random(int(abs(float(y_u.mean())) * 1e6) + lash_count)
        for _ in range(lash_count):
            i = rs.randrange(len(t_u))
            x0, y0 = float(t_u[i]), float(y_u[i])
            x1 = x0 + rs.uniform(-0.03, 0.03)
            y1 = y0 + rs.uniform(0.03, 0.07)
            if y1 > y0:
                ax.plot([x0, x1], [y0, y1], color=(0.05, 0.04, 0.03),
                        alpha=rs.uniform(0.35, 0.7), linewidth=rs.uniform(0.4, 0.9),
                        zorder=12)
    if crust:
        for _ in range(6):
            i = random.randrange(len(t_u))
            ax.add_patch(Ellipse((float(t_u[i]), float(y_u[i]) - 0.012),
                                 0.045, 0.022, facecolor=(0.78, 0.68, 0.45),
                                 alpha=0.75, zorder=12))


def draw_lid_crease(ax, t_u, y_u, skin_rgb, intensity=0.5) -> None:
    """Thin eyelid shadow below the crease (A9)."""
    if intensity <= 0.05:
        return
    band_top = y_u + 0.150
    band_bot = y_u + 0.055
    ax.fill(np.concatenate([t_u, t_u[::-1]]),
            np.concatenate([band_top, band_bot[::-1]]),
            color=np.clip(np.asarray(skin_rgb) * 0.62, 0, 1),
            alpha=0.20 + 0.18 * intensity, zorder=12, linewidth=0)
    ax.plot(t_u, y_u + 0.150, color=np.clip(np.asarray(skin_rgb) * 0.5, 0, 1),
            alpha=0.25 + 0.3 * intensity, linewidth=0.9, zorder=12)


def draw_eyebrow(ax, t_u, y_u, brow_rgb, present, density=1.0, tilt=0.0) -> None:
    """Randomised eyebrow presence/shape (A9)."""
    if not present:
        return
    tt = np.linspace(-0.80, 0.75, 60)
    arch = 0.62 + 0.10 * density
    base = arch * (1.0 - ((tt + 0.05) / 0.85) ** 2) + 0.10 * (tt / 0.85) + tilt
    thick = (0.045 + 0.035 * density) * np.clip(1.0 - ((tt + 0.05) / 0.85) ** 2, 0.1, 1)
    ax.fill(np.concatenate([tt, tt[::-1]]),
            np.concatenate([base + thick, (base - thick)[::-1]]),
            color=brow_rgb, alpha=0.85, zorder=12, linewidth=0)
    for k in range(26):
        i = int((k / 25) * (len(tt) - 1))
        ax.plot([tt[i], tt[i] + 0.04], [base[i], base[i] + 0.012],
                color=brow_rgb, alpha=0.5, linewidth=0.7, zorder=12)


def add_glasses_glare(ax, glare_spots) -> None:
    for (gx, gy, gw, gh, ga) in glare_spots:
        ax.add_patch(Ellipse((gx, gy), gw, gh, facecolor=(1, 1, 1), alpha=ga, zorder=15))


def add_eyelid_wrinkles(ax, t_u, y_u, skin_rgb, wrinkles) -> None:
    for (off, alpha, wob) in wrinkles:
        ax.plot(t_u, y_u + off + wob * np.sin(t_u * 7.0),
                color=np.clip(np.asarray(skin_rgb) * 0.55, 0, 1),
                alpha=alpha, linewidth=0.8, zorder=12)


# ---------------------------------------------------------------------------
# Stage 1 — sample_scene: ALL random parameters of one case, sampled ONCE (A1)
# ---------------------------------------------------------------------------
def _sample_vessels(rng, count: int) -> list:
    """Sample a fixed vessel layout so every view shows identical vessels."""
    layout = []
    for _ in range(max(1, count)):
        side = int(rng.choice([-1, 1]))
        x = side * rng.uniform(0.28, 0.82)
        y = rng.uniform(-0.28, 0.28)
        seg = []
        for s in range(int(rng.randint(2, 6))):
            x2 = x + rng.uniform(-0.12, 0.12) * (0.72 ** s)
            y2 = y + rng.uniform(-0.07, 0.07) * (0.72 ** s)
            if (x2 / EYE_RX) ** 2 + (y2 / EYE_RY) ** 2 < 1.0:
                seg.append(((x, y), (x2, y2), float(rng.uniform(0.4, 1.4))))
            x, y = x2, y2
        if seg:
            layout.append(seg)
    return layout


def _sample_lesion(cid: int, rng, scene: dict) -> dict:
    """Sample the lesion of one class into scene['lesion'] (positions fixed per case)."""
    lesion = {"kind": None, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0, "yolo_cls": None}
    if cid == 8:      # keratitis -> corneal_ulcer (LESION class 1)
        lesion.update(kind="keratitis", yolo_cls=1,
                      cx=float(rng.uniform(-0.12, 0.12)), cy=float(rng.uniform(-0.10, 0.10)),
                      w=float(rng.uniform(0.22, 0.42)) + 0.04,
                      h=float(rng.uniform(0.20, 0.38)) + 0.04,
                      alpha=float(rng.uniform(0.55, 0.75)))
    elif cid == 7:    # corneal scar — NO YOLO box (justified in CHANGES_v7.md)
        central = rng.random() < 0.60
        if central:
            cx, cy = float(rng.uniform(-0.10, 0.10)), float(rng.uniform(-0.10, 0.10))
            w, h = float(rng.uniform(0.14, 0.32)), float(rng.uniform(0.12, 0.28))
        else:
            side = int(rng.choice([-1, 1]))
            w, h = float(rng.uniform(0.12, 0.26)), float(rng.uniform(0.10, 0.22))
            rmax = 0.28 - w / 2
            cx = side * float(rng.uniform(0.12, max(0.13, rmax)))
            cy = float(rng.uniform(-0.10, 0.10))
        grade = int(rng.choice([1, 2, 3], p=[0.50, 0.30, 0.20]))
        if grade == 1:
            col = (rng.uniform(0.88, 0.94), rng.uniform(0.88, 0.93), rng.uniform(0.84, 0.90))
            alpha = rng.uniform(0.28, 0.48)
        elif grade == 2:
            col = (rng.uniform(0.84, 0.92), rng.uniform(0.84, 0.91), rng.uniform(0.78, 0.87))
            alpha = rng.uniform(0.55, 0.75)
        else:
            col = (rng.uniform(0.92, 0.97), rng.uniform(0.92, 0.96), rng.uniform(0.90, 0.95))
            alpha = rng.uniform(0.82, 0.95)
        lesion.update(kind="scar", yolo_cls=None, cx=cx, cy=cy, w=w, h=h,
                      color=col, alpha=alpha, grade=grade)
    elif cid == 9:    # pinguecula -> LESION class 4
        side = int(rng.choice([-1, 1]))
        lesion.update(kind="pinguecula", yolo_cls=4,
                      cx=side * float(rng.uniform(0.44, 0.74)), cy=float(rng.uniform(-0.10, 0.10)),
                      w=float(rng.uniform(0.14, 0.26)), h=float(rng.uniform(0.08, 0.15)),
                      alpha=float(rng.uniform(0.72, 0.90)))
    elif cid == 10:   # pterygium -> LESION class 3
        side = int(rng.choice([-1, 1]))
        length = float(rng.uniform(0.22, 0.55))
        base_x = side * 0.85
        tip_x = side * (0.85 - length)
        lesion.update(kind="pterygium", yolo_cls=3,
                      cx=(base_x + tip_x) / 2.0, cy=0.0,
                      w=abs(base_x - tip_x) + 0.05, h=float(rng.uniform(0.14, 0.26)),
                      side=side, length=length, base_x=base_x, tip_x=tip_x,
                      y0=float(rng.uniform(-0.14, 0.14)), y1=float(rng.uniform(-0.05, 0.05)),
                      y2=float(rng.uniform(-0.14, 0.14)),
                      alpha=float(rng.uniform(0.70, 0.88)))
    elif cid == 11:   # foreign body -> LESION class 2
        particles = []
        for k in range(int(rng.randint(1, 4))):
            ang = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0.05, 0.62)
            fx, fy = dist * math.cos(ang), dist * math.sin(ang) * 0.45
            fr = float(rng.uniform(0.018, 0.045))
            particles.append((float(fx), float(fy), fr))
            if k == 0:
                lesion.update(kind="foreign_body", yolo_cls=2, cx=float(fx), cy=float(fy),
                              w=fr * 3.0, h=fr * 3.0)
        lesion["particles"] = particles
    elif cid == 12:   # subconjunctival hemorrhage -> LESION class 5
        lesion.update(kind="hemorrhage", yolo_cls=5,
                      cx=float(rng.uniform(-0.35, 0.35)), cy=float(rng.uniform(-0.14, 0.14)),
                      w=float(rng.uniform(0.42, 0.75)), h=float(rng.uniform(0.16, 0.32)),
                      color=(float(rng.uniform(0.62, 0.74)), float(rng.uniform(0.05, 0.13)),
                             float(rng.uniform(0.05, 0.11))),
                      alpha=float(rng.uniform(0.78, 0.93)))
    elif cid == 4:    # blepharitis -> LESION class 6 (on the upper lid margin)
        t0 = float(rng.uniform(-0.55, 0.55))
        y_curve = (scene["arch_u"] * scene["aspect"] * (1.0 - (t0 / 0.90) ** 2)
                   + scene["asym_k"] * t0 + scene["corner_tilt"] * (t0 / 0.90) ** 2
                   - scene["occl_u"] + 0.015)
        lesion.update(kind="lid_margin", yolo_cls=6, cx=t0, cy=y_curve,
                      w=float(rng.uniform(0.10, 0.18)), h=0.07)
    return lesion


def sample_scene(cid: int, rng) -> dict:
    """Sample every random parameter of one case (A1, A4, A9).

    The returned dict fully determines the subject: skin, iris, lids, vessels,
    lesion, lighting/camera, quality machinery, questionnaire and eye side.
    """
    scene: dict = {"cid": cid}
    q = int(rng.choice([0, 1, 2, 3], p=[0.10, 0.10, 0.15, 0.65]))
    scene["quality_label"] = q

    # ---- skin (with undertone jitter) -------------------------------------------------
    scene["skin_hex"] = str(rng.choice(SKIN_TONES))
    skin = hex_to_rgb(scene["skin_hex"])
    undertone = rng.uniform(-0.035, 0.035, size=3)
    skin = np.clip(skin + undertone, 0.05, 0.98)
    scene["skin_rgb"] = skin
    scene["skin_hex"] = rgb_to_hex(skin)

    # ---- iris / pupil -----------------------------------------------------------------
    scene["iris_rgb"] = iris_color(rng)
    scene["iris_radius"] = float(rng.uniform(0.27, 0.34))
    scene["striations"] = [(float(rng.uniform(-0.8, -0.2)), float(rng.uniform(-0.4, 0.4)),
                            float(rng.uniform(0.6, 0.95)), float(rng.uniform(-0.3, 0.3)),
                            float(rng.uniform(0.05, 0.14))) for _ in range(int(rng.randint(8, 16)))]
    scene["hi_off"] = (float(rng.uniform(-0.03, 0.03)), float(rng.uniform(0.01, 0.04)))
    scene["hi_r"] = float(rng.uniform(0.022, 0.035))

    # ---- lids (A9: aspect, corner tilt, brow, crease) ---------------------------------
    scene["aspect"] = float(rng.uniform(0.92, 1.14))
    scene["arch_u"] = float(rng.uniform(0.30, 0.46))
    scene["arch_l"] = float(rng.uniform(0.11, 0.21))
    scene["asym_k"] = float(rng.uniform(0.02, 0.10))
    scene["corner_tilt"] = float(rng.uniform(-0.05, 0.05))
    scene["occl_u"] = float(rng.uniform(0.0, 0.20))
    scene["occl_l"] = float(rng.uniform(0.0, 0.05))
    scene["eyebrow"] = bool(rng.random() < 0.65)
    scene["brow_density"] = float(rng.uniform(0.3, 1.0))
    scene["brow_tilt"] = float(rng.uniform(-0.04, 0.04))
    scene["brow_rgb"] = tuple(np.clip(skin * rng.uniform(0.28, 0.5), 0.03, 0.4))
    scene["crease"] = float(rng.uniform(0.3, 1.0))
    scene["with_makeup"] = bool(rng.random() < 0.20)
    scene["lash_shadow"] = bool(rng.random() < 0.55)
    scene["lash_alpha"] = float(rng.uniform(0.12, 0.28))
    scene["lash_len"] = float(rng.uniform(0.035, 0.055))
    scene["lash_count"] = int(rng.randint(6, 18)) if rng.random() < 0.7 else 0
    scene["crust"] = False
    scene["wrinkles"] = [(float(rng.uniform(0.20, 0.34)), float(rng.uniform(0.15, 0.4)),
                          float(rng.uniform(-0.01, 0.01))) for _ in range(int(rng.randint(0, 3)))] \
        if rng.random() < 0.55 else []
    scene["glasses"] = bool(rng.random() < 0.12)
    scene["glare_spots"] = [(float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.5, 0.5)),
                             float(rng.uniform(0.15, 0.6)), float(rng.uniform(0.08, 0.28)),
                             float(rng.uniform(0.18, 0.48))) for _ in range(int(rng.randint(1, 3)))]

    # ---- vessels -----------------------------------------------------------------------
    scene["vessel_alpha"] = float(rng.uniform(0.28, 0.50))
    scene["vessel_n"] = int(rng.randint(6, 15))
    scene["vessels"] = _sample_vessels(rng, scene["vessel_n"])
    scene["sclera_tint"] = (0.96, 0.95, 0.93)

    # ---- strabismus offset (applied on top of every view) ------------------------------
    if cid == 5:
        scene["strabismus"] = (float(rng.choice([-1, 1])) * rng.uniform(0.18, 0.36),
                               float(rng.choice([-1, 1])) * rng.uniform(0.05, 0.13))
    else:
        scene["strabismus"] = (0.0, 0.0)

    # ---- lighting / camera / quality machinery (A4) -------------------------------------
    if q == 3:
        scene["blur_sigma"] = float(rng.uniform(0.1, 0.6))
        scene["lighting"] = str(rng.choice(["clinic", "clinic", "side_lighting"]))
        scene["camera"] = str(rng.choice(["phone", "phone", "slit_lamp"]))
        scene["light_gain"] = None
        scene["motion_blur"] = False
        scene["crop_active"] = False
    elif q == 2:
        scene["blur_sigma"] = float(rng.uniform(1.0, 2.0))
        scene["lighting"] = str(rng.choice(["low_light", "overexposed", "clinic",
                                            "glare", "side_lighting"]))
        scene["camera"] = str(rng.choice(CAMERA_TYPES))
        scene["light_gain"] = None
        scene["motion_blur"] = False
        scene["crop_active"] = False
    elif q == 1:
        scene["blur_sigma"] = float(rng.uniform(2.5, 4.0))
        scene["lighting"] = str(rng.choice(LIGHTING_TYPES))
        scene["camera"] = str(rng.choice(["phone", "phone", "low_res", "slit_lamp"]))
        scene["light_gain"] = None
        scene["motion_blur"] = True
        scene["crop_active"] = False
    else:  # q == 0 reject — produced with blur / darkness / crop, NEVER fake ptosis
        scene["blur_sigma"] = float(rng.uniform(4.5, 6.5))
        scene["lighting"] = str(rng.choice(LIGHTING_TYPES))
        scene["camera"] = str(rng.choice(CAMERA_TYPES))
        scene["motion_blur"] = bool(rng.random() < 0.5)
        mechanism = str(rng.choice(["dark", "crop", "dark_crop"], p=[0.4, 0.4, 0.2]))
        if "dark" in mechanism:
            scene["light_gain"] = float(rng.uniform(0.18, 0.30))
        else:
            scene["light_gain"] = None
        if "crop" in mechanism:
            ang = rng.uniform(0, 2 * math.pi)
            mag = float(rng.uniform(0.32, 0.45))
            scene["crop_active"] = True
            scene["crop_dx"] = float(mag * math.cos(ang))
            scene["crop_dy"] = float(mag * math.sin(ang) * 0.6)
        else:
            scene["crop_active"] = False
            scene["crop_dx"] = scene["crop_dy"] = 0.0
    scene["motion_len"] = float(rng.uniform(14, 30)) if scene["motion_blur"] else 0.0
    scene["motion_angle"] = float(rng.uniform(-35, 35)) if scene["motion_blur"] else 0.0

    # ---- pupil dilation follows the lighting -------------------------------------------
    dilation = {"clinic": (0.10, 0.14), "low_light": (0.16, 0.22),
                "overexposed": (0.07, 0.10), "side_lighting": (0.11, 0.16),
                "glare": (0.06, 0.09)}
    lo, hi = dilation.get(scene["lighting"], (0.10, 0.14))
    scene["pupil_r"] = float(rng.uniform(lo, hi))
    scene["pupil_r"] = min(scene["pupil_r"], scene["iris_radius"] - 0.03)

    # ---- ptosis keeps its own (disease-level) lid droop in every quality ---------------
    if cid == 6:
        target_yu = float(rng.uniform(0.02, 0.18))     # lid margin below iris top (0.30)
        scene["occl_u"] = max(0.12, scene["arch_u"] * scene["aspect"] - target_yu)
        scene["occl_l"] = float(rng.uniform(0.0, 0.05))
    else:
        # non-ptosis cases must never look like ptosis (A4)
        scene["occl_u"] = min(scene["occl_u"], 0.20)
        if scene["quality_label"] == 3:
            scene["occl_u"] = min(scene["occl_u"], 0.20)

    # ---- disease-specific sclera/vessels ------------------------------------------------
    if cid == 1:
        scene["sclera_tint"] = (float(rng.uniform(0.86, 0.94)), float(rng.uniform(0.66, 0.76)),
                                float(rng.uniform(0.66, 0.74)))
        scene["vessel_n"] = int(rng.randint(35, 55))
        scene["vessel_alpha"] = float(rng.uniform(0.60, 0.80))
        scene["vessels"] = _sample_vessels(rng, scene["vessel_n"])
        scene["occl_u"] = min(max(scene["occl_u"], float(rng.uniform(0.05, 0.18))), 0.20)
    elif cid == 2:
        scene["sclera_tint"] = (float(rng.uniform(0.88, 0.94)), float(rng.uniform(0.72, 0.82)),
                                float(rng.uniform(0.72, 0.80)))
        scene["vessel_n"] = int(rng.randint(24, 38))
        scene["vessel_alpha"] = float(rng.uniform(0.45, 0.65))
        scene["vessels"] = _sample_vessels(rng, scene["vessel_n"])
        scene["occl_u"] = min(max(scene["occl_u"], float(rng.uniform(0.06, 0.18))), 0.20)
    elif cid == 3:
        scene["vessel_n"] = int(rng.randint(10, 20))
        scene["vessels"] = _sample_vessels(rng, scene["vessel_n"])
        scene["vessel_alpha"] = float(rng.uniform(0.35, 0.55))
    elif cid == 4:
        scene["crust"] = True
        scene["lash_shadow"] = True

    # ---- misc per-case attributes ---------------------------------------------------------
    scene["eye_side"] = str(rng.choice(["left", "right"]))
    scene["tear_film"] = bool(rng.random() < 0.70)
    scene["skin_texture"] = bool(rng.random() < 0.80)
    scene["perlin_strength"] = float(rng.uniform(0.015, 0.055))
    scene["questionnaire"] = _synth_questionnaire(cid, rng)
    scene["lesion"] = _sample_lesion(cid, rng, scene)
    return scene


def _synth_questionnaire(class_id: int, rng) -> list:
    """Noisy, sometimes-missing synthetic questionnaire (A8).

    It is only a weak, leaky proxy of the label: per answer, with probability
    0.28 the value flips to a random {0, 0.5, 1}; gaussian noise sigma 0.30 is
    added otherwise; with probability 0.5 an answer is "missing" (random value).
    """
    q = []
    for qi in range(6):
        base = 1.0 if class_id in _QUESTION_TARGETS[qi] else 0.0
        val = base + float(rng.normal(0.0, 0.30))
        if rng.random() < 0.28:
            val = float(rng.choice([0.0, 0.5, 1.0]))
        val = float(np.clip(val, 0.0, 1.0))
        if rng.random() < 0.5:
            val = float(rng.choice([0.0, 0.5, 1.0]))      # answer missing
        q.append(val)
    return q


# ---------------------------------------------------------------------------
# Stage 2 — geometry: gaze clip, view affine, masks, YOLO boxes
# ---------------------------------------------------------------------------
def clip_gaze(base_x: float, base_y: float, dx: float, dy: float,
              r_iris: float) -> tuple:
    """Clip the gaze delta so the iris stays inside the sclera ellipse.

    The strabismus base offset is never destroyed: if the base itself sits
    outside the safe ellipse, the gaze delta is simply dropped for that view.
    """
    x_lim = max(0.10, EYE_RX - r_iris * 0.90)
    y_lim = max(0.05, EYE_RY - r_iris * 0.90)

    def inside(x: float, y: float) -> bool:
        return (x / x_lim) ** 2 + (y / y_lim) ** 2 <= 1.0

    tx, ty = base_x + dx, base_y + dy
    if inside(tx, ty):
        return dx, dy
    if inside(base_x, base_y):
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if inside(base_x + dx * mid, base_y + dy * mid):
                lo = mid
            else:
                hi = mid
        return dx * lo, dy * lo
    return 0.0, 0.0


def view_affine(view: str, scene: dict, iris_px: tuple, size: int):
    """Forward 3x3 affine (base pixels -> final pixels) for one view (A3, A10).

    Handles: horizontal mirror for left eyes, macro zoom 1.35 centred on the
    iris, around scale 0.80 with skin padding, and the reject-crop translation.
    Returns (A, effective_scale, final_iris_px).
    """
    w = h = float(size)
    A = np.eye(3)
    if scene["eye_side"] == "left":
        A = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ A
    ix, iy = iris_px
    p = A @ np.array([ix, iy, 1.0])
    cx, cy = float(p[0]), float(p[1])
    if view == "macro":
        s = MACRO_ZOOM
        A = np.array([[s, 0.0, cx - s * cx], [0.0, s, cy - s * cy], [0.0, 0.0, 1.0]]) @ A
        cx, cy = cx, cy
    elif view == "around":
        s = AROUND_SCALE
        mx, my = w / 2.0, h / 2.0
        A = np.array([[s, 0.0, mx - s * mx], [0.0, s, my - s * my], [0.0, 0.0, 1.0]]) @ A
    if scene.get("crop_active"):
        tx = scene["crop_dx"] / AXIS_SPAN * w
        ty = -scene["crop_dy"] / AXIS_SPAN * h
        A = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]) @ A
    p = A @ np.array([ix, iy, 1.0])
    scale = math.sqrt(abs(np.linalg.det(A[:2, :2])))
    return A, scale, (float(p[0]), float(p[1]))


def warp_image(arr: np.ndarray, A: np.ndarray, border_rgb) -> np.ndarray:
    """Apply the view affine to the image (bilinear)."""
    h, w = arr.shape[:2]
    border = tuple(int(np.clip(c, 0, 1) * 255) for c in border_rgb)
    return cv2.warpAffine(arr, A[:2].astype(np.float32), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def warp_masks(masks: dict, A: np.ndarray) -> dict:
    """Apply the exact same affine to every mask (nearest neighbour)."""
    out = {}
    first = next(iter(masks.values()))
    h, w = first.shape[:2]
    for name, m in masks.items():
        out[name] = cv2.warpAffine(m, A[:2].astype(np.float32), (w, h),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    return out


def make_segmentation_masks(size: int, iris_xy, r_iris: float, pupil_xy, r_pupil: float,
                            t_u, y_u, t_l, y_l) -> dict:
    """5-channel masks that match the VISIBLE eye (A6).

    Hidden regions are removed: open_region = NOT(upper_lid | lower_lid);
    iris/pupil/sclera are intersected with it, so pupil ⊆ iris and
    iris/sclera are disjoint by construction.
    """
    h = w = size

    def ellipse_mask(cx_axis, cy_axis, r_axis):
        mask = np.zeros((h, w), dtype=np.uint8)
        rx = max(1, int(round(r_axis / AXIS_SPAN * w)))
        ry = max(1, int(round(r_axis / AXIS_SPAN * h)))
        cxp, cyp = axis_to_px(cx_axis, cy_axis, size)
        cv2.ellipse(mask, (int(round(cxp)), int(round(cyp))), (rx, ry), 0, 0, 360, 255, -1)
        return mask

    iris = ellipse_mask(iris_xy[0], iris_xy[1], r_iris + LIMBUS_PAD)
    pupil = ellipse_mask(pupil_xy[0], pupil_xy[1], r_pupil)
    # sclera = full palpebral ellipse minus iris(+limbus)
    eye_full = np.zeros((h, w), dtype=np.uint8)
    rx = int(round(EYE_RX / AXIS_SPAN * w))
    ry = int(round(EYE_RY / AXIS_SPAN * h))
    cv2.ellipse(eye_full, (w // 2, h // 2), (rx, ry), 0, 0, 360, 255, -1)
    sclera = cv2.subtract(eye_full, iris)

    # lid masks cover the FULL area above/below the lid curve (A6)
    curve_u = [axis_to_px(float(t), float(y), size) for t, y in zip(t_u, y_u)]
    upper = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(upper, [np.array(curve_u + [(w + 10, -10), (-10, -10)], dtype=np.int32)], 255)
    curve_l = [axis_to_px(float(t), float(y), size) for t, y in zip(t_l, y_l)]
    lower = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lower, [np.array(curve_l + [(w + 10, h + 10), (-10, h + 10)], dtype=np.int32)], 255)

    open_region = np.logical_and(upper == 0, lower == 0)
    iris = np.logical_and(iris > 0, open_region)
    pupil = np.logical_and(pupil > 0, open_region)
    sclera = np.logical_and(sclera > 0, open_region)
    return {
        "iris": (iris.astype(np.uint8)) * 255,
        "sclera": (sclera.astype(np.uint8)) * 255,
        "pupil": (pupil.astype(np.uint8)) * 255,
        "upper_lid": upper,
        "lower_lid": lower,
    }


def build_yolo_line(yolo_cls, cx_axis: float, cy_axis: float, w_axis: float, h_axis: float,
                    A: np.ndarray, scale: float, size: int):
    """YOLO 'class cx cy w h' normalised to [0,1] and transformed to the view (A7).

    Returns None when the box is degenerate (area < 0.0004) or leaves the image.
    """
    if yolo_cls is None:
        return None, None
    px, py = axis_to_px(cx_axis, cy_axis, size)
    p = A @ np.array([px, py, 1.0])
    cx = float(p[0]) / size
    cy = float(p[1]) / size
    bw = w_axis / AXIS_SPAN * scale
    bh = h_axis / AXIS_SPAN * scale
    if cx < 0 or cx > 1 or cy < 0 or cy > 1:
        return None, None
    x1 = max(0.0, cx - bw / 2.0)
    x2 = min(1.0, cx + bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    y2 = min(1.0, cy + bh / 2.0)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or bw * bh < 0.0004:
        return None, None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return f"{int(yolo_cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}", (cx, cy, bw, bh)


# ---------------------------------------------------------------------------
# Post-processing (all centred on the real iris pixel position)
# ---------------------------------------------------------------------------
def apply_skin_gradient(arr: np.ndarray, skin_mask: np.ndarray, strength: float = 0.065) -> np.ndarray:
    """Subtle brow-darker / cheek-lighter shading on non-eye pixels only (A2)."""
    h, w = arr.shape[:2]
    row = (np.arange(h, dtype=np.float32) / max(1, h - 1)) * 2.0 - 1.0      # -1 top .. +1 bottom
    factor = (1.0 + strength * row)[:, None, None].astype(np.float32)
    out = arr.astype(np.float32)
    sel = skin_mask.astype(bool)
    out = np.where(sel[:, :, None], out * factor, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_lighting(arr: np.ndarray, lighting_type: str, gain_override=None, rng=None) -> np.ndarray:
    rs = rng if rng is not None else np.random.RandomState(0)
    if gain_override is not None:
        gain = gain_override
    elif lighting_type == "low_light":
        gain = rs.uniform(0.45, 0.65)
    elif lighting_type == "overexposed":
        gain = rs.uniform(1.20, 1.45)
    else:
        gain = rs.uniform(0.90, 1.10)
    out = np.clip(arr.astype(np.float32) / 255.0 * gain, 0, 1)
    if lighting_type == "glare":
        h, w = out.shape[:2]
        gx, gy = rs.uniform(0.1, 0.9) * w, rs.uniform(0.1, 0.9) * h
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.28 * w) ** 2)))
        out = np.clip(out + blob[:, :, None] * rs.uniform(0.12, 0.28), 0, 1)
    return (out * 255).astype(np.uint8)


def apply_camera_artifacts(arr: np.ndarray, camera_type: str, rng=None, quality_label=3) -> np.ndarray:
    rs = rng if rng is not None else np.random.RandomState(0)
    img = arr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    if camera_type == "phone" and quality_label != 3 and rs.random() < 0.35:
        sigma = rs.uniform(0.4, 1.6)
        ksize = int(6 * sigma + 1) | 1
        img = cv2.GaussianBlur((img * 255).astype(np.uint8), (ksize, ksize), sigma)
        img = img.astype(np.float32) / 255.0
    elif camera_type == "low_res":
        sw = int(rs.randint(52, 88))
        sh = int(rs.randint(52, 88))
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.NEAREST)
        img = np.array(pil, dtype=np.float32) / 255.0
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def add_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur — always applied with the scene's sigma (never skipped)."""
    if sigma <= 0:
        return arr
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(arr, (k, k), sigma)


def add_motion_blur(arr: np.ndarray, length: float, angle_deg: float, rng=None) -> np.ndarray:
    """Directional motion blur that is ALWAYS applied when called (A4)."""
    if length <= 0:
        return arr
    k = int(max(3, round(length)))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1.0, 1)
    ang = float(angle_deg) + (float(rng.uniform(-8, 8)) if rng is not None else 0.0)
    M = cv2.getRotationMatrix2D((k // 2, k // 2), ang, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k), flags=cv2.INTER_LINEAR, borderValue=0)
    s = kernel.sum()
    if s <= 0:
        return arr
    return cv2.filter2D(arr, -1, kernel / s)


def add_perlin_skin_texture(arr: np.ndarray, strength: float, rng) -> np.ndarray:
    """Cheap multi-scale skin texture noise (uses the per-case RNG only)."""
    h, w = arr.shape[:2]
    scales = [rng.uniform(0.4, 0.8), rng.uniform(1.2, 2.0), rng.uniform(3.0, 5.0)]
    weights = [0.55, 0.30, 0.15]
    noise = np.zeros((h, w), dtype=np.float32)
    for scale, weight in zip(scales, weights):
        nx = int(max(4, w * scale / 64))
        ny = int(max(4, h * scale / 64))
        raw = rng.normal(0, 1, (ny, nx)).astype(np.float32)
        noise += cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC) * weight
    noise = ndi.gaussian_filter(noise, sigma=rng.uniform(1.5, 3.5))
    noise = noise - noise.min()
    mx = noise.max()
    if mx > 0:
        noise /= mx
    noise = (noise - 0.5) * strength
    out = np.clip(arr.astype(np.float32) / 255.0 + noise[:, :, None], 0, 1)
    return (out * 255).astype(np.uint8)


def add_tear_film(arr: np.ndarray, cx_px: float, cy_px: float, r_px: float, rng) -> np.ndarray:
    """Tear-film sheen centred on the ACTUAL iris position, not the image centre (A3)."""
    h, w = arr.shape[:2]
    xg, yg = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    dist = np.sqrt((xg - cx_px / w) ** 2 + (yg - cy_px / h) ** 2)
    cornea = (dist < r_px / w).astype(np.float32)
    cornea = ndi.gaussian_filter(cornea, sigma=max(1.0, r_px * 0.12))
    shine_x = cx_px / w + rng.uniform(-0.04, 0.04)
    shine_y = cy_px / h + rng.uniform(-0.04, 0.04)
    d = np.sqrt((xg - shine_x) ** 2 + (yg - shine_y) ** 2)
    specular = np.exp(-d / max(1e-4, (r_px / w) * rng.uniform(0.10, 0.22))) * rng.uniform(0.08, 0.20)
    specular *= cornea
    out = np.clip(arr.astype(np.float32) / 255.0 + specular[:, :, None], 0, 1)
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Lesion drawing (positions come from the scene, so they are view-invariant)
# ---------------------------------------------------------------------------
def draw_lesion(ax, scene: dict, stage: str) -> None:
    lesion = scene.get("lesion")
    if not lesion or not lesion.get("kind"):
        return
    kind = lesion["kind"]
    if stage == "pre" and kind == "hemorrhage":
        ax.add_patch(Ellipse((lesion["cx"], lesion["cy"]), lesion["w"], lesion["h"],
                             facecolor=lesion["color"], alpha=lesion["alpha"], zorder=4))
    elif stage == "pre" and kind == "pinguecula":
        ax.add_patch(Ellipse((lesion["cx"], lesion["cy"]), lesion["w"], lesion["h"],
                             facecolor=(0.93, 0.90, 0.68), alpha=lesion["alpha"], zorder=4))
    elif stage == "pre" and kind == "pterygium":
        pts = np.array([[lesion["base_x"], lesion["y0"]],
                        [lesion["tip_x"], lesion["y1"]],
                        [lesion["base_x"], lesion["y2"]]])
        ax.add_patch(Polygon(pts, facecolor=(0.82, 0.78, 0.62),
                             alpha=lesion["alpha"], zorder=4))
    elif stage == "mid" and kind == "keratitis":
        ax.add_patch(Ellipse((lesion["cx"], lesion["cy"]), lesion["w"], lesion["h"],
                             facecolor=(0.80, 0.77, 0.58), alpha=lesion["alpha"], zorder=10))
    elif stage == "mid" and kind == "scar":
        n_verts = 12
        angles = np.linspace(0, 2 * math.pi, n_verts, endpoint=False)
        jit = np.linspace(0.72, 1.28, n_verts)
        verts = [(lesion["cx"] + lesion["w"] / 2 * j * math.cos(a),
                  lesion["cy"] + lesion["h"] / 2 * j * math.sin(a))
                 for j, a in zip(jit, angles)]
        ax.add_patch(Polygon(verts, facecolor=lesion["color"], alpha=lesion["alpha"],
                             edgecolor=(0.78, 0.75, 0.68), linewidth=0.5, zorder=10))
    elif stage == "mid" and kind == "foreign_body":
        for (fx, fy, fr) in lesion.get("particles", []):
            ax.add_patch(Ellipse((fx, fy), fr * 2, fr * 1.5,
                                 facecolor=(0.08, 0.06, 0.04), alpha=0.95, zorder=10.5))
    elif stage == "post" and kind == "lid_margin":
        ax.add_patch(Ellipse((lesion["cx"], lesion["cy"]), lesion["w"], lesion["h"],
                             facecolor=(0.80, 0.72, 0.50), alpha=0.9, zorder=12.5))
        for k in range(7):
            off = (k - 3) * lesion["w"] / 7.0
            ax.add_patch(Ellipse((lesion["cx"] + off, lesion["cy"] + 0.008),
                                 0.03, 0.02, facecolor=(0.72, 0.62, 0.40),
                                 alpha=0.85, zorder=12.5))


# ---------------------------------------------------------------------------
# render_scene — the single renderer used by every class (A1, A3, A9)
# ---------------------------------------------------------------------------
def render_scene(scene: dict, view: str, case_seed: int, train_px: int = 224) -> dict:
    """Render one gaze view of a sampled scene.

    Returns dict with keys: small, full, masks, yolo, box, iris_px, scale.
    All RNG used here derives from case_seed, so a re-render is identical.
    """
    cid = scene["cid"]
    size = GEN_SIZE
    _ = random.Random(case_seed)
    rn = np.random.RandomState(case_seed)

    # suppress unused warning
    _ = cid

    # --- gaze geometry (lids/sclera/skin stay fixed, iris moves) (A3) ------------------
    dx, dy = _GAZE_DELTAS.get(view, (0.0, 0.0))
    if scene["eye_side"] == "left":
        dx = -dx                     # pre-mirror so the FINAL direction is correct
    base_x, base_y = scene["strabismus"]
    gdx, gdy = clip_gaze(base_x, base_y, dx, dy, scene["iris_radius"])
    ix, iy = base_x + gdx, base_y + gdy

    # --- lid curves (shared by drawing and masks) ---------------------------------------
    t_u, y_u = eyelid_curve(upper=True, arch=scene["arch_u"], asym_k=scene["asym_k"],
                            occlusion=scene["occl_u"], corner_tilt=scene["corner_tilt"],
                            aspect=scene["aspect"])
    t_l, y_l = eyelid_curve(upper=False, arch=scene["arch_l"], asym_k=scene["asym_k"],
                            occlusion=scene["occl_l"], corner_tilt=scene["corner_tilt"],
                            aspect=scene["aspect"])

    # --- draw the scene ------------------------------------------------------------------
    fig, ax = make_fig_with_skin(scene["skin_rgb"])
    ax._ov_striations = scene["striations"]
    draw_sclera(ax, tint=scene["sclera_tint"], layout=scene["vessels"],
                va=scene["vessel_alpha"])
    draw_lesion(ax, scene, "pre")
    draw_iris(ax, cx=ix, cy=iy, r=scene["iris_radius"], color=scene["iris_rgb"])
    draw_pupil(ax, cx=ix, cy=iy, r=scene["pupil_r"],
               hi_off=scene["hi_off"], hi_r=scene["hi_r"])
    draw_lesion(ax, scene, "mid")
    draw_eyelids(ax, scene["skin_rgb"], t_u, y_u, t_l, y_l,
                 lash_shadow=scene["lash_shadow"], lash_len=scene["lash_len"],
                 lash_alpha=scene["lash_alpha"], crust=scene["crust"],
                 lash_count=scene["lash_count"])
    draw_lid_crease(ax, t_u, y_u, scene["skin_rgb"], intensity=scene["crease"])
    draw_eyebrow(ax, t_u, y_u, scene["brow_rgb"], scene["eyebrow"],
                 density=scene["brow_density"], tilt=scene["brow_tilt"])
    if scene["wrinkles"]:
        add_eyelid_wrinkles(ax, t_u, y_u, scene["skin_rgb"], scene["wrinkles"])
    if scene["glasses"]:
        add_glasses_glare(ax, scene["glare_spots"])
    draw_lesion(ax, scene, "post")
    arr = fig_to_arr(fig, size=size)

    # --- masks from the same geometry ----------------------------------------------------
    masks = make_segmentation_masks(size, (ix, iy), scene["iris_radius"],
                                    (ix, iy), scene["pupil_r"],
                                    t_u, y_u, t_l, y_l)

    # --- view affine (mirror / macro / around / reject-crop) applied to image AND masks ----
    iris_px_base = axis_to_px(ix, iy, size)
    A, scale, iris_px = view_affine(view, scene, iris_px_base, size)
    arr = warp_image(arr, A, scene["skin_rgb"])
    masks = warp_masks(masks, A)

    # --- skin shading gradient on non-eye pixels only (A2) ----------------------------------
    eye_any = np.zeros(arr.shape[:2], dtype=np.uint8)
    for ch in ("iris", "sclera", "pupil"):
        eye_any = np.maximum(eye_any, masks[ch])
    skin_mask = (eye_any == 0)
    arr = apply_skin_gradient(arr, skin_mask)

    # --- appearance / quality ----------------------------------------------------------------
    if scene["skin_texture"]:
        arr = add_perlin_skin_texture(arr, scene["perlin_strength"], rn)
    arr = apply_lighting(arr, scene["lighting"], gain_override=scene["light_gain"], rng=rn)
    arr = apply_camera_artifacts(arr, scene["camera"], rng=rn,
                                 quality_label=scene["quality_label"])
    arr = add_blur(arr, scene["blur_sigma"])
    if scene["motion_blur"]:
        arr = add_motion_blur(arr, scene["motion_len"], scene["motion_angle"], rng=rn)
    if scene["tear_film"]:
        r_px = scene["iris_radius"] / AXIS_SPAN * size * scale
        arr = add_tear_film(arr, iris_px[0], iris_px[1], r_px, rn)

    # --- YOLO box transformed into this view -------------------------------------------------
    lesion = scene.get("lesion") or {}
    yolo_str, box = build_yolo_line(lesion.get("yolo_cls"), lesion.get("cx", 0.0),
                                    lesion.get("cy", 0.0), lesion.get("w", 0.0),
                                    lesion.get("h", 0.0), A, scale, size)
    small = cv2.resize(arr, (train_px, train_px), interpolation=cv2.INTER_LANCZOS4)
    return {"small": small, "full": arr, "masks": masks, "yolo": yolo_str or "",
            "box": box, "iris_px": iris_px, "scale": scale, "A": A}


# --- thin per-class wrappers: scene sampled once, rendering per view (A1) -----------------
def _gen(view: str, train_px: int, scene, case_seed, cid: int):
    if scene is None:
        scene = sample_scene(cid, np.random.RandomState(case_seed or 0))
    if case_seed is None:
        case_seed = (cid * 100003) % (2 ** 31)
    out = render_scene(scene, view, case_seed, train_px)
    return out["small"], out["full"], out["masks"], out["yolo"]


def gen_normal(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 0)


def gen_conjunctivitis(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 1)


def gen_allergic_conjunctivitis(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 2)


def gen_dry_eye(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 3)


def gen_blepharitis(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 4)


def gen_strabismus(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 5)


def gen_ptosis(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 6)


def gen_corneal_scar(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 7)


def gen_keratitis(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 8)


def gen_pinguecula(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 9)


def gen_pterygium(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 10)


def gen_foreign_body(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 11)


def gen_subconjunctival_hemorrhage(view="center", train_px=224, scene=None, case_seed=None):
    return _gen(view, train_px, scene, case_seed, 12)


GENERATORS = {
    0: gen_normal, 1: gen_conjunctivitis, 2: gen_allergic_conjunctivitis,
    3: gen_dry_eye, 4: gen_blepharitis, 5: gen_strabismus, 6: gen_ptosis,
    7: gen_corneal_scar, 8: gen_keratitis, 9: gen_pinguecula,
    10: gen_pterygium, 11: gen_foreign_body, 12: gen_subconjunctival_hemorrhage,
}


# =========================================================================
# Part 3 — QC, worker, split, generate_dataset
# =========================================================================

def qc_check_case(records: list, cid: int, case_idx: int) -> tuple:
    """Per-case QC (B3).
    Returns (passed: bool, reasons: list[str]).
    Checks:
      - pupil ⊆ iris
      - iris ⊆ open_region
      - eye_area ≥ 8% of frame
      - no all-zero image / all-zero masks
      - anatomy checks per disease class (skipped for quality=0)
      - YOLO box inside image when present
    """
    reasons = []
    if not records:
        return False, ["no records"]

    # gather stats
    eye_areas = []
    zero_img = 0
    zero_masks = 0
    for r in records:
        masks = r.get("masks")
        if masks is None:
            continue
        iris = masks["iris"].astype(bool)
        pupil = masks["pupil"].astype(bool)
        upper = masks["upper_lid"].astype(bool)
        lower = masks["lower_lid"].astype(bool)
        open_region = np.logical_and(~upper, ~lower)

        # pupil ⊆ iris
        if pupil.any() and not np.all(pupil[~iris] == False):
            reasons.append(f"{r['view']}: pupil not subset of iris")

        # iris ⊆ open_region
        if iris.any() and not np.all(iris[~open_region] == False):
            reasons.append(f"{r['view']}: iris extends under lids")

        # eye area
        eye_any = np.logical_or(iris, masks["sclera"].astype(bool))
        eye_areas.append(float(eye_any.mean()))

        # blank checks
        if r["full"].sum() == 0:
            zero_img += 1
        if all(m.sum() == 0 for m in masks.values()):
            zero_masks += 1

        # box in image
        if r["yolo"]:
            cx, cy, bw, bh = r["box"]
            if cx < 0 or cx > 1 or cy < 0 or cy > 1 or bw <= 0 or bh <= 0:
                reasons.append(f"{r['view']}: YOLO box out of bounds")

    if zero_img == len(records):
        reasons.append("all images zero")
    if zero_masks == len(records):
        reasons.append("all masks zero")
    if eye_areas and np.mean(eye_areas) < 0.08:
        reasons.append(f"mean eye area {np.mean(eye_areas):.3f} < 0.08")

    # disease-specific anatomy checks (skip for quality 0 rejects)
    if records[0]["quality_label"] > 0:
        if cid == 5:  # strabismus
            for r in records:
                if r["view"] == "center":
                    ix, iy = r["iris_px"]
                    cx = (ix / GEN_SIZE - 0.5) * AXIS_SPAN
                    cy = (0.5 - iy / GEN_SIZE) * AXIS_SPAN
                    bx, by = scene_strabismus_base(cid, case_idx)
                    if abs(cx - bx) < 0.08 and abs(cy - by) < 0.03:
                        reasons.append("strabismus offset not visible in center view")
                    break
        if cid == 6:  # ptosis
            for r in records:
                if r["view"] == "center":
                    masks = r["masks"]
                    upper = masks["upper_lid"].astype(bool)
                    iris = masks["iris"].astype(bool)
                    if upper.any() and iris.any():
                        lid_bottom = np.max(np.where(upper)[0])
                        iris_top = np.min(np.where(iris)[0])
                        if lid_bottom >= iris_top - 2:
                            reasons.append("ptosis: upper lid does not cover iris top")
                    break

    return len(reasons) == 0, reasons


def scene_strabismus_base(cid: int, case_idx: int) -> tuple:
    """Recover the base strabismus offset used for this case (deterministic)."""
    if cid != 5:
        return 0.0, 0.0
    seed = (cid * 100003 + case_idx * 7919) % (2 ** 31)
    rng = np.random.RandomState(seed)
    return (float(rng.choice([-1, 1])) * float(rng.uniform(0.18, 0.36)),
            float(rng.choice([-1, 1])) * float(rng.uniform(0.05, 0.13)))


def _worker(args: tuple) -> dict:
    """Worker: generate all 7 views for one case, run QC, retry ≤3 times,
    write per-case meta.json, return summary for metadata aggregation.
    args = (cid, case_idx, n_cases, out_dir, train_px)
    """
    cid, case_idx, n_cases, out_dir, train_px = args
    class_name = CLASSES[cid]
    case_id = f"{class_name}_{case_idx:05d}"
    case_dir = Path(out_dir) / class_name / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # deterministic per-case seed (A1, A11)
    case_seed = (cid * 100003 + case_idx * 7919) % (2 ** 31)

    for attempt in range(1, 4):
        try:
            # sample scene ONCE
            rng = np.random.RandomState(case_seed)
            scene = sample_scene(cid, rng)
            # render all views
            records = []
            for view in GAZE_VIEWS:
                small, full, masks, yolo_str = GENERATORS[cid](
                    view=view, train_px=train_px, scene=scene, case_seed=case_seed)
                # recompute masks from render_scene (which also gives iris_px, scale, A)
                # We call render_scene again with same seed — deterministic.
                out = render_scene(scene, view, case_seed, train_px)
                out["class_id"] = cid
                out["class_name"] = class_name
                out["case_id"] = case_id
                out["case_idx"] = case_idx
                out["view"] = view
                out["quality_label"] = scene["quality_label"]
                out["eye_side"] = scene["eye_side"]
                out["questionnaire"] = scene["questionnaire"]
                out["sharpness"] = compute_sharpness(out["full"])
                records.append(out)

            # QC
            passed, reasons = qc_check_case(records, cid, case_idx)
            if passed:
                # success — write files
                for r in records:
                    view = r["view"]
                    # images
                    Image.fromarray(r["full"]).save(case_dir / f"{view}.jpg",
                                                   quality=CFG["jpeg_quality"], subsampling=0)
                    if CFG["save_full_res"]:
                        Image.fromarray(r["small"]).save(case_dir / f"{view}_small.jpg",
                                                         quality=CFG["jpeg_quality"], subsampling=0)
                    # masks
                    mdir = case_dir / "masks"
                    mdir.mkdir(exist_ok=True)
                    for ch in MASK_CHANNELS:
                        Image.fromarray(r["masks"][ch]).save(mdir / f"{view}_{ch}.png")
                    # YOLO label
                    if r["yolo"]:
                        (case_dir / f"{view}.txt").write_text(r["yolo"] + "\n")
                    else:
                        (case_dir / f"{view}.txt").write_text("")

                # per-case meta.json (all 7 records)
                meta = {
                    "case_id": case_id,
                    "class_id": cid,
                    "class_name": class_name,
                    "case_idx": case_idx,
                    "quality_label": scene["quality_label"],
                    "eye_side": scene["eye_side"],
                    "questionnaire": scene["questionnaire"],
                    "strabismus_base": scene["strabismus"],
                    "views": []
                }
                for r in records:
                    meta["views"].append({
                        "view": r["view"],
                        "sharpness": r["sharpness"],
                        "iris_px": r["iris_px"],
                        "scale": r["scale"],
                        "yolo": r["yolo"],
                        "box": r["box"],
                        "quality_label": r["quality_label"],
                    })
                (case_dir / "meta.json").write_text(json.dumps(meta, cls=NpEncoder, indent=2))

                return {"status": "ok", "cid": cid, "case_idx": case_idx,
                        "attempts": attempt, "records": records}

            # QC failed — log and retry
            print(f"  [QC retry {attempt}/3] {case_id}: {reasons}")
            if attempt == 3:
                # force-accept on 3rd failure (guarantees completion)
                print(f"  [QC force-accept] {case_id} after 3 attempts")
                for r in records:
                    view = r["view"]
                    Image.fromarray(r["full"]).save(case_dir / f"{view}.jpg",
                                                   quality=CFG["jpeg_quality"], subsampling=0)
                    mdir = case_dir / "masks"
                    mdir.mkdir(exist_ok=True)
                    for ch in MASK_CHANNELS:
                        Image.fromarray(r["masks"][ch]).save(mdir / f"{view}_{ch}.png")
                    if r["yolo"]:
                        (case_dir / f"{view}.txt").write_text(r["yolo"] + "\n")
                    else:
                        (case_dir / f"{view}.txt").write_text("")
                meta = {
                    "case_id": case_id,
                    "class_id": cid,
                    "class_name": class_name,
                    "case_idx": case_idx,
                    "quality_label": scene["quality_label"],
                    "eye_side": scene["eye_side"],
                    "questionnaire": scene["questionnaire"],
                    "strabismus_base": scene["strabismus"],
                    "views": [],
                    "qc_forced": True,
                    "qc_reasons": reasons
                }
                for r in records:
                    meta["views"].append({
                        "view": r["view"],
                        "sharpness": r["sharpness"],
                        "iris_px": r["iris_px"],
                        "scale": r["scale"],
                        "yolo": r["yolo"],
                        "box": r["box"],
                        "quality_label": r["quality_label"],
                    })
                (case_dir / "meta.json").write_text(json.dumps(meta, cls=NpEncoder, indent=2))
                return {"status": "forced", "cid": cid, "case_idx": case_idx,
                        "attempts": attempt, "reasons": reasons, "records": records}

        except Exception as e:
            print(f"  [Worker error] {case_id} attempt {attempt}: {e}")
            if attempt == 3:
                return {"status": "error", "cid": cid, "case_idx": case_idx,
                        "error": str(e)}

    return {"status": "error", "cid": cid, "case_idx": case_idx,
            "error": "exhausted retries"}


def split_for_case(cid: int, case_idx: int, n_cases: int) -> str:
    """Deterministic split keyed on (class, case_idx) (B1).
    train=70%, val=15%, test=15% stratified per class — exact counts.
    """
    # Create a deterministic permutation of indices for this class
    rng = random.Random(SPLIT_SEED * 1000003 + cid * 10007)
    indices = list(range(n_cases))
    rng.shuffle(indices)
    # Exact counts
    n_train = int(round(0.70 * n_cases))
    n_val = int(round(0.15 * n_cases))
    # Assign
    pos = indices.index(case_idx)
    if pos < n_train:
        return "train"
    if pos < n_train + n_val:
        return "val"
    return "test"


def plan_split(cids: list, n_cases: int) -> dict:
    """Compute how many cases per class per split, applying MIN_TEST_CASES floor for
    classes 5,6,11 (strabismus, ptosis, foreign_body) — only if test+val ≤ 50% of cases.
    """
    plan = {cid: {"train": [], "val": [], "test": []} for cid in cids}
    for cid in cids:
        for ci in range(n_cases):
            plan[cid][split_for_case(cid, ci, n_cases)].append(ci)

    # special floor for rare classes (B1)
    for cid in (5, 6, 11):
        n_test = len(plan[cid]["test"])
        _ = len(plan[cid]["val"])
        if n_test < MIN_TEST_CASES:
            needed = MIN_TEST_CASES - n_test
            # try to move from train → test
            available = len(plan[cid]["train"])
            move = min(needed, available)
            if move > 0:
                plan[cid]["test"].extend(plan[cid]["train"][-move:])
                plan[cid]["train"] = plan[cid]["train"][:-move]
                n_test = len(plan[cid]["test"])
        # warn if still short and test+val > 50%
        n_total = n_cases
        if len(plan[cid]["test"]) + len(plan[cid]["val"]) > n_total / 2:
            print(f"  [WARN] class {CLASSES[cid]}: test+val > 50% of cases "
                  f"({len(plan[cid]['test']) + len(plan[cid]['val'])}/{n_total}) "
                  f"— MIN_TEST_CASES not enforced")
    return plan


def generate_dataset(out_dir: str = None, cases_per_class: int = None,
                     train_px: int = None, resume: bool = True) -> dict:
    """Full staged generation:
      1) sample & render all cases (parallel workers, per-case meta.json)
      2) rebuild global metadata from per-case meta.json (atomic)
      3) write manifest + images.csv
    Returns summary dict.
    """
    out_dir = Path(out_dir or CFG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cases = cases_per_class or CFG["cases_per_class"]
    px = train_px or CFG["img_size"]
    workers = CFG["gen_workers"]

    print(f"\n{'='*60}")
    print("  OcuVision v7 — GENERATE DATASET")
    print(f"  Output: {out_dir}")
    print(f"  Classes: {len(CLASSES)} × {n_cases} cases = {len(CLASSES)*n_cases} cases")
    print(f"  Views: {GAZE_VIEWS}")
    print(f"  Workers: {workers}")
    print(f"{'='*60}\n")

    # prepare work list
    work = [(cid, ci, n_cases, str(out_dir), px) for cid in CLASSES for ci in range(n_cases)]

    # resume: skip cases that already have valid meta.json with 7 views
    if resume:
        pending = []
        for cid, ci, nc, od, tp in work:
            cname = CLASSES[cid]
            cdir = Path(od) / cname / f"{cname}_{ci:05d}"
            meta_p = cdir / "meta.json"
            if meta_p.exists():
                try:
                    with open(meta_p) as f:
                        m = json.load(f)
                    if len(m.get("views", [])) == 7:
                        continue
                except Exception:
                    pass
            pending.append((cid, ci, nc, od, tp))
        print(f"  Resume: {len(work) - len(pending)} cases already complete, "
              f"{len(pending)} pending")
        work = pending

    # split plan for metadata
    split_plan = plan_split(list(CLASSES.keys()), n_cases)

    # parallel generation
    total = len(work)
    completed = 0
    forced = 0
    errors = 0
    all_summaries = []

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_worker, work, chunksize=1):
            completed += 1
            all_summaries.append(result)
            if result["status"] == "forced":
                forced += 1
            elif result["status"] == "error":
                errors += 1
            if completed % 50 == 0 or completed == total:
                print(f"  [{completed}/{total}] ok={completed-forced-errors} "
                      f"forced={forced} errors={errors}")
            if _stop_requested.is_set():
                print("  Stop requested — finishing current batch...")
                break

    # rebuild global metadata.json from per-case meta.json (atomic)
    print("\n  Rebuilding global metadata.json...")
    metadata = {"cases": [], "split": split_plan, "config": {k: v for k, v in CFG.items()
                if not callable(v) and not k.startswith("_")}}
    metadata["config"]["num_cases_per_class"] = n_cases
    metadata["config"]["generation_version"] = GENERATOR_VERSION

    images_rows = []
    rejection_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for cid in CLASSES:
        cname = CLASSES[cid]
        for ci in range(n_cases):
            cdir = out_dir / cname / f"{cname}_{ci:05d}"
            meta_p = cdir / "meta.json"
            if not meta_p.exists():
                continue
            try:
                with open(meta_p) as f:
                    m = json.load(f)
            except Exception:
                continue

            # Use split from plan (exact stratified with MIN_TEST_CASES applied)
            if ci in split_plan[cid]["train"]:
                split = "train"
            elif ci in split_plan[cid]["val"]:
                split = "val"
            else:
                split = "test"

            case_entry = {
                "case_id": m["case_id"],
                "class_id": m["class_id"],
                "class_name": m["class_name"],
                "case_idx": m["case_idx"],
                "split": split,
                "quality_label": m["quality_label"],
                "eye_side": m["eye_side"],
                "questionnaire": m["questionnaire"],
                "strabismus_base": m["strabismus_base"],
                "views": m["views"]
            }
            metadata["cases"].append(case_entry)
            rejection_counts[m["quality_label"]] += 1

            for v in m["views"]:
                images_rows.append({
                    "case_id": m["case_id"],
                    "class_id": cid,
                    "class_name": cname,
                    "split": split,
                    "view": v["view"],
                    "quality_label": v["quality_label"],
                    "sharpness": v["sharpness"],
                    "iris_px": v["iris_px"],
                    "scale": v["scale"],
                    "has_yolo": bool(v["yolo"]),
                    "yolo_box": v["box"]
                })

    # atomic write metadata.json
    meta_path = out_dir / "metadata.json"
    tmp = meta_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(metadata, f, cls=NpEncoder, indent=2)
    tmp.replace(meta_path)

    # images.csv
    import csv
    csv_path = out_dir / "images.csv"
    tmp_csv = csv_path.with_suffix(".tmp")
    with open(tmp_csv, "w", newline="") as f:
        if images_rows:
            writer = csv.DictWriter(f, fieldnames=list(images_rows[0].keys()))
            writer.writeheader()
            for row in images_rows:
                writer.writerow(row)
    tmp_csv.replace(csv_path)

    # manifest with rejection rate
    total_cases = len(metadata["cases"])
    rejection_rate = {k: v / total_cases if total_cases else 0.0
                      for k, v in rejection_counts.items()}
    manifest = {
        "generation_version": GENERATOR_VERSION,
        "total_cases": total_cases,
        "total_images": len(images_rows),
        "cases_per_class": n_cases,
        "splits": {s: sum(1 for c in metadata["cases"] if c["split"] == s)
                   for s in ("train", "val", "test")},
        "quality_distribution": rejection_counts,
        "rejection_rate": rejection_rate,
        "forced_qc_count": forced,
        "errors": errors,
        "classes": {cid: cname for cid, cname in CLASSES.items()},
        "views": GAZE_VIEWS,
        "quality_labels": QUALITY_LABELS,
        "lesion_classes": LESION_CLASSES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    man_path = out_dir / "dataset_manifest.json"
    tmp_man = man_path.with_suffix(".tmp")
    with open(tmp_man, "w") as f:
        json.dump(manifest, f, cls=NpEncoder, indent=2)
    tmp_man.replace(man_path)

    # rejected previews under quality_control/rejected/
    if forced > 0:
        rej_dir = out_dir / "quality_control" / "rejected"
        rej_dir.mkdir(parents=True, exist_ok=True)
        for res in all_summaries:
            if res["status"] == "forced":
                for r in res["records"]:
                    view = r["view"]
                    dst = rej_dir / f"{res['case_idx']:05d}_{CLASSES[res['cid']]}_{view}.jpg"
                    Image.fromarray(r["full"]).save(dst, quality=80)

    print("\n  GENERATE complete:")
    print(f"    cases: {total_cases}  images: {len(images_rows)}")
    print(f"    splits: {manifest['splits']}")
    print(f"    quality dist: {rejection_counts}")
    print(f"    rejection rates: { {k: f'{v:.3f}' for k,v in rejection_rate.items()} }")
    print(f"    forced: {forced}  errors: {errors}")
    return manifest


# =========================================================================
# Part 4 — Datasets, Models (M1/M2/M4/Fusion), Training
# =========================================================================

class EyeDataset(Dataset):
    """Classification dataset (M1/M2) reading from metadata.json."""

    def __init__(self, meta_path: Path, split: str, img_size: int,
                 augment: bool = True, use_questionnaire: bool = False):
        with open(meta_path) as f:
            meta = json.load(f)
        self.records = [c for c in meta["cases"] if c["split"] == split]
        self.img_size = img_size
        self.augment = augment
        self.use_questionnaire = use_questionnaire
        self.split = split
        self.meta_path = meta_path
        self.base_dir = meta_path.parent

        self.tf_base = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_aug = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(0.15, 0.15, 0.15, 0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        case = self.records[idx]
        # pick a random view for training; fixed center for val/test
        if self.split == "train" and self.augment:
            view_rec = random.choice(case["views"])
        else:
            view_rec = next(v for v in case["views"] if v["view"] == "center")
        img_path = self.base_dir / case["class_name"] / case["case_id"] / f"{view_rec['view']}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = (self.tf_aug if (self.split == "train" and self.augment) else self.tf_base)(img)
        label = case["class_id"]
        q = torch.tensor(case["questionnaire"], dtype=torch.float32) if self.use_questionnaire else None
        return (img, q, label) if self.use_questionnaire else (img, label)


class SegDataset(Dataset):
    """Segmentation dataset (M4) reading full-res images + 5-channel masks."""

    def __init__(self, meta_path: Path, split: str, seg_size: int, augment: bool = True):
        with open(meta_path) as f:
            meta = json.load(f)
        self.records = []
        for c in meta["cases"]:
            if c["split"] != split:
                continue
            for v in c["views"]:
                self.records.append((c, v))
        self.seg_size = seg_size
        self.augment = augment
        self.base_dir = meta_path.parent
        self.tf_img = T.Compose([
            T.Resize((seg_size, seg_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_mask = T.Compose([
            T.Resize((seg_size, seg_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.tf_aug = T.Compose([
            T.Resize((seg_size, seg_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        case, view_rec = self.records[idx]
        img_path = self.base_dir / case["class_name"] / case["case_id"] / f"{view_rec['view']}.jpg"
        mdir = self.base_dir / case["class_name"] / case["case_id"] / "masks"
        img = Image.open(img_path).convert("RGB")
        masks = []
        for ch in MASK_CHANNELS:
            m = Image.open(mdir / f"{view_rec['view']}_{ch}.png")
            masks.append(self.tf_mask(m).squeeze(0))
        mask = torch.stack(masks, dim=0)  # (5, H, W)
        img = (self.tf_aug if (self.augment and random.random() < 0.5) else self.tf_img)(img)
        return img, mask


class FusionDataset(Dataset):
    """Fusion dataset: cached M1 logits (13) + M2 logits (13) + questionnaire (6) -> 32 dim."""

    def __init__(self, meta_path: Path, split: str, m1_logits: np.ndarray,
                 m2_logits: np.ndarray, questionnaire: np.ndarray):
        self.meta_path = meta_path
        with open(meta_path) as f:
            meta = json.load(f)
        self.records = [c for c in meta["cases"] if c["split"] == split]
        self.m1_logits = m1_logits
        self.m2_logits = m2_logits
        self.questionnaire = questionnaire
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        case = self.records[idx]
        # find global index in metadata (same order as cached arrays)
        # we stored cases in metadata in the same order; here we need a mapping
        # For simplicity, we rebuild a mapping from case_id to global idx
        if not hasattr(self, "_id_to_idx"):
            with open(self.meta_path) as f:
                all_cases = json.load(f)["cases"]
            self._id_to_idx = {c["case_id"]: i for i, c in enumerate(all_cases)}
        gid = self._id_to_idx[case["case_id"]]
        x = np.concatenate([
            self.m1_logits[gid],
            self.m2_logits[gid],
            self.questionnaire[gid]
        ], axis=0).astype(np.float32)
        y = case["class_id"]
        return torch.from_numpy(x), y


# --- Models ---
class M1_EyeNet(nn.Module):
    """M1: MobileNetV3-Small backbone, 13-class head (B2)."""

    def __init__(self, num_classes=13, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=0)
        # mobilenetv3_small_100 outputs 1024 after global pool when num_classes=0
        feat = 1024
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class M2_EfficientNet(nn.Module):
    """M2: EfficientNet-Lite0 backbone, 13-class head (B2)."""

    def __init__(self, num_classes=13, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_lite0", pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features  # 1280
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class M4_SegNet(nn.Module):
    """M4: U-Net (MobileNetV2 encoder) 3-channel in -> 5-channel mask (B2)."""

    def __init__(self, in_channels=3, out_channels=5):
        super().__init__()
        # encoder using timm features_only
        self.encoder = timm.create_model("mobilenetv2_100", pretrained=True, features_only=True)
        # feature channels from encoder: [16, 24, 32, 96, 320]
        self.enc_channels = [16, 24, 32, 96, 320]
        # decoder
        self.dec = nn.ModuleList()
        ch = self.enc_channels[-1]  # 320
        for i in range(len(self.enc_channels) - 2, -1, -1):
            self.dec.append(nn.Sequential(
                nn.Conv2d(ch + self.enc_channels[i], self.enc_channels[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(self.enc_channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.enc_channels[i], self.enc_channels[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(self.enc_channels[i]),
                nn.ReLU(inplace=True),
            ))
            ch = self.enc_channels[i]
        self.final = nn.Conv2d(ch, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        feats = self.encoder(x)  # list of 5 tensors
        x = feats[-1]
        for i, dec in enumerate(self.dec):
            x = self.upsample(x)
            skip = feats[-(i + 2)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        # Final upsample from 128x128 to 256x256
        x = self.upsample(x)
        return self.final(x)


class FusionMLP(nn.Module):
    """Fusion MLP: (13+13+6)=32 -> 256 -> 128 -> 13 (B2)."""

    def __init__(self, in_dim=32, num_classes=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# --- losses ---
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Dice loss for segmentation (5 channels)."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


# --- training loop ---
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion, optimizer, scheduler, device, epochs: int,
                model_name: str, grad_accum: int = 1,
                early_stop_patience: int = 6, mixed_precision: bool = True) -> dict:
    """Generic training with:
       - gradient accumulation
       - final partial batch handling
       - clip_grad_norm_ 1.0
       - best by val loss
       - early stopping (patience)
       - BCE+Dice for seg, CE for cls
       - autocast only on CUDA
       - per-epoch metrics (loss, acc, macro-F1 for cls; loss, mean IoU for seg)
    Returns history dict and best state dict.
    """
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_loss = float("inf")
    patience = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": [], "val_iou": []}
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            if len(batch) == 3:
                xb, _, yb = batch
            else:
                xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if isinstance(model, M4_SegNet):
                    out = model(xb)
                    loss = criterion(out, yb.to(device, non_blocking=True))
                else:
                    out = model(xb)
                    loss = criterion(out, yb)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * grad_accum * xb.size(0)
            if not isinstance(model, M4_SegNet):
                preds = out.argmax(dim=1)
                train_correct += (preds == yb).sum().item()
                train_total += yb.size(0)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(yb.cpu().numpy())

        train_loss /= max(1, train_total if train_total else len(train_loader.dataset))
        history["train_loss"].append(train_loss)
        if train_total:
            from sklearn.metrics import f1_score
            acc = train_correct / train_total
            f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)
            history["train_acc"].append(acc)
            history["train_f1"].append(f1)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        val_iou_sum = 0.0
        val_iou_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    xb, _, yb = batch
                else:
                    xb, yb = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if isinstance(model, M4_SegNet):
                        out = model(xb)
                        loss = criterion(out, yb)
                        # IoU
                        probs = torch.sigmoid(out)
                        preds = (probs > 0.5).float()
                        inter = (preds * yb).sum(dim=(0, 2, 3))
                        union = preds.sum(dim=(0, 2, 3)) + yb.sum(dim=(0, 2, 3)) - inter
                        iou = (inter / (union + 1e-6)).mean().item()
                        val_iou_sum += iou * xb.size(0)
                        val_iou_count += xb.size(0)
                    else:
                        out = model(xb)
                        loss = criterion(out, yb)
                        preds = out.argmax(dim=1)
                        val_correct += (preds == yb).sum().item()
                        val_total += yb.size(0)
                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(yb.cpu().numpy())
                val_loss += loss.item() * xb.size(0)

        val_loss /= max(1, val_total if val_total else len(val_loader.dataset))
        history["val_loss"].append(val_loss)
        if val_total:
            from sklearn.metrics import f1_score
            acc = val_correct / val_total
            f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
            history["val_acc"].append(acc)
            history["val_f1"].append(f1)
        if val_iou_count:
            history["val_iou"].append(val_iou_sum / val_iou_count)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # logging
        if isinstance(model, M4_SegNet):
            print(f"  {model_name} epoch {epoch:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_IoU={history['val_iou'][-1] if history['val_iou'] else 0:.4f}")
        else:
            print(f"  {model_name} epoch {epoch:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"train_acc={history['train_acc'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}  "
                  f"val_f1={history['val_f1'][-1]:.4f}")

        # early stopping / best save
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (patience {early_stop_patience})")
                break

    # restore best
    if best_state:
        model.load_state_dict(best_state)
    return history, best_state


def train_all(meta_path: Path, models_dir: Path, results_dir: Path,
              batch: int, batch_seg: int, epochs: int, lr: float,
              num_workers: int, mixed_precision: bool,
              use_questionnaire_in_fusion: bool, grad_accum: int,
              early_stop_patience: int) -> dict:
    """Train M1, M2, M4, FusionMLP (B2).
    Returns dict with paths to best checkpoints and metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training on {device}")
    workers = _loader_workers(num_workers)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- class weights from train split ----
    with open(meta_path) as f:
        meta = json.load(f)
    train_cases = [c for c in meta["cases"] if c["split"] == "train"]
    cls_counts = np.zeros(CFG["num_classes"])
    for c in train_cases:
        cls_counts[c["class_id"]] += 1
    cls_weights = 1.0 / np.clip(cls_counts, 1, None)
    cls_weights = cls_weights / cls_weights.sum() * CFG["num_classes"]
    cls_weights_t = torch.tensor(cls_weights, dtype=torch.float32, device=device)

    # ---- M1 ----
    print("\n  ==== M1: MobileNetV3-Small ====")
    m1 = M1_EyeNet(num_classes=CFG["num_classes"], pretrained=True).to(device)
    train_ds = EyeDataset(meta_path, "train", CFG["img_size"], augment=True, use_questionnaire=False)
    val_ds = EyeDataset(meta_path, "val", CFG["img_size"], augment=False, use_questionnaire=False)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW(m1.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=cls_weights_t)
    hist1, best1 = train_model(m1, train_loader, val_loader, crit, opt, sched, device,
                               epochs, "M1", grad_accum, early_stop_patience, mixed_precision)
    m1_path = models_dir / "m1_best.pt"
    torch.save({"model": m1.state_dict(), "config": {"arch": "mobilenetv3_small", "num_classes": CFG["num_classes"]}}, m1_path)

    # ---- M2 ----
    print("\n  ==== M2: EfficientNet-Lite0 ====")
    m2 = M2_EfficientNet(num_classes=CFG["num_classes"], pretrained=True).to(device)
    train_ds = EyeDataset(meta_path, "train", CFG["img_size"], augment=True, use_questionnaire=False)
    val_ds = EyeDataset(meta_path, "val", CFG["img_size"], augment=False, use_questionnaire=False)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW(m2.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=cls_weights_t)
    hist2, best2 = train_model(m2, train_loader, val_loader, crit, opt, sched, device,
                               epochs, "M2", grad_accum, early_stop_patience, mixed_precision)
    m2_path = models_dir / "m2_best.pt"
    torch.save({"model": m2.state_dict(), "config": {"arch": "efficientnet_lite0", "num_classes": CFG["num_classes"]}}, m2_path)

    # ---- M4 ----
    print("\n  ==== M4: U-Net Segmentation ====")
    m4 = M4_SegNet(in_channels=3, out_channels=5).to(device)
    train_ds = SegDataset(meta_path, "train", CFG["seg_size"], augment=True)
    val_ds = SegDataset(meta_path, "val", CFG["seg_size"], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_seg, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_seg, shuffle=False,
                            num_workers=workers, pin_memory=True, drop_last=False)
    opt = torch.optim.AdamW(m4.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    hist4, best4 = train_model(m4, train_loader, val_loader, bce_dice_loss, opt, sched, device,
                               epochs, "M4", grad_accum, early_stop_patience, mixed_precision)
    m4_path = models_dir / "m4_best.pt"
    torch.save({"model": m4.state_dict(), "config": {"arch": "unet_mobilenetv2", "in_channels": 3, "out_channels": 5}}, m4_path)

    # ---- Cache M1/M2 logits for Fusion ----
    print("\n  Caching M1/M2 logits for Fusion...")
    m1.eval()
    m2.eval()
    # Need all splits for fusion cache; iterate metadata order
    with open(meta_path) as f:
        all_cases = json.load(f)["cases"]
    m1_logits = np.zeros((len(all_cases), CFG["num_classes"]), dtype=np.float32)
    m2_logits = np.zeros((len(all_cases), CFG["num_classes"]), dtype=np.float32)
    questionnaire = np.zeros((len(all_cases), 6), dtype=np.float32)
    # Simpler: create a dataset that returns all cases in metadata order.
    class AllCasesDataset(Dataset):
        def __init__(self, meta_path, img_size):
            with open(meta_path) as f:
                self.cases = json.load(f)["cases"]
            self.img_size = img_size
            self.base_dir = Path(meta_path).parent
            self.tf = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        def __len__(self):
            return len(self.cases)
        def __getitem__(self, idx):
            case = self.cases[idx]
            v = next(v for v in case["views"] if v["view"] == "center")
            img_path = self.base_dir / case["class_name"] / case["case_id"] / f"{v['view']}.jpg"
            img = Image.open(img_path).convert("RGB")
            return self.tf(img), idx
    all_ds = AllCasesDataset(meta_path, CFG["img_size"])
    all_loader = DataLoader(all_ds, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
    with torch.no_grad():
        for xb, idxs in all_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=mixed_precision and device.type=="cuda"):
                m1_out = m1(xb)
                m2_out = m2(xb)
            for i, gid in enumerate(idxs.cpu().numpy()):
                m1_logits[gid] = m1_out[i].cpu().numpy()
                m2_logits[gid] = m2_out[i].cpu().numpy()
                questionnaire[gid] = all_cases[gid]["questionnaire"]

    # ---- FusionMLP ----
    print("\n  ==== FusionMLP ====")
    fusion_ds_train = FusionDataset(meta_path, "train", m1_logits, m2_logits, questionnaire)
    fusion_ds_val = FusionDataset(meta_path, "val", m1_logits, m2_logits, questionnaire)
    fusion_ds_test = FusionDataset(meta_path, "test", m1_logits, m2_logits, questionnaire)
    train_loader = DataLoader(fusion_ds_train, batch_size=512, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(fusion_ds_val, batch_size=512, shuffle=False,
                            num_workers=0, pin_memory=True)
    # test_loader created later for evaluation
    fusion = FusionMLP(in_dim=32, num_classes=CFG["num_classes"]).to(device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    hist_f, best_f = train_model(fusion, train_loader, val_loader, crit, opt, sched, device,
                                 epochs, "Fusion", grad_accum, early_stop_patience, mixed_precision)
    fusion_path = models_dir / "fusion_best.pt"
    torch.save({"model": fusion.state_dict(), "config": {"in_dim": 32, "num_classes": CFG["num_classes"]}}, fusion_path)

    # ---- Test evaluation with classification_report + confusion matrix ----
    print("\n  ==== TEST EVALUATION ====")
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    def eval_model(model, name, loader, use_questionnaire=False):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                if use_questionnaire:
                    xb, qb, yb = batch
                else:
                    xb, yb = batch
                xb = xb.to(device)
                with torch.cuda.amp.autocast(enabled=mixed_precision and device.type=="cuda"):
                    out = model(xb)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                targets.extend(yb.cpu().numpy())
        # Use labels parameter to handle missing classes in test set
        labels = list(range(CFG["num_classes"]))
        target_names = [CLASSES[i] for i in labels]
        report = classification_report(targets, preds, labels=labels, target_names=target_names,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=labels)
        # save confusion matrix PNG
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names,
                    yticklabels=target_names, cmap="Blues")
        plt.title(f"{name} Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(results_dir / f"{name.lower()}_confusion_test.png", dpi=150)
        plt.close()
        # print summary
        print(f"\n  {name} Test Report:")
        print(classification_report(targets, preds, labels=labels, target_names=target_names,
                                     zero_division=0, digits=4))
        return report, cm

    m1_test_ds = EyeDataset(meta_path, "test", CFG["img_size"], augment=False)
    m1_test_loader = DataLoader(m1_test_ds, batch_size=batch, shuffle=False,
                                num_workers=workers, pin_memory=True)
    m1_report, m1_cm = eval_model(m1, "M1", m1_test_loader)

    m2_test_ds = EyeDataset(meta_path, "test", CFG["img_size"], augment=False)
    m2_test_loader = DataLoader(m2_test_ds, batch_size=batch, shuffle=False,
                                num_workers=workers, pin_memory=True)
    m2_report, m2_cm = eval_model(m2, "M2", m2_test_loader)

    fusion_test_loader = DataLoader(fusion_ds_test, batch_size=512, shuffle=False,
                                    num_workers=0, pin_memory=True)
    fusion_report, fusion_cm = eval_model(fusion, "Fusion", fusion_test_loader)

    # warning for fusion questionnaire
    print("\n  [WARNING] FusionMLP accuracy on synthetic questionnaires is NOT meaningful.")
    print("  Synthetic questionnaires are noisy proxies (A8) and do not reflect real clinical data.")

    return {
        "m1_path": str(m1_path), "m2_path": str(m2_path),
        "m4_path": str(m4_path), "fusion_path": str(fusion_path),
        "m1_report": m1_report, "m2_report": m2_report,
        "fusion_report": fusion_report,
    }


# =========================================================================
# Part 5 — YOLO-prep, Export ONNX, Inference, Pipeline, CLI
# =========================================================================

def prepare_yolo_dataset(meta_path: Path, out_dir: Path, yolo_all_views: bool = False) -> dict:
    """Create YOLO-format dataset (images + labels) from metadata (A7).
    - Unique filenames: class_case_view.jpg
    - Wipe out_dir first
    - Only LESION_BOX_CLASSES get labels
    - Center-only unless yolo_all_views=True
    Returns summary dict.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sp in ("train", "val", "test"):
        (out_dir / "images" / sp).mkdir(parents=True)
        (out_dir / "labels" / sp).mkdir(parents=True)

    with open(meta_path) as f:
        meta = json.load(f)

    counts = {"train": 0, "val": 0, "test": 0}
    for case in meta["cases"]:
        cid = case["class_id"]
        if cid not in LESION_BOX_CLASSES:
            continue
        cname = case["class_name"]
        views_to_use = GAZE_VIEWS if yolo_all_views else ["center"]
        for v in case["views"]:
            if v["view"] not in views_to_use:
                continue
            if not v["yolo"]:
                continue
            src_img = Path(meta_path).parent / cname / case["case_id"] / f"{v['view']}.jpg"
            if not src_img.exists():
                continue
            dst_name = f"{cname}_{case['case_id']}_{v['view']}.jpg"
            dst_img = out_dir / "images" / case["split"] / dst_name
            shutil.copy2(src_img, dst_img)
            dst_lbl = out_dir / "labels" / case["split"] / f"{cname}_{case['case_id']}_{v['view']}.txt"
            dst_lbl.write_text(v["yolo"] + "\n")
            counts[case["split"]] += 1

    # data.yaml
    yaml_content = f"""path: {out_dir.absolute()}
train: images/train
val: images/val
test: images/test
nc: {len(LESION_CLASSES)}
names: {list(LESION_CLASSES.values())}
"""
    (out_dir / "data.yaml").write_text(yaml_content)

    print(f"  YOLO dataset: train={counts['train']} val={counts['val']} test={counts['test']}")
    return counts


def train_yolo(data_yaml: Path, models_dir: Path, epochs: int = 50, imgsz: int = 640,
               batch: int = 16, device: str = "0") -> Path:
    """Train YOLOv8n on the prepared dataset (B4). Returns best.pt path."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                          batch=batch, device=device, project=str(models_dir),
                          name="yolo_lesion", exist_ok=True, verbose=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, models_dir / "yolo_best.pt")
    return models_dir / "yolo_best.pt"


def _verify_model2_shape(model, input_shape=(1, 3, 224, 224)):
    """Ensure model output is [B, num_classes]."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy = torch.randn(input_shape, device=device)
        out = model(dummy)
    if out.shape != (input_shape[0], CFG["num_classes"]):
        raise ValueError(f"Model output shape {out.shape} != expected {(input_shape[0], CFG['num_classes'])}")


def export_onnx_models(models_dir: Path, meta_path: Path, results_dir: Path) -> dict:
    """Export M1, M2, M4, Fusion to ONNX (opset 17) with:
       - dynamic batch axis
       - onnx.checker verification
       - torch vs onnx max diff < 1e-3
       - INT8 QDQ (quantize_dynamic) parity check on 50 val images
       - export_manifest.json
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)

    # load checkpoints
    m1_ckpt = torch.load(models_dir / "m1_best.pt", map_location=device)
    m2_ckpt = torch.load(models_dir / "m2_best.pt", map_location=device)
    m4_ckpt = torch.load(models_dir / "m4_best.pt", map_location=device)
    fusion_ckpt = torch.load(models_dir / "fusion_best.pt", map_location=device)

    m1 = M1_EyeNet(num_classes=CFG["num_classes"], pretrained=False).to(device)
    m1.load_state_dict(m1_ckpt["model"])
    m1.eval()

    m2 = M2_EfficientNet(num_classes=CFG["num_classes"], pretrained=False).to(device)
    m2.load_state_dict(m2_ckpt["model"])
    m2.eval()

    m4 = M4_SegNet(in_channels=3, out_channels=5).to(device)
    m4.load_state_dict(m4_ckpt["model"])
    m4.eval()

    fusion = FusionMLP(in_dim=32, num_classes=CFG["num_classes"]).to(device)
    fusion.load_state_dict(fusion_ckpt["model"])
    fusion.eval()

    _verify_model2_shape(m1)
    _verify_model2_shape(m2)

    export_manifest = {
        "models": {},
        "onnx_version": onnx.__version__,
        "opset": 17,
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    def export_one(model, name, input_shape, dynamic_axes, input_names, output_names):
        onnx_path = models_dir / f"{name}.onnx"
        dummy = torch.randn(input_shape, device=device)
        torch.onnx.export(model, dummy, str(onnx_path),
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=17,
                          do_constant_folding=True, export_params=True)
        # verify
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        # torch vs onnx
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = model(dummy).cpu().numpy()
        ort_out = sess.run(output_names, {input_names[0]: dummy.cpu().numpy()})[0]
        max_diff = np.abs(torch_out - ort_out).max()
        print(f"  {name}: torch-vs-onnx max diff = {max_diff:.6f}")
        if max_diff >= 1e-3:
            print(f"  WARNING: {name} diff {max_diff} >= 1e-3")

        # INT8 QDQ (may fail for some architectures)
        int8_path = models_dir / f"{name}_int8.onnx"
        int8_ok = False
        int8_diff = None
        val_diff = None
        try:
            quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
            sess_int8 = ort.InferenceSession(str(int8_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            ort_out_int8 = sess_int8.run(output_names, {input_names[0]: dummy.cpu().numpy()})[0]
            int8_diff = np.abs(torch_out - ort_out_int8).max()
            print(f"  {name}: INT8 parity max diff = {int8_diff:.6f}")
            int8_ok = True

            # val set parity (50 images)
            val_diff_sum = 0.0
            val_count = 0
            if name in ("m1", "m2"):
                val_ds = EyeDataset(meta_path, "val", CFG["img_size"], augment=False)
                val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for xb, yb in val_loader:
                        if val_count >= 50:
                            break
                        xb = xb.to(device)
                        torch_out = model(xb).cpu().numpy()
                        ort_out = sess_int8.run(output_names, {input_names[0]: xb.cpu().numpy()})[0]
                        val_diff_sum += np.abs(torch_out - ort_out).max()
                        val_count += 1
                val_diff = val_diff_sum / max(1, val_count)
                print(f"  {name}: val (50) INT8 avg max diff = {val_diff:.6f}")
            elif name == "m4":
                val_ds = SegDataset(meta_path, "val", CFG["seg_size"], augment=False)
                val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for xb, yb in val_loader:
                        if val_count >= 50:
                            break
                        xb = xb.to(device)
                        torch_out = model(xb).cpu().numpy()
                        ort_out = sess_int8.run(output_names, {input_names[0]: xb.cpu().numpy()})[0]
                        val_diff_sum += np.abs(torch_out - ort_out).max()
                        val_count += 1
                val_diff = val_diff_sum / max(1, val_count)
                print(f"  {name}: val (50) INT8 avg max diff = {val_diff:.6f}")
        except Exception as e:
            print(f"  {name}: INT8 quantization skipped ({e})")
            if int8_path.exists():
                int8_path.unlink()

        export_manifest["models"][name] = {
            "fp32": str(onnx_path),
            "int8": str(int8_path) if int8_ok else None,
            "torch_onnx_max_diff": float(max_diff),
            "int8_parity_max_diff": float(int8_diff) if int8_diff is not None else None,
            "val_int8_avg_max_diff": float(val_diff) if val_diff is not None else None,
            "input_shape": list(input_shape),
            "output_shape": [int(x) for x in torch_out.shape]
        }
        return onnx_path

    print("\n  Exporting M1...")
    export_one(m1, "m1", (1, 3, CFG["img_size"], CFG["img_size"]),
               {"input": {0: "batch"}, "output": {0: "batch"}},
               ["input"], ["output"])
    print("  Exporting M2...")
    export_one(m2, "m2", (1, 3, CFG["img_size"], CFG["img_size"]),
               {"input": {0: "batch"}, "output": {0: "batch"}},
               ["input"], ["output"])
    print("  Exporting M4...")
    export_one(m4, "m4", (1, 3, CFG["seg_size"], CFG["seg_size"]),
               {"input": {0: "batch"}, "output": {0: "batch"}},
               ["input"], ["output"])
    print("  Exporting Fusion...")
    export_one(fusion, "fusion", (1, 32),
               {"input": {0: "batch"}, "output": {0: "batch"}},
               ["input"], ["output"])

    man_path = models_dir / "export_manifest.json"
    with open(man_path, "w") as f:
        json.dump(export_manifest, f, cls=NpEncoder, indent=2)
    print(f"  Export manifest: {man_path}")
    return export_manifest


def eigen_cam(model, img_tensor, target_class=None):
    """Eigen-CAM (raw RGB blend) for a classification model.
    Returns heatmap (H,W) in [0,1].
    """
    model.eval()
    # hook the last conv feature map
    features = []
    def hook(module, inp, out):
        features.append(out.detach())
    # find last conv layer (rough heuristic)
    target_layer = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            target_layer = mod
    if target_layer is None:
        return np.zeros(img_tensor.shape[-2:])
    h = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        out = model(img_tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
    h.remove()
    if not features:
        return np.zeros(img_tensor.shape[-2:])
    feat = features[0][0]  # (C, H, W)
    # SVD on (H*W, C)
    C, H, W = feat.shape
    feat_flat = feat.view(C, -1).t()  # (H*W, C)
    U, S, V = torch.svd(feat_flat)
    cam = (U[:, 0] * S[0]).view(H, W)  # first principal component
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def infer_single(image_path: str, models_dir: Path, meta_path: Path,
                 device: str = "cuda") -> dict:
    """Inference on one image (center.jpg) returning:
       - class probs (M1, M2, Fusion)
       - seg mask (M4) overlay
       - eigen_cam (M1)
       - YOLO detections
    """
    if device.isdigit():
        device = f"cuda:{device}"
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # load models
    m1_ckpt = torch.load(models_dir / "m1_best.pt", map_location=device)
    m2_ckpt = torch.load(models_dir / "m2_best.pt", map_location=device)
    m4_ckpt = torch.load(models_dir / "m4_best.pt", map_location=device)
    fusion_ckpt = torch.load(models_dir / "fusion_best.pt", map_location=device)

    m1 = M1_EyeNet(num_classes=CFG["num_classes"], pretrained=False).to(device)
    m1.load_state_dict(m1_ckpt["model"])
    m1.eval()
    m2 = M2_EfficientNet(num_classes=CFG["num_classes"], pretrained=False).to(device)
    m2.load_state_dict(m2_ckpt["model"])
    m2.eval()
    m4 = M4_SegNet(in_channels=3, out_channels=5).to(device)
    m4.load_state_dict(m4_ckpt["model"])
    m4.eval()
    fusion = FusionMLP(in_dim=32, num_classes=CFG["num_classes"]).to(device)
    fusion.load_state_dict(fusion_ckpt["model"])
    fusion.eval()

    # load YOLO
    yolo_path = models_dir / "yolo_best.pt"
    if yolo_path.exists():
        from ultralytics import YOLO
        yolo = YOLO(str(yolo_path))
    else:
        yolo = None

    # preprocess image
    img = Image.open(image_path).convert("RGB")
    tf = T.Compose([
        T.Resize((CFG["img_size"], CFG["img_size"]), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0).to(device)

    # M1, M2, Fusion (need cached logits for fusion — approximate with M1+M2+zero questionnaire)
    with torch.no_grad():
        m1_logits = m1(x)
        m2_logits = m2(x)
        m1_probs = F.softmax(m1_logits, dim=1).cpu().numpy()[0]
        m2_probs = F.softmax(m2_logits, dim=1).cpu().numpy()[0]
        # fusion input: concat M1 logits + M2 logits + questionnaire (zeros)
        fusion_in = torch.cat([m1_logits, m2_logits, torch.zeros(1, 6, device=device)], dim=1)
        fusion_logits = fusion(fusion_in)
        fusion_probs = F.softmax(fusion_logits, dim=1).cpu().numpy()[0]

        # M4 segmentation
        x_seg = F.interpolate(x, size=(CFG["seg_size"], CFG["seg_size"]), mode="bilinear", align_corners=False)
        seg_logits = m4(x_seg)
        seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0]  # (5, H, W)
        seg_mask = (seg_probs > 0.5).astype(np.uint8) * 255

        # Eigen-CAM on M1
        cam = eigen_cam(m1, x)

    # YOLO
    yolo_dets = []
    if yolo:
        res = yolo(image_path, verbose=False)[0]
        for box in res.boxes:
            yolo_dets.append({
                "class": LESION_CLASSES.get(int(box.cls.item()), "unknown"),
                "conf": float(box.conf.item()),
                "xyxy": box.xyxy[0].cpu().numpy().tolist()
            })

    return {
        "m1_probs": {CLASSES[i]: float(m1_probs[i]) for i in range(CFG["num_classes"])},
        "m2_probs": {CLASSES[i]: float(m2_probs[i]) for i in range(CFG["num_classes"])},
        "fusion_probs": {CLASSES[i]: float(fusion_probs[i]) for i in range(CFG["num_classes"])},
        "seg_mask": seg_mask,  # (5, H, W) binary
        "eigen_cam": cam,      # (H, W) float [0,1]
        "yolo_detections": yolo_dets,
    }


def run_pipeline(args):
    """Master pipeline dispatcher."""
    if args.cmd == "generate":
        generate_dataset(out_dir=args.out_dir, cases_per_class=args.cases,
                         train_px=args.img_size, resume=not args.no_resume)
    elif args.cmd == "train":
        meta_path = Path(args.out_dir) / "metadata.json"
        models_dir = Path(args.models_dir)
        results_dir = Path(args.results_dir)
        train_all(meta_path, models_dir, results_dir,
                  batch=args.batch, batch_seg=args.batch_seg, epochs=args.epochs,
                  lr=args.lr, num_workers=args.num_workers,
                  mixed_precision=args.mixed_precision,
                  use_questionnaire_in_fusion=args.use_questionnaire,
                  grad_accum=args.grad_accum,
                  early_stop_patience=args.early_stop_patience)
    elif args.cmd == "yolo-prep":
        meta_path = Path(args.out_dir) / "metadata.json"
        yolo_dir = Path(args.yolo_dir)
        prepare_yolo_dataset(meta_path, yolo_dir, yolo_all_views=args.yolo_all_views)
    elif args.cmd == "train-yolo":
        data_yaml = Path(args.yolo_dir) / "data.yaml"
        models_dir = Path(args.models_dir)
        train_yolo(data_yaml, models_dir, epochs=args.yolo_epochs, imgsz=args.img_size,
                   batch=args.batch, device=args.device)
    elif args.cmd == "export":
        models_dir = Path(args.models_dir)
        meta_path = Path(args.out_dir) / "metadata.json"
        results_dir = Path(args.results_dir)
        export_onnx_models(models_dir, meta_path, results_dir)
    elif args.cmd == "infer":
        models_dir = Path(args.models_dir)
        meta_path = Path(args.out_dir) / "metadata.json"
        result = infer_single(args.image, models_dir, meta_path, device=args.device)
        print(json.dumps(result, cls=NpEncoder, indent=2))
    elif args.cmd == "status":
        ckpt_status()
    elif args.cmd == "reset":
        ckpt_reset(args.stage)


def main():
    parser = argparse.ArgumentParser(description="OcuVisionAI v7 — Synthetic Eye Dataset & Pipeline")
    parser.add_argument("--out_dir", default=CFG["out_dir"], help="Dataset output directory")
    parser.add_argument("--models_dir", default=CFG["models_dir"], help="Models output directory")
    parser.add_argument("--results_dir", default=CFG["results_dir"], help="Results output directory")
    parser.add_argument("--yolo_dir", default="./yolo_dataset", help="YOLO dataset directory")
    parser.add_argument("--img_size", type=int, default=CFG["img_size"], help="Classification image size")
    parser.add_argument("--batch", type=int, default=CFG["batch"], help="Classification batch size")
    parser.add_argument("--batch_seg", type=int, default=CFG["batch_seg"], help="Segmentation batch size")
    parser.add_argument("--epochs", type=int, default=CFG["epochs"], help="Training epochs")
    parser.add_argument("--lr", type=float, default=CFG["lr"], help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=CFG["num_workers"], help="DataLoader workers")
    parser.add_argument("--mixed_precision", action="store_true", default=CFG["mixed_precision"],
                        help="Use AMP")
    parser.add_argument("--use_questionnaire", action="store_true", default=CFG["use_questionnaire_in_fusion"],
                        help="Use questionnaire in Fusion")
    parser.add_argument("--grad_accum", type=int, default=CFG["grad_accum"], help="Gradient accumulation steps")
    parser.add_argument("--early_stop_patience", type=int, default=CFG["early_stop_patience"],
                        help="Early stopping patience")
    parser.add_argument("--yolo_all_views", action="store_true", default=CFG["yolo_all_views"],
                        help="YOLO use all views (default center only)")
    parser.add_argument("--yolo_epochs", type=int, default=50, help="YOLO training epochs")
    parser.add_argument("--device", default="0", help="CUDA device for YOLO (e.g., '0' or 'cpu')")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume in generate")
    parser.add_argument("--stage", help="Checkpoint stage to reset")
    parser.add_argument("--image", help="Image path for infer command")

    sub = parser.add_subparsers(dest="cmd", required=True)

    gen_parser = sub.add_parser("generate", help="Generate dataset")
    gen_parser.add_argument("--cases", type=int, default=CFG["cases_per_class"],
                            help="Cases per class (default from CFG)")

    train_parser = sub.add_parser("train", help="Train M1, M2, M4, Fusion")
    train_parser.add_argument("--epochs", type=int, default=CFG["epochs"], help="Training epochs")
    train_parser.add_argument("--batch", type=int, default=CFG["batch"], help="Classification batch size")
    train_parser.add_argument("--batch_seg", type=int, default=CFG["batch_seg"], help="Segmentation batch size")
    train_parser.add_argument("--lr", type=float, default=CFG["lr"], help="Learning rate")
    train_parser.add_argument("--num_workers", type=int, default=CFG["num_workers"], help="DataLoader workers")
    train_parser.add_argument("--mixed_precision", action="store_true", default=CFG["mixed_precision"], help="Use AMP")
    train_parser.add_argument("--use_questionnaire", action="store_true", default=CFG["use_questionnaire_in_fusion"], help="Use questionnaire in Fusion")
    train_parser.add_argument("--grad_accum", type=int, default=CFG["grad_accum"], help="Gradient accumulation steps")
    train_parser.add_argument("--early_stop_patience", type=int, default=CFG["early_stop_patience"], help="Early stopping patience")

    yolo_prep_parser = sub.add_parser("yolo-prep", help="Prepare YOLO dataset")
    yolo_prep_parser.add_argument("--yolo_all_views", action="store_true", default=CFG["yolo_all_views"], help="Use all views")

    train_yolo_parser = sub.add_parser("train-yolo", help="Train YOLOv8n")
    train_yolo_parser.add_argument("--yolo_epochs", type=int, default=50, help="YOLO training epochs")
    train_yolo_parser.add_argument("--batch", type=int, default=16, help="YOLO batch size")
    train_yolo_parser.add_argument("--device", default="0", help="CUDA device (e.g., '0' or 'cpu')")

    sub.add_parser("export", help="Export models to ONNX")

    infer_parser = sub.add_parser("infer", help="Run inference on one image")
    infer_parser.add_argument("--image", required=True, help="Image path")
    infer_parser.add_argument("--device", default="0", help="CUDA device (e.g., '0' or 'cpu')")

    sub.add_parser("status", help="Show checkpoint status")

    reset_parser = sub.add_parser("reset", help="Reset checkpoint(s)")
    reset_parser.add_argument("--stage", help="Checkpoint stage to reset")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
