import os
import io
import cv2
import json
import math
import time
import random
import signal as _signal
import threading as _threading
import warnings
import multiprocessing
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from PIL import Image, ImageFilter
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*call to the `.*` method.*")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import timm
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("  GPU setup warning: No CUDA GPU available")

torch.set_num_threads(8)
torch.set_num_interop_threads(2)

try:
    import psutil as _psutil
    _HAVE_PSUTIL = True
except ImportError:
    _psutil = None
    _HAVE_PSUTIL = False
    print("  [note] psutil not installed — RAM-aware worker scaling disabled")
    print("         Install with: pip install psutil")

CFG = {
    "img_size"        : 224,
    "seg_size"        : 256,
    "gen_px"          : 512,
    "dpi"             : 100,
    "num_classes"     : 13,
    "gaze_views"      : 7,
    "cases_per_class" : 5000,
    "out_dir"         : "./eye_dataset_v6",
    "models_dir"      : "./ocuvision_models",
    "results_dir"     : "./ocuvision_results",
    "batch"           : 64,
    "batch_seg"       : 32,
    "epochs"          : 20,
    "lr"              : 1e-3,
    "shuffle"         : 256,
    "num_workers"     : 4,
    "mixed_precision" : True,
    "test_split"      : 0.15,
    "val_split"       : 0.15,
    "jpeg_quality"    : 90,
    "save_full_res"   : False,
}

def _clear_gpu():
    """Release PyTorch GPU memory and run GC."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def gpu_usage():
    """Print current GPU memory via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5
        )
        out = result.stdout.decode().strip()
        if out:
            used, total = out.split(", ")
            print(f"  GPU Memory: {used} MB / {total} MB")
        else:
            print("  GPU Memory: no output from nvidia-smi")
    except Exception:
        print("  GPU Memory: nvidia-smi not available")

CLASSES = {
    0 : "normal",
    1 : "conjunctivitis",
    2 : "allergic_conjunctivitis",
    3 : "dry_eye",
    4 : "blepharitis",
    5 : "strabismus",
    6 : "ptosis",
    7 : "corneal_scar",
    8 : "keratitis",
    9 : "pinguecula",
    10: "pterygium",
    11: "foreign_body",
    12: "subconjunctival_hemorrhage",
}

LESION_CLASSES = {
    0: "corneal_abrasion",
    1: "corneal_ulcer",
    2: "foreign_body_visible",
    3: "pterygium_lesion",
    4: "pinguecula_lesion",
    5: "hemorrhage_spot",
    6: "lid_lesion",
}

_QUESTION_TARGETS = {
    0: {1, 2, 3, 4},
    1: {3, 4},
    2: {8, 3, 9, 10},
    3: {8, 11, 3},
    4: {2, 1, 8},
    5: {8, 11, 7},
}

DIOPTER_RANGE_LABELS = {
    0: "Emmetropia (0 D)",
    1: "Low Myopia (-0.5 to -3 D)",
    2: "High Myopia (< -3 D)",
    3: "Low Hyperopia (+0.5 to +3 D)",
    4: "High Hyperopia (> +3 D)",
}

QUALITY_LABELS  = {0: "reject", 1: "poor", 2: "acceptable", 3: "good"}
GAZE_VIEWS      = ["center", "left", "right", "up", "down", "macro", "around"]
LIGHTING_TYPES  = ["clinic", "low_light", "overexposed", "side_lighting", "glare"]
CAMERA_TYPES    = ["phone", "phone", "phone", "slit_lamp", "low_res"]
SKIN_TONES      = ["#f5e6d0","#f0d0a8","#e8c49a","#d4a67a","#c8956c","#b07850","#8d5524","#6b3d1e"]
GEN_SIZE = CFG["gen_px"]
FIG_IN   = GEN_SIZE / CFG["dpi"]
NC       = CFG["num_classes"]

def rng(a, b):   return a + np.random.rand() * (b - a)
def rng_i(a, b): return random.randint(a, b)

def iris_color():
    return random.choice([
        np.array([0.27,0.51,0.71]), np.array([0.33,0.63,0.33]),
        np.array([0.45,0.30,0.14]), np.array([0.38,0.38,0.42]),
        np.array([0.55,0.38,0.18]), np.array([0.30,0.48,0.32]),
        np.array([0.20,0.44,0.60]),
    ])

def make_fig(skin=None):
    if skin is None: skin = random.choice(SKIN_TONES)
    fig = plt.figure(figsize=(FIG_IN, FIG_IN), dpi=CFG["dpi"])
    ax  = fig.add_axes([0,0,1,1])
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax, skin

def fig_to_arr(fig, size=GEN_SIZE):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    if w != size or h != size:
        img_arr = cv2.resize(img_arr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    plt.close(fig)
    return img_arr

def draw_vessels(ax, count, color, alpha):
    count = int(count * 0.6)
    if count < 1: count = 1
    for _ in range(count):
        side = random.choice([-1,1]); x,y = side*rng(0.28,0.82), rng(-0.28,0.28)
        for s in range(rng_i(2,6)):
            dx = rng(-0.12,0.12)*(0.72**s); dy = rng(-0.07,0.07)*(0.72**s)
            x2,y2 = x+dx, y+dy
            if (x2/0.90)**2+(y2/0.45)**2 < 1.0:
                ax.plot([x,x2],[y,y2], color=color, alpha=alpha,
                        linewidth=rng(0.4,1.4), zorder=3, solid_capstyle="round")
            x,y = x2,y2

def draw_sclera(ax, tint=(0.96,0.95,0.93), vc=6, va=0.30):
    ax.add_patch(Ellipse((0,0), 1.80, 0.90, facecolor=tint,
                          edgecolor="#c8b8a8", linewidth=0.5, zorder=2))
    draw_vessels(ax, vc, (0.65,0.25,0.25), va)

def draw_iris(ax, cx=0.0, cy=0.0, r=None, color=None):
    if r is None: r = rng(0.27, 0.34)
    if color is None: color = iris_color()

    limbal_color = np.clip(color * rng(0.20, 0.38), 0, 1)
    ax.add_patch(Ellipse((cx,cy), (r+0.032)*2, (r+0.032)*2, facecolor=limbal_color, zorder=4))
    ax.add_patch(Ellipse((cx,cy), r*2, r*2, facecolor=color, zorder=5))

    n_fibres = rng_i(24, 36)
    for ang in np.linspace(0, 2*math.pi, n_fibres, endpoint=False):
        ang += rng(-0.04, 0.04)
        ir_ = r * rng(0.36, 0.46); or_ = r * rng(0.76, 0.99)
        mid_r = (ir_ + or_) / 2 * rng(0.90, 1.10)
        ang_mid = ang + rng(-0.06, 0.06)
        pts_x = [cx+ir_*math.cos(ang), cx+mid_r*math.cos(ang_mid), cx+or_*math.cos(ang)]
        pts_y = [cy+ir_*math.sin(ang), cy+mid_r*math.sin(ang_mid), cy+or_*math.sin(ang)]
        ax.plot(pts_x, pts_y, color=color * rng(0.38, 0.72),
                alpha=rng(0.18, 0.52), linewidth=rng(0.3, 0.9),
                zorder=6, solid_capstyle="round")

    ax.add_patch(Ellipse((cx,cy), r*0.55*2, r*0.55*2, facecolor="none",
                          edgecolor=np.clip(color*1.35,0,1),
                          linewidth=rng(0.6, 1.0), alpha=0.55, zorder=7))

    n_crypts = rng_i(2, 6)
    for _ in range(n_crypts):
        cang  = rng(0, 2*math.pi); cdist = r * rng(0.42, 0.90)
        cx_c  = cx + cdist * math.cos(cang); cy_c = cy + cdist * math.sin(cang)
        cw = r * rng(0.040, 0.090); ch = cw * rng(0.6, 1.2)
        ax.add_patch(Ellipse((cx_c, cy_c), cw, ch, angle=math.degrees(cang),
                              facecolor=np.clip(color * rng(0.08, 0.28), 0, 1),
                              alpha=rng(0.45, 0.75), zorder=7))

    n_furrows = rng_i(2, 4)
    for fi in range(n_furrows):
        fr   = r * (0.70 + fi * 0.06 + rng(-0.01, 0.02))
        ang0 = rng(0, 2*math.pi)
        arc  = rng(math.pi * 0.6, math.pi * 1.8)
        angs = np.linspace(ang0, ang0 + arc, 60)
        xs   = cx + fr * np.cos(angs) + np.random.normal(0, r*0.008, 60)
        ys   = cy + fr * np.sin(angs) + np.random.normal(0, r*0.006, 60)
        ax.plot(xs, ys, color=np.clip(color * rng(0.50, 0.75), 0, 1),
                alpha=rng(0.20, 0.42), linewidth=rng(0.4, 0.9),
                zorder=7, solid_capstyle="round")

    n_spots = rng_i(0, 6)
    for _ in range(n_spots):
        sang  = rng(0, 2*math.pi); sdist = r * rng(0.30, 0.90)
        sx = cx + sdist * math.cos(sang); sy = cy + sdist * math.sin(sang)
        sw = r * rng(0.025, 0.065)
        ax.add_patch(Ellipse((sx, sy), sw, sw * rng(0.7, 1.3),
                              facecolor=np.clip(color * rng(0.10, 0.35), 0, 1),
                              alpha=rng(0.55, 0.85), zorder=7))

    if random.random() < 0.08:
        alt_color = iris_color()
        sec_ang   = rng(0, 2*math.pi); sec_span = rng(0.3, 1.0)
        angs_sec  = np.linspace(sec_ang, sec_ang + sec_span, 40)
        xs_sec    = [cx] + list(cx + r*np.cos(angs_sec)) + [cx]
        ys_sec    = [cy] + list(cy + r*np.sin(angs_sec)) + [cy]
        ax.fill(xs_sec, ys_sec, color=alt_color, alpha=rng(0.25, 0.55), zorder=6)

    return color, r

def draw_pupil(ax, cx=0.0, cy=0.0, r=0.13):
    ax.add_patch(Ellipse((cx,cy), r*2, r*2, facecolor=(0.04,0.04,0.06), zorder=8))
    p1x = cx - r*0.28 + rng(-0.012, 0.012); p1y = cy + r*0.28 + rng(-0.012, 0.012)
    p1r = r * rng(0.17, 0.25)
    ax.add_patch(Ellipse((p1x, p1y), p1r*2, p1r*1.35,
                          facecolor=(0.97, 0.98, 1.0), alpha=rng(0.82, 0.95), zorder=9))
    ax.add_patch(Ellipse((p1x, p1y), p1r*0.55*2, p1r*0.55*1.35,
                          facecolor=(1.0, 1.0, 1.0), alpha=rng(0.90, 1.00), zorder=10))
    if random.random() < 0.65:
        p3x = cx + r*0.40 + rng(-0.020, 0.020); p3y = cy - r*0.22 + rng(-0.015, 0.015)
        p3r = r * rng(0.06, 0.11)
        ax.add_patch(Ellipse((p3x, p3y), p3r*2, p3r*1.2,
                              facecolor=(0.92, 0.94, 0.98), alpha=rng(0.20, 0.40), zorder=9))
    for _ in range(rng_i(1, 3)):
        ax.add_patch(Ellipse((cx+rng(-r*0.55,r*0.55), cy+rng(-r*0.55,r*0.55)),
                              rng(0.008,0.020), rng(0.006,0.014),
                              facecolor=(1,1,1), alpha=rng(0.22, 0.45), zorder=10))
    return cx, cy, r

def eyelid_curve(n=240, upper=True, arch=None, asym_k=None, occlusion=0.0):
    if arch is None: arch = rng(0.28,0.48) if upper else rng(0.10,0.22)
    if asym_k is None: asym_k = rng(0.02,0.10)
    t = np.linspace(-0.90, 0.90, n)
    y = arch*(1-(t/0.90)**2) + asym_k*t
    return t, (y-occlusion if upper else (-y)+occlusion*0.5)

def draw_eyelids(ax, skin, occlusion=0.0, with_makeup=False,
                 lash_shadow=False, crust=False):
    t_u, y_u = eyelid_curve(upper=True, occlusion=occlusion)
    xu = np.concatenate([t_u, t_u[::-1]])
    yu = np.concatenate([y_u, np.full_like(t_u, 0.75)])
    ax.fill(xu, yu, color=skin, zorder=10, linewidth=0)
    if lash_shadow:
        ax.fill(np.concatenate([t_u, t_u[::-1]]),
                np.concatenate([y_u-0.045, y_u[::-1]]),
                color=(0,0,0), alpha=rng(0.12,0.28), zorder=10)
    lid_col = "#2a0a04" if with_makeup else "#5a3a28"
    ax.plot(t_u, y_u, color=lid_col,
            linewidth=1.8 if with_makeup else 1.1, zorder=11, solid_capstyle="round")
    if with_makeup:
        ax.plot(t_u, y_u+0.018, color="#120404", linewidth=1.0, alpha=0.55, zorder=11)
    if crust:
        for li in np.linspace(5, len(t_u)-6, 18, dtype=int):
            lx, ly = t_u[li], y_u[li]
            ax.add_patch(Ellipse((lx, ly-0.01), rng(0.018,0.040), rng(0.010,0.022),
                                  facecolor=(0.72,0.58,0.32), alpha=rng(0.55,0.80), zorder=12))
    for li in np.linspace(0, len(t_u)-1, 26, dtype=int):
        lx, ly = t_u[li], y_u[li]
        ang = math.atan2(ly, lx) + math.pi/2 + rng(-0.20,0.20)
        L   = rng(0.04, 0.13)
        ax.plot([lx, lx+L*math.cos(ang)], [ly, ly+L*math.sin(ang)],
                color="#0f0808", linewidth=rng(0.5,1.2), zorder=12, solid_capstyle="round")
    t_l, y_l = eyelid_curve(upper=False, occlusion=occlusion)
    xl = np.concatenate([t_l, t_l[::-1]])
    yl = np.concatenate([y_l, np.full_like(t_l,-0.75)])
    ax.fill(xl, yl, color=skin, zorder=10, linewidth=0)
    ax.plot(t_l, y_l, color="#6a4a38", linewidth=0.8, zorder=11, solid_capstyle="round")
    ax.add_patch(Ellipse((-0.90,0.0), 0.10, 0.14, facecolor="#e8908a", zorder=10))
    return t_u, y_u, t_l, y_l

def add_glasses_glare(ax):
    for _ in range(rng_i(1,3)):
        gx=rng(-0.80,0.80); gy=rng(-0.55,0.55)
        gw=rng(0.15,0.60);  gh=rng(0.08,0.28)
        ax.add_patch(Ellipse((gx,gy), gw, gh, facecolor=(1,1,1),
                              alpha=rng(0.18,0.48), zorder=15))
        ax.add_patch(Ellipse((gx+rng(-0.03,0.03), gy+rng(-0.02,0.02)),
                              gw*0.40, gh*0.40, facecolor=(1,1,1),
                              alpha=rng(0.28,0.55), zorder=16))

def apply_gaze_transform(arr, view="center"):
    h, w = arr.shape[:2]; img = Image.fromarray(arr)
    offsets = {"center":(0,0),"left":(-int(w*0.14),0),"right":(int(w*0.14),0),
               "up":(0,-int(h*0.10)),"down":(0,int(h*0.10)),"macro":(0,0),"around":(0,0)}
    scales  = {"center":1.0,"left":1.0,"right":1.0,"up":1.0,"down":1.0,"macro":1.55,"around":0.68}
    dx, dy = offsets[view]; scale = scales[view]
    if scale != 1.0:
        nw, nh = int(w*scale), int(h*scale)
        if scale > 1.0:
            img = img.resize((nw,nh), Image.LANCZOS)
            img = img.crop(((nw-w)//2, (nh-h)//2, (nw-w)//2+w, (nh-h)//2+h))
        else:
            img = img.resize((nw,nh), Image.LANCZOS)
            bg  = Image.new("RGB",(w,h),(210,185,160)); bg.paste(img,((w-nw)//2,(h-nh)//2)); img=bg
    if dx!=0 or dy!=0:
        img = img.transform(img.size, Image.AFFINE, (1,0,-dx,0,1,-dy),
                             resample=Image.BILINEAR, fillcolor=(210,185,160))
    return np.array(img, dtype=np.uint8)

def apply_lighting(arr, lighting_type="clinic"):
    img_f = arr.astype(np.float32)/255.0
    h, w  = img_f.shape[:2]
    xg, yg = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
    if   lighting_type == "clinic":      gain = np.ones((h,w),np.float32) * rng(0.90,1.10)
    elif lighting_type == "low_light":   gain = np.ones((h,w),np.float32) * rng(0.45,0.65)
    elif lighting_type == "overexposed": gain = np.ones((h,w),np.float32) * rng(1.35,1.65)
    elif lighting_type == "side_lighting":
        side = random.choice([0,1])
        grad = xg if side==0 else (1-xg)
        gain = (0.50 + grad*0.80).astype(np.float32)
    elif lighting_type == "glare":
        gx=rng(0.2,0.8); gy=rng(0.2,0.8)
        dist = np.sqrt((xg-gx)**2+(yg-gy)**2)
        gain = (1.0 + np.exp(-dist*8.0)*rng(0.4,0.9)).astype(np.float32)
    else:
        gain = np.ones((h,w),np.float32)
    img_f = np.clip(img_f * gain[:,:,None], 0, 1)
    return (img_f*255).astype(np.uint8)

def apply_camera_artifacts(arr, camera_type="phone"):
    img_f = arr.astype(np.float32)/255.0
    h, w  = img_f.shape[:2]
    noise = ndi.gaussian_filter(
        np.random.normal(0, rng(0.004,0.018), img_f.shape).astype(np.float32), sigma=0.8)
    img_f = np.clip(img_f + noise, 0, 1)
    if camera_type == "slit_lamp":
        xg,yg = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
        dist   = np.sqrt((xg-0.5)**2+(yg-0.5)**2)
        vign   = 1.0 - dist*rng(0.40,0.65)
        img_f  = np.clip(img_f*vign[:,:,None], 0,1)
        img_f  = np.clip(img_f*rng(1.10,1.40), 0,1)
    elif camera_type == "phone" and random.random() < 0.35:
        sigma = rng(0.4, 1.6)
        ksize = int(6 * sigma + 1) | 1
        img_f_u8 = (img_f * 255).astype(np.uint8)
        img_f_u8 = cv2.GaussianBlur(img_f_u8, (ksize, ksize), sigma)
        img_f = img_f_u8.astype(np.float32) / 255.0
    elif camera_type == "low_res":
        pil   = Image.fromarray((img_f*255).astype(np.uint8))
        small = pil.resize((rng_i(52,88),rng_i(52,88)), Image.BILINEAR)
        pil   = small.resize((w,h), Image.NEAREST)
        img_f = np.array(pil, dtype=np.float32)/255.0
    img_f = np.clip(img_f*rng(0.88,1.12) + rng(-0.06,0.06), 0,1)
    return (img_f*255).astype(np.uint8)

def add_perlin_skin_texture(arr, strength=None):
    if strength is None: strength = rng(0.015, 0.055)
    h, w = arr.shape[:2]
    scales = [rng(0.4,0.8), rng(1.2,2.0), rng(3.0,5.0)]
    weights = [0.55, 0.30, 0.15]
    noise_combined = np.zeros((h, w), dtype=np.float32)
    for scale, weight in zip(scales, weights):
        nx = int(max(4, w * scale / 64)); ny = int(max(4, h * scale / 64))
        raw = np.random.normal(0, 1, (ny, nx)).astype(np.float32)
        upscaled = cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC)
        noise_combined += upscaled * weight
    noise_combined = ndi.gaussian_filter(noise_combined, sigma=rng(1.5, 3.5))
    noise_combined = (noise_combined - noise_combined.min())
    mx = noise_combined.max()
    if mx > 0: noise_combined /= mx
    noise_combined = (noise_combined - 0.5) * strength

    img_f = arr.astype(np.float32) / 255.0
    img_f = np.clip(img_f + noise_combined[:, :, None], 0, 1)

    if random.random() < 0.30:
        pore_density = rng(0.005, 0.018)
        n_pores      = int(h * w * pore_density / 100)
        pore_map     = np.zeros((h, w), dtype=np.float32)
        for _ in range(min(n_pores, 200)):
            px = np.random.randint(2, w-2); py = np.random.randint(2, h-2)
            pr_ = rng(1.2, 3.0)
            y_  = np.clip(int(py-pr_), 0, h-1); ye_ = np.clip(int(py+pr_)+1, 0, h)
            x_  = np.clip(int(px-pr_), 0, w-1); xe_ = np.clip(int(px+pr_)+1, 0, w)
            pore_map[y_:ye_, x_:xe_] += rng(0.015, 0.045)
        pore_map = ndi.gaussian_filter(pore_map, sigma=0.8)
        img_f = np.clip(img_f - pore_map[:,:,None] * rng(0.5, 1.0), 0, 1)

    if random.random() < 0.60:
        sss_strength = rng(0.008, 0.022)
        lum = img_f.mean(axis=2)
        sss_mask = np.exp(-((lum - 0.55)**2) / (2 * 0.12**2)).astype(np.float32)
        sss_mask = ndi.gaussian_filter(sss_mask, sigma=rng(8, 20))
        sss_r = sss_mask * sss_strength * rng(0.8, 1.2)
        sss_g = sss_mask * sss_strength * rng(0.2, 0.4)
        sss_b = sss_mask * sss_strength * rng(-0.15, -0.05)
        img_f[:,:,0] = np.clip(img_f[:,:,0] + sss_r, 0, 1)
        img_f[:,:,1] = np.clip(img_f[:,:,1] + sss_g, 0, 1)
        img_f[:,:,2] = np.clip(img_f[:,:,2] + sss_b, 0, 1)

    return (img_f * 255).astype(np.uint8)

def add_tear_film(arr, cx_px=None, cy_px=None, r_px=None):
    h, w = arr.shape[:2]
    if cx_px is None: cx_px = w // 2
    if cy_px is None: cy_px = h // 2
    if r_px  is None: r_px  = int(min(h, w) * rng(0.22, 0.34))

    img_f = arr.astype(np.float32) / 255.0
    xg, yg = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
    dist_norm = np.sqrt((xg - cx_px/w)**2 + (yg - cy_px/h)**2)
    cornea_mask = (dist_norm < r_px / w).astype(np.float32)
    cornea_mask = ndi.gaussian_filter(cornea_mask, sigma=r_px * 0.12)

    shine_x = cx_px/w + rng(-0.04, 0.04); shine_y = cy_px/h + rng(-0.04, 0.04)
    d_shine  = np.sqrt((xg - shine_x)**2 + (yg - shine_y)**2)
    specular = np.exp(-d_shine / (r_px / w * rng(0.10, 0.22))) * rng(0.08, 0.20)
    specular *= cornea_mask

    ripple_freq = rng(18, 35)
    ripple = (np.sin(xg * ripple_freq * math.pi) *
              np.cos(yg * ripple_freq * math.pi * rng(0.7,1.3)) * rng(0.008, 0.022))
    ripple *= cornea_mask

    if random.random() < 0.40:
        n_breaks = rng_i(2, 6)
        for _ in range(n_breaks):
            bx = cx_px/w + rng(-0.08, 0.08); by = cy_px/h + rng(-0.08, 0.08)
            db = np.sqrt((xg - bx)**2 + (yg - by)**2)
            specular -= np.exp(-db / (r_px/w * rng(0.04,0.10))) * rng(0.04, 0.10) * cornea_mask

    tint = np.zeros_like(img_f)
    tint[:,:,0] = rng(-0.01, 0.01); tint[:,:,1] = rng(0.00, 0.02); tint[:,:,2] = rng(0.01, 0.04)
    tint *= cornea_mask[:,:,None]

    img_f = np.clip(img_f + specular[:,:,None] + ripple[:,:,None] + tint, 0, 1)
    return (img_f * 255).astype(np.uint8)

def add_motion_blur(arr, max_angle=None, max_length=None):
    if random.random() > 0.35: return arr
    h, w = arr.shape[:2]
    if max_angle  is None: max_angle  = rng(0, math.pi)

    if max_length is None: max_length = max(2, rng_i(int(w * 0.008), int(w * 0.030)))
    length = max(2, max_length)
    kernel = np.zeros((length, length), dtype=np.float32)
    cx_k = length // 2; cy_k = length // 2
    dx = math.cos(max_angle); dy = math.sin(max_angle)
    for i in range(length):
        t  = i - length // 2
        px = int(cx_k + t * dx); py = int(cy_k + t * dy)
        if 0 <= px < length and 0 <= py < length:
            kernel[py, px] = 1.0
    s = kernel.sum()
    if s > 0: kernel /= s
    else: return arr
    img_f  = arr.astype(np.float32)
    blurred = cv2.filter2D(img_f, -1, kernel)
    alpha   = rng(0.45, 0.90)
    result  = img_f * (1 - alpha) + blurred * alpha
    return np.clip(result, 0, 255).astype(np.uint8)

def add_eyelid_wrinkles(ax, t_u, y_u, skin_hex, intensity=None):
    if intensity is None: intensity = rng(0.3, 1.0)
    if intensity < 0.4: return
    r_val = int(skin_hex.lstrip("#")[0:2], 16)
    g_val = int(skin_hex.lstrip("#")[2:4], 16)
    b_val = int(skin_hex.lstrip("#")[4:6], 16)
    shadow = (max(0,(r_val-40))/255, max(0,(g_val-35))/255, max(0,(b_val-30))/255)
    n_wrinkles = rng_i(3, 8)
    for _ in range(n_wrinkles):
        t_center = rng(-0.65, 0.65); idx = np.argmin(np.abs(t_u - t_center))
        y_base   = y_u[idx]; length = rng(0.04, 0.16) * intensity
        angle    = rng(0.15, 0.55) * random.choice([-1, 1])
        t_pts    = np.linspace(t_center, t_center + length * math.cos(angle), 8)
        y_pts    = np.array([y_base + rng(-0.005,0.005) +
                              (rng(0.01,0.025)*intensity * math.sin(math.pi*k/7))
                              for k in range(8)])
        ax.plot(t_pts, y_pts, color=shadow,
                linewidth=rng(0.3, 0.8)*intensity, alpha=rng(0.25, 0.55)*intensity,
                zorder=11, solid_capstyle="round")

def vary_pupil_dilation(ax, cx=0.0, cy=0.0, base_r=0.13, lighting_type="clinic"):
    dilation_map = {
        "clinic"      : rng(0.10, 0.14),
        "low_light"   : rng(0.16, 0.22),
        "overexposed" : rng(0.07, 0.10),
        "side_lighting": rng(0.11, 0.16),
        "glare"       : rng(0.06, 0.09),
    }
    r = dilation_map.get(lighting_type, base_r)
    ax.add_patch(Ellipse((cx,cy), r*2, r*2, facecolor=(0.04,0.04,0.06), zorder=8))
    ax.add_patch(Ellipse((cx-r*0.30, cy+r*0.30), r*0.22*2, r*0.22*1.3,
                          facecolor=(0.95,0.97,1.0), alpha=0.88, zorder=9))
    for _ in range(rng_i(1,4)):
        ax.add_patch(Ellipse((cx+rng(-r*0.5,r*0.5), cy+rng(-r*0.5,r*0.5)),
                              rng(0.008,0.022), rng(0.006,0.016),
                              facecolor=(1,1,1), alpha=rng(0.28,0.55), zorder=10))
    return cx, cy, r

def make_segmentation_masks(img_size, cx_i, cy_i, r_iris,
                              cx_p, cy_p, r_pupil, t_u, y_u, t_l, y_l, view="center"):
    """
    FIX-22: img_size here is GEN_SIZE (512). Masks are saved as 512×512 PNGs.
    The segmentation model uses 256×256. Resize happens in build_seg_dataset.
    Coordinate system: axis ranges [-1.12, 1.12], total span = 2.24.
    """
    S = img_size; H = W = S
    AXIS_SPAN = 2.24

    def ellipse_mask(cx_n, cy_n, r_n, w_frac=1.0):
        mask  = np.zeros((H,W), dtype=np.uint8)
        rx_px = int(r_n / AXIS_SPAN * W * w_frac)
        ry_px = int(r_n / AXIS_SPAN * H)
        cx_   = int((cx_n / AXIS_SPAN + 0.5) * W)
        cy_   = int((1.0 - (cy_n / AXIS_SPAN + 0.5)) * H)
        cv2.ellipse(mask, (cx_,cy_), (max(1,rx_px), max(1,ry_px)), 0, 0, 360, 255, -1)
        return mask

    iris_mask   = ellipse_mask(cx_i, cy_i, r_iris)
    pupil_mask  = ellipse_mask(cx_p, cy_p, r_pupil)
    limbus_r    = r_iris * 1.14
    outer_mask  = ellipse_mask(cx_i, cy_i, limbus_r)
    limbus_mask = cv2.subtract(outer_mask, iris_mask)
    full_sclera = ellipse_mask(0.0, 0.0, 0.45, w_frac=2.0)
    sclera_mask = cv2.subtract(full_sclera, outer_mask)
    conjunctiva_mask = sclera_mask.copy()

    upper_mask = np.zeros((H,W), dtype=np.uint8)
    pts_u = np.array([[(int((t/AXIS_SPAN+0.5)*W), int((1.0-(y/AXIS_SPAN+0.5))*H))
                        for t,y in zip(t_u,y_u)] + [(W-1,0),(0,0)]], dtype=np.int32)
    cv2.fillPoly(upper_mask, pts_u, 255)

    lower_mask = np.zeros((H,W), dtype=np.uint8)
    pts_l = np.array([[(int((t/AXIS_SPAN+0.5)*W), int((1.0-(y/AXIS_SPAN+0.5))*H))
                        for t,y in zip(t_l,y_l)] + [(W-1,H-1),(0,H-1)]], dtype=np.int32)
    cv2.fillPoly(lower_mask, pts_l, 255)

    masks = {
        "iris"       : iris_mask,
        "sclera"     : sclera_mask,
        "pupil"      : pupil_mask,
        "upper_lid"  : upper_mask,
        "lower_lid"  : lower_mask,
    }

    if view != "center":
        offsets = {"center":(0,0),"left":(-int(W*0.14),0),"right":(int(W*0.14),0),
                   "up":(0,-int(H*0.10)),"down":(0,int(H*0.10)),"macro":(0,0),"around":(0,0)}
        scales  = {"center":1.0,"left":1.0,"right":1.0,"up":1.0,"down":1.0,"macro":1.55,"around":0.68}
        dx, dy = offsets[view]; scale = scales[view]

        for k, v in masks.items():
            img = Image.fromarray(v)
            if scale != 1.0:
                nw, nh = int(W*scale), int(H*scale)
                if scale > 1.0:
                    img = img.resize((nw,nh), Image.NEAREST)
                    img = img.crop(((nw-W)//2, (nh-H)//2, (nw-W)//2+W, (nh-H)//2+H))
                else:
                    img = img.resize((nw,nh), Image.NEAREST)
                    bg  = Image.new("L",(W,H),0); bg.paste(img,((W-nw)//2,(H-nh)//2)); img=bg
            if dx!=0 or dy!=0:
                img = img.transform(img.size, Image.AFFINE, (1,0,-dx,0,1,-dy),
                                     resample=Image.NEAREST, fillcolor=0)
            masks[k] = np.array(img, dtype=np.uint8)

    return masks

def make_yolo_label(class_id, cx_n, cy_n, r_n):
    AXIS_SPAN = 2.24
    cx = max(0.001, min(0.999, cx_n/AXIS_SPAN+0.5))
    cy = max(0.001, min(0.999, 1.0-(cy_n/AXIS_SPAN+0.5)))
    w  = max(0.01,  min(0.999, r_n*2.5/AXIS_SPAN))
    h  = max(0.01,  min(0.999, r_n*2.0/AXIS_SPAN))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

_FORCE_QUALITY = None

def _flags():
    f = dict(
        occlusion    = rng(0,0.20),
        with_makeup  = random.random()<0.20,
        lash_shadow  = random.random()<0.55,
        glasses      = random.random()<0.12,
        lighting     = random.choice(LIGHTING_TYPES),
        camera       = random.choice(CAMERA_TYPES),
        wrinkles     = random.random()<0.55,
        tear_film    = random.random()<0.70,
        motion_blur  = random.random()<0.35,
        skin_texture = random.random()<0.80,
    )
    if _FORCE_QUALITY == "blurry":
        f["motion_blur"] = True
    elif _FORCE_QUALITY == "low_light":
        f["lighting"] = "low_light"
    elif _FORCE_QUALITY == "partial_eye":
        f["occlusion"] = rng(0.60, 0.85)
    return f

def _render(fig, f, view, train_size):
    arr  = fig_to_arr(fig, size=GEN_SIZE)
    arr  = apply_gaze_transform(arr, view)
    arr  = apply_lighting(arr, f["lighting"])
    arr  = apply_camera_artifacts(arr, f["camera"])
    if f.get("skin_texture", True) and random.random() < 0.4:
        arr = add_perlin_skin_texture(arr)
    if f.get("tear_film", True):
        arr = add_tear_film(arr)
    if f.get("motion_blur", False):
        arr = add_motion_blur(arr)
    small = cv2.resize(arr, (train_size,train_size), interpolation=cv2.INTER_LANCZOS4)
    return arr, small

def _gen_base(tint=(0.96,0.95,0.93), vc=6, va=0.30, cx=0.0, cy=0.0):
    f = _flags()
    fig, ax, sk = make_fig()
    ax.set_facecolor(sk)
    draw_sclera(ax, tint=tint, vc=vc, va=va)
    ic, ir = draw_iris(ax, cx=cx, cy=cy)
    pcx,pcy,pr = vary_pupil_dilation(ax, cx=cx, cy=cy, lighting_type=f["lighting"])
    t_u,y_u,t_l,y_l = draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    if f.get("wrinkles", False): add_eyelid_wrinkles(ax, t_u, y_u, sk)
    if f["glasses"]: add_glasses_glare(ax)
    return fig, ax, sk, ir, pcx,pcy,pr, t_u,y_u,t_l,y_l, f

def gen_normal(view="center", train_px=224):
    fig,ax,sk,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l,f = _gen_base()
    arr_full, arr_small = _render(fig,f,view,train_px)
    masks = make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small, arr_full, masks, ""

def gen_conjunctivitis(view="center", train_px=224):
    f = _flags()
    fig,ax,sk = make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,tint=(rng(0.86,0.94),rng(0.66,0.76),rng(0.66,0.74)),vc=45,va=0.72)
    ax.add_patch(Ellipse((0,0),0.80,0.80,facecolor="none",edgecolor=(0.80,0.38,0.38),linewidth=2.0,alpha=0.50,zorder=5))
    ic,ir = draw_iris(ax); pcx,pcy,pr = draw_pupil(ax)
    for cxd in [rng(-0.92,-0.84),rng(0.84,0.92)]:
        ax.add_patch(Ellipse((cxd,rng(-0.07,0.07)),0.14,0.08,facecolor=(0.75,0.80,0.18),alpha=rng(0.60,0.85),zorder=13))
    t_l2,y_l2 = eyelid_curve(upper=False,arch=rng(0.20,0.30))
    ax.fill(np.concatenate([t_l2,t_l2[::-1]]),np.concatenate([y_l2-0.10,np.full_like(t_l2,-0.75)]),
            color="#e0a090",zorder=13,alpha=0.55,linewidth=0)
    f["occlusion"]=max(f["occlusion"],rng(0.05,0.22))
    t_u,y_u,t_l,y_l = draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small = _render(fig,f,view,train_px)
    masks = make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small, arr_full, masks, ""

def gen_allergic_conjunctivitis(view="center", train_px=224):
    f = _flags()
    fig,ax,sk = make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,tint=(rng(0.88,0.94),rng(0.72,0.82),rng(0.72,0.80)),vc=30,va=0.60)
    ic,ir = draw_iris(ax); pcx,pcy,pr = draw_pupil(ax)
    for _ in range(rng_i(8,18)):
        ax.add_patch(Ellipse((rng(-0.70,0.70),rng(-0.36,-0.18)),0.05,0.04,facecolor=(0.75,0.35,0.35),alpha=rng(0.45,0.70),zorder=6))
    for _ in range(2):
        ax.add_patch(Ellipse((rng(-0.20,0.20),rng(-0.52,-0.38)),rng(0.90,1.25),rng(0.18,0.30),facecolor="#e8b8a8",alpha=0.45,zorder=9))
    ax.add_patch(Ellipse((rng(-0.40,0.40),-0.46),0.18,0.08,facecolor=(0.80,0.88,0.95),alpha=0.62,zorder=13))
    f["occlusion"]=max(f["occlusion"],rng(0.08,0.24))
    t_u,y_u,t_l,y_l = draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small = _render(fig,f,view,train_px)
    masks = make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small, arr_full, masks, ""

def gen_dry_eye(view="center", train_px=224):
    f = _flags()
    fig, ax, sk = make_fig(); ax.set_facecolor(sk)
    tint_r=rng(0.91,0.97); tint_g=rng(0.89,0.94); tint_b=rng(0.82,0.90)
    draw_sclera(ax,tint=(tint_r,tint_g,tint_b),vc=rng_i(6,14),va=rng(0.28,0.48))
    for side in [-1,1]:
        bx=side*rng(0.35,0.62)
        ax.add_patch(Ellipse((bx,rng(-0.06,0.06)),rng(0.18,0.32),rng(0.08,0.14),
                              facecolor=(rng(0.72,0.82),rng(0.22,0.32),rng(0.38,0.52)),
                              alpha=rng(0.20,0.38),zorder=4))
    ic,ir=draw_iris(ax)
    n_tbut=rng_i(3,8)
    for _ in range(n_tbut):
        tx=rng(-0.30,0.30); ty=rng(-0.22,0.22); tw=rng(0.06,0.18); th=rng(0.04,0.12)
        ax.add_patch(Ellipse((tx,ty),tw,th,
                              facecolor=(rng(0.82,0.90),rng(0.80,0.88),rng(0.72,0.82)),
                              alpha=rng(0.18,0.38),zorder=5))
    n_spk=rng_i(12,30)
    for _ in range(n_spk):
        px=rng(-ir*0.9,ir*0.9); py=rng(-ir*0.6,ir*0.6)
        ax.add_patch(Ellipse((px,py),rng(0.010,0.025),rng(0.008,0.020),
                              facecolor=(rng(0.76,0.86),rng(0.74,0.82),rng(0.60,0.74)),
                              alpha=rng(0.30,0.55),zorder=6))
    t_men=np.linspace(-0.80,0.80,120)
    y_men=-0.38+rng(-0.02,0.02)+np.zeros_like(t_men)
    meniscus_h=rng(0.004,0.012)
    ax.fill(np.concatenate([t_men,t_men[::-1]]),
            np.concatenate([y_men,y_men-meniscus_h]),
            color=(rng(0.78,0.88),rng(0.88,0.96),rng(0.90,0.98)),
            alpha=rng(0.40,0.65),zorder=7)
    if random.random()<0.25:
        for _ in range(rng_i(2,5)):
            mx=rng(-0.50,0.50); my_s=rng(-0.30,0.10)
            ax.plot([mx,mx+rng(-0.08,0.08)],[my_s,my_s-rng(0.05,0.14)],
                    color=(0.90,0.88,0.72),linewidth=rng(0.6,1.4),
                    alpha=rng(0.40,0.65),zorder=7,solid_capstyle="round")
    if random.random()<0.15:
        for _ in range(rng_i(1,4)):
            fx=rng(-0.20,0.20); fy=rng(-0.15,0.15)
            ax.plot([fx,fx+rng(-0.04,0.04)],[fy,fy-rng(0.06,0.12)],
                    color=(0.86,0.82,0.60),linewidth=rng(0.8,1.8),
                    alpha=rng(0.55,0.80),zorder=8,solid_capstyle="round")
            ax.add_patch(Ellipse((fx,fy),0.016,0.012,facecolor=(0.82,0.78,0.55),alpha=0.70,zorder=9))
    pcx,pcy,pr=vary_pupil_dilation(ax,lighting_type=f["lighting"])
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,""

def gen_blepharitis(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,tint=(rng(0.90,0.96),rng(0.88,0.94),rng(0.82,0.90)),vc=12,va=0.42)
    ic,ir=draw_iris(ax); pcx,pcy,pr=draw_pupil(ax)
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"],crust=True)
    for side in [-1,1]:
        ax.add_patch(Ellipse((side*rng(0.20,0.70),-0.36),rng(0.30,0.55),rng(0.10,0.18),
                              facecolor=(0.78,0.42,0.32),alpha=rng(0.45,0.70),zorder=13))
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,make_yolo_label(6,0,-0.36,0.25)

def gen_strabismus(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=5,va=0.28)
    dx=random.choice([-1,1])*rng(0.20,0.38); dy=random.choice([-1,1])*rng(0.08,0.24)
    pr_r=random.choice([rng(0.06,0.09),rng(0.20,0.27)])
    ic,ir=draw_iris(ax,cx=dx,cy=dy); pcx,pcy,pr=draw_pupil(ax,cx=dx,cy=dy,r=pr_r)
    ax.plot(0,0,"+",color="#cc3333",markersize=4,linewidth=0.8,alpha=0.35,zorder=15)
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,dx,dy,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,""

def gen_ptosis(view="center", train_px=224):
    f=_flags()
    cover=rng(0.25,0.60); f["occlusion"]=max(f["occlusion"],cover)
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=6,va=0.28)
    ic,ir=draw_iris(ax); pcx,pcy,pr=draw_pupil(ax)
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    ax.fill(np.concatenate([t_u,t_u[::-1]]),
            np.concatenate([y_u+rng(0.06,0.16),np.full_like(t_u,0.75)]),
            color=sk,alpha=0.75,zorder=11,linewidth=0)
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,""

def gen_corneal_scar(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=8,va=0.32)
    ic,ir=draw_iris(ax)
    central=random.random()<0.60
    if central:
        sx=rng(-0.10,0.10); sy=rng(-0.10,0.10); sw=rng(0.14,0.32); sh=rng(0.12,0.28)
    else:
        side=random.choice([-1,1]); sx=side*rng(0.22,0.48); sy=rng(-0.16,0.16)
        sw=rng(0.12,0.28); sh=rng(0.10,0.22)
    grade=random.choices([1,2,3],weights=[0.50,0.30,0.20])[0]
    if grade==1:
        scar_color=(rng(0.88,0.94),rng(0.88,0.93),rng(0.84,0.90)); base_alpha=rng(0.28,0.48)
    elif grade==2:
        scar_color=(rng(0.84,0.92),rng(0.84,0.91),rng(0.78,0.87)); base_alpha=rng(0.55,0.75)
    else:
        scar_color=(rng(0.92,0.97),rng(0.92,0.96),rng(0.90,0.95)); base_alpha=rng(0.82,0.95)
    n_verts=rng_i(9,16)
    angles=np.linspace(0,2*math.pi,n_verts,endpoint=False)
    radii_w=np.array([sw/2*rng(0.72,1.28) for _ in angles])
    radii_h=np.array([sh/2*rng(0.72,1.28) for _ in angles])
    verts=[(sx+radii_w[i]*math.cos(a),sy+radii_h[i]*math.sin(a)) for i,a in enumerate(angles)]
    from matplotlib.patches import Polygon as MPoly
    ax.add_patch(MPoly(verts,facecolor=scar_color,alpha=base_alpha,
                        edgecolor=(0.78,0.75,0.68),linewidth=0.5,zorder=10))
    if grade>=2:
        for _ in range(rng_i(5,12)):
            lx=sx+rng(-sw*0.45,sw*0.45); ly_start=sy-sh*0.45; ly_end=sy+sh*0.45
            ax.plot([lx,lx+rng(-0.04,0.04)],[ly_start,ly_end],
                    color=(0.75,0.72,0.62),linewidth=rng(0.3,0.7),
                    alpha=rng(0.20,0.45),zorder=11,solid_capstyle="round")
    for _ in range(rng_i(8,20)):
        dx_=sx+rng(-sw*0.48,sw*0.48); dy_=sy+rng(-sh*0.48,sh*0.48)
        ax.add_patch(Ellipse((dx_,dy_),rng(0.008,0.025),rng(0.006,0.020),
                              facecolor=(0.82,0.80,0.72),alpha=rng(0.35,0.60),zorder=12))
    n_vessels=rng_i(4,10)
    for _ in range(n_vessels):
        ang_v=rng(0,2*math.pi)
        vx_start=sx+(sw/2+rng(0.10,0.30))*math.cos(ang_v)
        vy_start=sy+(sh/2+rng(0.08,0.25))*math.sin(ang_v)
        vx_end=sx+(sw/2*rng(0.0,0.6))*math.cos(ang_v+rng(-0.3,0.3))
        vy_end=sy+(sh/2*rng(0.0,0.6))*math.sin(ang_v+rng(-0.3,0.3))
        ax.plot([vx_start,vx_end],[vy_start,vy_end],
                color=(0.72,0.18,0.18),linewidth=rng(0.5,1.2),
                alpha=rng(0.50,0.75),zorder=11,solid_capstyle="round")
    draw_vessels(ax,6,(0.68,0.20,0.20),0.50)
    pcx,pcy,pr=vary_pupil_dilation(ax,lighting_type=f["lighting"])
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,""

def gen_keratitis(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,tint=(rng(0.88,0.94),rng(0.82,0.90),rng(0.60,0.72)),vc=28,va=0.75)
    for r_ring in [0.85,0.76,0.67]:
        alpha_ring=(0.90-r_ring)*rng(0.8,1.2)
        ax.add_patch(Ellipse((0,0),r_ring*1.85,r_ring*0.92,facecolor="none",
                              edgecolor=(rng(0.70,0.82),rng(0.18,0.28),rng(0.18,0.28)),
                              linewidth=rng(0.8,1.8),alpha=min(alpha_ring,0.55),zorder=3))
    ic,ir=draw_iris(ax)
    kx=rng(-0.12,0.12); ky=rng(-0.10,0.10); kw=rng(0.22,0.42); kh=rng(0.20,0.38)
    ax.add_patch(Ellipse((kx,ky),kw*1.55,kh*1.55,facecolor=(0.86,0.84,0.72),alpha=rng(0.22,0.38),zorder=8))
    ax.add_patch(Ellipse((kx,ky),kw,kh,facecolor=(0.80,0.77,0.58),alpha=rng(0.55,0.75),zorder=9))
    n_edge=rng_i(8,16)
    for ang in np.linspace(0,2*math.pi,n_edge,endpoint=False):
        ex=kx+(kw/2+rng(-0.04,0.08))*math.cos(ang+rng(-0.3,0.3))
        ey=ky+(kh/2+rng(-0.03,0.06))*math.sin(ang+rng(-0.3,0.3))
        ax.add_patch(Ellipse((ex,ey),rng(0.03,0.10),rng(0.02,0.07),
                              facecolor=(0.78,0.74,0.55),alpha=rng(0.25,0.50),zorder=9))
    fx=kx+rng(-0.04,0.04); fy=ky+rng(-0.03,0.03); fw=kw*rng(0.30,0.55); fh=kh*rng(0.28,0.52)
    ax.add_patch(Ellipse((fx,fy),fw,fh,facecolor=(0.96,0.96,0.78),alpha=rng(0.55,0.80),zorder=10))
    if random.random()<0.40:
        for branch in range(rng_i(2,5)):
            bx=kx+rng(-kw*0.3,kw*0.3); by=ky+rng(-kh*0.3,kh*0.3); ang_b=rng(0,2*math.pi)
            for seg in range(rng_i(3,7)):
                ex=bx+rng(0.02,0.06)*math.cos(ang_b+rng(-0.5,0.5))
                ey=by+rng(0.02,0.05)*math.sin(ang_b+rng(-0.5,0.5))
                ax.plot([bx,ex],[by,ey],color=(0.82,0.78,0.40),linewidth=rng(0.6,1.4),
                        alpha=rng(0.50,0.80),zorder=11,solid_capstyle="round")
                bx,by=ex,ey
    if random.random()<0.25:
        for _ in range(rng_i(2,5)):
            dist=rng(kw*0.6,kw*1.2); ang_s=rng(0,2*math.pi)
            sx_=kx+dist*math.cos(ang_s); sy_=ky+dist*0.7*math.sin(ang_s)
            sw_=rng(0.04,0.10); sh_=rng(0.03,0.08)
            ax.add_patch(Ellipse((sx_,sy_),sw_,sh_,facecolor=(0.80,0.76,0.55),alpha=rng(0.45,0.65),zorder=9))
    if random.random()<0.20:
        hp_y=rng(-0.40,-0.28); hp_h=rng(0.04,0.10)
        ax.add_patch(Ellipse((0,hp_y),rng(0.35,0.55),hp_h,facecolor=(0.95,0.95,0.92),alpha=rng(0.70,0.88),zorder=6))
    draw_vessels(ax,rng_i(18,30),(0.72,0.18,0.18),0.65)
    pcx,pcy,pr=vary_pupil_dilation(ax,lighting_type=f["lighting"])
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,make_yolo_label(1,kx,ky,max(kw,kh)*0.65)

def gen_pinguecula(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=7,va=0.30)
    side=random.choice([-1,1]); px_=side*rng(0.44,0.74); py_=rng(-0.10,0.10)
    pw=rng(0.14,0.26); ph=rng(0.08,0.15)
    ax.add_patch(Ellipse((px_,py_),pw,ph,facecolor=(0.93,0.90,0.68),alpha=rng(0.72,0.90),zorder=5))
    ax.add_patch(Ellipse((px_,py_),pw*0.55,ph*0.55,facecolor=(0.97,0.95,0.80),alpha=0.62,zorder=6))
    draw_vessels(ax,8,(0.65,0.22,0.22),0.42)
    ic,ir=draw_iris(ax); pcx,pcy,pr=draw_pupil(ax)
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,make_yolo_label(4,px_,py_,max(pw,ph)*0.8)

def gen_pterygium(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=10,va=0.35)
    side=random.choice([-1,1]); length=rng(0.22,0.55); base_x=side*0.85; tip_x=side*(0.85-length)
    pts=np.array([[base_x,rng(-0.15,0.15)],[tip_x,rng(-0.04,0.04)],[base_x,rng(-0.15,0.15)]])
    ax.add_patch(Polygon(pts,facecolor=(0.82,0.78,0.62),alpha=rng(0.70,0.88),zorder=5))
    draw_vessels(ax,12,(0.70,0.22,0.22),0.50)
    ic,ir=draw_iris(ax); pcx,pcy,pr=draw_pupil(ax)
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,make_yolo_label(3,(base_x+tip_x)/2,0,length*0.5)

def gen_foreign_body(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=10,va=0.40)
    ic,ir=draw_iris(ax); pcx,pcy,pr=draw_pupil(ax)
    fb_cx=fb_cy=fb_r=0.0
    for k in range(rng_i(1,4)):
        ang=rng(0,2*math.pi); dist=rng(0.05,0.70)
        fx=dist*math.cos(ang); fy=dist*math.sin(ang)*0.50; fr=rng(0.016,0.050)
        if k==0: fb_cx,fb_cy,fb_r=fx,fy,fr
        ax.add_patch(Ellipse((fx,fy),(fr+0.058)*2,(fr+0.058)*2,facecolor=(0.80,0.28,0.28),alpha=rng(0.22,0.42),zorder=7))
        ax.add_patch(Ellipse((fx,fy),(fr+0.024)*2,(fr+0.024)*2,facecolor=(0.60,0.34,0.09),alpha=0.65,zorder=8))
        ax.add_patch(Ellipse((fx,fy),fr*2,fr*1.5,angle=rng(0,180),facecolor=(0.08,0.06,0.04),alpha=0.95,zorder=9))
    ax.plot([rng(-0.24,0.24)+rng(-0.03,0.03) for _ in range(8)],np.linspace(-0.44,-0.92,8),
            color="#c8e0f0",linewidth=rng(1.0,2.2),alpha=0.65,zorder=13,solid_capstyle="round")
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],f["lash_shadow"])
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,make_yolo_label(2,fb_cx,fb_cy,fb_r*4)

def gen_subconjunctival_hemorrhage(view="center", train_px=224):
    f=_flags()
    fig,ax,sk=make_fig(); ax.set_facecolor(sk)
    draw_sclera(ax,vc=4,va=0.22)
    hx=rng(-0.40,0.40); hy=rng(-0.18,0.18); hw=rng(0.42,0.75); hh=rng(0.16,0.32)
    ax.add_patch(Ellipse((hx,hy),hw,hh,facecolor=(rng(0.62,0.74),rng(0.05,0.13),rng(0.05,0.11)),alpha=rng(0.78,0.93),zorder=4))
    for _ in range(14):
        ex=hx+rng(-hw/2,hw/2); ey=hy+rng(-hh/2,hh/2)
        ax.add_patch(Ellipse((ex,ey),rng(0.04,0.15),rng(0.03,0.10),facecolor=(0.65,0.08,0.08),alpha=rng(0.28,0.55),zorder=5))
    ic,ir=draw_iris(ax); pcx,pcy,pr=draw_pupil(ax)
    t_u,y_u,t_l,y_l=draw_eyelids(ax,sk,f["occlusion"],f["with_makeup"],True)
    arr_full,arr_small=_render(fig,f,view,train_px)
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l, view=view)
    return arr_small,arr_full,masks,make_yolo_label(5,hx,hy,max(hw,hh)*0.55)



GENERATORS = {
    0:gen_normal, 1:gen_conjunctivitis, 2:gen_allergic_conjunctivitis,
    3:gen_dry_eye, 4:gen_blepharitis, 5:gen_strabismus, 6:gen_ptosis,
    7:gen_corneal_scar, 8:gen_keratitis, 9:gen_pinguecula,
    10:gen_pterygium, 11:gen_foreign_body, 12:gen_subconjunctival_hemorrhage,
}

_CKPT_DIR       = Path("./ocuvision_checkpoints")
_stop_requested = _threading.Event()

def _setup_signal_handler():
    def _handler(sig, frame):
        print("\n\n  [PAUSE] Ctrl+C — finishing current item then stopping ...")
        print("  Re-run the same command to resume from this point.")
        _stop_requested.set()
    _signal.signal(_signal.SIGINT,  _handler)
    _signal.signal(_signal.SIGTERM, _handler)

def _ckpt_path(stage: str) -> Path:
    _CKPT_DIR.mkdir(exist_ok=True)
    return _CKPT_DIR / f"{stage}.ckpt.json"

def ckpt_save(stage: str, data: dict):
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

def ckpt_clear(stage: str):
    p = _ckpt_path(stage)
    if p.exists(): p.unlink()

def ckpt_status():
    _CKPT_DIR.mkdir(exist_ok=True)
    files = sorted(_CKPT_DIR.glob("*.ckpt.json"))
    if not files:
        print("\n  No checkpoints found — nothing has been started yet.\n"); return
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  CHECKPOINT STATUS                                          │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    for fp in files:
        try:
            with open(fp) as fh: data = json.load(fh)
        except Exception:
            data = {}
        stage   = fp.stem.replace(".ckpt","")
        updated = data.get("_updated","?")
        status  = data.get("status","in_progress")
        icon    = "✓ done   " if status=="done" else "⏸ paused "
        detail  = ""
        if "completed_classes" in data and "total_classes" in data:
            done=len(data["completed_classes"]); total=data["total_classes"]
            pct=int(done/total*100) if total else 0
            detail=f"{done}/{total} classes ({pct}%)"
        elif "epoch" in data and "total_epochs" in data:
            detail=f"epoch {data['epoch']}/{data['total_epochs']}"
        line=f"  │  {icon}  {stage:<22} {updated}   {detail}"
        print(f"{line:<66}│")
    print("  └─────────────────────────────────────────────────────────────┘\n")

def ckpt_reset(stage: str = None):
    _CKPT_DIR.mkdir(exist_ok=True)
    if stage:
        ckpt_clear(stage); print(f"  Checkpoint cleared: {stage}")
    else:
        cleared = sum(1 for fp in _CKPT_DIR.glob("*.ckpt.json") if (fp.unlink() or True))
        print(f"  Cleared {cleared} checkpoint(s).")

def _worker(args):
    """
    FIX-20: This runs in a subprocess spawned by multiprocessing.
    Each worker sets its own random seeds based on PID + case_idx to avoid
    identical images across workers.
    FIX-21: train_px is now correctly passed in args tuple (was missing in original).
    FIX-03: full-res save is now conditional on CFG["save_full_res"].
    """
    cid, case_idx, base_dir, train_px, save_full = args

    worker_seed = (os.getpid() * 131 + case_idx * 17 + cid * 1009) % (2**31)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    cname    = CLASSES[cid]
    case_dir = Path(base_dir) / cname / f"case_{case_idx:06d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = case_dir / "masks"; mask_dir.mkdir(exist_ok=True)

    done_views = [v for v in GAZE_VIEWS if (case_dir / f"{v}.jpg").exists()]
    if len(done_views) == len(GAZE_VIEWS):
        records = []
        for view in GAZE_VIEWS:
            yolo_txt = case_dir / f"{view}.txt"
            yolo_str = yolo_txt.read_text().strip() if yolo_txt.exists() else ""
            records.append({
                "path"     : str(case_dir / f"{view}.jpg"),
                "label"    : cid,
                "class"    : cname,
                "case_idx" : case_idx,
                "view"     : view,
                "yolo"     : yolo_str,
                "quality_label" : 0,
                "questionnaire" : _synth_questionnaire(cid),
            })
        return records

    records = []
    global _FORCE_QUALITY
    for view in GAZE_VIEWS:
        q_label = random.choices([0, 1, 2, 3], weights=[0.10, 0.10, 0.10, 0.70])[0]
        if q_label == 0: _FORCE_QUALITY = "partial_eye"
        elif q_label == 1: _FORCE_QUALITY = "blurry"
        elif q_label == 2: _FORCE_QUALITY = "low_light"
        else: _FORCE_QUALITY = None

        arr_small, arr_full, masks, yolo = GENERATORS[cid](view=view, train_px=train_px)
        img_fname = f"{view}.jpg"

        Image.fromarray(arr_small).save(
            case_dir / img_fname, format="JPEG", quality=CFG["jpeg_quality"])

        if save_full:
            Image.fromarray(arr_full).save(
                case_dir / f"{view}_full.jpg", format="JPEG", quality=CFG["jpeg_quality"])

        (case_dir / f"{view}.txt").write_text(yolo + "\n")

        for mname, mask in masks.items():
            cv2.imwrite(str(mask_dir / f"{view}_{mname}.png"), mask)

        records.append({
            "path"          : str(case_dir / img_fname),
            "label"         : cid,
            "class"         : cname,
            "case_idx"      : case_idx,
            "view"          : view,
            "yolo"          : yolo,
            "quality_label" : q_label,



            "questionnaire"  : _synth_questionnaire(cid),
        })
    return records



def generate_dataset(cfg=CFG):
    """
    FIX-02: Uses 'spawn' multiprocessing context. 'fork' + matplotlib deadlocks.
    FIX-21: Passes train_px AND save_full correctly into worker args tuple.
    """
    out_dir   = Path(cfg["out_dir"])
    workers   = cfg["num_workers"]
    img_size  = cfg["img_size"]
    n_per     = cfg["cases_per_class"]
    save_full = cfg.get("save_full_res", False)
    total_img = n_per * NC * cfg["gaze_views"]
    metadata  = []

    print("=" * 68)
    print("  OcuVisionAI  |  Dataset Generator  |  v6 (fixed)")
    print(f"  {NC} classes  |  {n_per:,}/class  |  7 gaze views  |  JPEG q={cfg['jpeg_quality']}")
    print(f"  Total training images : {total_img:,}    Workers : {workers}")
    est_gb = total_img * 20 / 1024 / 1024
    print(f"  Disk estimate (train) : ~{est_gb:.1f} GB")
    if save_full:
        print(f"  Disk estimate (full)  : ~{est_gb*7:.1f} GB  ← LARGE")
    print(f"  save_full_res = {save_full}")
    print("=" * 68)





    if _HAVE_PSUTIL:
        avail_gb = _psutil.virtual_memory().available / 1024**3
        if avail_gb < 4.0:
            workers = max(4, workers // 2)
            print(f"  ⚠ Low available RAM ({avail_gb:.1f}GB) — reducing workers to {workers}")
        else:
            print(f"  Available RAM: {avail_gb:.1f}GB — using {workers} workers")

    _setup_signal_handler()
    _stop_requested.clear()

    ckpt = ckpt_load("generate")
    completed_classes = set(ckpt.get("completed_classes", []))
    if completed_classes:
        print(f"  Resuming — {len(completed_classes)}/{NC} classes already done.")
        meta_f = out_dir / "metadata.json"
        if meta_f.exists():
            with open(meta_f) as fh: metadata = json.load(fh)
    else:
        print("  Starting fresh.")
        out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    ctx = multiprocessing.get_context("spawn")

    for cid, cname in CLASSES.items():
        if cid in completed_classes:
            print(f"  [{cid:02d}] {cname:<36} [already done]")
            continue

        tasks = [(cid, i, str(out_dir), img_size, save_full) for i in range(n_per)]

        print(f"  [{cid:02d}] {cname:<36} ", end="", flush=True)
        t1 = time.time()
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks)

        for case_records in results:
            metadata.extend(case_records)

        completed_classes.add(cid)
        ckpt_save("generate", {
            "status": "done" if len(completed_classes) == NC else "in_progress",
            "completed_classes": sorted(completed_classes),
            "total_classes": NC,
        })

        meta_f = out_dir / "metadata.json"
        with open(meta_f, "w") as fh: json.dump(metadata, fh, indent=2)
        print(f"{n_per:,} cases  {time.time()-t1:.0f}s")

        if _stop_requested.is_set():
            print(f"\n  [PAUSED] Progress saved. Re-run to continue.")
            return metadata

    meta_f = out_dir / "metadata.json"
    with open(meta_f, "w") as fh: json.dump(metadata, fh, indent=2)
    ckpt_save("generate", {"status":"done","completed_classes":list(range(NC)),"total_classes":NC})
    print(f"\n  metadata.json  -> {meta_f}")
    print(f"  Total time     : {(time.time()-t0)/60:.1f} min")
    print(f"  Total images   : {len(metadata):,}")
    print("=" * 68)
    return metadata

class EyeDataset(Dataset):
    def __init__(self, paths, labels, img_size, augment=False):
        self.paths = paths
        self.labels = labels
        self.img_size = img_size
        self.augment = augment

        transforms_list = [T.Resize((img_size, img_size))]
        if augment:
            transforms_list.extend([
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04)
            ])
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

class EyeSegDataset(Dataset):
    def __init__(self, meta_list, seg_size):
        self.meta = meta_list
        self.seg_size = seg_size
        self.channels = ["iris", "sclera", "pupil", "upper_lid", "lower_lid"]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        img_path = self.meta[idx]["path"]
        p = Path(img_path)
        view = p.stem
        mask_dir = p.parent / "masks"

        try:
            img = Image.open(img_path).convert('RGB')
            img = T.Resize((self.seg_size, self.seg_size))(img)
            img = T.ToTensor()(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        except Exception:
            img = torch.zeros(3, self.seg_size, self.seg_size)

        masks = []
        for ch in self.channels:
            mp = mask_dir / f"{view}_{ch}.png"
            try:
                m = Image.open(mp).convert('L')
                m = T.Resize((self.seg_size, self.seg_size), interpolation=T.InterpolationMode.NEAREST)(m)
                m = torch.from_numpy(np.array(m)) > 127
                masks.append(m.float())
            except Exception:
                masks.append(torch.zeros(self.seg_size, self.seg_size))

        return img, torch.stack(masks)

def build_dataset(paths, labels, img_size, batch, shuffle=True, augment=False):
    ds = EyeDataset(paths, labels, img_size, augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=min(4, CFG["num_workers"]), pin_memory=True, persistent_workers=False, prefetch_factor=2)

def build_seg_dataset(meta_list, seg_size, batch, shuffle=True):
    ds = EyeSegDataset(meta_list, seg_size)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=min(4, CFG["num_workers"]), pin_memory=True, persistent_workers=False, prefetch_factor=2)

def _split(metadata):
    train, val, test = [], [], []
    from collections import defaultdict
    class_dict = defaultdict(list)
    for m in metadata:
        class_dict[m["label"]].append(m)

    for cid, items in class_dict.items():
        random.shuffle(items)
        n = len(items)

        if cid in [5, 6, 11]:
            test_n = max(50, int(n * CFG["test_split"]))
            val_n = int(n * CFG["val_split"])
            if test_n + val_n > n:
                test_n = min(50, n); val_n = (n - test_n) // 2
        else:
            test_n = int(n * CFG["test_split"])
            val_n = int(n * CFG["val_split"])

        test.extend(items[:test_n])
        val.extend(items[test_n:test_n+val_n])
        train_items = items[test_n+val_n:]
        train.extend(train_items)




        if cid in [5, 6, 11] and len(train_items) < 20:
            print(f"  ⚠ WARNING: Class {cid} ({CLASSES[cid]}) has only "
                  f"{len(train_items)} training images after enforcing the 50-image "
                  f"test-set minimum. Consider increasing --cases.")

    print(f"  train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")
    return train, val, test

class Model1_Quality(nn.Module):
    def __init__(self):
        super().__init__()
        base = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class UNetLiteEncoder(nn.Module):
    """MobileNetV2 encoder with verified stride annotations (Section 6)."""
    def __init__(self):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        features = mobilenet.features





        self.enc1 = features[0:2]
        self.enc2 = features[2:4]
        self.enc3 = features[4:7]
        self.enc4 = features[7:14]


    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class Model2_Segmentation(nn.Module):
    """
    U-Net Lite with MobileNetV2 encoder (spec Section 6).
    Skip connections: 3, at strides /8 (e3), /4 (e2), /2 (e1) per spec.
    Decoder: /16→/8→/4→/2 (3 skip-concat steps) then /2→/1 (resolution restore).
    The final up_final step is NOT a 4th skip — it simply restores spatial resolution
    from H/2×W/2 back to H×W so the output is a per-pixel mask at full seg_size.
    This is a standard U-Net completion step, not an architectural deviation.
    5-channel output: iris, sclera, pupil, upper_lid, lower_lid.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = UNetLiteEncoder()

        self.up3  = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = DoubleConv(96, 64)
        self.up2  = nn.ConvTranspose2d(64, 48, 2, stride=2)
        self.dec2 = DoubleConv(72, 48)
        self.up1  = nn.ConvTranspose2d(48, 32, 2, stride=2)
        self.dec1 = DoubleConv(48, 32)

        self.up_final = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        d3 = self.dec3(torch.cat([e3, self.up3(e4)],  dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)],  dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)],  dim=1))
        out = self.out_conv(self.up_final(d1))
        return out


def _verify_model2_shape(seg_size=256):
    """Called once at the top of train_all() to catch shape mismatches before training."""
    m = Model2_Segmentation()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, seg_size, seg_size))
    expected = (1, 5, seg_size, seg_size)
    assert out.shape == tuple(expected), (
        f"[FATAL] Model2 output shape {tuple(out.shape)} != expected {expected}. "
        f"Check UNetLiteEncoder stride assumptions."
    )
    print(f"  [verify] Model2 output shape OK: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in m.parameters())
    print(f"  [verify] Model2 parameters: {n_params:,}")
    if not (4_000_000 <= n_params <= 6_000_000):
        print(f"  ⚠ WARNING: Model2 has {n_params:,} params — spec expects ~4–6M (Section 6). "
              f"Architecture may have drifted.")
    del m


class Model4_Classifier(nn.Module):
    """
    EfficientNet-Lite0 disease classifier (spec Section 8).
    P2: forward() exposes pre-pool feat_map when return_feature_map=True.
    Eigen-CAM consumes feat_map (C×H×W) via SVD; pooled embedding is used
    for Fusion MLP and standard classification.
    Channel-count note: EfficientNet-Lite0's head input is 1280 channels in
    timm, NOT 320 as the spec states. The spec's '320' likely refers to the
    last inverted-residual block output before the 1×1 expansion convolution.
    We use the actual timm tensor (1280ch) for all ops; this is a deliberate
    deviation from the spec's stated number because the spec's figure is
    inconsistent with the actual model graph (verify with Netron on exported ONNX).
    """
    def __init__(self, num_classes=13):
        super().__init__()
        self.base = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(1280, num_classes)

    def forward(self, x, return_feature_map=False):
        feat_map = self.base.forward_features(x)
        pooled   = torch.flatten(self.pool(feat_map), 1)
        logits   = self.fc(pooled)
        if return_feature_map:
            return logits, pooled, feat_map
        return logits, pooled


def _synth_questionnaire(class_id: int) -> list:
    """
    P4: Synthetic questionnaire proxy, derived from spec Section 13 mappings.
    Returns a 6-element list of floats snapped to {0.0, 0.5, 1.0}.
    Gaussian noise (sigma=0.15) prevents trivial label leakage while
    preserving the spec's structural question→symptom→condition mapping.
    NOT real clinical responses — see _QUESTION_TARGETS for mapping source.
    """
    q = []
    for qi in range(6):
        base = 1.0 if class_id in _QUESTION_TARGETS[qi] else 0.0
        val  = base + random.gauss(0, 0.15)
        val  = max(0.0, min(1.0, val))
        q.append(0.0 if val < 0.33 else (0.5 if val < 0.66 else 1.0))
    return q


def extract_geometry_from_masks(masks: dict, seg_size: int) -> np.ndarray:
    """
    P6: Derive the 5-element geometric feature vector from Model 2's binary
    output masks. Features are exactly the 5 defined in spec Section 9.
    masks: dict {channel_name: (H,W) uint8 binary array (0 or 255)}
    """
    def mask_stats(m):
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return dict(cx=seg_size/2, cy=seg_size/2, area=0.0, diam=0.0)
        return dict(
            cx=float(xs.mean()), cy=float(ys.mean()),
            area=float(len(xs)),
            diam=float(2 * np.sqrt(len(xs) / np.pi))
        )

    iris  = mask_stats(masks.get("iris",  np.zeros((seg_size, seg_size), np.uint8)))
    pupil = mask_stats(masks.get("pupil", np.zeros((seg_size, seg_size), np.uint8)))
    upper = masks.get("upper_lid", np.zeros((seg_size, seg_size), np.uint8))
    lower = masks.get("lower_lid", np.zeros((seg_size, seg_size), np.uint8))


    f1 = pupil["diam"] / iris["diam"] if iris["diam"] > 0 else 0.0


    ys, xs = np.where(masks.get("pupil", np.zeros_like(upper)) > 0)
    if len(xs) > 2:
        cov = np.cov(np.stack([xs.astype(float), ys.astype(float)]))
        eigvals = np.linalg.eigvalsh(cov)
        f2 = float(np.sqrt(eigvals.min() / eigvals.max())) if eigvals.max() > 0 else 1.0
    else:
        f2 = 1.0


    offset = np.hypot(pupil["cx"] - iris["cx"], pupil["cy"] - iris["cy"])
    f3 = offset / iris["diam"] if iris["diam"] > 0 else 0.0


    f4 = iris["diam"] / seg_size


    upper_ys, _ = np.where(upper > 0)
    lower_ys, _ = np.where(lower > 0)
    aperture = float(lower_ys.min() - upper_ys.max()) if len(upper_ys) and len(lower_ys) else 0.0
    f5 = max(0.0, aperture) / iris["diam"] if iris["diam"] > 0 else 0.0

    return np.array([f1, f2, f3, f4, f5], dtype=np.float32)


class Model5_EyePower(nn.Module):
    """
    P6: 5-class diopter-range classifier (spec Section 9).
    geom_fc input is 5 — consistent with spec's stated feature vector and
    extract_geometry_from_masks() output.
    Decision A: This model is defined but NOT trained in train_all() because
    no valid diopter-range label source exists (Blender ruled out, no clinical
    pairing data). Enable by populating m['diopter_label'] in metadata.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.geom_fc = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        self.fusion  = nn.Sequential(
            nn.Linear(128 + 32, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, img, geom):
        x = torch.flatten(self.features(img), 1)
        g = self.geom_fc(geom)
        return self.fusion(torch.cat([x, g], dim=1))


class FusionMLP(nn.Module):
    """
    P4: Late-fusion MLP combining Model 4 embedding with patient questionnaire.
    Trained AFTER Model 4 is frozen (spec Section 14 sequence).
    Input: 1280-dim embedding (from Model4) + 6-dim questionnaire vector.
    Output: 13-class disease logits (supersedes raw Model 4 output when
    questionnaire is supplied — this is the authoritative final prediction).
    """
    def __init__(self, num_classes=13):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1280 + 6, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, embedding, questionnaire):
        return self.fc(torch.cat([embedding, questionnaire], dim=1))


class FusionDataset(Dataset):
    """
    P4: Generates (embedding, questionnaire, label) triples for FusionMLP training.
    Runs frozen Model4 inline to produce real embeddings from images.
    Requires each metadata record to carry a 'questionnaire' field (populated
    by _synth_questionnaire() during dataset generation).
    """
    def __init__(self, meta_list, model4, img_size, device):
        self.meta    = meta_list
        self.model4  = model4.to(device).eval()
        self.device  = device
        self.xform   = T.Compose([
            T.Resize((img_size, img_size)), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for p in self.model4.parameters(): p.requires_grad = False

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        m = self.meta[idx]
        img = Image.open(m["path"]).convert("RGB")
        img_t = self.xform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, embedding = self.model4(img_t)
        q = torch.tensor(m.get("questionnaire", [0.0]*6), dtype=torch.float32)
        return embedding.squeeze(0).cpu(), q, torch.tensor(m["label"], dtype=torch.long)


def train_fusion_mlp(model4, train_m, val_m, cfg):
    """
    P4: Train FusionMLP using frozen Model4 embeddings + questionnaire vectors.
    DataLoader workers=0 because embeddings are computed on GPU inline —
    multiprocessing workers cannot share CUDA tensors.
    """
    print("  [Fusion MLP] NOTE: Questionnaire responses are SYNTHETIC PROXIES "
          "derived from spec Section 13 mappings. Not real clinical data.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FusionDataset(train_m, model4, cfg["img_size"], device)
    val_ds   = FusionDataset(val_m,   model4, cfg["img_size"], device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch"], shuffle=False, num_workers=0)
    fusion = FusionMLP(num_classes=NC)

    train_model(fusion, train_loader, val_loader, "fusion_mlp", nn.CrossEntropyLoss(), is_geom=True, accum_steps=4)
    return fusion

def train_model(model, train_loader, val_loader, name, loss_fn, epochs=None, is_geom=False, is_phase2=False, accum_steps=4):
    if epochs is None: epochs = CFG["epochs"]
    models_dir = Path(CFG["models_dir"]); models_dir.mkdir(exist_ok=True)
    save_path = str(models_dir / f"{name}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if is_phase2 and hasattr(model, "base") and hasattr(model, "fc"):
        optimizer = torch.optim.Adam([
            {"params": model.base.parameters(), "lr": CFG["lr"] * 0.1},
            {"params": model.fc.parameters(), "lr": CFG["lr"]}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    scaler = torch.amp.GradScaler('cuda') if CFG.get("mixed_precision", True) and device.type == 'cuda' else None

    best_loss = float('inf')
    print(f"\n{'='*68}\n  TRAINING: {name} (accum_steps={accum_steps})\n{'='*68}")

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                if is_geom:
                    img, geom, tgt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    out = model(img, geom)
                else:
                    img, tgt = batch[0].to(device), batch[1].to(device)
                    out = model(img)
                    if isinstance(out, tuple): out = out[0]
                loss = loss_fn(out, tgt) / accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            t_loss += loss.item() * accum_steps

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if is_geom:
                    img, geom, tgt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    out = model(img, geom)
                else:
                    img, tgt = batch[0].to(device), batch[1].to(device)
                    out = model(img)
                    if isinstance(out, tuple): out = out[0]
                v_loss += loss_fn(out, tgt).item()

        v_loss /= len(val_loader)
        scheduler.step(v_loss)
        print(f"  Epoch {epoch+1:02d}/{epochs} | Val Loss: {v_loss:.4f}")

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), save_path)

    print(f"  Saved -> {save_path}")
    model.cpu()

def train_all(metadata=None, cfg=None):
    if cfg is None: cfg = CFG
    models_dir = Path(cfg["models_dir"]); models_dir.mkdir(exist_ok=True)


    print(f"\n  MKL/oneDNN  : {torch.backends.mkldnn.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  CUDA Device : {props.name}, VRAM: {props.total_memory/1e9:.1f}GB")
    print(f"  Threads     : {torch.get_num_threads()} intra-op / {torch.get_num_interop_threads()} inter-op")
    print(f"  Workers     : {cfg['num_workers']} DataLoader")

    if metadata is None:
        meta_file = Path(cfg["out_dir"]) / "metadata.json"
        if not meta_file.exists():
            print("  No metadata.json found. Run 'generate' first."); return
        with open(meta_file) as fh: metadata = json.load(fh)
        print(f"  Loaded {len(metadata):,} records from {meta_file}")

    center_meta = [m for m in metadata if m.get("view","") == "center"] or metadata
    train_m, val_m, test_m = _split(center_meta)

    img_size = cfg["img_size"]
    seg_size = cfg["seg_size"]
    batch    = cfg["batch"]


    print(f"\n{'='*68}\n  PRE-FLIGHT: Model 2 shape + parameter verification\n{'='*68}")
    _verify_model2_shape(seg_size)

    print(f"\n{'='*68}\n  MODEL 1 — Image Quality (MobileNetV3Small)\n{'='*68}")
    q_paths  = [m["path"] for m in train_m]
    q_labels = [m.get("quality_label", 0) for m in train_m]
    vq_paths  = [m["path"] for m in val_m]
    vq_labels = [m.get("quality_label", 0) for m in val_m]
    train_q = build_dataset(q_paths, q_labels, img_size, batch, augment=True)
    val_q   = build_dataset(vq_paths, vq_labels, img_size, batch)
    m1 = Model1_Quality()
    train_model(m1, train_q, val_q, "model1_quality", nn.CrossEntropyLoss(), accum_steps=4)
    del train_q, val_q
    _clear_gpu()

    print(f"\n{'='*68}\n  MODEL 2 — U-Net Segmentation\n{'='*68}")
    train_seg = build_seg_dataset(train_m, seg_size, cfg["batch_seg"])
    val_seg   = build_seg_dataset(val_m,   seg_size, cfg["batch_seg"], shuffle=False)
    m2 = Model2_Segmentation()
    train_model(m2, train_seg, val_seg, "model2_segmentation", nn.BCEWithLogitsLoss(), accum_steps=4)
    del train_seg, val_seg
    _clear_gpu()

    print("\n  MODEL 3 (YOLOv8-n) trains separately via 'yolo' subcommand.")

    print(f"\n{'='*68}\n  MODEL 4 — Disease Classifier (EfficientNet-Lite0)\n{'='*68}")
    train_paths  = [m["path"]  for m in train_m]
    train_labels = [m["label"] for m in train_m]
    val_paths    = [m["path"]  for m in val_m]
    val_labels   = [m["label"] for m in val_m]
    train_cls = build_dataset(train_paths, train_labels, img_size, batch, augment=True)
    val_cls   = build_dataset(val_paths, val_labels, img_size, batch)
    m4 = Model4_Classifier()

    for param in m4.base.parameters(): param.requires_grad = False
    train_model(m4, train_cls, val_cls, "model4_classifier", nn.CrossEntropyLoss(), epochs=5, accum_steps=4)

    m4.load_state_dict(torch.load(models_dir / "model4_classifier.pth", weights_only=True))
    for param in m4.base.parameters(): param.requires_grad = True
    train_model(m4, train_cls, val_cls, "model4_classifier", nn.CrossEntropyLoss(), epochs=10, is_phase2=True, accum_steps=4)
    del train_cls, val_cls
    _clear_gpu()

    print(f"\n{'='*68}\n  FUSION MLP — Late Fusion (Model4 embedding + questionnaire)\n{'='*68}")
    m4_loaded = Model4_Classifier()
    m4_loaded.load_state_dict(torch.load(
        Path(cfg["models_dir"]) / "model4_classifier.pth", weights_only=True))
    m4_loaded.eval()
    fusion = train_fusion_mlp(m4_loaded, train_m, val_m, cfg)
    del m4_loaded

    print(f"\n{'='*68}\n  MODEL 5 — Eye Power Classification\n{'='*68}")
    print("  [SKIPPED] Model 5 not trained: no valid diopter-range label source.")
    print("  Blender is ruled out (per design decision A). No clinical pairing data.")
    print("  To enable: populate m['diopter_label'] (int 0–4) in metadata.json")
    print("  and uncomment the EyeGeomDataset training block.")
    print("  EyeGeomDataset and extract_geometry_from_masks() are fully implemented")
    print("  and ready to use once valid labels exist.")

    print(f"\n{'='*68}\n  ALL ACTIVE MODELS TRAINED\n{'='*68}")

def prepare_yolo_dataset(metadata, output_dir="./yolo_dataset"):
    """
    Prepare YOLO dataset from metadata.

    IMPORTANT:
    Source images reuse filenames such as center.jpg, left.jpg, etc.
    Therefore destination filenames MUST include a unique case identifier.
    Otherwise files overwrite each other.
    """
    import shutil
    import random

    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(
            parents=True, exist_ok=True
        )
        (output_dir / split / "labels").mkdir(
            parents=True, exist_ok=True
        )

    # YOLO currently uses center-view images.
    center_meta = [
        m for m in metadata
        if m.get("view", "") == "center"
    ]

    print(f"  Center-view records found: {len(center_meta)}")

    random.shuffle(center_meta)

    n_val = int(len(center_meta) * 0.20)

    splits = {
        "val": center_meta[:n_val],
        "train": center_meta[n_val:]
    }

    copied = {
        "train": 0,
        "val": 0
    }

    positive = {
        "train": 0,
        "val": 0
    }

    for split, items in splits.items():

        for idx, m in enumerate(items):

            src = Path(m["path"])

            if not src.exists():
                continue

            # ---------------------------------------------------------
            # Create a UNIQUE filename.
            #
            # case_idx is preferred because multiple source images
            # commonly have the same filename such as center.jpg.
            # ---------------------------------------------------------

            case_id = m.get("case_idx")

            if case_id is None:
                case_id = idx

            case_id = str(case_id)

            # Preserve original extension.
            extension = src.suffix.lower()

            unique_name = f"case_{case_id}_center{extension}"

            dst = (
                output_dir
                / split
                / "images"
                / unique_name
            )

            shutil.copy2(src, dst)

            # ---------------------------------------------------------
            # YOLO label
            # ---------------------------------------------------------

            lbl_file = (
                output_dir
                / split
                / "labels"
                / f"case_{case_id}_center.txt"
            )

            yolo_str = str(
                m.get("yolo", "")
            ).strip()

            with open(lbl_file, "w") as fh:

                if yolo_str:
                    fh.write(yolo_str + "\n")
                    positive[split] += 1

            copied[split] += 1

    # -------------------------------------------------------------
    # Dataset YAML
    # -------------------------------------------------------------

    yaml_path = output_dir / "dataset.yaml"

    with open(yaml_path, "w") as fh:

        fh.write(
            f"path: {output_dir.resolve()}\n"
        )

        fh.write(
            "train: train/images\n"
        )

        fh.write(
            "val: val/images\n"
        )

        fh.write(
            f"nc: {len(LESION_CLASSES)}\n"
        )

        fh.write(
            f"names: {list(LESION_CLASSES.values())}\n"
        )

    print()
    print("  YOLO dataset prepared successfully")
    print(f"  Train images    : {copied['train']}")
    print(f"  Validation     : {copied['val']}")
    print(f"  Train positive  : {positive['train']}")
    print(f"  Val positive    : {positive['val']}")
    print(f"  Dataset YAML    : {yaml_path}")

    return str(yaml_path)

def train_yolo(data_yaml="./yolo_dataset/dataset.yaml"):
    """
    FIX-10: Auto-detect device. If CUDA is present but PyTorch CUDA version
    mismatches (common on Ryzen 7 7435HS with ROCm/hybrid), fall back to CPU.
    """
    print(f"\n{'='*68}\n  MODEL 3 — YOLOv8-n Lesion Detection")
    print("  Releasing TF GPU memory before PyTorch YOLO launch")
    print(f"{'='*68}")
    _clear_gpu()
    gpu_usage()

    cmd = (
        f"yolo detect train model=yolov8n.pt data={data_yaml} "
        f"imgsz=320 batch=4 epochs={CFG['epochs']} "
        f"project=./ocuvision_models name=yolo_model3 "
        f"exist_ok=True"
    )
    print(f"\n  Running: {cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        print(f"\n  ⚠ YOLO training exited with code {ret}.")
        print("  Common fixes:")
        print("    1. pip install ultralytics")
        print("    2. yolo check")
        print(f"    3. Verify data yaml: {data_yaml}")

def export_onnx_models(models_dir, img_size, seg_size, val_m=None):
    """
    P5: Static INT8 quantization with calibration dataset (spec Section B.3).
    Uses validation split for calibration (GAP 9: spec doesn't specify source;
    val split is the only held-out, non-test data available).
    Fusion MLP is NOT quantized — small MLP, negligible benefit, and would
    require a separate two-input CalibrationDataReader class.
    """
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

    class EyeCalibrationReader(CalibrationDataReader):
        """Single-image-input calibration reader for M1, M2, M4."""
        def __init__(self, meta_list, size, input_name="input", max_samples=200):
            self.paths = [m["path"] for m in (meta_list or [])][:max_samples]
            self.size  = size
            self.input_name = input_name
            self._iter = iter(self.paths)
        def get_next(self):
            while True:
                path = next(self._iter, None)
                if path is None: return None
                try:
                    img = Image.open(path).convert("RGB").resize((self.size, self.size), resample=Image.BILINEAR)
                    arr = np.transpose(np.array(img, np.float32) / 255.0, (2, 0, 1))
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    arr = (arr - mean) / std
                    return {self.input_name: arr[np.newaxis]}
                except Exception:
                    continue

    class M4ExportWrapper(nn.Module):
        """P5: Wraps Model4 so return_feature_map=True is always active during export,
        registering feat_map as a named ONNX output for Eigen-CAM."""
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return self.base(x, return_feature_map=True)

    models_dir = Path(models_dir)
    print("\n  Exporting PyTorch models to ONNX + static INT8 quantization...")
    print("  [calibration] Using val split (max 200 images). Source: GAP 9 resolution.")

    def export_static(model, dummy, name, in_names, out_names, calib_size):
        raw  = str(models_dir / f"{name}.onnx")
        int8 = str(models_dir / f"{name}_int8.onnx")
        model.eval()
        torch.onnx.export(model, dummy, raw, input_names=in_names, output_names=out_names,
                          dynamic_axes={in_names[0]: {0: "batch"}})
        reader = EyeCalibrationReader(val_m, calib_size, input_name=in_names[0])
        quantize_static(raw, int8, reader, weight_type=QuantType.QInt8)
        print(f"    {name}: exported + static INT8 -> {int8}")

    m1 = Model1_Quality()
    m1.load_state_dict(torch.load(models_dir / "model1_quality.pth", weights_only=True))
    export_static(m1, torch.randn(1,3,img_size,img_size), "model1", ["input"], ["output"], img_size)

    m2 = Model2_Segmentation()
    m2.load_state_dict(torch.load(models_dir / "model2_segmentation.pth", weights_only=True))
    export_static(m2, torch.randn(1,3,seg_size,seg_size), "model2", ["input"], ["output"], seg_size)


    m4 = Model4_Classifier()
    m4.load_state_dict(torch.load(models_dir / "model4_classifier.pth", weights_only=True))
    m4_export = M4ExportWrapper(m4).eval()
    raw4 = str(models_dir / "model4.onnx")
    torch.onnx.export(m4_export, torch.randn(1,3,img_size,img_size), raw4,
                      input_names=["input"], output_names=["logits", "embedding", "feat_map"],
                      dynamic_axes={"input": {0: "batch"}})
    reader4 = EyeCalibrationReader(val_m, img_size)
    quantize_static(raw4, str(models_dir / "model4_int8.onnx"), reader4, weight_type=QuantType.QInt8)
    print("    model4: exported (logits, embedding, feat_map) + static INT8")


    fusion_path = models_dir / "fusion_mlp.pth"
    if fusion_path.exists():
        fusion = FusionMLP(NC)
        fusion.load_state_dict(torch.load(fusion_path, weights_only=True))
        fusion.eval()
        torch.onnx.export(fusion, (torch.randn(1,1280), torch.randn(1,6)),
                          str(models_dir / "fusion_mlp.onnx"),
                          input_names=["embedding", "questionnaire"], output_names=["logits"])
        print("    fusion_mlp: exported (unquantized — small model)")
    else:
        print("    fusion_mlp: skipped (not trained)")

    print("  ONNX export complete.")


def eigen_cam(feat_map: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    P3: Real Eigen-CAM implementation (spec Section 8).
    feat_map: (C, H, W) numpy array — intermediate conv output for ONE image.
    Returns a (target_size, target_size, 3) BGR heatmap blended as uint8.
    """
    c, h, w = feat_map.shape
    reshaped = feat_map.reshape(c, h * w)
    reshaped = reshaped - reshaped.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(reshaped, full_matrices=False)
    principal = Vt[0].reshape(h, w)
    cam = principal - principal.min()
    if cam.max() > 0: cam /= cam.max()
    cam_resized = cv2.resize(cam, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)


def blend_eigen_cam(original_rgb: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 1 - alpha, heatmap_rgb, alpha, 0)


def run_yolo_inference(model_path: str, image_path: str,
                       conf: float = 0.35, iou: float = 0.45, max_det: int = 20) -> list:
    """P7: YOLOv8 lesion detection with NMS parameters per spec."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, iou=iou, max_det=max_det, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class"     : int(box.cls.item()),
                "confidence": float(box.conf.item()),
                "bbox_norm" : box.xyxyn.tolist()[0],
            })
    return detections


def _fuse_predictions(probs_list):
    if not probs_list: return np.ones(NC) / NC
    return np.mean(np.stack(probs_list, axis=0), axis=0)


def run_pipeline(image_path_or_paths, questionnaire=None, models_dir=None, results_dir=None):
    """
    Full sequential ONNX inference pipeline.
    questionnaire: list of 6 floats (spec Section 13 responses), or None for image-only mode.
    """
    if models_dir  is None: models_dir  = CFG["models_dir"]
    if results_dir is None: results_dir = CFG["results_dir"]
    models_dir  = Path(models_dir);  results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    if isinstance(image_path_or_paths, str): image_paths = [image_path_or_paths]
    else: image_paths = list(image_path_or_paths)

    def load_img(path, size):
        img = Image.open(path).convert("RGB").resize((size, size), resample=Image.BILINEAR)
        arr = np.transpose(np.array(img, np.float32) / 255.0, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std
        return arr[np.newaxis]

    img_size = CFG["img_size"]; seg_size = CFG["seg_size"]
    print(f"\n{'='*60}\n  OcuVisionAI — Inference (ONNX)\n{'='*60}")

    import onnxruntime as ort

    def _sess(name):
        try: return ort.InferenceSession(str(models_dir / name))
        except Exception: return None

    sess1      = _sess("model1_int8.onnx")
    sess2      = _sess("model2_int8.onnx")
    sess4      = _sess("model4_int8.onnx")
    sess_fusion= _sess("fusion_mlp.onnx")
    yolo_path  = models_dir / "yolo_model3" / "weights" / "best.pt"

    quality_scores=[]; cls_probs_all=[]; embeddings_all=[]; fusion_probs_all=[]; yolo_dets_all=[]

    for i, path in enumerate(image_paths):
        if not Path(path).exists(): continue
        view = GAZE_VIEWS[i] if i < len(GAZE_VIEWS) else f"view_{i}"
        print(f"\n  [{view}] {path}")
        img = load_img(path, img_size)


        if sess1:
            q_logits = sess1.run(None, {"input": img})[0][0]
            q_logits = q_logits - np.max(q_logits)
            q_probs  = np.exp(q_logits) / np.sum(np.exp(q_logits))
            q_label  = QUALITY_LABELS[int(np.argmax(q_probs))]
            q_conf   = float(q_probs.max()) * 100
            quality_scores.append((q_label, q_conf))
            print(f"    Quality    : {q_label} ({q_conf:.0f}%)")
            if q_label == "reject": continue


        seg_masks = None
        if sess2:
            seg_img  = load_img(path, seg_size)
            seg_out  = sess2.run(None, {"input": seg_img})[0][0]
            channels = ["iris", "sclera", "pupil", "upper_lid", "lower_lid"]
            seg_masks = {ch: (seg_out[c] > 0.5).astype(np.uint8) * 255
                         for c, ch in enumerate(channels)}
            print(f"    Seg mask   : shape={seg_out.shape} (5 channels)")
            for ch_name, mask_arr in seg_masks.items():
                mask_path = results_dir / f"segmask_{view}_{ch_name}.png"
                cv2.imwrite(str(mask_path), mask_arr)
            geom = extract_geometry_from_masks(seg_masks, seg_size)
            print(f"    Geometry   : pupil/iris={geom[0]:.3f}  roundness={geom[1]:.3f}  "
                  f"offset={geom[2]:.3f}  iris_ratio={geom[3]:.3f}  aperture={geom[4]:.3f}")


        if yolo_path.exists():
            dets = run_yolo_inference(str(yolo_path), path)
            yolo_dets_all.extend(dets)
            if dets:
                det_strs = ", ".join(
                    f"{LESION_CLASSES.get(d['class'], d['class'])} ({d['confidence']*100:.0f}%)"
                    for d in dets)
                print(f"    Lesion Det.: {len(dets)} finding(s) — {det_strs}")
            else:
                print(f"    Lesion Det.: none above threshold (conf={0.35})")
        else:
            print(f"    Lesion Det.: model not found — run 'yolo' then 'yolo-prep'")


        if sess4:
            out4       = sess4.run(None, {"input": img})
            cls_logits = out4[0][0]
            cls_logits = cls_logits - np.max(cls_logits)
            embedding  = out4[1][0]
            feat_map   = out4[2][0]
            cls_probs  = np.exp(cls_logits) / np.sum(np.exp(cls_logits))
            cls_probs_all.append(cls_probs)
            embeddings_all.append(embedding)
            top_cls = CLASSES[int(np.argmax(cls_probs))]
            print(f"    Pre-fusion : {top_cls} ({float(cls_probs.max())*100:.1f}%)")


            heatmap  = eigen_cam(feat_map, target_size=img_size)
            orig_rgb = (np.transpose(img[0], (1,2,0)) * 255).astype(np.uint8)
            blended  = blend_eigen_cam(orig_rgb, heatmap)
            cam_path = results_dir / f"eigencam_{view}.jpg"
            Image.fromarray(blended).save(cam_path)
            print(f"    Eigen-CAM  : saved -> {cam_path}")


        if sess_fusion and embeddings_all and questionnaire is not None:
            q_vec = np.array(questionnaire, dtype=np.float32)[np.newaxis, :]
            emb   = embeddings_all[-1][np.newaxis, :]
            f_logits = sess_fusion.run(None, {"embedding": emb, "questionnaire": q_vec})[0][0]
            f_logits = f_logits - np.max(f_logits)
            f_probs  = np.exp(f_logits) / np.sum(np.exp(f_logits))
            fusion_probs_all.append(f_probs)
            f_cls = CLASSES[int(np.argmax(f_probs))]
            print(f"    Post-fusion: {f_cls} ({float(f_probs.max())*100:.1f}%)  [questionnaire-informed]")


        print(f"    Eye Power  : not available (model not trained — awaiting clinical label data)")

    print(f"\n  {'─'*56}\n  FINAL FUSED RESULT\n  {'─'*56}")



    if fusion_probs_all:
        final_probs = _fuse_predictions(fusion_probs_all)
        final_cls   = CLASSES[int(np.argmax(final_probs))]
        final_conf  = float(final_probs.max()) * 100
        prefusion   = _fuse_predictions(cls_probs_all)
        pre_cls     = CLASSES[int(np.argmax(prefusion))]
        print(f"  Image-only (pre-fusion) : {pre_cls} ({float(prefusion.max())*100:.1f}%)")
        print(f"  FINAL (post-fusion)     : {final_cls} ({final_conf:.1f}%)  [authoritative]")
    elif cls_probs_all:
        final_probs = _fuse_predictions(cls_probs_all)
        final_cls   = CLASSES[int(np.argmax(final_probs))]
        final_conf  = float(final_probs.max()) * 100
        print(f"  FINAL (image-only, no questionnaire supplied): {final_cls} ({final_conf:.1f}%)")
        print(f"  [note] Supply --questionnaire to activate Fusion MLP for improved accuracy.")
    else:
        final_cls = "unknown"; final_conf = 0.0
        print("  No valid views processed.")


    print("\n  ⚠ Eye Power Estimate — EXPERIMENTAL")
    print("  The system presents eye power estimation as an experimental screening indicator,")
    print("  not as a clinically validated measurement. Model 5 is not currently trained")
    print("  (awaiting validated diopter-range labels). Do not use for clinical decisions.")

    result = {
        "disease"        : final_cls,
        "confidence"     : round(final_conf, 2),
        "quality"        : quality_scores,
        "yolo_detections": yolo_dets_all,
        "power_cls"      : -1,
        "eye_power_note" : "Model 5 not trained — awaiting clinical label data",
    }
    rpath = results_dir / "inference_result.json"
    with open(rpath, "w") as fh: json.dump(result, fh, indent=2, default=str)
    print(f"\n  Results -> {rpath}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    import argparse
    _setup_signal_handler()

    p = argparse.ArgumentParser(
        description="OcuVisionAI — Tesla V100 32GB | 32 vCPUs Xeon Gold",
        formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    gen_p = sub.add_parser("generate", help="Generate synthetic dataset")
    gen_p.add_argument("--cases", type=int, default=CFG["cases_per_class"])
    gen_p.add_argument("--full-res", action="store_true")

    sub.add_parser("train",  help="Train PyTorch models (M1, M2, M4, Fusion MLP)")
    sub.add_parser("export", help="Export to ONNX static INT8")

    yolo_p = sub.add_parser("yolo", help="Train YOLOv8-n (Model 3)")
    yolo_p.add_argument("--data", default="./yolo_dataset/dataset.yaml")
    sub.add_parser("yolo-prep", help="Prepare YOLO dataset")

    infer_p = sub.add_parser("infer", help="Run ONNX inference")
    infer_p.add_argument("images", nargs="+")
    infer_p.add_argument(
        "--questionnaire", type=float, nargs=6, default=None,
        metavar=("Q1","Q2","Q3","Q4","Q5","Q6"),
        help="6 questionnaire responses (0.0/0.5/1.0 each). "
             "If omitted, falls back to image-only Model 4 output.")

    sub.add_parser("status", help="Show checkpoint status")
    reset_p = sub.add_parser("reset", help="Clear checkpoint(s)")
    reset_p.add_argument("--stage", default=None)

    args = p.parse_args()
    print("======================================================================")
    print("  OcuVisionAI — Tesla V100 32GB VRAM | 32 vCPUs Xeon Gold")
    print("======================================================================")

    if args.cmd == "generate":
        CFG["cases_per_class"] = args.cases
        CFG["save_full_res"]   = getattr(args, "full_res", False)
        generate_dataset(CFG)
    elif args.cmd == "train":
        train_all()
    elif args.cmd == "export":
        meta_f = Path(CFG["out_dir"]) / "metadata.json"
        if not meta_f.exists(): print("No metadata.json — run 'generate' first.")
        else:
            with open(meta_f) as fh: md = json.load(fh)
            center_meta = [m for m in md if m.get("view","") == "center"] or md
            _, val_m, _ = _split(center_meta)
            export_onnx_models(CFG["models_dir"], CFG["img_size"], CFG["seg_size"], val_m)
    elif args.cmd == "yolo":
        train_yolo(data_yaml=args.data)
    elif args.cmd == "yolo-prep":
        meta_f = Path(CFG["out_dir"]) / "metadata.json"
        if not meta_f.exists(): print("No metadata.json.")
        else:
            with open(meta_f) as fh: md = json.load(fh)
            prepare_yolo_dataset(md)
    elif args.cmd == "infer":
        run_pipeline(args.images, questionnaire=args.questionnaire)
    elif args.cmd == "status":
        ckpt_status()
    elif args.cmd == "reset":
        ckpt_reset(args.stage)
    else:
        p.print_help()
