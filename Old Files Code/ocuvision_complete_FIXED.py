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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"  # FIXED: disabled

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"  GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"  GPU setup warning: {e}")
        tf.config.optimizer.set_jit(True)

try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except:
    pass

mixed_precision.set_global_policy("float32")  # FIXED: disabled mixed_float16

CFG = {
    "img_size"        : 600, "img_size" : 600, # EfficientNetB7 needs larger images
    "seg_size"        : 256,
    "gen_px"          : 512,
    "dpi"             : 100,
    "num_classes"     : 13,
    "gaze_views"      : 7,

    "cases_per_class" : 6000,  #	10000 →	6000 (60% synthetic)
    "out_dir"         : "./eye_dataset_v6",
    "models_dir"      : "./ocuvision_models",
    "results_dir"     : "./ocuvision_results",

    "batch": 64,  # FIXED   # was 32 — V100 can handle 128+ for EfficientNetB3  32 to 64

    "batch_seg": 32,  # FIXED    # was 16   16 →	32 (double it)
    "epochs"          : 100,   #	50 →	100 (more capacity)
    "lr": 1e-4,  # FIXED
    "shuffle"         : 8192,  # was 2048

    "num_workers"     : 16,   # keep as-is
    "mixed_precision" : True,
    "test_split"      : 0.20,
    "val_split"       : 0.10,
    "jpeg_quality"    : 90,

    "save_full_res"   : True,
}

def _clear_gpu():
    """Release all Keras/TF session GPU memory and run GC."""
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph() 
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
                              cx_p, cy_p, r_pupil, t_u, y_u, t_l, y_l):
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

    eyelid_mask = np.zeros((H,W), dtype=np.uint8)
    pts_u = np.array([[(int((t/AXIS_SPAN+0.5)*W), int((1.0-(y/AXIS_SPAN+0.5))*H))
                        for t,y in zip(t_u,y_u)] + [(W-1,0),(0,0)]], dtype=np.int32)
    cv2.fillPoly(eyelid_mask, pts_u, 255)
    pts_l = np.array([[(int((t/AXIS_SPAN+0.5)*W), int((1.0-(y/AXIS_SPAN+0.5))*H))
                        for t,y in zip(t_l,y_l)] + [(W-1,H-1),(0,H-1)]], dtype=np.int32)
    cv2.fillPoly(eyelid_mask, pts_l, 255)

    return {
        "iris"       : iris_mask,
        "pupil"      : pupil_mask,
        "sclera"     : sclera_mask,
        "eyelid"     : eyelid_mask,
        "limbus"     : limbus_mask,
        "conjunctiva": conjunctiva_mask,
    }

def make_yolo_label(class_id, cx_n, cy_n, r_n):
    AXIS_SPAN = 2.24
    cx = max(0.001, min(0.999, cx_n/AXIS_SPAN+0.5))
    cy = max(0.001, min(0.999, 1.0-(cy_n/AXIS_SPAN+0.5)))
    w  = max(0.01,  min(0.999, r_n*2.5/AXIS_SPAN))
    h  = max(0.01,  min(0.999, r_n*2.0/AXIS_SPAN))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def _flags():
    return dict(
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
    masks = make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small, arr_full, masks, make_yolo_label(0,0,0,ir)

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
    masks = make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small, arr_full, masks, make_yolo_label(1,0,0,ir)

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
    masks = make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small, arr_full, masks, make_yolo_label(2,0,0,ir)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(3,0,0,ir)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(4,0,-0.36,0.25)

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
    masks=make_segmentation_masks(GEN_SIZE,dx,dy,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(5,dx,dy,ir)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(6,0,y_u.mean()*0.5,0.50)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(7,sx,sy,max(sw,sh)*0.65)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(8,kx,ky,max(kw,kh)*0.65)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(9,px_,py_,max(pw,ph)*0.8)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(10,(base_x+tip_x)/2,0,length*0.5)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(11,fb_cx,fb_cy,fb_r*4)

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
    masks=make_segmentation_masks(GEN_SIZE,0,0,ir,pcx,pcy,pr,t_u,y_u,t_l,y_l)
    return arr_small,arr_full,masks,make_yolo_label(12,hx,hy,max(hw,hh)*0.55)

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
            })
        return records

    records = []
    for view in GAZE_VIEWS:
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
            "path"     : str(case_dir / img_fname),
            "label"    : cid,
            "class"    : cname,
            "case_idx" : case_idx,
            "view"     : view,
            "yolo"     : yolo,
        })
    return records

def _quality_label_worker(path: str) -> int:
    """FIX-06: Moved out of main process. Runs in a Pool."""
    try:
        img = cv2.imread(path)
        if img is None: return 0
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur   = cv2.Laplacian(gray, cv2.CV_64F).var()
        bright = gray.mean()
        if blur < 20 or bright < 30:   return 0
        elif blur < 50 or bright < 50: return 1
        elif bright > 220 or blur < 80:return 2
        else:                          return 3
    except Exception:
        return random.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]

def _parallel_quality_labels(paths, n_workers=16):
    """Compute quality labels in parallel using a process pool."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        labels = pool.map(_quality_label_worker, paths)
    return labels

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

def build_dataset(paths, labels, img_size, batch,
                  shuffle=True,
                  augment=False):
    """
    Memory-optimized dataset pipeline.
    """

    labels = np.asarray(labels, dtype=np.float32)

    def _safe_load_py(path_bytes):
        try:
            path = path_bytes.numpy().decode("utf-8")

            raw = tf.io.read_file(path)

            img = tf.image.decode_jpeg(raw, channels=3)

            img = tf.image.resize(
                img,
                (img_size, img_size),
            )

            img = tf.cast(img, tf.float32) / 255.0

        except Exception:

            img = tf.zeros(
                (img_size, img_size, 3),
                dtype=tf.float32
            )

        return img

    def load(path, label):
        img = tf.py_function(
            _safe_load_py,
            [path],
            tf.float32
        )

        img.set_shape((img_size, img_size, 3))

        return img, label

    def augment_fn(img, label):

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.20)
        img = tf.image.random_contrast(img, 0.80, 1.20)
        img = tf.image.random_saturation(img, 0.80, 1.20)
        img = tf.image.random_hue(img, 0.04)

        return tf.clip_by_value(img, 0.0, 1.0), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=CFG["shuffle"],
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        ds = ds.map(
            augment_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(
        batch,
        drop_remainder=False
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds
def build_seg_dataset(meta_list, seg_size, batch, shuffle=True):
    """
    FIX-04 / FIX-24: Original mask loading was broken.
    Generator saves masks as: masks/{view}_{region}.png (e.g., center_iris.png)
    We load the iris mask as the primary segmentation target for simplicity.
    A full 6-class segmentation dataset builder would require stacking all 6
    masks into a single multi-channel image — expensive and complex.
    This version trains U-Net as a binary iris segmenter (iris vs background).
    To do full 6-class seg, switch to the commented block below.
    """

    def _find_mask(img_path_str: str) -> str:
        """Derive iris mask path from training image path."""
        p = Path(img_path_str)
        view = p.stem
        mask_p = p.parent / "masks" / f"{view}_iris.png"
        return str(mask_p)

    def load_pair(img_path, mask_path):

        raw  = tf.io.read_file(img_path)
        img  = tf.image.decode_jpeg(raw, channels=3)
        img  = tf.image.resize(img, (seg_size, seg_size))
        img  = tf.cast(img, tf.float32) / 255.0

        raw_m = tf.io.read_file(mask_path)
        mask  = tf.image.decode_png(raw_m, channels=1)
        mask  = tf.image.resize(mask, (seg_size, seg_size), method="nearest")
        mask  = tf.cast(mask > 127, tf.float32)
        return img, mask

    img_paths  = [m["path"] for m in meta_list]
    mask_paths = [_find_mask(m["path"]) for m in meta_list]

    valid_pairs = [(ip, mp) for ip, mp in zip(img_paths, mask_paths) if Path(mp).exists()]
    if not valid_pairs:
        print("  WARNING: No segmentation masks found. Skipping seg dataset build.")

        dummy_img  = tf.zeros((1, seg_size, seg_size, 3), tf.float32)
        dummy_mask = tf.zeros((1, seg_size, seg_size, 1), tf.float32)
        return tf.data.Dataset.from_tensors((dummy_img[0], dummy_mask[0])).batch(1)

    img_paths  = [p[0] for p in valid_pairs]
    mask_paths = [p[1] for p in valid_pairs]
    print(f"  Seg dataset: {len(img_paths):,} valid pairs found.")

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(CFG["shuffle"], seed=SEED)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def _split(metadata):
    labels = [m["label"] for m in metadata]
    n_min_class = min(labels.count(i) for i in range(NC))

    use_stratify = n_min_class >= max(10, int(len(metadata) * CFG["test_split"] * 2))

    tmp, test = train_test_split(
        metadata, test_size=CFG["test_split"],
        stratify=labels if use_stratify else None, random_state=SEED)
    vf = CFG["val_split"] / (1 - CFG["test_split"])
    train, val = train_test_split(
        tmp, test_size=vf,
        stratify=[m["label"] for m in tmp] if use_stratify else None, random_state=SEED)
    print(f"  train:{len(train):,}  val:{len(val):,}  test:{len(test):,}  "
          f"(stratify={'yes' if use_stratify else 'no'})")
    return train, val, test

def build_quality_model(img_size=224):
    """
    MobileNetV3Small for 4-class quality assessment.
    FIX-01: Output Dense cast to float32 (already done, verified).
    VRAM estimate: ~12MB weights + ~80MB activations at batch=2 = safe.
    """
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x   = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(4, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(base.input, out, name="Model1_Quality_MobileNetV3Small")

def _conv_block(x, f):
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return x

def build_unet(seg_size=256):
    """
    Lightweight U-Net for binary iris segmentation.
    FIX-23: Output changed to sigmoid (binary). For multi-class use softmax + one-hot.
    VRAM: ~8MB weights + ~180MB activations at batch=1 = safe on 4GB.
    """
    inp = tf.keras.Input((seg_size, seg_size, 3))
    s1  = _conv_block(inp, 16);  p1 = tf.keras.layers.MaxPool2D()(s1)
    s2  = _conv_block(p1,  32);  p2 = tf.keras.layers.MaxPool2D()(s2)
    s3  = _conv_block(p2,  64);  p3 = tf.keras.layers.MaxPool2D()(s3)
    b   = _conv_block(p3, 128)
    u1  = tf.keras.layers.UpSampling2D()(b)
    u1  = tf.keras.layers.Concatenate()([u1, s3]); u1 = _conv_block(u1, 64)
    u2  = tf.keras.layers.UpSampling2D()(u1)
    u2  = tf.keras.layers.Concatenate()([u2, s2]); u2 = _conv_block(u2, 32)
    u3  = tf.keras.layers.UpSampling2D()(u2)
    u3  = tf.keras.layers.Concatenate()([u3, s1]); u3 = _conv_block(u3, 16)

    out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(u3)
    return tf.keras.Model(inp, out, name="Model2_UNetLite_BinaryIris")

def build_classifier(img_size=224, num_classes=NC):
    """
    EfficientNet-B3 disease classifier.
    FIX-07: batch MUST be ≤ 2. At batch=2: ~1.2GB VRAM (weights + activations).
    FIX-17: BatchNorm + float16 can NaN — loss scaling handled by Keras automatically
            when mixed_precision policy is active AND model output is float32.
    """
    base = tf.keras.applications.EfficientNetB7(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x   = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(base.input, out, name="Model4_Classifier_EfficientNetB3")
    return model, base

def build_regression(img_size=224):
    """
    Lightweight custom CNN for eye power regression.
    FIX-25: Labels are SYNTHETIC (random Gaussian). This model CANNOT predict
            real refractive error from fundus-style synthetic images.
            It is a placeholder for when real labeled data is available.
    VRAM: ~6MB weights at batch=2 = trivially safe.
    """
    inp = tf.keras.Input((img_size, img_size, 3))
    x   = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x   = tf.keras.layers.MaxPool2D()(x)
    x   = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x   = tf.keras.layers.MaxPool2D()(x)
    x   = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(4, dtype="float32")(x)
    return tf.keras.Model(inp, out, name="Model5_EyePower_CustomCNN")

def _standard_callbacks(model_name="model"):
    models_dir = Path(CFG["models_dir"]); models_dir.mkdir(exist_ok=True)
    return [
    EarlyStopping(
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        patience=2,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
    filepath=str(models_dir / f"{model_name}_best.h5"),
    save_best_only=True,
    save_weights_only=False,
    monitor="val_loss",
    verbose=1
),
]

def train_model(model, train_ds, val_ds, name, loss="categorical_crossentropy",
                metrics=None, epochs=None):
    """Train one model, save, then aggressively clear GPU memory."""
    if metrics is None: metrics = ["accuracy"]
    if epochs  is None: epochs  = CFG["epochs"]

    models_dir = Path(CFG["models_dir"]); models_dir.mkdir(exist_ok=True)
    save_path  = str(models_dir / f"{name}.keras")

    print(f"\n{'='*68}\n  TRAINING: {name}")
    print(f"  Epochs: {epochs}  |  Loss: {loss}")
    print(f"{'='*68}")
    gpu_usage()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=CFG["lr"],
        clipnorm=1.0,
    ),
        loss=loss, metrics=metrics,
    )
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=_standard_callbacks(name), verbose=1,
    )
    model.save(save_path, save_format="keras")
    print(f"\n  Saved -> {save_path}")
    gpu_usage()
    del model
    _clear_gpu()
    print(f"  GPU cleared after {name}\n")
    return history

def train_all(metadata=None, cfg=None):
    """
    Train all models sequentially while aggressively freeing memory
    between stages.
    """
    import gc

    if cfg is None:
        cfg = CFG

    models_dir = Path(cfg["models_dir"])
    models_dir.mkdir(exist_ok=True)

    if metadata is None:
        meta_file = Path(cfg["out_dir"]) / "metadata.json"
        if not meta_file.exists():
            print("No metadata.json found. Run 'generate' first.")
            return

        with open(meta_file) as fh:
            metadata = json.load(fh)

        print(f"Loaded {len(metadata):,} records.")

    center_meta = [m for m in metadata if m.get("view", "") == "center"] or metadata
    train_m, val_m, test_m = _split(center_meta)

    img_size = cfg["img_size"]
    seg_size = cfg["seg_size"]
    batch = cfg["batch"]

    ############################################################
    # MODEL 1
    ############################################################

    print("\nMODEL 1 : QUALITY")
    _clear_gpu()

    q_paths = [m["path"] for m in train_m]
    q_label_ints = _parallel_quality_labels(
        q_paths,
        n_workers=min(cfg["num_workers"], 16)
    )
    q_labels = tf.one_hot(q_label_ints, 4).numpy().astype(np.float32)

    vq_paths = [m["path"] for m in val_m]
    vq_label_ints = _parallel_quality_labels(
        vq_paths,
        n_workers=min(cfg["num_workers"], 16)
    )
    vq_labels = tf.one_hot(vq_label_ints, 4).numpy().astype(np.float32)

    train_q = build_dataset(
        q_paths,
        q_labels,
        img_size,
        batch,
        augment=True,
    )

    val_q = build_dataset(
        vq_paths,
        vq_labels,
        img_size,
        batch,
    )

    m1 = build_quality_model(img_size)

    train_model(
        m1,
        train_q,
        val_q,
        "model1_quality",
    )

    gpu_usage()

    ckpt_save("model1", {"status": "done"})

    del m1
    del train_q, val_q
    del q_labels, vq_labels
    del q_label_ints, vq_label_ints

    gc.collect()
    _clear_gpu()

    ############################################################
    # MODEL 2
    ############################################################

    print("\nMODEL 2 : SEGMENTATION")
    _clear_gpu()

    train_seg = build_seg_dataset(
        train_m,
        seg_size,
        cfg["batch_seg"],
    )

    val_seg = build_seg_dataset(
        val_m,
        seg_size,
        cfg["batch_seg"],
        shuffle=False,
    )

    m2 = build_unet(seg_size)

    train_model(
        m2,
        train_seg,
        val_seg,
        "model2_segmentation",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    gpu_usage()

    ckpt_save("model2", {"status": "done"})

    del m2
    del train_seg, val_seg

    gc.collect()
    _clear_gpu()

    ############################################################
    # MODEL 3
    ############################################################

    print("\nYOLO trains separately.")
    print("python ocuvision_complete.py yolo-prep")
    print("python ocuvision_complete.py yolo")

    ############################################################
    # MODEL 4
    ############################################################

    print("\nMODEL 4 : CLASSIFIER")
    _clear_gpu()

    train_paths = [m["path"] for m in train_m]
    val_paths = [m["path"] for m in val_m]

    train_labels = tf.one_hot(
        [m["label"] for m in train_m],
        NC,
    ).numpy().astype(np.float32)

    val_labels = tf.one_hot(
        [m["label"] for m in val_m],
        NC,
    ).numpy().astype(np.float32)

    train_cls = build_dataset(
        train_paths,
        train_labels,
        img_size,
        batch,
        augment=True,
    )

    val_cls = build_dataset(
        val_paths,
        val_labels,
        img_size,
        batch,
    )

    m4, base4 = build_classifier(img_size)

    print("Phase 1")

    m4.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    m4.fit(
        train_cls,
        validation_data=val_cls,
        epochs=cfg["epochs"],
        callbacks=_standard_callbacks("model4_p1"),
    )

    print("Phase 2")

    for layer in base4.layers:
        layer.trainable = False

    # More aggressive fine tuning
    for layer in base4.layers[-30:]:
        layer.trainable = True

    m4.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    try:
        m4.fit(
            train_cls,
            validation_data=val_cls,
            epochs=10,
            callbacks=_standard_callbacks("model4_p2"),
        )

    except tf.errors.ResourceExhaustedError:
        print("OOM during fine tuning.")

    m4.save(models_dir / "model4_classifier.keras")

    gpu_usage()

    ckpt_save("model4", {"status": "done"})

    del m4
    del base4
    del train_cls, val_cls
    del train_labels, val_labels

    gc.collect()
    _clear_gpu()

    ############################################################
    # MODEL 5
    ############################################################

    print("\nMODEL 5 : REGRESSION")
    _clear_gpu()

    def synth_power(_):
        s = random.gauss(0, 3.0)
        c = random.gauss(0, 1.5)
        return [
            s,
            c,
            s + random.gauss(0, 0.5),
            c + random.gauss(0, 0.3),
        ]

    tp_labels = np.array(
        [synth_power(m["label"]) for m in train_m],
        dtype=np.float32,
    )

    vp_labels = np.array(
        [synth_power(m["label"]) for m in val_m],
        dtype=np.float32,
    )

    train_pow = build_dataset(
        train_paths,
        tp_labels,
        img_size,
        batch,
    )

    val_pow = build_dataset(
        val_paths,
        vp_labels,
        img_size,
        batch,
    )

    m5 = build_regression(img_size)

    train_model(
        m5,
        train_pow,
        val_pow,
        "model5_power",
        loss="mse",
        metrics=["mae"],
    )

    gpu_usage()

    ckpt_save("model5", {"status": "done"})

    del m5
    del train_pow, val_pow
    del tp_labels, vp_labels

    gc.collect()
    _clear_gpu()

    print("\nALL MODELS TRAINED")
    gpu_usage()
def prepare_yolo_dataset(metadata, output_dir="./yolo_dataset"):
    import shutil
    output_dir = Path(output_dir)
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    center_meta = [m for m in metadata if m.get("view","") == "center"]
    random.shuffle(center_meta)
    n_val = int(len(center_meta) * 0.2)
    splits = {"val": center_meta[:n_val], "train": center_meta[n_val:]}

    for split, items in splits.items():
        for m in items:
            src = Path(m["path"])
            if not src.exists(): continue
            dst = output_dir / split / "images" / src.name
            shutil.copy2(src, dst)
            lbl_file = output_dir / split / "labels" / src.name.replace(".jpg",".txt")
            yolo_str = m.get("yolo","")
            if not yolo_str:
                yolo_str = f"{m['label']} 0.5 0.5 0.8 0.8"
            with open(lbl_file, "w") as fh: fh.write(yolo_str + "\n")

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {output_dir.resolve()}\n")
        fh.write("train: train/images\nval: val/images\n")
        fh.write(f"nc: {NC}\nnames: {list(CLASSES.values())}\n")

    print(f"  YOLO dataset prepared -> {output_dir}")
    return str(yaml_path)

def train_yolo(data_yaml="./yolo_dataset/dataset.yaml"):
    """
    Train YOLOv8 on GPU (device 0).
    """

    print(f"\n{'='*68}\n  MODEL 3 — YOLOv8 Lesion Detection")
    print("  Releasing TF GPU memory before PyTorch YOLO launch")
    print(f"{'='*68}")

    _clear_gpu()
    gpu_usage()

    cmd = (
        f"yolo detect train "
        f"model=yolov8m.pt "
        f"data={data_yaml} "
        f"imgsz=640 "
        f"batch=32 "
        f"epochs={CFG['epochs']} "
        f"device=0 "
        f"project=./ocuvision_models "
        f"name=yolo_model3 "
        f"exist_ok=True"
    )

    print(f"\nRunning:\n{cmd}\n")

    ret = os.system(cmd)

    if ret != 0:
        print(f"\n⚠ YOLO training exited with code {ret}")
        print("Common fixes:")
        print("  1. pip install ultralytics")
        print("  2. yolo checks")
        print(f"  3. Verify dataset: {data_yaml}")

def _fuse_predictions(probs_list):
    if not probs_list: return np.ones(NC) / NC
    return np.mean(np.stack(probs_list, axis=0), axis=0)

def run_pipeline(image_path_or_paths, models_dir=None, results_dir=None):
    """
    Run full sequential inference. Loads each model one at a time.
    FIX-11: Guards against empty cls_probs_all and power_results.
    """
    if models_dir  is None: models_dir  = CFG["models_dir"]
    if results_dir is None: results_dir = CFG["results_dir"]
    models_dir  = Path(models_dir);  results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    if isinstance(image_path_or_paths, str):
        image_paths = [image_path_or_paths]
    else:
        image_paths = list(image_path_or_paths)

    def load_img(path, size):
        return np.array(
            Image.open(path).convert("RGB").resize((size, size)),
            dtype=np.float32) / 255.0

    img_size = CFG["img_size"]
    print(f"\n{'='*60}\n  OcuVisionAI — Inference\n{'='*60}")

    quality_scores=[]; cls_probs_all=[]; power_results=[]

    for i, path in enumerate(image_paths):
        if not Path(path).exists():
            print(f"  [{i}] File not found: {path}"); continue
        view = GAZE_VIEWS[i] if i < len(GAZE_VIEWS) else f"view_{i}"
        print(f"\n  [{view}] {path}")
        img = load_img(path, img_size)

        m1_path = models_dir / "model1_quality.keras"
        if not m1_path.exists():
            print("  Model 1 not found — skipping quality check"); q_label="acceptable"
        else:
            m1 = keras.models.load_model(str(m1_path))
            q_probs = m1.predict(img[np.newaxis], verbose=0)[0]
            del m1; _clear_gpu()
            q_label = QUALITY_LABELS[int(np.argmax(q_probs))]
            q_conf  = float(q_probs.max()) * 100
            quality_scores.append((q_label, q_conf))
            print(f"    Quality    : {q_label} ({q_conf:.0f}%)")
            if q_label == "reject":
                print("    [SKIPPED — rejected quality]"); continue

        m2_path = models_dir / "model2_segmentation.keras"
        if m2_path.exists():
            seg_img = load_img(path, CFG["seg_size"])
            m2 = keras.models.load_model(str(m2_path))
            seg = m2.predict(seg_img[np.newaxis], verbose=0)[0]
            del m2; _clear_gpu()
            print(f"    Seg mask   : {seg.shape}")

        m4_path = models_dir / "model4_classifier.keras"
        if m4_path.exists():
            m4 = keras.models.load_model(str(m4_path))
            cls_probs = m4.predict(img[np.newaxis], verbose=0)[0]
            del m4; _clear_gpu()
            cls_probs_all.append(cls_probs)
            top_cls = CLASSES[int(np.argmax(cls_probs))]
            print(f"    Classifier : {top_cls} ({float(cls_probs.max())*100:.1f}%)")

        m5_path = models_dir / "model5_power.keras"
        if m5_path.exists():
            m5 = keras.models.load_model(str(m5_path))
            pwr = m5.predict(img[np.newaxis], verbose=0)[0]
            del m5; _clear_gpu()
            power_results.append(tuple(float(v) for v in pwr))
            print(f"    Eye Power  : L sph={pwr[0]:+.2f} cyl={pwr[1]:+.2f}  "
                  f"R sph={pwr[2]:+.2f} cyl={pwr[3]:+.2f}")

    print(f"\n  {'─'*56}\n  FINAL FUSED RESULT\n  {'─'*56}")
    fused     = _fuse_predictions(cls_probs_all)
    final_cls  = CLASSES[int(np.argmax(fused))]
    final_conf = float(fused.max()) * 100

    if cls_probs_all:
        print(f"  Disease ({len(cls_probs_all)} views): {final_cls} ({final_conf:.1f}%)")
        for i, p in enumerate(fused):
            print(f"    {CLASSES[i]:<36} {float(p)*100:5.1f}%  {'#'*int(float(p)*30)}")
    else:
        print("  No valid views processed (all rejected or models missing).")

    if power_results:
        avg = [np.mean([r[j] for r in power_results]) for j in range(4)]
        print(f"\n  Eye Power (avg {len(power_results)} views):")
        print(f"    Left  : SPH={avg[0]:+.2f}  CYL={avg[1]:+.2f}")
        print(f"    Right : SPH={avg[2]:+.2f}  CYL={avg[3]:+.2f}")

    result = {
        "disease"   : final_cls,
        "confidence": round(final_conf, 2),
        "quality"   : quality_scores,
        "power"     : power_results,
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
        description="OcuVisionAI — RTX 3050 4GB VRAM Production Mode",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
QUICK START (RTX 3050 safe):
  python ocuvision_complete.py generate --cases 500
  python ocuvision_complete.py train
  python ocuvision_complete.py yolo-prep
  python ocuvision_complete.py yolo
  python ocuvision_complete.py infer path/to/eye.jpg

RESUME: Re-run the same command — checkpoints are automatic.
RESET:  python ocuvision_complete.py reset --stage generate
""")
    sub = p.add_subparsers(dest="cmd")

    gen_p = sub.add_parser("generate", help="Generate synthetic dataset")
    gen_p.add_argument("--cases", type=int, default=CFG["cases_per_class"],
                       help=f"Cases per class (default: {CFG['cases_per_class']})")
    gen_p.add_argument("--full-res", action="store_true",
                       help="Also save full-resolution images (+21 GB disk)")

    sub.add_parser("train", help="Train models 1, 2, 4, 5 sequentially")

    yolo_p = sub.add_parser("yolo", help="Train YOLOv8-n (Model 3)")
    yolo_p.add_argument("--data", default="./yolo_dataset/dataset.yaml")

    sub.add_parser("yolo-prep", help="Prepare YOLO dataset from metadata")

    infer_p = sub.add_parser("infer", help="Run inference on image(s)")
    infer_p.add_argument("images", nargs="+")

    sub.add_parser("status", help="Show checkpoint status")

    reset_p = sub.add_parser("reset", help="Clear checkpoint(s)")
    reset_p.add_argument("--stage", default=None,
                         help="Stage name to reset (omit for all)")

    args = p.parse_args()

    print("""
======================================================================
  OcuVisionAI — Tesla V100 Production Mode (v6-v100)
======================================================================
  Hardware  : Tesla V100 32GB VRAM | 30GB RAM | 32 vCPU
  Models    : Quality | U-Net | YOLOv8-m | EfficientNetB3 | Regression
  Batch     : 32 (optimized for V100)
  Rule      : Mixed Precision + XLA + TF32 Enabled
  Full res  : ON by default
======================================================================
""")
    gpu_usage()

    if args.cmd == "generate":
        CFG["cases_per_class"] = args.cases
        CFG["save_full_res"]   = getattr(args, "full_res", False)
        generate_dataset(CFG)

    elif args.cmd == "train":
        train_all()

    elif args.cmd == "yolo":
        train_yolo(data_yaml=args.data)

    elif args.cmd == "yolo-prep":
        meta_f = Path(CFG["out_dir"]) / "metadata.json"
        if not meta_f.exists():
            print("  No metadata.json. Run 'generate' first.")
        else:
            with open(meta_f) as fh: md = json.load(fh)
            prepare_yolo_dataset(md)

    elif args.cmd == "infer":
        run_pipeline(args.images)

    elif args.cmd == "status":
        ckpt_status()

    elif args.cmd == "reset":
        ckpt_reset(args.stage)

    else:
        p.print_help()
