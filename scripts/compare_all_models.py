"""
compare_all_models.py — NAFNet-Alpha vs BM3D vs DnCNN

NAFNet-Alpha: EXPB_FINAL_a0_42.60.pth
BM3D:         sigma=25/255, classical patch denoiser
DnCNN:        dncnn_color_blind.pth from KAIR (nb=20, 3ch, pretrained)
              trained on synthetic Gaussian noise, so on real phone images
              it outputs near-identity (SSIM~0.99 vs noisy input) — expected.

Metrics are against the noisy input since no clean ground truth exists.
Higher PSNR here means the output is closer to the noisy original.

Results saved to: NAFNet-image denoising-final model/results/comparison/
"""

import os
import sys
import shutil
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd

print("╔" + "=" * 100 + "╗")
print("║" + " Complete Denoising Comparison".center(100) + "║")
print("║" + " NAFNet-Alpha vs BM3D vs DnCNN".center(100) + "║")
print("╚" + "=" * 100 + "╝\n")

from google.colab import drive, files
drive.mount("/content/drive", force_remount=False)

NAF_DIR = "/content/drive/MyDrive/NAFNet-image denoising-final model"
DEVICE  = "cuda" if __import__("torch").cuda.is_available() else "cpu"

if not os.path.exists(NAF_DIR):
    raise FileNotFoundError(
        f"\nProject folder not found at:\n  {NAF_DIR}\n"
        "Update NAF_DIR at the top of this script to match your Drive path."
    )

print(f"Project folder: {NAF_DIR}")
print(f"Device: {DEVICE}\n")

os.system("pip install -q bm3d scikit-image scipy pandas requests")

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NAFNet-Alpha ---

print("[MODEL 1] NAFNet-Alpha...")

sys.path.insert(0, NAF_DIR)
from model import NAFNetAlpha, run_model

ckpt_path = os.path.join(NAF_DIR, "checkpoints", "EXPB_FINAL_a0_42.60.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

naf_model = NAFNetAlpha(width=32).to(DEVICE)
ck    = torch.load(ckpt_path, map_location=DEVICE)
state = ck.get("model_state", ck)
naf_model.load_state_dict(state, strict=False)
naf_model.eval()

print("✓ Checkpoint loaded\n")

# --- BM3D ---

print("[MODEL 2] BM3D...")

try:
    from bm3d import bm3d
    HAVE_BM3D = True
    print("✓ Ready\n")
except Exception:
    HAVE_BM3D = False
    print("✗ Skipped\n")

# --- DnCNN (pretrained, KAIR color-blind, nb=20) ---

print("[MODEL 3] DnCNN (pretrained color-blind)...")

# Matches dncnn_color_blind.pth from KAIR exactly.
# nb=20: Conv+ReLU | 18x Conv+BN+ReLU | Conv, stored as model.0..19
def build_kair_dncnn(in_nc=3, out_nc=3, nc=64, nb=20):
    layers = []
    # first: Conv + ReLU, no BN
    layers.append(nn.Conv2d(in_nc, nc, 3, padding=1, bias=True))
    layers.append(nn.ReLU(inplace=True))
    # middle: Conv + BN + ReLU
    for _ in range(nb - 2):
        layers.append(nn.Conv2d(nc, nc, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(nc, momentum=0.9, eps=1e-04, affine=True))
        layers.append(nn.ReLU(inplace=True))
    # last: Conv only
    layers.append(nn.Conv2d(nc, out_nc, 3, padding=1, bias=True))

    class _DnCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return x - self.model(x)  # output = noisy - predicted residual

    return _DnCNN()


DNCNN_CKPT = "/content/dncnn_color_blind.pth"
DNCNN_URL  = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"

HAVE_DNCNN = False
dncnn = None

try:
    if not os.path.exists(DNCNN_CKPT):
        print("  Downloading dncnn_color_blind.pth (~1.4 MB)...")
        r = requests.get(DNCNN_URL, timeout=60)
        r.raise_for_status()
        with open(DNCNN_CKPT, "wb") as f:
            f.write(r.content)
        print("  Done.")

    dncnn = build_kair_dncnn(in_nc=3, out_nc=3, nc=64, nb=20).to(DEVICE)
    ck_d  = torch.load(DNCNN_CKPT, map_location=DEVICE)
    dncnn.load_state_dict(ck_d, strict=True)
    dncnn.eval()
    HAVE_DNCNN = True
    print("✓ Pretrained weights loaded\n")

except Exception as e:
    HAVE_DNCNN = False
    print(f"✗ DnCNN failed: {e}")
    print("  Continuing without DnCNN.\n")

# --- Upload images ---

print("[UPLOAD] Test Images")
print("-" * 60)
print("Upload 2-4 noisy smartphone photos\n")

uploaded = files.upload()

test_dir = "/content/test_images"
os.makedirs(test_dir, exist_ok=True)

for name in list(uploaded.keys()):
    dst = os.path.join(test_dir, os.path.basename(name))
    if os.path.abspath(name) != os.path.abspath(dst):
        shutil.move(name, dst)

img_paths = sorted([
    os.path.join(test_dir, f) for f in os.listdir(test_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
])

if not img_paths:
    print("No images found — exiting.")
    sys.exit(1)

print(f"✓ {len(img_paths)} image(s) ready\n")

# --- Helpers ---

def center_crop(img_np, ratio=0.4):
    h, w   = img_np.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img_np[y0:y0 + ch, x0:x0 + cw]


def compute_metrics(ref, out):
    try:
        p = psnr_fn(ref, out, data_range=1.0)
    except Exception:
        p = 0.0
    try:
        s = ssim_fn(ref, out, data_range=1.0, channel_axis=2)
    except Exception:
        s = 0.0
    return p, s


def save_grid(outputs_dict, model_order, grid_type, img_name, out_dir):
    present = [n for n in model_order if n in outputs_dict]
    if not present:
        return

    ncols = min(3, len(present))
    nrows = (len(present) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    label = "Full" if grid_type == "full" else "Crop 40%"
    fig.suptitle(f"{label}: {img_name}", fontsize=16, fontweight="bold")

    for idx, name in enumerate(present):
        r, c = divmod(idx, ncols)
        data = outputs_dict[name] if grid_type == "full" else center_crop(outputs_dict[name])
        axes[r][c].imshow(data)
        axes[r][c].set_title(name, fontsize=12, fontweight="bold")
        axes[r][c].axis("off")

    for idx in range(len(present), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    suffix = "00_FULL" if grid_type == "full" else "01_CROP"
    plt.savefig(os.path.join(out_dir, f"{img_name}_{suffix}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


# --- Run comparison ---

print("[RUN] Denoising Comparison")
print("-" * 60)
print()

output_dir = os.path.join(NAF_DIR, "results", "comparison")
os.makedirs(output_dir, exist_ok=True)

MODEL_ORDER = ["Original", "NAFNet-α=0", "NAFNet-α=1", "BM3D", "DnCNN"]
all_results = []

for img_idx, img_path in enumerate(img_paths, 1):
    img_name = Path(img_path).stem
    print(f"[{img_idx}/{len(img_paths)}] {img_name}")

    img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    if max(img_pil.size) > 1600:
        scale   = 1600 / max(img_pil.size)
        img_pil = img_pil.resize(
            (int(round(img_pil.size[0] * scale)), int(round(img_pil.size[1] * scale))),
            Image.LANCZOS
        )

    img_np = np.array(img_pil)                 # uint8 HxWx3
    img_01 = img_np.astype(np.float32) / 255.0 # float32 [0,1]

    outputs = {"Original": img_np}

    try:
        outputs["NAFNet-α=0"] = np.array(run_model(naf_model, img_pil, alpha=0.0, device=DEVICE))
        outputs["NAFNet-α=1"] = np.array(run_model(naf_model, img_pil, alpha=1.0, device=DEVICE))
    except Exception as e:
        print(f"  NAFNet error: {e}")

    if HAVE_BM3D:
        try:
            bm3d_out = bm3d(img_01, sigma_psd=25 / 255.0)
            outputs["BM3D"] = (np.clip(bm3d_out, 0, 1) * 255).astype(np.uint8)
        except Exception as e:
            print(f"  BM3D error: {e}")

    if HAVE_DNCNN:
        try:
            img_t = torch.from_numpy(img_01).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out_t = dncnn(img_t)
            outputs["DnCNN"] = (torch.clamp(out_t, 0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        except Exception as e:
            print(f"  DnCNN error: {e}")

    print(f"  {'Model':<20} | {'PSNR (dB)':>10} | {'SSIM':>8}")
    print(f"  {'-' * 44}")

    for name in ["NAFNet-α=0", "NAFNet-α=1", "BM3D", "DnCNN"]:
        if name not in outputs:
            continue
        res_01         = outputs[name].astype(np.float32) / 255.0
        p_full, s_full = compute_metrics(img_01, res_01)
        p_crop, s_crop = compute_metrics(center_crop(img_01), center_crop(res_01))
        print(f"  {name:<20} | {p_full:10.2f} | {s_full:8.4f}")
        all_results.append({
            "image": img_name, "model": name,
            "psnr_full": p_full, "ssim_full": s_full,
            "psnr_crop": p_crop, "ssim_crop": s_crop,
        })

    print()

    for grid_type in ("full", "crop"):
        save_grid(outputs, MODEL_ORDER, grid_type, img_name, output_dir)

    print(f"  ✓ Grids saved\n")


# --- Summary ---

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60 + "\n")

if all_results:
    df = pd.DataFrame(all_results)

    print("Full image — average PSNR per model:")
    print(
        df.groupby("model")[["psnr_full"]]
          .mean()
          .sort_values("psnr_full", ascending=False)
          .rename(columns={"psnr_full": "PSNR (dB)"})
          .to_string()
    )
    print()

    print("Center crop 40% — average PSNR per model:")
    print(
        df.groupby("model")[["psnr_crop"]]
          .mean()
          .sort_values("psnr_crop", ascending=False)
          .rename(columns={"psnr_crop": "PSNR (dB)"})
          .to_string()
    )

    csv_path = os.path.join(output_dir, "RESULTS.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
else:
    print("No results.")

print(f"\nOutputs: {output_dir}")
