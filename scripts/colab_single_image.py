# this cell loads model.py from the root folder , and downloads the result.
#You can use this code to run the model and test it on a photo you want

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys, os, torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F

NAF_DIR = "/content/drive/MyDrive/NAFNet-image denoising-final model"

# Load model.py
with open(NAF_DIR + "/model.py", 'r') as f:
    exec(f.read(), globals())
print("✓ model loaded")

# Load checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NAFNetAlpha(width=32).to(DEVICE)
ck = torch.load(NAF_DIR + "/checkpoints/EXPB_FINAL_a0_42.60.pth", map_location=DEVICE)
model.load_state_dict(ck.get("model_state", ck), strict=False)
model.eval()
print("✓ checkpoint loaded")

# ── RUN ON YOUR IMAGE ──
from google.colab import files
uploaded = files.upload()
img_name = list(uploaded.keys())[0]

result = run_model(model, Image.open(img_name), alpha=0.20, device=DEVICE)
result.save("/content/result.png")
print("✓ Done — saved to /content/result.png")
files.download("/content/result.png")