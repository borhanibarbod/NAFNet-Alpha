"""
run_inference.py — run NAFNet-Alpha on one image.

OPTION A — paste into a Colab cell:

    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)

    import sys, torch
    from PIL import Image
    from google.colab import files

    NAF_DIR = "/content/drive/MyDrive/NAFNet-image denoising-final model"
    sys.path.insert(0, NAF_DIR)

    from model import NAFNetAlpha, run_model

    DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_PATH = NAF_DIR + "/checkpoints/EXPB_FINAL_a0_42.60.pth"
    ALPHA     = 0.20  # 0.00=clean  0.20=natural  0.50=grain  1.00=max grain

    model = NAFNetAlpha(width=32).to(DEVICE)
    ck = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ck.get("model_state", ck), strict=False)
    model.eval()
    print("model ready")

    uploaded = files.upload()
    img_name = list(uploaded.keys())[0]
    result = run_model(model, Image.open(img_name), alpha=ALPHA, device=DEVICE)
    result.save("/content/result.png")
    files.download("/content/result.png")

OPTION B — command line (model.py must be in the same folder):

    python run_inference.py \
        --input  path/to/image.jpg \
        --output path/to/result.png \
        --checkpoint path/to/EXPB_FINAL_a0_42.60.pth \
        --alpha 0.20

Alpha: 0.00=clean  0.20=natural (default)  0.50=grain  1.00=max grain
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def _load_model(checkpoint_path, device):
    # model.py lives in ../src relative to this script; also accept a copy
    # placed next to the script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, "..", "src"))
    for p in (src_dir, script_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from model import NAFNetAlpha
    except ImportError as exc:
        raise ImportError(
            f"Cannot import NAFNetAlpha — make sure model.py is in src/ "
            f"or next to run_inference.py.\nLooked in: {src_dir} and {script_dir}\n{exc}"
        )

    model = NAFNetAlpha(width=32).to(device)
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck.get("model_state", ck), strict=False)
    model.eval()
    return model


def _pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def _pad_to_multiple(x, multiple=8):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w


def _resize_if_needed(img, max_side=1600):
    w, h = img.size
    biggest = max(w, h)
    if biggest <= max_side:
        return img
    scale = max_side / biggest
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)


def run(input_path, output_path, checkpoint_path, alpha=0.20):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = _load_model(checkpoint_path, device)
    print(f"Loaded: {checkpoint_path}")

    img = ImageOps.exif_transpose(Image.open(input_path)).convert("RGB")
    img = _resize_if_needed(img)
    print(f"Input: {input_path}  size: {img.size}  alpha: {alpha}")

    x = _pil_to_tensor(img).to(device)
    x_pad, h, w = _pad_to_multiple(x)
    alpha_t = torch.tensor([[float(alpha)]], device=device)

    with torch.no_grad():
        y_pad = model(x_pad, alpha_t)

    result = _tensor_to_pil(y_pad[:, :, :h, :w])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAFNet-Alpha — denoise one image.")
    parser.add_argument("--input",      required=True,              help="noisy input image")
    parser.add_argument("--output",     required=True,              help="output path (.png)")
    parser.add_argument("--checkpoint", required=True,              help="path to .pth file")
    parser.add_argument("--alpha",      type=float, default=0.20,   help="0.0–1.0 (default 0.20)")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be between 0.0 and 1.0")

    run(args.input, args.output, args.checkpoint, args.alpha)
