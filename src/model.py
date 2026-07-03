"""  
NAFNet-Alpha — Controllable Smartphone Image Denoiser
Architecture: NAFNet backbone with FiLM conditioning and alpha-gated skip connections.
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from PIL import Image, ImageOps  
import numpy as np  
import os, glob  
from pathlib import Path  


class SimpleGate(nn.Module):  
    def forward(self, x):  
        x1, x2 = x.chunk(2, dim=1)  
        return x1 * x2  

  
class NAFBlock(nn.Module):  
    def __init__(self, ch, dw_expand=1, ffn_expand=2):  
        super().__init__()  
        dw_ch = ch * dw_expand  
        ffn_ch = ch * ffn_expand  
          
        self.norm1 = nn.LayerNorm(ch)  
        self.conv1 = nn.Conv2d(ch, dw_ch, 1)  
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)  
        self.gate1 = SimpleGate()  
        self.sca = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),  
        )  
        self.conv3 = nn.Conv2d(dw_ch // 2, ch, 1)  
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1))  
          
        self.norm2 = nn.LayerNorm(ch)  
        self.conv4 = nn.Conv2d(ch, ffn_ch, 1)  
        self.gate2 = SimpleGate()  
        self.conv5 = nn.Conv2d(ffn_ch // 2, ch, 1)  
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))  
          
    def forward(self, x):  
        inp = x  
        x = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)  
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.gate1(x)  
        x = x * torch.sigmoid(self.sca(x))  
        x = self.conv3(x)  
        y = inp + x * self.beta  
          
        x = self.norm2(y.permute(0,2,3,1)).permute(0,3,1,2)  
        x = self.conv4(x)  
        x = self.gate2(x)  
        x = self.conv5(x)  
        return y + x * self.gamma  

  
class AlphaFiLM(nn.Module):  
    def __init__(self, ch, hidden=64):  
        super().__init__()  
        self.ch = ch  
        self.net = nn.Sequential(  
            nn.Linear(1, hidden), nn.SiLU(),  
            nn.Linear(hidden, hidden), nn.SiLU(),  
            nn.Linear(hidden, ch * 2),  
        )  
          
    def forward(self, x, alpha):  
        if alpha.dim() == 1: alpha = alpha.view(-1, 1)  
        if alpha.dim() == 4: alpha = alpha.view(alpha.shape[0], 1)  
        p = self.net(alpha)  
        g, b = p.chunk(2, dim=1)  
        g = g.view(-1, self.ch, 1, 1)  
        b = b.view(-1, self.ch, 1, 1)  
        return x * (1 + torch.tanh(g)) + b  

  
class AlphaGatedSkip(nn.Module):  
    def __init__(self, ch):  
        super().__init__()  
        self.gate = nn.Sequential(  
            nn.Linear(1, ch),  
            nn.Sigmoid(),  
        )  
          
    def forward(self, skip, alpha):  
        if alpha.dim() == 1: alpha = alpha.view(-1, 1)  
        g = self.gate(alpha).view(-1, skip.shape[1], 1, 1)  
        return skip * g  

  
class NAFNetAlpha(nn.Module):  
    def __init__(self, width=32, enc_blks=(2, 2, 4, 8), middle_blk=12, dec_blks=(2, 2, 2, 2)):  
        super().__init__()  
        self.intro = nn.Conv2d(3, width, 3, padding=1)  
        self.ending = nn.Conv2d(width, 3, 3, padding=1)  
          
        self.encoders = nn.ModuleList()  
        self.downs = nn.ModuleList()  
        ch = width  
        for n in enc_blks:  
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))  
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, 2))  
            ch *= 2  
          
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk)])  
        self.film_middle = AlphaFiLM(ch)  
          
        self.ups = nn.ModuleList()  
        self.decoders = nn.ModuleList()  
        self.film_dec = nn.ModuleList()  
        self.skip_gates = nn.ModuleList()  
          
        for n in dec_blks:  
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1), nn.PixelShuffle(2)))  
            ch = ch // 2  
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))  
            self.film_dec.append(AlphaFiLM(ch))  
            self.skip_gates.append(AlphaGatedSkip(ch))  
          
    def forward(self, x, alpha):  
        if alpha.dim() == 1: alpha = alpha.view(-1, 1)  
          
        inp = x  
        x = self.intro(x)  
          
        encs = []  
        for encoder, down in zip(self.encoders, self.downs):  
            x = encoder(x)  
            encs.append(x)  
            x = down(x)  
          
        x = self.middle(x)  
        x = self.film_middle(x, alpha)  
          
        for up, decoder, film, gate, enc_skip in zip(
            self.ups, self.decoders, self.film_dec, self.skip_gates, reversed(encs)):  
            
            x = up(x)  
            
            # Align spatial dimensions before adding skip connection
            if x.shape[2:] != enc_skip.shape[2:]:
                x = F.interpolate(x, size=enc_skip.shape[2:], mode='bilinear', align_corners=False)
            
            gated = gate(enc_skip, alpha)  
            x = x + gated  
            x = decoder(x)  
            x = film(x, alpha)  
          
        residual = self.ending(x)  
        alpha_scale = (1.0 - 0.8 * alpha).view(-1, 1, 1, 1)  
        out = inp - residual * alpha_scale  
        return out.clamp(0, 1)  

  
VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]  

def pil_to_tensor(img):  
    img = ImageOps.exif_transpose(img).convert("RGB")  
    arr = np.array(img).astype(np.float32) / 255.0  
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  

def tensor_to_pil(tensor):  
    tensor = tensor.detach().float().cpu().clamp(0, 1)  
    if tensor.dim() == 4: tensor = tensor[0]  
    return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8))  

def pad_to_multiple(x, multiple=8):  
    _, _, h, w = x.shape  
    ph = (multiple - h % multiple) % multiple  
    pw = (multiple - w % multiple) % multiple  
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), h, w  

def resize_for_inference(img, max_side=1600):  
    w, h = img.size  
    biggest = max(w, h)  
    if biggest <= max_side: return img  
    scale = max_side / biggest  
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)  

@torch.no_grad()  
def run_model(model, img_pil, alpha=0.0, device="cuda", max_side=1600):  
    img_pil = resize_for_inference(img_pil, max_side=max_side)  
    x = pil_to_tensor(img_pil).to(device)  
    x_pad, h, w = pad_to_multiple(x, multiple=8)  
    alpha_t = torch.tensor([[float(alpha)]], device=device)  
    y_pad = model(x_pad, alpha_t)  
    return tensor_to_pil(y_pad[:, :, :h, :w])  

def collect_images(folder):  
    paths = []  
    for ext in VALID_EXTS:  
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))  
        paths.extend(glob.glob(os.path.join(folder, "**", f"*{ext.upper()}"), recursive=True))  
    return sorted(set(paths))