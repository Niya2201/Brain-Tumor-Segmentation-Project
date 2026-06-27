"""
backend.py — Model, preprocessing, inference, and visualization
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Model (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class SwinUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, base=32,
                 backbone_name="swin_tiny_patch4_window7_224", img_size=256):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name, pretrained=False, in_chans=in_ch,
            features_only=True, out_indices=(0, 1, 2, 3), img_size=img_size,
        )
        c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self.bottleneck = ConvBlock(c4, c4)
        self.up3  = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.up2  = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1  = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_ch, 1)

    def _to_nchw(self, t):
        if t.ndim == 4 and t.shape[1] < t.shape[-1]:
            t = t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x):
        f1, f2, f3, f4 = [self._to_nchw(f) for f in self.encoder(x)]
        b  = self.bottleneck(f4)
        d3 = self.dec3(torch.cat([self.up3(b),  f3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), f2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), f1], 1))
        return F.interpolate(self.out_conv(d1), size=x.shape[2:],
                             mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 256

val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])


def load_model(ckpt_path: str, device: str = "cpu"):
    model = SwinUNet(in_ch=4)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_h5(path: str):
    """Load an H5 slice -> (image_hwc float32, mask_hw uint8)."""
    with h5py.File(path, "r") as f:
        img  = f["image"][()].astype(np.float32)
        mask = f["mask"][()].astype(np.uint8)
    # fix channel order if needed
    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4) and img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))
    # collapse multi-channel mask to binary
    if mask.ndim == 3:
        mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)
    return img, mask


def preprocess(img_hwc: np.ndarray) -> torch.Tensor:
    """Apply val transforms -> (1, C, H, W) tensor."""
    out = val_tfms(image=img_hwc, mask=np.zeros(img_hwc.shape[:2], dtype=np.uint8))
    return out["image"].unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: str, threshold: float = 0.4):
    logits = model(tensor.to(device))[0, 0].cpu()
    prob = torch.sigmoid(logits.float()).numpy()
    pred = (prob > threshold).astype(np.uint8)
    return prob, pred


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    # Resize gt to match pred if shapes differ (e.g. 240×240 vs 256×256)
    if pred.shape != gt.shape:
        from skimage.transform import resize
        gt = (resize(gt.astype(np.float32), pred.shape, order=0,
                     preserve_range=True) > 0.5).astype(np.uint8)
    tp = (pred * gt).sum()
    return {
        "dice":        float((2 * tp + eps) / (pred.sum() + gt.sum() + eps)),
        "sensitivity": float((tp + eps) / (gt.sum() + eps)),
        "precision":   float((tp + eps) / (pred.sum() + eps)),
        "iou":         float((tp + eps) / (pred.sum() + gt.sum() - tp + eps)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def make_vis_image(tensor_chw: torch.Tensor) -> np.ndarray:
    """Max-project all MRI channels -> grayscale (H, W) in [0, 1]."""
    channels = []
    for c in range(tensor_chw.shape[0]):
        ch = tensor_chw[c].cpu().numpy().astype(np.float32)
        lo, hi = np.percentile(ch, [1, 99])
        span = max(hi - lo, 1e-8)
        channels.append(np.clip((ch - lo) / span, 0, 1))
    return np.max(np.stack(channels), axis=0)


def make_figure(img_vis, pred_mask):
    """3-panel figure: Original MRI, Predicted Mask, Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0a0e1a")

    def _setup(ax, title):
        ax.axis("off")
        ax.set_title(title, color="#94a3b8", fontsize=10, pad=6)

    # 1 — Original MRI
    axes[0].imshow(img_vis, cmap="gray", vmin=0, vmax=1)
    _setup(axes[0], "Original MRI")

    # 2 — Predicted Mask
    axes[1].imshow(pred_mask, cmap="gray")
    _setup(axes[1], "Predicted Mask")

    # 3 — Overlay (prediction on MRI)
    axes[2].imshow(img_vis, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    overlay[pred_mask == 1] = [1.0, 0.2, 0.2, 0.6]
    axes[2].imshow(overlay)
    axes[2].legend(
        handles=[mpatches.Patch(color="#ef4444", alpha=0.8, label="Prediction")],
        loc="lower right", fontsize=7, framealpha=0.5, labelcolor="white",
    )
    _setup(axes[2], "Overlay")

    plt.tight_layout(pad=0.5)
    return fig
