"""
Flow-Matching Fusion with Disparity + Cost Volume (Full runnable demo)
=====================================================================

Your setup:
- Monocular depth dm: great geometry/semantics, bad absolute scale
- Stereo (SGM-like) disparity disp_s: metric-correct (via fx*B/disp), but occlusions/artifacts
- You CAN provide stereo disparity + cost volume
- You have calibrated intrinsics K and baseline B
- You do NOT have an explicit confidence map -> we will compute it from cost volume (+ optional LR consistency)

We will:
1) Create pseudo data (RGB, GT depth, monocular depth, stereo disparity, cost volume).
2) Compute stereo confidence c(x) from:
   - cost-volume "peakiness" (softmax entropy, peak/second-peak margin)
   - disparity validity
   - (optional) LR consistency if you provide right disparity (here pseudo can generate it; code supports it)
3) Align monocular inverse depth to stereo inverse depth using robust affine fit on high-confidence pixels.
4) Train a rectified-flow / flow-matching model on residual space:
      z0 = metric inverse depth from stereo
      z1 = aligned monocular inverse depth
      r1 = (1-c) * (z1 - z0)
   Learn velocity field v_theta(r_t, t, cond) ≈ r1
5) Sample via ODE solver (Heun) starting r(0)=0 to get fused depth.

Dependencies: torch, numpy, matplotlib(optional)
Run:
  python flow_fusion_costvol.py --device cuda
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_inverse_depth(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1.0 / torch.clamp(d, min=eps)


def depth_from_disp(
    disp: torch.Tensor, fx: float, baseline: float, eps: float = 1e-6
) -> torch.Tensor:
    return fx * baseline / torch.clamp(disp, min=eps)


def disp_from_depth(
    depth: torch.Tensor, fx: float, baseline: float, eps: float = 1e-6
) -> torch.Tensor:
    return fx * baseline / torch.clamp(depth, min=eps)


def sobel_mag(img: torch.Tensor) -> torch.Tensor:
    """img: (B,3,H,W) in [0,1] -> (B,1,H,W)"""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device
    ).view(1, 1, 3, 3)
    gray = img.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def depth_grad_mag(d: torch.Tensor) -> torch.Tensor:
    """d: (B,1,H,W) -> (B,1,H,W)"""
    dx = d[..., :, 1:] - d[..., :, :-1]
    dy = d[..., 1:, :] - d[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-8)


def mono_to_inverse_depth(dm: torch.Tensor, mono_is_inverse: bool) -> torch.Tensor:
    if mono_is_inverse:
        return torch.clamp(dm, min=1e-6)
    return safe_inverse_depth(torch.clamp(dm, min=1e-6))


def mono_to_disparity(
    dm: torch.Tensor, fx: float, baseline: float, mono_is_inverse: bool
) -> torch.Tensor:
    if mono_is_inverse:
        return torch.clamp(fx * baseline * torch.clamp(dm, min=1e-6), min=0.0)
    return disp_from_depth(torch.clamp(dm, min=1e-6), fx, baseline)


def warp_right_to_left(
    img_r: torch.Tensor, disp_l: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp right image to left view using left disparity."""
    B, _, H, W = img_r.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=img_r.device, dtype=img_r.dtype),
        torch.arange(W, device=img_r.device, dtype=img_r.dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, H, W).expand(B, 1, H, W)
    yy = yy.view(1, 1, H, W).expand(B, 1, H, W)

    xr = xx - disp_l
    xn = 2.0 * xr / max(W - 1, 1) - 1.0
    yn = 2.0 * yy / max(H - 1, 1) - 1.0
    grid = torch.cat([xn, yn], dim=1).permute(0, 2, 3, 1)

    warped = F.grid_sample(
        img_r, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    inside = (xr >= 0.0) & (xr <= float(W - 1))
    return warped, inside.float()


def ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-pixel SSIM map in [0,1], shape (B,1,H,W)."""
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sig_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2)
    ssim = num / (den + 1e-6)
    ssim = torch.clamp((1.0 - ssim) * 0.5, 0.0, 1.0)
    return ssim.mean(dim=1, keepdim=True)


def stereo_reprojection_loss(
    img_l: torch.Tensor,
    img_r: torch.Tensor,
    disp_l: torch.Tensor,
    valid: torch.Tensor,
    conf: torch.Tensor,
    ssim_weight: float = 0.85,
    lowconf_power: float = 1.5,
) -> torch.Tensor:
    """Photometric reprojection loss on low-confidence valid stereo pixels."""
    rec_l, inside = warp_right_to_left(img_r, disp_l)
    l1 = torch.mean(torch.abs(img_l - rec_l), dim=1, keepdim=True)
    ssim = ssim_map(img_l, rec_l)
    photo = (1.0 - ssim_weight) * l1 + ssim_weight * ssim

    low_conf = torch.clamp(1.0 - conf, 0.0, 1.0) ** lowconf_power
    w = valid * inside * low_conf
    denom = w.sum() + 1e-6
    return (photo * w).sum() / denom


def load_left_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1)).astype(np.float32)  # (3,H,W)


def load_disp_png(path: str, disp_scale: float) -> np.ndarray:
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Cannot read disparity: {path}")
    disp = disp.astype(np.float64)
    if disp_scale > 0:
        disp = disp / disp_scale
    disp = np.clip(disp, 0.0, None)
    return disp


def load_disp_any(path: str, disp_scale: float) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".exr"):
        # Some OpenCV builds require this switch at runtime for EXR support.
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Cannot read disparity: {path}")
    if disp.ndim == 3:
        pos_counts = [(disp[:, :, i] > 0).sum() for i in range(disp.shape[2])]
        best_ch = int(np.argmax(pos_counts))
        disp = disp[:, :, best_ch]
    disp = disp.astype(np.float64)
    if disp_scale > 0:
        disp = disp / disp_scale
    disp = np.clip(disp, 0.0, None)
    return disp


def make_costvol_from_disp(
    img_chw: np.ndarray,
    disp: np.ndarray,
    D: int,
    disp_max: float,
) -> np.ndarray:
    """
    Build a pseudo cost volume from disparity + image edges.
    Lower cost is better.
    """
    gray = img_chw.mean(axis=0)
    gx = np.pad(gray[:, 1:] - gray[:, :-1], ((0, 0), (0, 1)))
    gy = np.pad(gray[1:, :] - gray[:-1, :], ((0, 1), (0, 0)))
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (edge.max() + 1e-6)

    invalid = (disp <= 1e-6).astype(np.float32)
    sharp = 2.4 + 1.6 * (1.0 - edge)
    sharp = sharp * (1.0 - 0.7 * invalid)
    sharp = np.clip(sharp, 0.6, 4.0).astype(np.float32)

    disp_candidates = np.linspace(0.0, disp_max, D, dtype=np.float32).reshape(D, 1, 1)
    cost = (disp_candidates - disp.reshape(1, *disp.shape)) ** 2
    cost = cost * sharp.reshape(1, *disp.shape)
    cost += 0.08 * edge.reshape(1, *edge.shape)
    return cost.astype(np.float32)


def robust_affine_fit_weighted(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    iters: int = 6,
    huber_delta: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit y ≈ a*x + b using IRLS with Huber weights.
    x,y,w are (N,)
    """
    w_ = torch.clamp(w, 0.0, 1.0).clone()

    def weighted_ls(xv, yv, ww):
        s1 = torch.sum(ww * xv * xv)
        s2 = torch.sum(ww * xv)
        s3 = torch.sum(ww)
        t1 = torch.sum(ww * xv * yv)
        t2 = torch.sum(ww * yv)
        det = s1 * s3 - s2 * s2
        det = torch.clamp(det, min=eps)
        a = (t1 * s3 - t2 * s2) / det
        b = (s1 * t2 - s2 * t1) / det
        return a, b

    a, b = weighted_ls(x, y, w_)

    for _ in range(iters):
        r = a * x + b - y
        abs_r = torch.abs(r)
        huber_w = torch.where(
            abs_r > huber_delta, huber_delta / (abs_r + eps), torch.ones_like(abs_r)
        )
        ww = w_ * huber_w
        a, b = weighted_ls(x, y, ww)

    return a, b


# ----------------------------
# Real Data
# ----------------------------


@dataclass
class Camera:
    fx: float
    baseline: float


class RealFusionDataset(torch.utils.data.Dataset):
    def __init__(self, sample: Tuple[torch.Tensor, ...], n: int):
        self.sample = sample
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return clones to avoid in-place side effects.
        return tuple(t.clone() for t in self.sample)


# ----------------------------
# Confidence from Cost Volume (+ optional LR)
# ----------------------------


@torch.no_grad()
def confidence_from_cost_volume(
    costvol: torch.Tensor,  # (B,D,H,W) lower is better
    disp_s: torch.Tensor,  # (B,1,H,W)
    valid: torch.Tensor,  # (B,1,H,W)
    disp_r: Optional[torch.Tensor] = None,  # (B,1,H,W) optional
    alpha: float = 3.0,
    lr_thresh: float = 1.0,
) -> torch.Tensor:
    """
    Build c in [0,1] from:
      - entropy of softmin distribution (peaky => confident)
      - margin between best and 2nd best cost (bigger margin => confident)
      - optional LR consistency term
    """
    B, D, H, W = costvol.shape
    device = costvol.device

    # softmin probabilities over disparity candidates
    # p(d) ∝ exp(-alpha * cost)
    logits = -alpha * costvol
    probs = torch.softmax(logits, dim=1)  # (B,D,H,W)

    # entropy normalized
    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)  # (B,1,H,W)
    ent_norm = ent / (math.log(D) + 1e-8)  # in [0,1] roughly
    c_ent = 1.0 - ent_norm

    # margin: best vs second best cost
    # Get top2 smallest costs -> use torch.topk on negative?
    # easiest: sort along D (cost lower better) - but expensive. Use topk with largest on (-cost)
    top2 = torch.topk(-costvol, k=2, dim=1)  # larger (-cost) means smaller cost
    best = -top2.values[:, 0:1]  # (B,1,H,W) best cost
    second = -top2.values[:, 1:2]
    margin = torch.clamp(second - best, min=0.0)  # >0
    # normalize margin by robust scale
    med = margin.flatten(2).median(dim=2).values.view(B, 1, 1, 1)
    mad = (torch.abs(margin - med)).flatten(2).median(dim=2).values.view(
        B, 1, 1, 1
    ) + 1e-6
    margin_n = (margin - med) / (3.0 * mad)
    margin_n = torch.clamp(margin_n, min=0.0)
    c_margin = torch.tanh(margin_n)  # squashed [0,1)

    # base confidence
    c = 0.55 * c_ent + 0.45 * c_margin
    c = torch.clamp(c, 0.0, 1.0)
    c = c * valid

    # optional LR consistency: penalize pixels where left/right disagree
    if disp_r is not None:
        # Here we don't have true warping; we approximate:
        #   consistency = |disp_left - disp_right| small => confident
        lr_err = torch.abs(disp_s - disp_r)
        c_lr = torch.exp(-lr_err / lr_thresh)
        c = c * c_lr

    # light spatial smoothing to reduce structured banding
    c = F.avg_pool2d(c, kernel_size=3, stride=1, padding=1)

    return torch.clamp(c, 0.0, 1.0)


@torch.no_grad()
def align_mono_to_stereo_inverse_depth(
    dm: torch.Tensor,  # (B,1,H,W)
    z0: torch.Tensor,  # (B,1,H,W) stereo inverse depth
    c: torch.Tensor,  # (B,1,H,W) confidence
    mono_is_inverse: bool = False,
    min_pixels: int = 256,
) -> torch.Tensor:
    """
    Fit z0 ≈ a*z_m + b on high-confidence pixels per batch element.
    Return z1 = a*z_m + b (aligned mono inverse depth).
    """
    B = dm.shape[0]
    z_m = mono_to_inverse_depth(dm, mono_is_inverse)
    z1 = torch.zeros_like(z0)

    for b in range(B):
        m = (c[b, 0] > 0.6).reshape(-1)
        if m.sum().item() < min_pixels:
            m = (c[b, 0] > 0.3).reshape(-1)
        if m.sum().item() < min_pixels:
            z1[b : b + 1] = z_m[b : b + 1]
            continue

        xv = z_m[b, 0].reshape(-1)[m]
        yv = z0[b, 0].reshape(-1)[m]
        ww = c[b, 0].reshape(-1)[m]

        # normalize x for stability
        x_mean = xv.mean()
        x_std = xv.std() + 1e-6
        xv_n = (xv - x_mean) / x_std

        a_n, b_n = robust_affine_fit_weighted(xv_n, yv, ww, iters=6, huber_delta=0.5)
        a = a_n / x_std
        bb = b_n - a_n * (x_mean / x_std)

        z1[b : b + 1] = a * z_m[b : b + 1] + bb

    return z1


@torch.no_grad()
def edge_aware_smooth_map(
    x: torch.Tensor,
    img: torch.Tensor,
    iters: int = 2,
    edge_sigma: float = 10.0,
) -> torch.Tensor:
    """Simple 4-neighbor edge-aware smoothing guided by image gradients."""
    if iters <= 0:
        return x

    y = x
    g = img.mean(dim=1, keepdim=True)
    for _ in range(iters):
        wx = torch.exp(-edge_sigma * torch.abs(g[:, :, :, 1:] - g[:, :, :, :-1]))
        wy = torch.exp(-edge_sigma * torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :]))

        num = y.clone()
        den = torch.ones_like(y)

        num[:, :, :, :-1] += wx * y[:, :, :, 1:]
        den[:, :, :, :-1] += wx
        num[:, :, :, 1:] += wx * y[:, :, :, :-1]
        den[:, :, :, 1:] += wx

        num[:, :, :-1, :] += wy * y[:, :, 1:, :]
        den[:, :, :-1, :] += wy
        num[:, :, 1:, :] += wy * y[:, :, :-1, :]
        den[:, :, 1:, :] += wy

        y = num / (den + 1e-6)

    return y


@torch.no_grad()
def inpaint_from_valid_stereo_inverse_depth(
    z0: torch.Tensor,
    valid: torch.Tensor,
    img: torch.Tensor,
    iters: int = 6,
    edge_sigma: float = 10.0,
) -> torch.Tensor:
    """Propagate valid stereo inverse-depth to nearby invalid regions with edge-aware weights."""
    z = z0.clone()
    m = (valid > 0.5).float()
    g = img.mean(dim=1, keepdim=True)

    for _ in range(max(iters, 0)):
        wx = torch.exp(-edge_sigma * torch.abs(g[:, :, :, 1:] - g[:, :, :, :-1]))
        wy = torch.exp(-edge_sigma * torch.abs(g[:, :, 1:, :] - g[:, :, :-1, :]))

        num = z * m
        den = m.clone()

        num[:, :, :, :-1] += wx * (z[:, :, :, 1:] * m[:, :, :, 1:])
        den[:, :, :, :-1] += wx * m[:, :, :, 1:]
        num[:, :, :, 1:] += wx * (z[:, :, :, :-1] * m[:, :, :, :-1])
        den[:, :, :, 1:] += wx * m[:, :, :, :-1]

        num[:, :, :-1, :] += wy * (z[:, :, 1:, :] * m[:, :, 1:, :])
        den[:, :, :-1, :] += wy * m[:, :, 1:, :]
        num[:, :, 1:, :] += wy * (z[:, :, :-1, :] * m[:, :, :-1, :])
        den[:, :, 1:, :] += wy * m[:, :, :-1, :]

        z = torch.where(den > 1e-6, num / (den + 1e-6), z)
        m = torch.where(den > 1e-6, torch.ones_like(m), m)

    return z


@torch.no_grad()
def conservative_fuse_inverse_depth(
    img: torch.Tensor,
    z0: torch.Tensor,
    z1: torch.Tensor,
    c: torch.Tensor,
    valid: torch.Tensor,
    gate_power: float = 2.0,
    valid_correction_max: float = 0.20,
    lowconf_thresh: float = 0.45,
    lowconf_boost: float = 0.25,
    mono_hole_weight: float = 0.35,
    mono_hole_clip_ratio: float = 0.40,
    edge_smooth_iters: int = 2,
    edge_sigma: float = 10.0,
    residual_clip: float = 1.5,
) -> torch.Tensor:
    """
    Conservative fusion:
      - Keep stereo z0 as anchor.
      - Use mono only where confidence is low.
      - Smooth only residual term to suppress cloth-like ripple artifacts.
    """
    gate_valid = torch.clamp(1.0 - c, 0.0, 1.0) ** gate_power
    if lowconf_thresh > 1e-6 and lowconf_boost > 0.0:
        low_conf = torch.clamp((lowconf_thresh - c) / lowconf_thresh, 0.0, 1.0)
        gate_valid = gate_valid + lowconf_boost * low_conf * valid
    gate_valid = torch.clamp(gate_valid, 0.0, valid_correction_max)
    u = (1.0 - valid) + valid * gate_valid

    z0_hole = inpaint_from_valid_stereo_inverse_depth(
        z0=z0,
        valid=valid,
        img=img,
        iters=max(edge_smooth_iters + 2, 3),
        edge_sigma=edge_sigma,
    )
    lo_h = (1.0 - mono_hole_clip_ratio) * z0_hole
    hi_h = (1.0 + mono_hole_clip_ratio) * z0_hole
    z1_hole = torch.clamp(z1, min=lo_h, max=hi_h)
    z_hole = (1.0 - mono_hole_weight) * z0_hole + mono_hole_weight * z1_hole
    z_target = valid * z1 + (1.0 - valid) * z_hole

    delta = z_target - z0
    res = u * delta

    if residual_clip > 0:
        B = res.shape[0]
        scale = (
            torch.quantile(torch.abs(delta).reshape(B, -1), 0.90, dim=1).view(
                B, 1, 1, 1
            )
            + 1e-6
        )
        res = torch.clamp(res, min=-residual_clip * scale, max=residual_clip * scale)

    res = edge_aware_smooth_map(
        res, img, iters=edge_smooth_iters, edge_sigma=edge_sigma
    )
    zhat = z0 + res

    lo = torch.minimum(z0, z1) - 0.10 * torch.abs(delta)
    hi = torch.maximum(z0, z1) + 0.10 * torch.abs(delta)
    zhat = torch.clamp(zhat, min=lo, max=hi)

    return zhat


# ----------------------------
# Tiny UNet (velocity field)
# ----------------------------


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TinyUNet(nn.Module):
    """
    Condition on:
      rt (1), z0 (1), z1 (1), c (1), img (3) -> 7 channels + time channel = 8 total.
    Output: v (1)
    """

    def __init__(self, in_ch: int = 7, base: int = 48):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch + 1, base, 3, padding=1)

        self.down1 = nn.Sequential(ResBlock(base), ResBlock(base))
        self.down2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            ResBlock(base * 2),
            ResBlock(base * 2),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            ResBlock(base * 4),
            ResBlock(base * 4),
        )

        self.mid = nn.Sequential(ResBlock(base * 4), ResBlock(base * 4))

        # Use upsample + conv to avoid transposed-conv stripe artifacts.
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            ResBlock(base * 2),
            ResBlock(base * 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, base, 3, padding=1),
            ResBlock(base),
            ResBlock(base),
        )

        self.conv_out = nn.Conv2d(base, 1, 3, padding=1)

    @staticmethod
    def time_channel(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return t.view(-1, 1, 1, 1).expand(-1, 1, H, W)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        tc = self.time_channel(t, H, W)
        x = torch.cat([x, tc], dim=1)

        h0 = self.conv_in(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        hm = self.mid(h3)
        u3 = self.up3(hm)
        u2 = self.up2(u3)
        return self.conv_out(u2)


# ----------------------------
# Flow Matching Loss + Sampling
# ----------------------------


def fm_loss(
    model: nn.Module,
    img: torch.Tensor,
    img_r: torch.Tensor,
    z0: torch.Tensor,
    z1: torch.Tensor,
    c: torch.Tensor,
    valid: torch.Tensor,
    fx_baseline: float,
    gate_power: float = 2.0,
    valid_correction_max: float = 0.20,
    lowconf_thresh: float = 0.45,
    lowconf_boost: float = 0.25,
    residual_clip: float = 1.5,
    photo_weight: float = 0.15,
    photo_ssim_weight: float = 0.85,
    photo_lowconf_power: float = 1.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Rectified flow in residual space, with conservative target gating to preserve stereo details.
    """
    B = img.shape[0]
    device = img.device

    t = torch.rand(B, device=device)
    gate_valid = torch.clamp(1.0 - c, 0.0, 1.0) ** gate_power
    if lowconf_thresh > 1e-6 and lowconf_boost > 0.0:
        low_conf = torch.clamp((lowconf_thresh - c) / lowconf_thresh, 0.0, 1.0)
        gate_valid = gate_valid + lowconf_boost * low_conf * valid
    gate_valid = torch.clamp(gate_valid, 0.0, valid_correction_max)
    gate = (1.0 - valid) + valid * gate_valid
    delta = z1 - z0
    r1 = gate * delta

    if residual_clip > 0:
        scale = (
            torch.quantile(torch.abs(delta).reshape(B, -1), 0.90, dim=1).view(
                B, 1, 1, 1
            )
            + 1e-6
        )
        r1 = torch.clamp(r1, min=-residual_clip * scale, max=residual_clip * scale)

    rt = t.view(B, 1, 1, 1) * r1

    x = torch.cat([rt, z0, z1, c, img], dim=1)
    v = model(x, t)

    loss_fm = F.smooth_l1_loss(v, r1, beta=0.02)

    loss_anchor = torch.mean(c * torch.abs(v))

    zhat_proxy = z0 + rt
    grad_zh = depth_grad_mag(zhat_proxy)
    grad_z1 = depth_grad_mag(z1)
    loss_grad = torch.mean(gate * torch.abs(grad_zh - grad_z1))

    g_img = sobel_mag(img)
    w_smooth = torch.exp(-3.0 * (g_img / (g_img.amax(dim=(2, 3), keepdim=True) + 1e-6)))
    loss_smooth = torch.mean(gate * w_smooth * grad_zh)

    loss_photo = torch.tensor(0.0, device=img.device, dtype=img.dtype)
    if photo_weight > 0.0:
        disp_proxy = torch.clamp(fx_baseline * zhat_proxy, min=0.0)
        loss_photo = stereo_reprojection_loss(
            img_l=img,
            img_r=img_r,
            disp_l=disp_proxy,
            valid=valid,
            conf=c,
            ssim_weight=photo_ssim_weight,
            lowconf_power=photo_lowconf_power,
        )

    loss = (
        loss_fm
        + 0.35 * loss_anchor
        + 0.20 * loss_grad
        + 0.12 * loss_smooth
        + photo_weight * loss_photo
    )

    stats = {
        "loss": float(loss.item()),
        "loss_fm": float(loss_fm.item()),
        "loss_anchor": float(loss_anchor.item()),
        "loss_grad": float(loss_grad.item()),
        "loss_smooth": float(loss_smooth.item()),
        "loss_photo": float(loss_photo.item()),
        "c_mean": float(c.mean().item()),
        "gate_mean": float(gate.mean().item()),
    }
    return loss, stats


@torch.no_grad()
def sample_flow(
    model: nn.Module,
    img: torch.Tensor,
    z0: torch.Tensor,
    z1: torch.Tensor,
    c: torch.Tensor,
    valid: torch.Tensor,
    steps: int = 20,
    method: str = "heun",
    gate_power: float = 2.0,
    valid_correction_max: float = 0.20,
    lowconf_thresh: float = 0.45,
    lowconf_boost: float = 0.25,
    residual_clip: float = 1.5,
) -> torch.Tensor:
    """
    ODE: dr/dt = v_theta(r,t,cond), r(0)=0, t in [0,1].
    Conservative gating/clipping is applied before output.
    """
    B, _, _, _ = z0.shape
    device = img.device
    r = torch.zeros_like(z0)

    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((B,), i * dt, device=device)
        x0 = torch.cat([r, z0, z1, c, img], dim=1)
        v0 = model(x0, t0)

        if method == "euler":
            r = r + dt * v0
        else:
            r_pred = r + dt * v0
            t1 = torch.full((B,), (i + 1) * dt, device=device)
            x1 = torch.cat([r_pred, z0, z1, c, img], dim=1)
            v1 = model(x1, t1)
            r = r + 0.5 * dt * (v0 + v1)

    gate_valid = torch.clamp(1.0 - c, 0.0, 1.0) ** gate_power
    if lowconf_thresh > 1e-6 and lowconf_boost > 0.0:
        low_conf = torch.clamp((lowconf_thresh - c) / lowconf_thresh, 0.0, 1.0)
        gate_valid = gate_valid + lowconf_boost * low_conf * valid
    gate_valid = torch.clamp(gate_valid, 0.0, valid_correction_max)
    gate = (1.0 - valid) + valid * gate_valid
    r = gate * r

    if residual_clip > 0:
        delta = z1 - z0
        scale = (
            torch.quantile(torch.abs(delta).reshape(B, -1), 0.90, dim=1).view(
                B, 1, 1, 1
            )
            + 1e-6
        )
        r = torch.clamp(r, min=-residual_clip * scale, max=residual_clip * scale)

    return z0 + r


# ----------------------------
# Main
# ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--disp_max", type=float, default=None)
    parser.add_argument("--train_n", type=int, default=1000)
    parser.add_argument("--test_n", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="outputs_flow_fusion_costvol")
    parser.add_argument(
        "--use_lr",
        action="store_true",
        help="use LR-consistency term (requires disp_r)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Real data dir. If set, use real-data mode.",
    )
    parser.add_argument("--left_name", type=str, default="l_00004.png")
    parser.add_argument("--right_name", type=str, default="r_00004.png")
    parser.add_argument("--mono_name", type=str, default="l_00004_vitl.npy")
    parser.add_argument("--disp_name", type=str, default="nvpd_00004.png")
    parser.add_argument(
        "--gt_disp_name",
        type=str,
        default="d_00004.exr",
        help="Optional GT disparity file name.",
    )
    parser.add_argument(
        "--disp_scale", type=float, default=64.0, help="PNG disparity scale divisor."
    )
    parser.add_argument(
        "--gt_disp_scale", type=float, default=1.0, help="GT disparity scale divisor."
    )
    parser.add_argument("--fx", type=float, default=420.0)
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument(
        "--mono_is_inverse",
        action="store_true",
        help="Treat mono input as inverse depth (1/m) instead of metric depth",
    )
    parser.add_argument(
        "--real_w",
        type=int,
        default=480,
        help="Resize width in real-data mode. <=0 disables resize.",
    )
    parser.add_argument(
        "--real_h",
        type=int,
        default=272,
        help="Resize height in real-data mode. <=0 disables resize.",
    )
    parser.add_argument(
        "--invalid_depth_fill",
        type=float,
        default=80.0,
        help="Depth fill value (meters) for invalid stereo pixels before inversion",
    )
    parser.add_argument(
        "--fuse_mode",
        type=str,
        default="hybrid",
        choices=["base", "model", "hybrid"],
        help="base: deterministic conservative blend; model: learned residual only; hybrid: mix model and base",
    )
    parser.add_argument(
        "--model_blend",
        type=float,
        default=0.35,
        help="In hybrid mode, output = model_blend*model + (1-model_blend)*base",
    )
    parser.add_argument(
        "--gate_power",
        type=float,
        default=2.2,
        help="Larger value makes mono correction more restricted to low-confidence regions",
    )
    parser.add_argument(
        "--residual_clip",
        type=float,
        default=1.5,
        help="Residual clipping scale in inverse-depth space",
    )
    parser.add_argument(
        "--valid_correction_max",
        type=float,
        default=0.00,
        help="Max correction strength on valid stereo pixels (0 keeps stereo unchanged)",
    )
    parser.add_argument(
        "--mono_hole_weight",
        type=float,
        default=0.35,
        help="In invalid stereo regions, blend ratio toward mono (0: stereo inpaint only, 1: mono only)",
    )
    parser.add_argument(
        "--mono_hole_clip_ratio",
        type=float,
        default=0.40,
        help="Clip mono inverse-depth in holes around stereo-inpaint by +/- ratio",
    )
    parser.add_argument(
        "--edge_smooth_iters",
        type=int,
        default=2,
        help="Edge-aware smoothing iterations on residual/base fusion",
    )
    parser.add_argument(
        "--edge_sigma", type=float, default=10.0, help="Edge-aware smoothing strength"
    )
    parser.add_argument(
        "--lowconf_thresh",
        type=float,
        default=0.45,
        help="Confidence threshold below which mono guidance is boosted on valid stereo pixels",
    )
    parser.add_argument(
        "--lowconf_boost",
        type=float,
        default=0.25,
        help="Additional gate boost for low-confidence valid stereo pixels",
    )
    parser.add_argument(
        "--photo_weight",
        type=float,
        default=0.15,
        help="Weight for stereo reprojection loss during training",
    )
    parser.add_argument(
        "--photo_ssim_weight",
        type=float,
        default=0.85,
        help="SSIM ratio in reprojection loss",
    )
    parser.add_argument(
        "--photo_lowconf_power",
        type=float,
        default=1.5,
        help="Low-confidence emphasis power for reprojection loss",
    )
    parser.add_argument(
        "--real_train",
        action="store_true",
        help="Enable model training in real-data mode (default off to avoid overfitting artifacts)",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.enabled = False

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    if not args.real_train:
        if args.fuse_mode != "base":
            print(
                "[Info] real mode without --real_train: force fuse_mode=base to avoid single-scene overfit artifacts."
            )
            args.fuse_mode = "base"
        args.epochs = 1

    left_path = os.path.join(args.data_dir, args.left_name)
    right_path = os.path.join(args.data_dir, args.right_name)
    mono_path = os.path.join(args.data_dir, args.mono_name)
    disp_path = os.path.join(args.data_dir, args.disp_name)
    gt_disp_path = os.path.join(args.data_dir, args.gt_disp_name)

    img = load_left_rgb(left_path)
    img_r = load_left_rgb(right_path)

    dm = np.load(mono_path).astype(np.float32)
    print(f"mono mode: {'inverse-depth' if args.mono_is_inverse else 'depth'}")
    # Decode disparity as uint16->float64/disp_scale, then cast to float32 for Torch.
    disp_s = load_disp_any(disp_path, args.disp_scale).astype(np.float32)

    print(
        f"disp_s stats: dtype={disp_s.dtype}, min={float(np.min(disp_s)):.4f}, max={float(np.max(disp_s)):.4f}, scale={args.disp_scale}"
    )

    if dm.shape != disp_s.shape:
        dm = cv2.resize(
            dm, (disp_s.shape[1], disp_s.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    if args.real_w > 0 and args.real_h > 0:
        img_hwc = np.transpose(img, (1, 2, 0))
        img_hwc = cv2.resize(
            img_hwc, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR
        )
        img = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32)

        img_r_hwc = np.transpose(img_r, (1, 2, 0))
        img_r_hwc = cv2.resize(
            img_r_hwc, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR
        )
        img_r = np.transpose(img_r_hwc, (2, 0, 1)).astype(np.float32)

        dm = cv2.resize(dm, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR)
        disp_s = cv2.resize(
            disp_s, (args.real_w, args.real_h), interpolation=cv2.INTER_LINEAR
        )

    h, w = disp_s.shape
    disp_max = (
        args.disp_max
        if args.disp_max is not None
        else max(32.0, float(np.nanpercentile(disp_s, 99.5) * 1.5))
    )
    valid = (disp_s > 1e-6).astype(np.float32)
    costvol = make_costvol_from_disp(img, disp_s, args.D, disp_max)
    disp_r = disp_s.copy()
    gt = np.zeros_like(disp_s, dtype=np.float32)
    if os.path.isfile(gt_disp_path):
        gt_disp = load_disp_any(gt_disp_path, args.gt_disp_scale).astype(np.float32)
        if gt_disp.shape != disp_s.shape:
            gt_disp = cv2.resize(
                gt_disp,
                (disp_s.shape[1], disp_s.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        gt_valid = gt_disp > 1e-6
        gt[gt_valid] = (args.fx * args.baseline / np.clip(gt_disp[gt_valid], 1e-6, None)).astype(np.float32)

    sample = (
        torch.from_numpy(img),  # (3,H,W)
        torch.from_numpy(img_r),  # (3,H,W)
        torch.from_numpy(dm).unsqueeze(0),  # (1,H,W)
        torch.from_numpy(disp_s).unsqueeze(0),  # (1,H,W)
        torch.from_numpy(costvol),  # (D,H,W)
        torch.from_numpy(disp_r).unsqueeze(0),  # (1,H,W)
        torch.from_numpy(gt).unsqueeze(0),  # (1,H,W)
        torch.from_numpy(valid).unsqueeze(0),  # (1,H,W)
    )

    train_ds = RealFusionDataset(sample=sample, n=max(args.train_n, args.batch))
    test_ds = RealFusionDataset(sample=sample, n=max(args.test_n, 1))
    args.disp_max = disp_max
    # calibrated camera
    cam = Camera(
        fx=args.fx,
        baseline=args.baseline,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=min(args.batch, len(test_ds)), shuffle=False
    )

    model = TinyUNet(in_ch=7, base=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | D={args.D} | mode=real")

    for ep in range(args.epochs):
        do_train = args.real_train
        model.train(do_train)
        running = {}
        if do_train:
            for it, batch in enumerate(train_loader):
                img, img_r, dm, disp_s, costvol, disp_r, gt, valid = [
                    b.to(device) for b in batch
                ]
                costvol = costvol  # (B,D,H,W)

                # stereo metric depth and inverse depth
                ds = depth_from_disp(disp_s, cam.fx, cam.baseline)  # (B,1,H,W)
                # fill invalid with something safe for inversion (won't matter if confidence 0)
                ds_filled = torch.where(
                    valid > 0.5, ds, torch.full_like(ds, args.invalid_depth_fill)
                )
                z0 = safe_inverse_depth(ds_filled)

                # confidence from cost volume
                c = confidence_from_cost_volume(
                    costvol=costvol,
                    disp_s=disp_s,
                    valid=valid,
                    disp_r=disp_r,
                    alpha=3.0,
                    lr_thresh=1.0,
                )

                # align mono to stereo in inverse-depth
                z1 = align_mono_to_stereo_inverse_depth(
                    dm=dm, z0=z0, c=c, mono_is_inverse=args.mono_is_inverse
                )

                loss, stats = fm_loss(
                    model,
                    img,
                    img_r,
                    z0,
                    z1,
                    c,
                    valid,
                    fx_baseline=cam.fx * cam.baseline,
                    gate_power=args.gate_power,
                    valid_correction_max=args.valid_correction_max,
                    lowconf_thresh=args.lowconf_thresh,
                    lowconf_boost=args.lowconf_boost,
                    residual_clip=args.residual_clip,
                    photo_weight=args.photo_weight,
                    photo_ssim_weight=args.photo_ssim_weight,
                    photo_lowconf_power=args.photo_lowconf_power,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                for k, v in stats.items():
                    running[k] = running.get(k, 0.0) + v

                if (it + 1) % 50 == 0:
                    denom = 50
                    msg = " | ".join(
                        [
                            f"{k}:{running[k]/denom:.4f}"
                            for k in [
                                "loss",
                                "loss_fm",
                                "loss_anchor",
                                "loss_grad",
                                "loss_smooth",
                                "loss_photo",
                                "c_mean",
                                "gate_mean",
                            ]
                        ]
                    )
                    print(
                        f"[ep {ep+1}/{args.epochs}] iter {it+1}/{len(train_loader)} :: {msg}"
                    )
                    running = {}
        # Eval
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            img, img_r, dm, disp_s, costvol, disp_r, gt, valid = [
                b.to(device) for b in batch
            ]
            ds = depth_from_disp(disp_s, cam.fx, cam.baseline)
            ds_filled = torch.where(
                valid > 0.5, ds, torch.full_like(ds, args.invalid_depth_fill)
            )
            z0 = safe_inverse_depth(ds_filled)

            c = confidence_from_cost_volume(
                costvol, disp_s, valid, disp_r if args.use_lr else None
            )
            z1 = align_mono_to_stereo_inverse_depth(
                dm, z0, c, mono_is_inverse=args.mono_is_inverse
            )

            z_base = conservative_fuse_inverse_depth(
                img=img,
                z0=z0,
                z1=z1,
                c=c,
                valid=valid,
                gate_power=args.gate_power,
                valid_correction_max=args.valid_correction_max,
                lowconf_thresh=args.lowconf_thresh,
                lowconf_boost=args.lowconf_boost,
                mono_hole_weight=args.mono_hole_weight,
                mono_hole_clip_ratio=args.mono_hole_clip_ratio,
                edge_smooth_iters=args.edge_smooth_iters,
                edge_sigma=args.edge_sigma,
                residual_clip=args.residual_clip,
            )
            z_model = sample_flow(
                model,
                img,
                z0,
                z1,
                c,
                valid,
                steps=args.steps,
                method="heun",
                gate_power=args.gate_power,
                valid_correction_max=args.valid_correction_max,
                lowconf_thresh=args.lowconf_thresh,
                lowconf_boost=args.lowconf_boost,
                residual_clip=args.residual_clip,
            )
            if args.fuse_mode == "base":
                zhat = z_base
            elif args.fuse_mode == "model":
                zhat = z_model
            else:
                zhat = args.model_blend * z_model + (1.0 - args.model_blend) * z_base

            dhat = 1.0 / torch.clamp(zhat, min=1e-6)
            d1 = 1.0 / torch.clamp(z1, min=1e-6)
            disp_gt = disp_from_depth(torch.clamp(gt, min=1e-6), cam.fx, cam.baseline)
            disp_f = disp_from_depth(torch.clamp(dhat, min=1e-6), cam.fx, cam.baseline)
            disp_m = disp_from_depth(torch.clamp(d1, min=1e-6), cam.fx, cam.baseline)
            valid_gt = (disp_gt > 0.0).float()
            if valid_gt.mean().item() > 1e-6:
                abs_rel_f = torch.mean(
                    valid_gt
                    * torch.abs(disp_f - disp_gt)
                    / torch.clamp(disp_gt, min=1e-6)
                ) / (valid_gt.mean() + 1e-6)
                abs_rel_s = torch.mean(
                    valid_gt
                    * torch.abs(disp_s - disp_gt)
                    / torch.clamp(disp_gt, min=1e-6)
                ) / (valid_gt.mean() + 1e-6)
                abs_rel_m = torch.mean(
                    valid_gt
                    * torch.abs(disp_m - disp_gt)
                    / torch.clamp(disp_gt, min=1e-6)
                ) / (valid_gt.mean() + 1e-6)

                mask_stereo = (valid_gt > 0.5) & (valid > 0.5)
                mask_hole = (valid_gt > 0.5) & (valid <= 0.5)
                if mask_stereo.float().mean().item() > 1e-6:
                    abs_rel_f_stereo = (
                        torch.abs(disp_f - disp_gt) / torch.clamp(disp_gt, min=1e-6)
                    )[mask_stereo].mean()
                    abs_rel_s_stereo = (
                        torch.abs(disp_s - disp_gt) / torch.clamp(disp_gt, min=1e-6)
                    )[mask_stereo].mean()
                else:
                    abs_rel_f_stereo = torch.tensor(float("nan"), device=disp_f.device)
                    abs_rel_s_stereo = torch.tensor(float("nan"), device=disp_s.device)
                if mask_hole.float().mean().item() > 1e-6:
                    abs_rel_f_hole = (
                        torch.abs(disp_f - disp_gt) / torch.clamp(disp_gt, min=1e-6)
                    )[mask_hole].mean()
                    abs_rel_s_hole = (
                        torch.abs(disp_s - disp_gt) / torch.clamp(disp_gt, min=1e-6)
                    )[mask_hole].mean()
                else:
                    abs_rel_f_hole = torch.tensor(float("nan"), device=disp_f.device)
                    abs_rel_s_hole = torch.tensor(float("nan"), device=disp_s.device)

                print(
                    f"[ep {ep+1}] real disp abs-rel: fused={abs_rel_f.item():.4f} | stereo={abs_rel_s.item():.4f} | mono_aligned={abs_rel_m.item():.4f}"
                )
                print(
                    f"[ep {ep+1}] split metrics: stereo_valid_ratio={mask_stereo.float().mean().item():.4f} | hole_ratio={mask_hole.float().mean().item():.4f} | fused@stereo={abs_rel_f_stereo.item():.4f} | stereo@stereo={abs_rel_s_stereo.item():.4f} | fused@hole={abs_rel_f_hole.item():.4f} | stereo@hole={abs_rel_s_hole.item():.4f} | conf_mean={c.mean().item():.4f}"
                )
            else:
                print(
                    f"[ep {ep+1}] real-data eval: fused disparity generated (GT missing)."
                )

            # Save a quick viz
            try:
                import matplotlib.pyplot as plt

                b0 = 0
                I = img[b0].detach().cpu().permute(1, 2, 0).numpy()
                GT = disp_gt[b0, 0].detach().cpu().numpy()
                DS = disp_s[b0, 0].detach().cpu().numpy()
                DM = (
                    mono_to_disparity(
                        dm[b0 : b0 + 1], cam.fx, cam.baseline, args.mono_is_inverse
                    )[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                D1 = disp_m[b0, 0].detach().cpu().numpy()
                DH = disp_f[b0, 0].detach().cpu().numpy()
                C = c[b0, 0].detach().cpu().numpy()

                def norm_vis(x):
                    x = x.copy()
                    x[x <= 0] = np.nan
                    vmin = np.nanpercentile(x, 2)
                    vmax = np.nanpercentile(x, 98)
                    x = np.clip((x - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    x = np.nan_to_num(x, nan=0.0)
                    return x

                fig = plt.figure(figsize=(14, 7))
                ax = plt.subplot(2, 4, 1)
                ax.set_title("RGB")
                ax.imshow(I)
                ax.axis("off")
                has_gt = np.any(GT > 0.0)
                if has_gt:
                    ax = plt.subplot(2, 4, 2)
                    ax.set_title("GT disparity")
                    ax.imshow(norm_vis(GT))
                    ax.axis("off")
                else:
                    ax = plt.subplot(2, 4, 2)
                    ax.set_title("GT disparity (N/A)")
                    ax.imshow(np.zeros_like(DH))
                    ax.axis("off")
                ax = plt.subplot(2, 4, 3)
                ax.set_title("Stereo disparity")
                ax.imshow(norm_vis(DS))
                ax.axis("off")
                ax = plt.subplot(2, 4, 4)
                ax.set_title("Mono disparity")
                ax.imshow(norm_vis(DM))
                ax.axis("off")
                ax = plt.subplot(2, 4, 5)
                ax.set_title("Aligned mono disp")
                ax.imshow(norm_vis(D1))
                ax.axis("off")
                ax = plt.subplot(2, 4, 6)
                ax.set_title("CostVol conf c")
                ax.imshow(np.clip(C, 0, 1))
                ax.axis("off")
                ax = plt.subplot(2, 4, 7)
                ax.set_title("Fused disparity")
                ax.imshow(norm_vis(DH))
                ax.axis("off")
                if has_gt:
                    ax = plt.subplot(2, 4, 8)
                    ax.set_title("|Fused disp - GT disp|")
                    ax.imshow(np.abs(DH - GT), cmap="magma")
                    ax.axis("off")
                else:
                    ax = plt.subplot(2, 4, 8)
                    ax.set_title("Valid stereo mask")
                    ax.imshow(valid[b0, 0].detach().cpu().numpy(), cmap="gray")
                    ax.axis("off")
                plt.tight_layout()
                out_path = os.path.join(args.save_dir, f"viz_ep{ep+1:03d}.png")
                plt.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"Saved: {out_path}")

                fused_depth_path = os.path.join(args.save_dir, "fused_depth.npy")
                fused_disp_path = os.path.join(args.save_dir, "fused_disparity.npy")
                fused_conf_path = os.path.join(args.save_dir, "confidence.npy")
                np.save(
                    fused_depth_path,
                    dhat[b0, 0].detach().cpu().numpy().astype(np.float32),
                )
                np.save(fused_disp_path, DH.astype(np.float32))
                np.save(fused_conf_path, C.astype(np.float32))
                print(f"Saved: {fused_depth_path}")
                print(f"Saved: {fused_disp_path}")
                print(f"Saved: {fused_conf_path}")
            except Exception as e:
                print(f"Visualization skipped: {e}")

    ckpt_path = os.path.join(args.save_dir, "tiny_unet_flow_fusion_costvol.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()