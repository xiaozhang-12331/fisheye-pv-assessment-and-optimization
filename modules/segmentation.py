# -*- coding: utf-8 -*-
"""
Four-Class Semantic Segmentation of Fisheye Images - Extreme-Precision Version
(Minimal Circle + GPU Maximization + Image Rotation + Tree-Gap Sky Preservation)
==============================================================================
Goal: Under strong GPU conditions, exploit the full detail potential of the
      model, preserve small sky gaps between leaves, and support automatic
      image rotation according to the shooting azimuth so that the result
      conforms to north-up / south-down orientation.
Usage: Modify the paths and parameters below, then run: python <this_script>.py

New-Architecture Notes:
- This module corresponds to Step 1 of the pipeline: it takes fisheye images
  as input and outputs four-class masks, overlays, metadata, and an optional
  soft4.npz.
- Routine parameters should preferably be adjusted in ../config.py;
  pipeline.py injects them into this module before execution.
- To debug the segmentation algorithm independently, this file can also be
  run directly, in which case the default parameters defined here are used.
"""

import os, glob, math, json, random, argparse, warnings
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    SegformerForSemanticSegmentation,
)

# Attempt to import pydensecrf; automatically disable CRF if unavailable.
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_DENSECRF = True
except ImportError:
    HAS_DENSECRF = False
    dcrf = None
    unary_from_softmax = None
    warnings.warn("pydensecrf is not installed; CRF post-processing will be skipped. Installation is recommended for optimal boundary refinement.")


# ============================================================================
# ============ All Tunable Parameters (centralized; extreme precision + rotation) ====
# ============================================================================

# --- Path configuration (make sure to change to your own paths) ---
IN_DIR = r"D:\projects\image_recog\in"
OUT_DIR = r"D:\projects\image_recog\results\2423\segmentation"
ADE_MODEL_DIR = r"D:\projects\image_recog\models\nvidia_segformer_b4_ade"
BIN_BASE_MODEL_DIR = r"D:\projects\image_recog\models\nvidia_segformer_b4_ade"
BIN_CKPT_PATH = r"D:\projects\image_recog\models\ckpts\best_segformer_b4_binary.pt"

# --- Hardware / inference ---
FP16 = True                       # GPU half precision: negligible accuracy loss, but substantially faster and saves VRAM
SEED = 42
DETERMINISTIC = False             # Disable determinism to let cuDNN select the fastest algorithms

# ============================================================================
# * Image rotation correction (to make the fisheye image north-up / south-down) *
# If enabled, the cropped square image is rotated about its center so that the
# top of the image points to true north.
# The user must provide the azimuth that the top of the image actually points
# to (degrees; 0 = north, increasing clockwise).
# Example: if the top of the image actually points east (90 deg), the script
# rotates it counterclockwise by 90 deg to correct it.
# Azimuth convention: 0 deg = north, increasing clockwise; 90 deg = east,
# 180 deg = south, 270 deg = west.
# The script rotates the image counterclockwise by that angle about the square
# center so that the top of the image points to true north.
# ============================================================================
ENABLE_ROTATION = True            # Whether to enable rotation correction (set False first to run once and confirm the offset)
IMAGE_TOP_AZIMUTH = 0.0
# ============================================================================
# * Sliding-window TTA parameters - directly affect the receptive field and edge smoothness *
# Larger TILE: the model sees more context at once, which helps recognize
# small objects (e.g., sky in gaps between branches and leaves)
# Larger OVERLAP: adjacent tiles overlap more, feathered fusion is smoother,
# and stitching artifacts are eliminated
# More SCALES: the same region is observed at multiple scales; small scales
# capture the global structure, large scales capture details
# USE_TTA_HFLIP: horizontal-flip augmentation; steadily improves accuracy and
# is harmless for sky/vegetation
# ============================================================================
TILE = 896                       # Increase the sliding-window size (896 -> 1280) to provide richer context
OVERLAP = 192                     # Substantially increase the overlap (192 -> 512), almost completely eliminating stitching artifacts
SCALES = [0.5, 1.0, 1.5]   # 5 scales (previously 3) for thorough multi-scale fusion
USE_TTA_HFLIP = True

# --- Binary model inference size ---
# Larger values let the binary model see more detail (more accurate
# discrimination between foliage and buildings), but consume more VRAM
BIN_INFER_SIZE = 768             # Increased from 768 to 1280 to match the large sliding window

# --- Minimal circle definition (based on the short side) ---
INNER_MARGIN = 0.01               # Inward margin ratio; 0 means flush with the short side
FORCE_CENTER = True
CENTER_BIAS_X = 0.0
CENTER_BIAS_Y = 0.0

# --- Spatial prior (building enhancement in the lower half) ---
USE_SPATIAL_PRIOR = True
PRIOR_STRENGTH = 1.5
PRIOR_SHARPNESS = 6.0
PRIOR_BUILDING_BIAS = 0.0

# ============================================================================
# * CRF post-processing - key change: weaken Gaussian smoothing, preserve tiny fragments *
# To preserve small sky patches between leaves, the spatial smoothing strength
# must be reduced.
# - CRF_SXY_GAUSS and CRF_COMPAT_GAUSS control the strength of the Gaussian
#   term; smaller values impose less label consistency on neighboring pixels,
#   so small fragments are preserved.
# - CRF_ITERS is reduced to avoid over-smoothing.
# - NARROW_BAND_WIDTH remains moderate, refining edges without enlarging the
#   smoothing region.
# The color bilateral filtering remains relatively strong, exploiting color
# information to refine the leaf/sky boundaries.
# ============================================================================
USE_NARROWBAND_CRF = True         # Narrow-band CRF; a balance between speed and accuracy
FULL_IMAGE_CRF = False            # Full-image CRF is extremely slow; not used
CRF_ITERS = 5                     # Reduced from 10 to 5 to decrease smoothing iterations
NARROW_BAND_WIDTH = 11            # Reduced from 13 to 11 to focus processing on edges
CRF_SXY_GAUSS = 1                 # Reduced from 3 to 1, substantially weakening spatial smoothing (key to preserving small fragments!)
CRF_COMPAT_GAUSS = 1              # Reduced from 3 to 1, correspondingly lowering the label compatibility penalty
CRF_SXY_BILAT = 40                # Bilateral spatial standard deviation; keep the original value
CRF_SRGB_BILAT = 5                # Bilateral color standard deviation; kept fairly strict to distinguish leaf/sky by color
CRF_COMPAT_BILAT = 7              # Bilateral compatibility; keep the original value

# ============================================================================
# * Binary building fusion *
# Setting BIN_HARD_THRESH to 1.0 completely disables high-confidence hard
# overwrite, relying only on soft fusion and avoiding mistakenly removing
# small sky reflections on building walls or gaps between branches and leaves.
# ============================================================================
BIN_FUSE_MODE = "logit_add"       # Soft fusion; the most stable
BIN_LOGIT_GAIN = 2.0
BIN_HARD_THRESH = 1.0             # Raised to 1.0 to completely disable hard overwrite (since the probability never reaches 1.0)
BIN_HARD_DILATE = 0               # Dilation disabled as well

# ============================================================================
# * Morphological post-processing - key to preserving tiny sky patches *
# The original parameters removed sky fragments smaller than 64 pixels, wiping
# out small gaps between leaves.
# Now the kernel and min_area of all classes are reduced to very low values or
# 0, performing only minimal smoothing (or none) and fully preserving the
# original classification of the soft probability map.
# ============================================================================
MORPH_PARAMS = {
    0: dict(kernel=0, min_area=0),    # Sky: no morphological operation, preserving all details
    1: dict(kernel=1, min_area=4),    # Vegetation: remove only isolated tiny noise (area < 4 pixels)
    2: dict(kernel=2, min_area=32),   # Building: mild denoising
    3: dict(kernel=2, min_area=32),   # Other: mild denoising
}

# --- Output options ---
SAVE_SOFT_NPZ = True              # Save the soft probability map for irradiance computation (strongly recommended)
KEEP_OTHER_CLASS = False          # When False, "other" is fully merged into "building", yielding three output classes
SAVE_CIRCLE_DEBUG = True          # Draw a green circle to verify circle detection correctness
CIRCLE_THICKNESS = 8
OVERLAY_ALPHA = 0.40

# --- Colors (RGB format) ---
COLOR_SKY  = ( 65, 105, 225)      # Sky: blue
COLOR_VEG  = ( 34, 139,  34)      # Vegetation: green
COLOR_BLD  = (220,  20,  60)      # Building: red
COLOR_OTH  = (255, 215,   0)      # Other: yellow


# ============================================================================
# ============================== Utility Functions ===========================
# ============================================================================
def _imwrite(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def collect_input_images(in_path: str) -> List[str]:
    if in_path is None: return []
    p = os.path.abspath(os.path.expanduser(in_path))
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if os.path.isfile(p):
        if os.path.splitext(p)[1].lower() not in exts:
            raise ValueError(f"Not an image file: {p}")
        return [p]
    if os.path.isdir(p):
        imgs = []
        for pat in ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff",
                    "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF"]:
            imgs.extend(glob.glob(os.path.join(p, pat)))
        return sorted(list(dict.fromkeys(imgs)))
    raise FileNotFoundError(f"Path does not exist: {p}")

def build_fourclass_mapping(id2label: Dict[int, str]) -> Dict[int, int]:
    # Map the 150 ADE20K classes to the four-class scheme (sky/vegetation/building/other).
    SKY_SET = {"sky"}
    VEG_SET = {"tree","grass","plant","flower","palm"}
    BLD_SET = {
        "wall","building","house","skyscraper","door",
        "windowpane","column","fence","railing","tower",
        "bridge","grandstand","booth","canopy","awning",
        "bannister","escalator","stairs","stairway","stage","pier",
        "glass","mirror","tent",
    }
    mapping = {}
    for k, v in id2label.items():
        name = str(v).strip().lower()
        if name in SKY_SET:   mapping[int(k)] = 0
        elif name in VEG_SET: mapping[int(k)] = 1
        elif name in BLD_SET: mapping[int(k)] = 2
        else:                 mapping[int(k)] = 3
    return mapping

# ============================================================================
# =================== Minimal Circle Definition and Crop =====================
# ============================================================================
def define_fisheye_circle(rgb: np.ndarray) -> Tuple[Tuple[float, float], float]:
    H, W = rgb.shape[:2]
    cx = W / 2.0 + CENTER_BIAS_X
    cy = H / 2.0 + CENTER_BIAS_Y
    raw_radius = min(H, W) / 2.0
    r_eff = raw_radius * (1.0 - INNER_MARGIN)
    r_eff = max(r_eff, 1.0)
    return (cx, cy), r_eff

def circular_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    y, x = np.ogrid[:h, :w]
    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    return (dist2 <= r ** 2).astype(np.uint8)

def crop_to_square(img: np.ndarray, cx: float, cy: float, r: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = img.shape[:2]
    side = int(round(2 * r))
    x0 = int(round(cx - r))
    y0 = int(round(cy - r))
    x0 = max(0, min(x0, W - side))
    y0 = max(0, min(y0, H - side))
    crop = img[y0:y0 + side, x0:x0 + side].copy()
    return crop, (x0, y0)

# ============================================================================
# ===================== Sliding-Window Stitching Helpers =====================
# ============================================================================
def build_feather_weight(tile: int, overlap: int) -> torch.Tensor:
    wy = torch.ones(tile, dtype=torch.float32)
    wx = torch.ones(tile, dtype=torch.float32)
    ramp = torch.linspace(0, math.pi, overlap, dtype=torch.float32)
    edge = (1 - torch.cos(ramp)) * 0.5
    wy[:overlap] = edge; wy[-overlap:] = edge.flip(0)
    wx[:overlap] = edge; wx[-overlap:] = edge.flip(0)
    return wy[:, None] * wx[None, :]

def tile_boxes(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    stride = tile - 2 * overlap
    if stride <= 0: raise ValueError("OVERLAP is too large")
    ys, y = [], 0
    while True:
        if y + tile >= H: ys.append(max(0, H - tile)); break
        ys.append(y); y += stride
    xs, x = [], 0
    while True:
        if x + tile >= W: xs.append(max(0, W - tile)); break
        xs.append(x); x += stride
    return [(y0, min(H, y0+tile), x0, min(W, x0+tile)) for y0 in ys for x0 in xs]

# ============================================================================
# ==================== Deep Learning Model Inference (GPU) ===================
# ============================================================================
@torch.inference_mode()
def _infer_logprob_single(rgb_tile: np.ndarray, processor, model, device) -> torch.Tensor:
    h, w = rgb_tile.shape[:2]
    pil = Image.fromarray(rgb_tile)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    use_fp16 = (device.type == "cuda" and FP16)
    with torch.amp.autocast('cuda', enabled=use_fp16):
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        logprob = torch.log_softmax(logits, dim=1)
    return logprob.squeeze(0)

def infer_tile_prob_tta(rgb_tile, processor, model, device, scales, use_hflip):
    h, w = rgb_tile.shape[:2]
    logps = []
    for s in scales:
        tile_s = rgb_tile if s==1.0 else cv2.resize(rgb_tile, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        lp = _infer_logprob_single(tile_s, processor, model, device)
        lp = F.interpolate(lp.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        logps.append(lp)
        if use_hflip:
            tile_sf = cv2.flip(tile_s, 1)
            lp_f = _infer_logprob_single(tile_sf, processor, model, device)
            lp_f = torch.flip(lp_f, dims=[2])
            lp_f = F.interpolate(lp_f.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            logps.append(lp_f)
    logp_mean = torch.stack(logps, dim=0).mean(dim=0)
    return torch.softmax(logp_mean, dim=0)

def probs_to_four_probs(probs: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    C, H, W = probs.shape
    out = torch.zeros((4, H, W), dtype=probs.dtype, device=probs.device)
    for cid in range(C):
        out[mapping.get(cid, 3)] += probs[cid]
    out = out / (out.sum(dim=0, keepdim=True) + 1e-6)
    return out

@torch.inference_mode()
def infer_binary_building(rgb_sq: np.ndarray, processor, bin_model, device):
    H, W = rgb_sq.shape[:2]
    pil = Image.fromarray(rgb_sq)
    inputs = processor(images=pil, return_tensors="pt", size=BIN_INFER_SIZE).to(device)
    use_fp16 = (device.type == "cuda" and FP16)
    with torch.amp.autocast('cuda', enabled=use_fp16):
        out = bin_model(pixel_values=inputs["pixel_values"])
        logits = out.logits
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[0, 1]
    return probs

# ============================================================================
# ==================== GPU Fusion & Spatial Prior ============================
# ============================================================================
def fuse_building_prob_gpu(P4: torch.Tensor, pb: torch.Tensor,
                           mode="logit_add", logit_gain=2.0) -> torch.Tensor:
    eps = 1e-6
    P = P4.clamp(eps, 1.0)
    pb = pb.clamp(eps, 1.0)
    nb = 1.0 - pb

    if mode == "logit_add":
        def _logit(p): return (p / (1.0 - p + eps)).log()
        def _sigmoid(x): return 1.0 / (1.0 + (-x).exp())
        log_b = _logit(P[..., 2]); log_s = _logit(P[..., 0])
        log_v = _logit(P[..., 1]); log_o = _logit(P[..., 3])
        log_pb = _logit(pb)
        log_b = log_b + logit_gain * log_pb
        damp = 0.25 * (-logit_gain) * log_pb
        log_s = log_s + damp; log_v = log_v + damp; log_o = log_o + damp
        Pb = _sigmoid(log_b); Ps = _sigmoid(log_s)
        Pv = _sigmoid(log_v); Po = _sigmoid(log_o)
        S = (Ps + Pv + Pb + Po) + eps
        P4_out = torch.stack([Ps/S, Pv/S, Pb/S, Po/S], dim=-1)

    elif mode == "mul":
        strength = logit_gain
        w_b = (pb ** strength) + eps
        w_n = (nb ** strength) + eps
        P4_out = P.clone()
        P4_out[..., 2] *= w_b; P4_out[..., 0] *= w_n
        P4_out[..., 1] *= w_n; P4_out[..., 3] *= w_n
        P4_out = P4_out / (P4_out.sum(dim=-1, keepdim=True) + eps)

    elif mode == "max":
        P4_out = P.clone()
        P4_out[..., 2] = torch.maximum(P4_out[..., 2], pb)
        P4_out = P4_out / (P4_out.sum(dim=-1, keepdim=True) + eps)

    return P4_out

def apply_spatial_prior_gpu(P4: torch.Tensor, circle_mask_tensor: torch.Tensor) -> torch.Tensor:
    if not USE_SPATIAL_PRIOR:
        return P4
    H, W = P4.shape[:2]
    yy, xx = torch.meshgrid(torch.arange(H, device=P4.device, dtype=torch.float32),
                            torch.arange(W, device=P4.device, dtype=torch.float32), indexing='ij')
    y_rel = (yy - H/2.0) / (H/2.0 + 1e-6)
    lower_boost = torch.sigmoid(PRIOR_SHARPNESS * y_rel)
    lower_boost = torch.clamp((lower_boost - 0.5) * 2.0, 0.0, 1.0)
    lower_boost = lower_boost * circle_mask_tensor

    logP = P4.clamp(1e-6, 1.0).log()
    logs = logP.clone()
    logs[..., 2] = logP[..., 2] + PRIOR_BUILDING_BIAS + PRIOR_STRENGTH * lower_boost
    logs = logs - logs.max(dim=-1, keepdim=True).values
    P4_boost = logs.exp()
    P4_boost = P4_boost / (P4_boost.sum(dim=-1, keepdim=True) + 1e-6)
    return P4_boost

# ============================================================================
# ================ CRF Post-Processing (if pydensecrf is available) ==========
# ============================================================================
def densecrf_refine(img_rgb, probs4_hwk, iters=6):
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    H, W, K = probs4_hwk.shape
    P = np.clip(probs4_hwk, 1e-6, 1.0)
    P = P / P.sum(axis=-1, keepdims=True)
    P_chw = np.transpose(P, (2, 0, 1)).copy()
    d = dcrf.DenseCRF2D(W, H, K)
    unary = unary_from_softmax(P_chw)
    d.setUnaryEnergy(unary.astype(np.float32, copy=False))
    d.addPairwiseGaussian(sxy=CRF_SXY_GAUSS, compat=CRF_COMPAT_GAUSS)
    d.addPairwiseBilateral(sxy=CRF_SXY_BILAT, srgb=CRF_SRGB_BILAT,
                           rgbim=img_rgb, compat=CRF_COMPAT_BILAT)
    Q = np.array(d.inference(iters), dtype=np.float32).reshape(K, H, W)
    cls = np.argmax(Q, axis=0).astype(np.int32)
    Q = np.transpose(Q, (1, 2, 0))
    return cls, Q

def densecrf_refine_narrowband(img_rgb, probs4_hwk, band_w=9):
    H, W, K = probs4_hwk.shape
    P = np.clip(probs4_hwk, 1e-6, 1.0)
    P = P / P.sum(axis=-1, keepdims=True)
    p_sorted = np.sort(P, axis=-1)
    margin = p_sorted[..., -1] - p_sorted[..., -2]
    thr_m = max(0.12, float(np.quantile(margin, 0.30)))
    band_m = (margin < thr_m)
    entropy = -(P * np.log(P + 1e-8)).sum(axis=-1)
    thr_H = float(np.quantile(entropy, 0.70))
    band_H = (entropy > thr_H)
    argmax = P.argmax(axis=-1).astype(np.uint8)
    edge = cv2.Canny(argmax * 63, 0, 0)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    band = (band_m | band_H | (edge > 0)).astype(np.uint8)
    band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_w, band_w)))
    if band.sum() == 0:
        return P.argmax(axis=-1).astype(np.int32), P
    ys, xs = np.where(band > 0)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    img_crop = img_rgb[y0:y1, x0:x1, :]
    P_crop   = P[y0:y1, x0:x1, :]
    cls_crop, Q_crop = densecrf_refine(img_crop, P_crop, iters=CRF_ITERS)
    cls = P.argmax(axis=-1).astype(np.int32); Q = P.copy()
    cls[y0:y1, x0:x1] = cls_crop
    Q  [y0:y1, x0:x1, :] = Q_crop
    return cls, Q

def postprocess_binary(mask, kernel_size=3, min_area=64):
    out = mask.copy()
    if kernel_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    filt = np.zeros_like(out)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filt[labels == i] = 255
    return filt

# ============================================================================
# ============ Single-Image Processing Pipeline (extreme precision + rotation correction) ====
# ============================================================================
def process_image(img_path, ade_processor, ade_model, bin_processor, bin_model,
                  device, ade_to_four, out_dir):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[WARN] Failed to read image: {img_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    base = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*60}\nProcessing: {base}\n{'='*60}")

    # ---------- 1. Minimal circle definition ----------
    (cx, cy), r = define_fisheye_circle(rgb)
    print(f"[CIRCLE] Center=({cx:.1f}, {cy:.1f}), Radius={r:.1f} (inward margin={INNER_MARGIN})")

    # ---------- 2. Square crop + circular mask ----------
    rgb_sq, (x0, y0) = crop_to_square(rgb, cx, cy, r)
    H, W = rgb_sq.shape[:2]
    # Circle center coordinates in the square (theoretically equal to r)
    cx_sq = cx - x0
    cy_sq = cy - y0
    # Build the circular mask (same size as the square image, centered at (cx_sq, cy_sq))
    circle_mask_np = circular_mask(H, W, cx_sq, cy_sq, r)

    # ---------- 2.5 Image rotation correction (optional) ----------
    if ENABLE_ROTATION:
        # In OpenCV, counterclockwise is positive, which corresponds exactly
        # to the rotation required to make the top point to "north".
        angle = IMAGE_TOP_AZIMUTH
        # Use the square center as the rotation center
        center_rot = (W / 2.0, H / 2.0)
        M = cv2.getRotationMatrix2D(center_rot, angle, 1.0)
        rgb_sq = cv2.warpAffine(rgb_sq, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        print(f"[ROTATE] Image rotated counterclockwise by {angle:.1f} deg so that the top points to true north")
        # Note: after rotation, the circle center theoretically remains at the
        # square center; negligible small coordinate offsets are ignored, and
        # the circular mask circle_mask_np does not need to change since it is
        # still centered on the square center.

    # Blacken pixels outside the circle as model input
    work = rgb_sq.copy()
    work[circle_mask_np == 0] = 0
    _imwrite(os.path.join(out_dir, f"{base}_square.png"), cv2.cvtColor(work, cv2.COLOR_RGB2BGR))
    # Convert to a GPU tensor
    circle_mask_tensor = torch.from_numpy(circle_mask_np).to(device).float()

    # Optionally save the circle detection visualization (green circle drawn on the rotated image)
    if SAVE_CIRCLE_DEBUG:
        dbg = work.copy()  # Use the blackened image; using rgb_sq alone also works
        cv2.circle(dbg, (int(cx_sq), int(cy_sq)), int(r), (0, 255, 0), CIRCLE_THICKNESS)
        _imwrite(os.path.join(out_dir, f"{base}_circle_dbg.png"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # ---------- 3. ADE four-class probabilities (sliding-window TTA, GPU accumulation) ----------
    boxes = tile_boxes(H, W, TILE, OVERLAP)
    weight = build_feather_weight(TILE, OVERLAP).to(device)

    acc4 = torch.zeros((4, H, W), dtype=torch.float32, device=device)
    wsum = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    print("[MODEL] ADE four-class inference...")
    for (y0b, y1b, x0b, x1b) in tqdm(boxes, desc="ADE tiles"):
        tile = work[y0b:y1b, x0b:x1b, :]
        h_t, w_t = tile.shape[:2]
        wmask = weight[:h_t, :w_t]
        probs_ade = infer_tile_prob_tta(tile, ade_processor, ade_model, device,
                                        SCALES, USE_TTA_HFLIP)
        probs4 = probs_to_four_probs(probs_ade, ade_to_four)
        acc4[:, y0b:y1b, x0b:x1b] += probs4 * wmask.unsqueeze(0)
        wsum[0, y0b:y1b, x0b:x1b] += wmask

    P4_gpu = acc4 / wsum.clamp_min(1e-6)
    P4_gpu = P4_gpu * circle_mask_tensor.unsqueeze(0)
    P4_gpu = P4_gpu.permute(1, 2, 0)   # (H,W,4)

    # ---------- 4. Binary building probability ----------
    print("[MODEL] Binary building inference...")
    pb_gpu = infer_binary_building(work, bin_processor, bin_model, device)
    pb_gpu = pb_gpu * circle_mask_tensor

    # ---------- 5. High-confidence hard overwrite (disabled) ----------
    strong_mask = (pb_gpu >= BIN_HARD_THRESH).float()
    if BIN_HARD_DILATE > 0:
        strong_np = strong_mask.cpu().numpy().astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BIN_HARD_DILATE*2+1, BIN_HARD_DILATE*2+1))
        strong_np = cv2.morphologyEx(strong_np, cv2.MORPH_CLOSE, k)
        strong_mask = torch.from_numpy(strong_np).to(device).float()
    P4_gpu[..., 2] = torch.maximum(P4_gpu[..., 2], strong_mask * 0.95)
    P4_gpu[..., 0] *= (1.0 - strong_mask)
    P4_gpu[..., 1] *= (1.0 - strong_mask)
    P4_gpu[..., 3] *= (1.0 - strong_mask)
    P4_gpu = P4_gpu / (P4_gpu.sum(dim=-1, keepdim=True) + 1e-6)

    # ---------- 6. Logit fusion ----------
    P4_gpu = fuse_building_prob_gpu(P4_gpu, pb_gpu, mode=BIN_FUSE_MODE, logit_gain=BIN_LOGIT_GAIN)
    P4_gpu = P4_gpu * circle_mask_tensor.unsqueeze(-1)

    # ---------- 7. Spatial prior ----------
    P4_gpu = apply_spatial_prior_gpu(P4_gpu, circle_mask_tensor)
    P4_gpu = P4_gpu * circle_mask_tensor.unsqueeze(-1)

    # ---------- 8. Convert to numpy for CRF ----------
    P4_np = P4_gpu.cpu().numpy()

    # ---------- 9. CRF post-processing ----------
    if HAS_DENSECRF:
        if USE_NARROWBAND_CRF and not FULL_IMAGE_CRF:
            print(f"[CRF] Narrow-band CRF (band width {NARROW_BAND_WIDTH}, iterations {CRF_ITERS}, Gaussian smoothing weakened)")
            cls_ref, Q4_np = densecrf_refine_narrowband(work, P4_np, band_w=NARROW_BAND_WIDTH)
        elif FULL_IMAGE_CRF:
            print("[CRF] Full-image CRF...")
            cls_ref, Q4_np = densecrf_refine(work, P4_np, iters=CRF_ITERS)
        else:
            cls_ref = P4_np.argmax(axis=-1).astype(np.int32)
            Q4_np = P4_np
    else:
        if USE_NARROWBAND_CRF or FULL_IMAGE_CRF:
            print("[INFO] pydensecrf is not installed; skipping CRF.")
        cls_ref = P4_np.argmax(axis=-1).astype(np.int32)
        Q4_np = P4_np

    # ---------- 10. Light smoothing of vegetation ----------
    Q4_smooth = Q4_np.copy()
    veg_chan = Q4_smooth[..., 1]
    veg_u8 = np.clip(veg_chan * 255.0, 0, 255).astype(np.uint8)
    veg_u8 = cv2.medianBlur(veg_u8, 3)
    Q4_smooth[..., 1] = veg_u8.astype(np.float32) / 255.0
    Q4_smooth = Q4_smooth / (Q4_smooth.sum(axis=-1, keepdims=True) + 1e-6)

    # With KEEP_OTHER_CLASS=False, "other" is merged into "building" (three output classes).
    if not KEEP_OTHER_CLASS:
        Q4_smooth[..., 2] += Q4_smooth[..., 3]
        Q4_smooth[..., 3] = 0.0
        Q4_smooth = Q4_smooth / (Q4_smooth.sum(axis=-1, keepdims=True) + 1e-6)

    cls_final = Q4_smooth.argmax(axis=-1).astype(np.int32)
    cls_final[circle_mask_np == 0] = 3

    # ---------- 11. Statistics ----------
    valid = (circle_mask_np > 0)
    total_valid = int(valid.sum())
    if total_valid > 0:
        ratios = []
        for cid in range(4):
            cnt = int(((cls_final == cid) & valid).sum())
            ratios.append(cnt / total_valid)
        print(f"[RATIO] Sky={ratios[0]*100:.1f}%  Vegetation={ratios[1]*100:.1f}%  "
              f"Building={ratios[2]*100:.1f}%  Other={ratios[3]*100:.1f}%")

    # ---------- 12. Morphological post-processing (extremely light) ----------
    def pp(mask, cls_id):
        p = MORPH_PARAMS.get(cls_id, dict(kernel=3, min_area=64))
        return postprocess_binary(mask, p["kernel"], p["min_area"])

    sky = pp(((cls_final == 0) & valid).astype(np.uint8)*255, 0)
    veg = pp(((cls_final == 1) & valid).astype(np.uint8)*255, 1)
    bld = pp(((cls_final == 2) & valid).astype(np.uint8)*255, 2)
    oth = pp(((cls_final == 3) & valid).astype(np.uint8)*255, 3)
    stack = np.stack([sky, veg, bld, oth], axis=-1).astype(np.float32)/255.0
    cls_final = stack.argmax(axis=-1).astype(np.int32)
    if not KEEP_OTHER_CLASS:
        cls_final[(cls_final == 3) & valid] = 2
    cls_final[circle_mask_np == 0] = 3

    color_map_arr = np.array([COLOR_SKY, COLOR_VEG, COLOR_BLD, COLOR_OTH], dtype=np.uint8)
    color_layer = color_map_arr[np.clip(cls_final, 0, 3)]
    overlay = (work * (1-OVERLAY_ALPHA) + color_layer * OVERLAY_ALPHA).astype(np.uint8)
    overlay[circle_mask_np == 0] = 0

    sky_mask = (cls_final == 0).astype(np.uint8) * 255
    veg_mask = (cls_final == 1).astype(np.uint8) * 255
    bld_mask = (cls_final == 2).astype(np.uint8) * 255
    _imwrite(os.path.join(out_dir, f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    _imwrite(os.path.join(out_dir, f"{base}_mask_sky.png"), sky_mask)
    _imwrite(os.path.join(out_dir, f"{base}_mask_veg.png"), veg_mask)
    _imwrite(os.path.join(out_dir, f"{base}_mask_bld.png"), bld_mask)
    if KEEP_OTHER_CLASS:
        oth_mask = (cls_final == 3).astype(np.uint8) * 255
        _imwrite(os.path.join(out_dir, f"{base}_mask_oth.png"), oth_mask)
    else:
        oth_path = os.path.join(out_dir, f"{base}_mask_oth.png")
        if os.path.exists(oth_path):
            os.remove(oth_path)

    if SAVE_SOFT_NPZ:
        np.savez_compressed(os.path.join(out_dir, f"{base}_soft4.npz"),
                            soft4=Q4_smooth.astype(np.float32))
        print(f"[SAVE] Soft probability map -> {base}_soft4.npz")
    print(f"[DONE] {base} finished")

# ============================================================================
# ================================ Main Entry ================================
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Four-class segmentation of fisheye images, extreme-precision version (with rotation correction)")
    parser.add_argument("--in_dir", default=IN_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--ade_model_dir", default=ADE_MODEL_DIR)
    parser.add_argument("--bin_base_model_dir", default=BIN_BASE_MODEL_DIR)
    parser.add_argument("--bin_ckpt", default=BIN_CKPT_PATH)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.deterministic:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("[INFO] Deterministic mode enabled")
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}, Half precision: {FP16 if device.type=='cuda' else False}")

    # Load models
    print("[INFO] Loading ADE20K four-class model...")
    try:
        ade_processor = AutoImageProcessor.from_pretrained(args.ade_model_dir, local_files_only=True, use_fast=True)
        ade_model = AutoModelForSemanticSegmentation.from_pretrained(
            args.ade_model_dir, local_files_only=True,
            torch_dtype=(torch.float16 if device.type=="cuda" and FP16 else torch.float32),
            low_cpu_mem_usage=True
        ).to(device).eval()
    except Exception as e:
        print(f"[ERROR] Failed to load ADE model: {e}")
        return
    id2label = {int(k): v for k, v in ade_model.config.id2label.items()}
    ade_to_four = build_fourclass_mapping(id2label)

    print("[INFO] Loading binary building model...")
    try:
        bin_processor = AutoImageProcessor.from_pretrained(args.bin_base_model_dir, local_files_only=True, use_fast=True)
        bin_model = SegformerForSemanticSegmentation.from_pretrained(
            args.bin_base_model_dir, num_labels=2, ignore_mismatched_sizes=True,
            local_files_only=True, low_cpu_mem_usage=False
        )
        ckpt = torch.load(args.bin_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        bin_model.load_state_dict(state, strict=True)
        bin_model = bin_model.to(device).eval()
        print("[INFO] Binary weights loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load binary model: {e}")
        return

    images = collect_input_images(args.in_dir)
    print(f"[INFO] Found {len(images)} images")
    if not images: return

    os.makedirs(args.out_dir, exist_ok=True)
    for i, p in enumerate(images, 1):
        print(f"\n>>> Progress: {i}/{len(images)}")
        process_image(p, ade_processor, ade_model, bin_processor, bin_model,
                      device, ade_to_four, args.out_dir)

    print("\n[DONE] All images processed!")

if __name__ == "__main__":
    main()
