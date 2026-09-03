# -*- coding: utf-8 -*-
"""
Fisheye solar trajectory overlay (stable enhanced full version)
=================================================
Features:
- Equisolid projection
- North at top, south at bottom / east on left, west on right
- Continuous solid lines / dashed lines
- Automatic merging of continuous segments
- Trajectory break repair
- Occlusion priority repair
- Low solar altitude filtering
- Hour markers / cross lines / NESW directions / legend
- Occlusion CSV output

New architecture notes:
- This module corresponds to step 2 of the pipeline: it reads the segmentation
  output, computes the solar trajectory, and performs obstruction determination.
- Daily parameters should preferably be adjusted in ../config.py; pipeline.py
  injects them into CFG before running.
- The output directory structure is kept as results/<image_name>/sunpath/<image_name>/.
"""

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import pvlib


# =========================================================
# Configuration (all parameters centrally managed)
# =========================================================
class CFG:
    # ---- Paths ----
    POINT_DIR = Path(r"D:\projects\image_recog\results\2423\segmentation")
    OUTPUT_ROOT = Path(r"D:\projects\image_recog\results\2423\sunpath")
    IMAGE_NAME = "2423"
    SEG_SUBDIR = ""

    # ---- Geography ----
    LAT = 38.8825
    LON = 115.5556
    ALT_M = 14.0
    TZ = "Asia/Shanghai"

    # ---- Dates ----
    SINGLE_DATE = "2026-04-15"
    TIME_STEP_MIN = 1
    DRAW_FREQ_MIN = 5

    # ---- Projection ----
    PROJECTION_MODEL = "equisolid"

    # ---- Image orientation ----
    IMG_TOP_AZIMUTH_DEG = 0.0
    MIRROR_AZIMUTH = True

    # ---- Trajectory style ----
    LINE_COLOR_BGR = (0, 165, 255)
    LINE_THICKNESS = 4
    DASH_LENGTH = 8
    DASH_GAP = 6

    # ---- Occlusion ----
    OCCLUDED_CLASSES = {"building", "vegetation"}
    KEEP_OTHER_CLASS = False
    MIN_ALTITUDE_DEG = 3.0
    MAX_JUMP_PX = 80

    # ---- Hour markers ----
    HOUR_MARKER_ENABLED = True
    HOUR_MARKER_RADIUS = 8
    HOUR_MARKER_COLOR_BGR = (0, 255, 255)
    HOUR_MARKER_SHOW_LABEL = True
    HOUR_LABEL_FONT_SCALE = 0.8
    HOUR_LABEL_THICKNESS = 2
    HOUR_LABEL_OFFSET = (15, -10)

    # ---- Cross lines ----
    CROSS_ENABLED = True
    CROSS_THICK = 4
    CROSS_DASH_LEN = 18
    CROSS_GAP_LEN = 12
    CROSS_COLOR = (255, 255, 255)

    # ---- NESW directions ----
    NESW_ENABLED = True
    NESW_FONT_SCALE = 3.2
    NESW_OUTLINE_THICK = 7
    NESW_FILL_THICK = 4
    NESW_RADIUS_RATIO = 0.96
    NESW_COLOR_FILL = (255, 255, 255)
    NESW_COLOR_OUTLINE = (0, 0, 0)
    NESW_OFFSET = {"N": (-30, 55), "S": (-30, -35), "E": (-80, 25), "W": (20, 25)}

    # ---- Legend ----
    LEGEND_ENABLED = True
    LEGEND_W = 500
    LEGEND_H = 300
    LEGEND_MARGIN = 40
    LEGEND_TITLE_SCALE = 1.5
    LEGEND_ITEM_SCALE = 1.2
    LEGEND_TEXT_THICK = 2
    LEGEND_LINE_X_START_RATIO = 0.1
    LEGEND_LINE_X_END_RATIO = 0.4
    LEGEND_TEXT_X_OFFSET = 25
    LEGEND_TEXT_Y_OFFSET = 8
    LEGEND_FIRST_Y_OFFSET = 100
    LEGEND_LINE_STEP = 45

    # ---- Output ----
    SAVE_CSV = True
    SAVE_IMAGE = True


# =========================================================
# Load overlay and geometry information
# =========================================================
def load_overlay_and_geometry(point_dir, image_name, seg_subdir=""):
    seg_dir = point_dir / seg_subdir if seg_subdir else point_dir
    overlay_path = seg_dir / f"{image_name}_overlay.png"
    overlay = cv2.imread(str(overlay_path))
    if overlay is None:
        raise FileNotFoundError(f"Failed to read: {overlay_path}")

    meta_path = seg_dir / f"{image_name}_segmentation_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cx, cy = meta["circle_center_square_xy"]
        radius = meta["circle_radius_square_px"]
    else:
        h, w = overlay.shape[:2]
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        radius = min(h, w) / 2.0

    return overlay, (cx, cy), radius


# =========================================================
# Load the four class masks
# =========================================================
def load_masks(seg_dir, image_name):
    masks = {}
    for name in ["sky", "veg", "bld", "oth"]:
        fpath = seg_dir / f"{image_name}_mask_{name}.png"
        if name == "oth" and not fpath.exists():
            ref = next((m for m in (masks.get("sky"), masks.get("veg"), masks.get("bld")) if m is not None), None)
            if ref is None:
                raise FileNotFoundError(fpath)
            masks[name] = np.zeros_like(ref, dtype=bool)
            continue
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(fpath)
        masks[name] = img > 127
    if not CFG.KEEP_OTHER_CLASS:
        masks["bld"] = masks["bld"] | masks["oth"]
        masks["oth"] = np.zeros_like(masks["oth"], dtype=bool)
    return {
        "sky": masks["sky"],
        "vegetation": masks["veg"],
        "building": masks["bld"],
        "other": masks["oth"]
    }


# =========================================================
# Project solar position onto fisheye image coordinates
# =========================================================
def sun_to_pixel(alt, azi, cx, cy, R, img_top_az=0.0, mirror=False):
    alt = np.asarray(alt, dtype=float)
    azi = np.asarray(azi, dtype=float)
    theta_rad = np.radians(90.0 - alt)

    if CFG.PROJECTION_MODEL == "equidistant":
        f = R / (math.pi / 2.0)
        rho = f * theta_rad
    elif CFG.PROJECTION_MODEL == "equisolid":
        f = R / math.sqrt(2.0)
        rho = 2.0 * f * np.sin(theta_rad / 2.0)
    else:
        raise ValueError("Unknown projection model")

    az_rot = azi - img_top_az
    if mirror:
        az_rot = -az_rot
    phi = np.radians(az_rot % 360.0)

    u = cx + rho * np.sin(phi)
    v = cy - rho * np.cos(phi)
    return u, v


# =========================================================
# Occlusion classification (with priority)
# =========================================================
def classify_occlusion(masks, u_arr, v_arr, h, w):
    occ = np.full(len(u_arr), "outside", dtype=object)
    valid = (u_arr >= 0) & (u_arr < w) & (v_arr >= 0) & (v_arr < h)
    if not np.any(valid):
        return occ

    u_int = np.clip(u_arr[valid].astype(int), 0, w - 1)
    v_int = np.clip(v_arr[valid].astype(int), 0, h - 1)
    occ_valid = np.full(len(u_int), "outside", dtype=object)

    priority = ["building", "vegetation", "other", "sky"] if CFG.KEEP_OTHER_CLASS else ["building", "vegetation", "sky"]
    for name in priority:
        hit = masks[name][v_int, u_int]
        update = hit & (occ_valid == "outside")
        occ_valid[update] = name

    if not CFG.KEEP_OTHER_CLASS:
        occ_valid[occ_valid == "other"] = "building"

    occ[valid] = occ_valid
    return occ


# =========================================================
# Build the per-day occlusion data table
# =========================================================
def compute_day_occlusion(geometry, date_str, freq_min, tz):
    date = pd.Timestamp(date_str).tz_localize(tz)
    end = date + pd.Timedelta(days=1) - pd.Timedelta(minutes=freq_min)
    times = pd.date_range(date, end, freq=f"{freq_min}min", tz=tz)

    solpos = pvlib.solarposition.get_solarposition(times, CFG.LAT, CFG.LON)
    alt = solpos["apparent_elevation"].values
    azi = solpos["azimuth"].values

    cx, cy = geometry["circle_center"]
    R = geometry["circle_radius"]
    h, w = geometry["masks"]["sky"].shape

    u, v = sun_to_pixel(alt, azi, cx, cy, R, CFG.IMG_TOP_AZIMUTH_DEG, CFG.MIRROR_AZIMUTH)
    occ = classify_occlusion(geometry["masks"], u, v, h, w)

    df = pd.DataFrame({
        "date": times.strftime("%Y-%m-%d"),
        "time": times.strftime("%H:%M"),
        "altitude_deg": alt,
        "zenith_deg": 90.0 - alt,
        "azimuth_deg": azi % 360.0,
        "occlusion": occ,
        "u_px": u,
        "v_py": v
    })
    return df


# =========================================================
# Helper functions for drawing
# =========================================================
def resample_for_drawing(df, draw_freq_min):
    if draw_freq_min <= 1:
        return df.copy()
    idx = pd.to_datetime(df["date"] + " " + df["time"])
    df_idx = df.set_index(idx).sort_index()
    resampled = df_idx.resample(f"{draw_freq_min}min").nearest()
    return resampled.dropna().reset_index(drop=True)


def draw_dashed_polyline(img, points, color, thickness, dash_len, gap_len):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            continue
        cos_a = (pt2[0] - pt1[0]) / dist
        sin_a = (pt2[1] - pt1[1]) / dist
        pos = 0.0
        while pos < dist:
            start_x = int(round(pt1[0] + pos * cos_a))
            start_y = int(round(pt1[1] + pos * sin_a))
            end_pos = min(pos + dash_len, dist)
            end_x = int(round(pt1[0] + end_pos * cos_a))
            end_y = int(round(pt1[1] + end_pos * sin_a))
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
            pos += dash_len + gap_len


def iter_continuous_segments(draw_df, mask, max_jump_px):
    """Yield point segments without drawing artificial cross-gap connectors."""
    selected = draw_df.loc[mask, ["u_px", "v_py"]]
    if selected.empty:
        return

    prev_index = None
    prev_point = None
    current = []
    for index, row in selected.iterrows():
        point = np.array([float(row["u_px"]), float(row["v_py"])], dtype=float)
        continuous_index = prev_index is not None and index == prev_index + 1
        continuous_space = (
            prev_point is not None
            and math.hypot(point[0] - prev_point[0], point[1] - prev_point[1]) <= max_jump_px
        )
        if current and not (continuous_index and continuous_space):
            if len(current) >= 2:
                yield np.rint(np.asarray(current)).astype(np.int32)
            current = []
        current.append(point)
        prev_index = index
        prev_point = point

    if len(current) >= 2:
        yield np.rint(np.asarray(current)).astype(np.int32)


def draw_cross(img, cx, cy, r_eff):
    if not CFG.CROSS_ENABLED:
        return
    draw_dashed_polyline(img, np.array([[cx - r_eff, cy], [cx + r_eff, cy]]),
                         CFG.CROSS_COLOR, CFG.CROSS_THICK, CFG.CROSS_DASH_LEN, CFG.CROSS_GAP_LEN)
    draw_dashed_polyline(img, np.array([[cx, cy - r_eff], [cx, cy + r_eff]]),
                         CFG.CROSS_COLOR, CFG.CROSS_THICK, CFG.CROSS_DASH_LEN, CFG.CROSS_GAP_LEN)


def draw_nesw(img, cx, cy, r_eff):
    if not CFG.NESW_ENABLED:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_r = CFG.NESW_RADIUS_RATIO * r_eff
    for txt, azi in [("N", 0), ("E", 270), ("S", 180), ("W", 90)]:
        rad = math.radians(azi)
        u = cx + label_r * math.sin(rad)
        v = cy - label_r * math.cos(rad)
        ui, vi = int(round(u)), int(round(v))
        dx, dy = CFG.NESW_OFFSET.get(txt, (0, 0))
        pos = (ui + dx, vi + dy)
        cv2.putText(img, txt, pos, font, CFG.NESW_FONT_SCALE, CFG.NESW_COLOR_OUTLINE, CFG.NESW_OUTLINE_THICK, cv2.LINE_AA)
        cv2.putText(img, txt, pos, font, CFG.NESW_FONT_SCALE, CFG.NESW_COLOR_FILL, CFG.NESW_FILL_THICK, cv2.LINE_AA)


def draw_legend(img, date_str):
    # Simplified version: following the original logic, only the background box is drawn;
    # detailed legend content is not drawn (the legend drawing details were never fully
    # implemented in the original script).
    if not CFG.LEGEND_ENABLED:
        return img
    H, W = img.shape[:2]
    lw = CFG.LEGEND_W
    lh = CFG.LEGEND_H
    x0 = max(W - lw - CFG.LEGEND_MARGIN, 0)
    y0 = max(H - lh - CFG.LEGEND_MARGIN, 0)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + lw, y0 + lh), (0, 0, 0), -1)
    img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    return img


# =========================================================
# Main workflow (called by the master controller)
# =========================================================
def main():
    # 1. Load overlay and geometry
    overlay, (cx, cy), radius = load_overlay_and_geometry(CFG.POINT_DIR, CFG.IMAGE_NAME, CFG.SEG_SUBDIR)
    h_img, w_img = overlay.shape[:2]

    # 2. Load masks
    seg_dir = CFG.POINT_DIR / CFG.SEG_SUBDIR if CFG.SEG_SUBDIR else CFG.POINT_DIR
    masks = load_masks(seg_dir, CFG.IMAGE_NAME)
    geometry = {"circle_center": (cx, cy), "circle_radius": radius, "masks": masks}

    # 3. Compute the per-day occlusion data
    df_occ = compute_day_occlusion(geometry, CFG.SINGLE_DATE, CFG.TIME_STEP_MIN, CFG.TZ)

    # 4. Save CSV
    out_dir = CFG.OUTPUT_ROOT / CFG.IMAGE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"sun_occlusion_{CFG.IMAGE_NAME}_{CFG.SINGLE_DATE}.csv"
    df_occ.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] Saved: {csv_path}")

    # 5. Draw the trajectory image
    draw_df = resample_for_drawing(df_occ, CFG.DRAW_FREQ_MIN)
    overlay_copy = overlay.copy()

    # Filter visible sun points (not occluded, not outside, altitude above the
    # minimum threshold)
    visible_mask = (draw_df["occlusion"].isin(["sky"])) & (draw_df["altitude_deg"] >= CFG.MIN_ALTITUDE_DEG)
    for points_visible in iter_continuous_segments(draw_df, visible_mask, CFG.MAX_JUMP_PX):
        cv2.polylines(
            overlay_copy,
            [points_visible],
            isClosed=False,
            color=CFG.LINE_COLOR_BGR,
            thickness=CFG.LINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Draw occluded segments as dashed lines (simplified: unoccluded segments are
    # drawn as solid lines, the rest as dashed lines)
    occluded_mask = (draw_df["occlusion"].isin(CFG.OCCLUDED_CLASSES)) & (draw_df["altitude_deg"] >= CFG.MIN_ALTITUDE_DEG)
    for points_occluded in iter_continuous_segments(draw_df, occluded_mask, CFG.MAX_JUMP_PX):
        draw_dashed_polyline(overlay_copy, points_occluded, CFG.LINE_COLOR_BGR, CFG.LINE_THICKNESS, CFG.DASH_LENGTH, CFG.DASH_GAP)

    # Hour markers
    if CFG.HOUR_MARKER_ENABLED:
        hour_df = draw_df[(draw_df["time"].str.endswith(":00")) & (draw_df["altitude_deg"] >= CFG.MIN_ALTITUDE_DEG)]
        for _, row in hour_df.iterrows():
            u, v = int(round(row["u_px"])), int(round(row["v_py"]))
            cv2.circle(overlay_copy, (u, v), CFG.HOUR_MARKER_RADIUS, CFG.HOUR_MARKER_COLOR_BGR, -1)
            if CFG.HOUR_MARKER_SHOW_LABEL:
                label = row["time"].strftime("%H") if hasattr(row["time"], "strftime") else row["time"][:2]
                cv2.putText(overlay_copy, label, (u + CFG.HOUR_LABEL_OFFSET[0], v + CFG.HOUR_LABEL_OFFSET[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, CFG.HOUR_LABEL_FONT_SCALE, (0, 255, 255), CFG.HOUR_LABEL_THICKNESS, cv2.LINE_AA)

    # Cross lines, directions, legend
    draw_cross(overlay_copy, cx, cy, radius)
    draw_nesw(overlay_copy, cx, cy, radius)
    overlay_copy = draw_legend(overlay_copy, CFG.SINGLE_DATE)

    # Save image
    img_path = out_dir / f"{CFG.IMAGE_NAME}_sunpath.png"
    cv2.imwrite(str(img_path), overlay_copy)
    print(f"[IMG] Saved: {img_path}")

if __name__ == "__main__":
    main()
