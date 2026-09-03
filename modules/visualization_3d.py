# -*- coding: utf-8 -*-
"""Three-dimensional visualization of the fisheye image and PV panel pose.

Implementation notes:
- A single fisheye image contains no real depth, so the 3D geometry of
  buildings/trees cannot be reliably recovered.
- This module maps the fisheye image onto a 3D sky hemisphere according to the
  projection model, forming an "upright" fisheye dome.
- The PV panel is placed as an independent 3D plane at the center of the
  hemisphere and rotated according to the optimal tilt/azimuth angles.

Design references:
- Fisheye-to-sphere/equirectangular mappings typically first project pixels to
  3D direction vectors.
- The original solar trajectory module of this project uses equisolid/equidistant
  projections; the same projection relations are reused here.
"""

from __future__ import annotations

import json
import math
import shutil
import base64
from pathlib import Path
from typing import Any

import numpy as np


def _load_rgb_image(path: Path) -> np.ndarray:
    """Read an image as RGB. Prefer cv2; fall back to PIL when unavailable.

    On Windows, OpenCV sometimes fails to read paths containing non-ASCII
    characters, so np.fromfile + cv2.imdecode is used here to avoid a blank
    interactive visualization for image directories with such names.
    """
    try:
        import cv2

        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))


def _load_gray_image(path: Path) -> np.ndarray:
    """Read a single-channel grayscale image, mainly for the sky mask."""
    try:
        import cv2

        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"))


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Write an RGBA PNG, also compatible with non-ASCII paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        ok, buf = cv2.imencode(".png", bgra)
        if not ok:
            raise OSError(f"Failed to encode PNG: {path}")
        buf.tofile(str(path))
    except ImportError:
        from PIL import Image

        Image.fromarray(rgba, mode="RGBA").save(path)


def _image_base_name(path: Path) -> str:
    """Recover the image base name from an overlay/square file name to find matching masks."""
    stem = path.stem
    for suffix in ["_overlay", "_square", "_circle_dbg", "_sunpath"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_sky_mask(background_path: Path) -> Path | None:
    """Locate the sky mask produced by the segmentation module. Returns None and falls back gracefully if not found."""
    base = _image_base_name(background_path)
    candidates = [
        background_path.parent / f"{base}_mask_sky.png",
        background_path.parent / f"{base}_mask_sky.jpg",
        background_path.parent / f"{base}_mask_sky.jpeg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resize_for_interactive_texture(rgb: np.ndarray, max_px: int = 2048) -> np.ndarray:
    """Downsample specifically for the interactive HTML to avoid embedding the full oversized original image."""
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_px:
        return rgb
    scale = max_px / float(long_edge)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    try:
        import cv2

        return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize(new_size, resample=Image.Resampling.LANCZOS))


def _resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize the mask to the texture size; nearest-neighbor is kept to avoid blending class boundaries."""
    try:
        import cv2

        return cv2.resize(mask, size_wh, interpolation=cv2.INTER_NEAREST)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(mask).resize(size_wh, resample=Image.Resampling.NEAREST))


def _make_sky_and_obstacle_textures(background_path: Path, out_dir: Path,
                                    max_texture_px: int = 2048) -> tuple[Path, Path, bool]:
    """Split the fisheye texture into a sky layer and a non-sky layer.

    - Sky layer: only pixels inside the sky mask are opaque; the opacity is
      controlled by the HTML slider.
    - Non-sky layer: non-sky pixels such as vegetation, buildings, and others
      remain fully opaque and are unaffected by the slider.

    If no sky mask is found, the full image degrades to the sky layer and the
    non-sky layer is empty.
    """
    rgb = _resize_for_interactive_texture(_load_rgb_image(background_path), max_texture_px)
    h, w = rgb.shape[:2]
    sky_mask_path = _find_sky_mask(background_path)

    if sky_mask_path is None:
        sky_alpha = np.full((h, w), 255, dtype=np.uint8)
        obstacle_alpha = np.zeros((h, w), dtype=np.uint8)
        has_mask = False
    else:
        sky_mask = _load_gray_image(sky_mask_path)
        if sky_mask.shape[:2] != (h, w):
            sky_mask = _resize_mask(sky_mask, (w, h))
        sky_alpha = (sky_mask > 127).astype(np.uint8) * 255
        obstacle_alpha = (sky_mask <= 127).astype(np.uint8) * 255
        has_mask = True

    sky_rgba = np.dstack([rgb, sky_alpha]).astype(np.uint8)
    obstacle_rgba = np.dstack([rgb, obstacle_alpha]).astype(np.uint8)

    sky_path = out_dir / "fisheye_sky_texture.png"
    obstacle_path = out_dir / "fisheye_obstacle_texture.png"
    _write_rgba_png(sky_path, sky_rgba)
    _write_rgba_png(obstacle_path, obstacle_rgba)
    return sky_path, obstacle_path, has_mask


def _file_data_uri(path: Path, mime: str = "image/png") -> str:
    """Embed the image as a data URI to avoid texture loading failures caused by browser cross-origin restrictions on local files."""
    return "data:%s;base64,%s" % (
        mime,
        base64.b64encode(Path(path).read_bytes()).decode("ascii"),
    )


def _sample_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample at floating-point pixel coordinates and return RGB in 0..1."""
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = u - x0
    wy = v - y0

    c00 = img[y0, x0].astype(float)
    c10 = img[y0, x1].astype(float)
    c01 = img[y1, x0].astype(float)
    c11 = img[y1, x1].astype(float)
    c0 = c00 * (1 - wx[..., None]) + c10 * wx[..., None]
    c1 = c01 * (1 - wx[..., None]) + c11 * wx[..., None]
    return (c0 * (1 - wy[..., None]) + c1 * wy[..., None]) / 255.0


def _hemisphere_mesh(img: np.ndarray, projection_model: str = "equisolid",
                     n_alt: int = 90, n_az: int = 181):
    """Sample the fisheye image onto a 3D upper-hemisphere mesh.

    Coordinate conventions:
    - X points east, Y points north, Z points to the zenith.
    - Azimuth: 0=north, 90=east, 180=south, 270=west.
    """
    h, w = img.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    radius = min(h, w) / 2.0

    alt = np.linspace(0.0, math.pi / 2.0, n_alt)[:, None]
    az = np.linspace(0.0, 2.0 * math.pi, n_az)[None, :]

    x = np.cos(alt) * np.sin(az)
    y = np.cos(alt) * np.cos(az)
    z = np.sin(alt) * np.ones_like(az)

    theta = math.pi / 2.0 - alt
    if projection_model == "equidistant":
        rho = radius * theta / (math.pi / 2.0)
    else:
        # Consistent with the equisolid projection of the existing solar
        # trajectory script: rho = sqrt(2) * R * sin(theta/2).
        rho = math.sqrt(2.0) * radius * np.sin(theta / 2.0)

    u = cx + rho * np.sin(az)
    v = cy - rho * np.cos(az)
    colors = _sample_bilinear(img, u, v)
    return x, y, z, colors


def _panel_mesh(tilt_deg: float, azimuth_deg: float,
                width: float = 0.55, height: float = 0.34):
    """Generate a 3D PV panel plane rotated by the tilt/azimuth angles."""
    local = np.array([
        [-width / 2, -height / 2, 0.0],
        [ width / 2, -height / 2, 0.0],
        [ width / 2,  height / 2, 0.0],
        [-width / 2,  height / 2, 0.0],
    ])

    tilt = math.radians(float(tilt_deg))
    az = math.radians(float(azimuth_deg))

    # The pvlib surface_azimuth denotes the horizontal projection direction of
    # the panel normal: 0=north, 90=east, 180=south, 270=west. This must remain
    # consistent with the definition used in the optimization computation.
    across = np.array([math.cos(az), -math.sin(az), 0.0])
    normal = np.array([
        math.sin(tilt) * math.sin(az),
        math.sin(tilt) * math.cos(az),
        math.cos(tilt),
    ])
    # The slope is the in-plane vector along the facing direction; together with
    # across it spans the panel plane.
    slope = np.cross(normal, across)
    center = np.array([0.0, 0.0, 0.10])
    pts = []
    for px, py, _ in local:
        pts.append(center + across * px + slope * py)
    pts = np.asarray(pts)
    return pts[:, 0], pts[:, 1], pts[:, 2]


def render_best_panel_scene(background_path: Path, out_path: Path, best: dict[str, Any],
                            projection_model: str = "equisolid") -> None:
    """Render a 3D hemisphere fisheye image plus a 3D PV panel pose PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    img = _load_rgb_image(background_path)
    x, y, z, colors = _hemisphere_mesh(img, projection_model=projection_model)

    fig = plt.figure(figsize=(10, 9), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors,
                    linewidth=0, antialiased=False, shade=False, alpha=0.96)

    px, py, pz = _panel_mesh(float(best["tilt_deg"]), float(best["azimuth_deg"]))
    verts = [list(zip(px, py, pz))]
    panel = Poly3DCollection(verts, facecolors=(0.05, 0.22, 0.52, 0.88),
                             edgecolors=(0.0, 0.0, 0.0, 1.0), linewidths=1.8)
    ax.add_collection3d(panel)

    # In-plane grid lines to improve the readability of the 3D panel.
    for t in np.linspace(0.2, 0.8, 4):
        p1 = np.array([px[0], py[0], pz[0]]) * (1 - t) + np.array([px[3], py[3], pz[3]]) * t
        p2 = np.array([px[1], py[1], pz[1]]) * (1 - t) + np.array([px[2], py[2], pz[2]]) * t
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="white", linewidth=1.0)

    ax.text2D(
        0.03,
        0.94,
        "\n".join([
            "3D fisheye hemisphere and optimal PV panel pose",
            f"Optimal tilt angle: {float(best['tilt_deg']):.1f}\u00b0",
            f"Optimal azimuth angle: {float(best['azimuth_deg']):.1f}\u00b0",
            f"P10 daily PV yield: {float(best['daily_p10']):.3f}",
            f"Annual total PV yield: {float(best['annual_total']):.3f}",
        ]),
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="0.4"),
    )

    ax.text(0, 1.08, 0.03, "N", fontsize=12, ha="center")
    ax.text(1.08, 0, 0.03, "E", fontsize=12, ha="center")
    ax.text(0, -1.08, 0.03, "S", fontsize=12, ha="center")
    ax.text(-1.08, 0, 0.03, "W", fontsize=12, ha="center")

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_zlim(0.0, 1.05)
    ax.set_box_aspect((1, 1, 0.65))
    ax.view_init(elev=24, azim=-48)
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _panel_frame_for_mechanical_scene(tilt_deg: float, azimuth_deg: float,
                                      center: np.ndarray,
                                      width: float = 1.35,
                                      height: float = 0.82) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return panel corners, normal, width axis and slope axis for an engineering view."""
    tilt = math.radians(float(tilt_deg))
    az = math.radians(float(azimuth_deg) % 360.0)
    normal = np.array([
        math.sin(tilt) * math.sin(az),
        math.sin(tilt) * math.cos(az),
        math.cos(tilt),
    ], dtype=float)
    width_axis = np.array([math.cos(az), -math.sin(az), 0.0], dtype=float)
    slope_axis = np.cross(normal, width_axis)
    slope_axis = slope_axis / max(np.linalg.norm(slope_axis), 1e-9)

    corners = np.array([
        center - width_axis * width / 2 - slope_axis * height / 2,
        center + width_axis * width / 2 - slope_axis * height / 2,
        center + width_axis * width / 2 + slope_axis * height / 2,
        center - width_axis * width / 2 + slope_axis * height / 2,
    ])
    return corners, normal, width_axis, slope_axis


def _draw_3d_arc(ax, points: np.ndarray, color: str, linewidth: float = 1.8) -> None:
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth)


def _arc_arrow(ax, p1: np.ndarray, p2: np.ndarray, color: str) -> None:
    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm <= 1e-9:
        return
    ax.quiver(
        p1[0], p1[1], p1[2],
        direction[0], direction[1], direction[2],
        length=1.0,
        normalize=False,
        arrow_length_ratio=0.55,
        color=color,
        linewidth=1.4,
    )


def render_best_panel_scene(background_path: Path, out_path: Path, best: dict[str, Any],
                            projection_model: str = "equisolid") -> None:
    """Render a precise orthographic axonometric diagram of the optimal panel pose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    tilt = float(best["tilt_deg"])
    azimuth = float(best["azimuth_deg"]) % 360.0
    pole_top = np.array([0.0, 0.0, 0.95], dtype=float)
    panel_center = pole_top + np.array([0.0, 0.0, 0.10], dtype=float)
    corners, normal, width_axis, slope_axis = _panel_frame_for_mechanical_scene(
        tilt, azimuth, panel_center
    )

    fig = plt.figure(figsize=(10.5, 8.5), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    fig.patch.set_facecolor("#f6f7f8")
    ax.set_facecolor("#f6f7f8")

    grid_color = "#b8bec6"
    axis_color = "#30343b"
    measure_color = "#d46b08"
    panel_edge = "#103a5c"
    panel_face = "#5aa6c8"

    # Ground grid, drawn as a mechanical drafting reference plane.
    extent = 1.8
    ticks = np.arange(-extent, extent + 1e-9, 0.3)
    for x in ticks:
        lw = 1.1 if abs(x) < 1e-9 else 0.45
        ax.plot([x, x], [-extent, extent], [0, 0], color=grid_color, linewidth=lw, alpha=0.88)
    for y in ticks:
        lw = 1.1 if abs(y) < 1e-9 else 0.45
        ax.plot([-extent, extent], [y, y], [0, 0], color=grid_color, linewidth=lw, alpha=0.88)

    # Cardinal axes: X east, Y north, Z up.
    ax.quiver(0, 0, 0, 1.45, 0, 0, color=axis_color, linewidth=1.4, arrow_length_ratio=0.06)
    ax.quiver(0, 0, 0, 0, 1.45, 0, color=axis_color, linewidth=1.4, arrow_length_ratio=0.06)
    ax.quiver(0, 0, 0, 0, 0, 1.45, color=axis_color, linewidth=1.4, arrow_length_ratio=0.06)
    ax.text(1.55, 0, 0.02, "E", color=axis_color, fontsize=10, ha="center")
    ax.text(0, 1.55, 0.02, "N", color=axis_color, fontsize=10, ha="center")
    ax.text(0, 0, 1.55, "Z", color=axis_color, fontsize=10, ha="center")

    # Central support pole and small hub.
    ax.plot([0, 0], [0, 0], [0, pole_top[2]], color="#20242a", linewidth=4.0, solid_capstyle="round")
    ax.scatter([0], [0], [pole_top[2]], color="#20242a", s=34, depthshade=False)

    # Panel face, border, and fine mechanical grid lines.
    panel = Poly3DCollection(
        [corners],
        facecolors=(90 / 255, 166 / 255, 200 / 255, 0.72),
        edgecolors=panel_edge,
        linewidths=1.8,
    )
    ax.add_collection3d(panel)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=panel_edge, linewidth=2.1)
    for t in np.linspace(-0.3, 0.3, 3):
        p1 = panel_center + slope_axis * t - width_axis * 0.675
        p2 = panel_center + slope_axis * t + width_axis * 0.675
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#e8f4f8", linewidth=0.9)
    for t in np.linspace(-0.45, 0.45, 4):
        p1 = panel_center + width_axis * t - slope_axis * 0.41
        p2 = panel_center + width_axis * t + slope_axis * 0.41
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#e8f4f8", linewidth=0.9)

    # Normal vector and horizontal projection.
    n_end = panel_center + normal * 0.72
    h = np.array([normal[0], normal[1], 0.0], dtype=float)
    if np.linalg.norm(h) > 1e-9:
        h = h / np.linalg.norm(h)
    else:
        h = np.array([0.0, 1.0, 0.0])
    h_end = panel_center + h * 0.78
    ax.quiver(
        panel_center[0], panel_center[1], panel_center[2],
        normal[0], normal[1], normal[2],
        length=0.72,
        normalize=True,
        color="#0f5c8c",
        linewidth=2.0,
        arrow_length_ratio=0.12,
    )
    ax.plot(
        [panel_center[0], h_end[0]],
        [panel_center[1], h_end[1]],
        [panel_center[2], h_end[2]],
        color="#0f5c8c",
        linewidth=1.2,
        linestyle="--",
    )

    # Azimuth angle arc on ground, measured clockwise from north to the panel normal projection.
    az_rad = math.radians(azimuth)
    az_steps = np.linspace(0.0, az_rad, 80)
    r_az = 0.62
    az_arc = np.column_stack([
        r_az * np.sin(az_steps),
        r_az * np.cos(az_steps),
        np.full_like(az_steps, 0.035),
    ])
    _draw_3d_arc(ax, az_arc, measure_color, 2.0)
    _arc_arrow(ax, az_arc[-3], az_arc[-1], measure_color)
    mid = az_arc[len(az_arc) // 2]
    ax.text(mid[0], mid[1], mid[2] + 0.05, f"Azimuth {azimuth:.1f}\u00b0", color=measure_color, fontsize=10, ha="center")

    # Tilt angle arc in the vertical plane of the panel normal.
    tilt_steps = np.linspace(0.0, math.radians(tilt), 70)
    r_tilt = 0.44
    tilt_arc = np.column_stack([
        panel_center[0] + r_tilt * np.sin(tilt_steps) * h[0],
        panel_center[1] + r_tilt * np.sin(tilt_steps) * h[1],
        panel_center[2] + r_tilt * np.cos(tilt_steps),
    ])
    _draw_3d_arc(ax, tilt_arc, measure_color, 2.0)
    _arc_arrow(ax, tilt_arc[-3], tilt_arc[-1], measure_color)
    tm = tilt_arc[len(tilt_arc) // 2]
    ax.text(tm[0], tm[1], tm[2] + 0.04, f"Tilt {tilt:.1f}\u00b0", color=measure_color, fontsize=10, ha="center")

    # A few restrained annotations, like a technical plate.
    ax.text2D(
        0.03, 0.94,
        "Optimal PV Panel Pose  Orthographic Axonometric",
        transform=ax.transAxes,
        fontsize=13,
        color="#1f252c",
        weight="bold",
    )
    ax.text2D(
        0.03, 0.89,
        f"Tilt {tilt:.1f}\u00b0   Azimuth {azimuth:.1f}\u00b0   P10 {float(best.get('daily_p10', 0.0)):.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color="#3a414a",
    )

    ax.view_init(elev=28, azim=-42)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_zlim(0.0, 1.75)
    ax.set_box_aspect((1, 1, 0.72))
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_interactive_sky_dome_html(background_path: Path, html_path: Path, best: dict[str, Any],
                                     projection_model: str = "equisolid",
                                     sun_path: list[dict[str, float]] | None = None) -> None:
    """Render a draggable Three.js 3D sky dome HTML.

    Outputs:
    - interactive_sky_dome.html: the interactive model page.
    - fisheye_texture.png: backup of the original fisheye texture.
    - fisheye_sky_texture.png: transparent texture containing only the sky
      region, controlled by the HTML slider.
    - fisheye_obstacle_texture.png: texture containing only non-sky regions;
      buildings/vegetation/other stay opaque.

    Interactions:
    - Left-button drag rotates the view.
    - Mouse wheel zooms.
    - Right-button pans.
    - Clicking "Reset view" restores the default camera.
    """
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path = html_path.parent / "fisheye_texture.png"
    shutil.copyfile(background_path, texture_path)
    sky_texture_path, obstacle_texture_path, has_sky_mask = _make_sky_and_obstacle_textures(
        background_path,
        html_path.parent,
    )

    payload = {
        "skyTexture": _file_data_uri(sky_texture_path, "image/png"),
        "obstacleTexture": _file_data_uri(obstacle_texture_path, "image/png"),
        "hasSkyMask": bool(has_sky_mask),
        "projectionModel": projection_model,
        "tiltDeg": float(best["tilt_deg"]),
        "azimuthDeg": float(best["azimuth_deg"]),
        "dailyP10": float(best.get("daily_p10", 0.0)),
        "annualTotal": float(best.get("annual_total", 0.0)),
        "panelCount": int(best.get("panel_count", 1)),
        "modulePowerW": float(best.get("module_p_stc_w", 0.0)),
        "prTotal": float(best.get("pr_total", 0.0)),
        "pvTech": str(best.get("pv_tech", "")),
        "sunPath": sun_path or [],
    }
    payload_js = json.dumps(payload, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive 3D Sky Dome and PV Panel Pose</title>
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #0b1020;
      font-family: "Segoe UI", Arial, "Helvetica Neue", sans-serif;
    }}
    #scene {{
      position: fixed;
      inset: 0;
    }}
    #panel {{
      position: fixed;
      top: 18px;
      right: 18px;
      width: 320px;
      color: #f7fbff;
      background: rgba(8, 14, 28, 0.78);
      border: 1px solid rgba(255,255,255,0.22);
      box-shadow: 0 14px 40px rgba(0,0,0,0.35);
      border-radius: 8px;
      padding: 14px 16px;
      line-height: 1.55;
      backdrop-filter: blur(8px);
    }}
    #panel h1 {{
      margin: 0 0 10px;
      font-size: 17px;
      font-weight: 700;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      border-top: 1px solid rgba(255,255,255,0.12);
      padding-top: 6px;
      margin-top: 6px;
    }}
    .label {{
      color: #b9c8df;
    }}
    .value {{
      color: #ffffff;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    #reset {{
      margin-top: 12px;
      width: 100%;
      height: 34px;
      border: 1px solid rgba(255,255,255,0.28);
      background: rgba(255,255,255,0.08);
      color: white;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
    }}
    .slider-row {{
      margin-top: 10px;
      font-size: 12px;
      color: #dbe7f7;
    }}
    #opacity {{
      width: 100%;
      margin-top: 6px;
    }}
    #hint {{
      position: fixed;
      left: 18px;
      bottom: 16px;
      color: rgba(255,255,255,0.78);
      background: rgba(8, 14, 28, 0.58);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
    }}
    #error {{
      position: fixed;
      left: 18px;
      top: 18px;
      max-width: 520px;
      color: #fff4f4;
      background: rgba(130, 24, 24, 0.86);
      border: 1px solid rgba(255,255,255,0.25);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 13px;
      line-height: 1.55;
      display: none;
      z-index: 10;
    }}
  </style>
</head>
<body>
  <div id="scene"></div>
  <div id="error"></div>
  <aside id="panel">
    <h1>Optimal PV Panel Pose</h1>
    <div class="row"><span class="label">Optimal tilt angle</span><span class="value" id="tilt"></span></div>
    <div class="row"><span class="label">Optimal azimuth angle</span><span class="value" id="azimuth"></span></div>
    <div class="row"><span class="label">P10 daily PV yield</span><span class="value" id="p10"></span></div>
    <div class="row"><span class="label">Annual total PV yield</span><span class="value" id="annual"></span></div>
    <div class="row"><span class="label">Number of modules</span><span class="value" id="count"></span></div>
    <div class="row"><span class="label">Module rated power</span><span class="value" id="power"></span></div>
    <div class="row"><span class="label">System performance ratio (PR)</span><span class="value" id="pr"></span></div>
    <div class="row"><span class="label">Module type</span><span class="value" id="tech"></span></div>
    <div class="slider-row">
      Sky opacity
      <input id="opacity" type="range" min="0.12" max="0.92" step="0.02" value="0.42" />
    </div>
    <button id="reset">Reset view</button>
  </aside>
  <div id="hint">Left-drag to rotate, scroll to zoom, right-drag to pan. Sky opacity is adjustable; buildings, vegetation and other obstructions remain opaque.</div>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }}
  }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

    const DATA = {payload_js};
    const errorBox = document.getElementById("error");
    window.addEventListener("error", (event) => {{
      errorBox.style.display = "block";
      errorBox.textContent = "Failed to load the 3D model: " + event.message + ". If module loading fails, make sure this computer can reach the Three.js CDN, or switch to an offline vendor build.";
    }});
    const container = document.getElementById("scene");

    document.getElementById("tilt").textContent = `${{DATA.tiltDeg.toFixed(1)}}°`;
    document.getElementById("azimuth").textContent = `${{DATA.azimuthDeg.toFixed(1)}}°`;
    document.getElementById("p10").textContent = `${{DATA.dailyP10.toFixed(3)}} kWh`;
    document.getElementById("annual").textContent = `${{DATA.annualTotal.toFixed(3)}} kWh`;
    document.getElementById("count").textContent = `${{DATA.panelCount}}`;
    document.getElementById("power").textContent = `${{DATA.modulePowerW.toFixed(1)}} W`;
    document.getElementById("pr").textContent = `${{DATA.prTotal.toFixed(2)}}`;
    document.getElementById("tech").textContent = DATA.pvTech || "-";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);

    const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.01, 50);
    const defaultCameraPosition = new THREE.Vector3(2.35, -3.05, 1.75);
    camera.position.copy(defaultCameraPosition);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.45);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.9;
    controls.maxDistance = 7.0;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x243050, 1.5));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(2, -3, 5);
    scene.add(dirLight);

    function buildSkyDomeGeometry(radialSegments = 192, heightSegments = 72) {{
      const radius = 1.8;
      const positions = [];
      const uvs = [];
      const indices = [];
      for (let iy = 0; iy <= heightSegments; iy++) {{
        const alt = (iy / heightSegments) * Math.PI / 2.0;
        const theta = Math.PI / 2.0 - alt;
        let rho;
        if (DATA.projectionModel === "equidistant") {{
          rho = theta / (Math.PI / 2.0) * 0.5;
        }} else {{
          rho = Math.SQRT2 * 0.5 * Math.sin(theta / 2.0);
        }}
        for (let ix = 0; ix <= radialSegments; ix++) {{
          const az = (ix / radialSegments) * Math.PI * 2.0;
          const x = radius * Math.cos(alt) * Math.sin(az);
          const y = radius * Math.cos(alt) * Math.cos(az);
          const z = radius * Math.sin(alt);
          positions.push(x, y, z);
          const uu = 0.5 + rho * Math.sin(az);
          const vv = 0.5 - rho * Math.cos(az);
          uvs.push(uu, vv);
        }}
      }}
      for (let iy = 0; iy < heightSegments; iy++) {{
        for (let ix = 0; ix < radialSegments; ix++) {{
          const a = iy * (radialSegments + 1) + ix;
          const b = a + radialSegments + 1;
          const c = b + 1;
          const d = a + 1;
          indices.push(a, b, d);
          indices.push(b, c, d);
        }}
      }}
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      return geometry;
    }}

    async function loadTexture(src) {{
      const tex = await new THREE.TextureLoader().loadAsync(src);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      return tex;
    }}

    const domeGeometry = buildSkyDomeGeometry();
    const skyTexture = await loadTexture(DATA.skyTexture);
    const obstacleTexture = await loadTexture(DATA.obstacleTexture);

    const skyMaterial = new THREE.MeshBasicMaterial({{
      map: skyTexture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.42,
      alphaTest: 0.01,
      depthWrite: false
    }});
    const obstacleMaterial = new THREE.MeshBasicMaterial({{
      map: obstacleTexture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 1.0,
      alphaTest: 0.01,
      depthWrite: false
    }});

    const skyDome = new THREE.Mesh(domeGeometry, skyMaterial);
    skyDome.renderOrder = 1;
    scene.add(skyDome);

    const obstacleDome = new THREE.Mesh(domeGeometry, obstacleMaterial);
    obstacleDome.renderOrder = 2;
    scene.add(obstacleDome);

    const ground = new THREE.GridHelper(3.8, 24, 0x6f7f99, 0x293348);
    ground.position.z = -0.005;
    ground.material.transparent = true;
    ground.material.opacity = 0.38;
    scene.add(ground);

    function makeTextSprite(text, position, color = "#ffffff") {{
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 128;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "bold 54px Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 8;
      ctx.strokeStyle = "rgba(0,0,0,0.72)";
      ctx.fillStyle = color;
      ctx.strokeText(text, 128, 64);
      ctx.fillText(text, 128, 64);
      const tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true }});
      const sprite = new THREE.Sprite(mat);
      sprite.position.copy(position);
      sprite.scale.set(0.26, 0.13, 1);
      return sprite;
    }}

    scene.add(makeTextSprite("N", new THREE.Vector3(0, 1.98, 0.08)));
    scene.add(makeTextSprite("E", new THREE.Vector3(1.98, 0, 0.08)));
    scene.add(makeTextSprite("S", new THREE.Vector3(0, -1.98, 0.08)));
    scene.add(makeTextSprite("W", new THREE.Vector3(-1.98, 0, 0.08)));

    function sunPointToVector(altDeg, azDeg, radius = 1.84) {{
      const alt = THREE.MathUtils.degToRad(Math.max(0, Math.min(90, altDeg)));
      const az = THREE.MathUtils.degToRad(azDeg);
      return new THREE.Vector3(
        radius * Math.cos(alt) * Math.sin(az),
        radius * Math.cos(alt) * Math.cos(az),
        radius * Math.sin(alt)
      );
    }}

    function buildSunPath() {{
      const pts = DATA.sunPath
        .filter(p => Number.isFinite(p.altitude_deg) && p.altitude_deg > 0)
        .map(p => sunPointToVector(p.altitude_deg, p.azimuth_deg));
      if (pts.length < 2) return;
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(
        geom,
        new THREE.LineBasicMaterial({{ color: 0xffd84d, linewidth: 3, transparent: true, opacity: 0.98 }})
      );
      line.renderOrder = 4;
      scene.add(line);

      const markerGeom = new THREE.SphereGeometry(0.035, 16, 12);
      const markerMat = new THREE.MeshStandardMaterial({{ color: 0xffef85, emissive: 0xffc928, emissiveIntensity: 0.65 }});
      const first = new THREE.Mesh(markerGeom, markerMat);
      first.position.copy(pts[0]);
      scene.add(first);
      const last = new THREE.Mesh(markerGeom, markerMat);
      last.position.copy(pts[pts.length - 1]);
      scene.add(last);
      scene.add(makeTextSprite("Solar trajectory", pts[Math.floor(pts.length / 2)].clone().multiplyScalar(1.03), "#ffeb7a"));
    }}
    buildSunPath();

    function buildPanel() {{
      const group = new THREE.Group();
      const width = 0.62;
      const height = 0.38;
      const thickness = 0.025;
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, thickness),
        [
          new THREE.MeshStandardMaterial({{ color: 0x0c244d, roughness: 0.55, metalness: 0.08 }}),
          new THREE.MeshStandardMaterial({{ color: 0x0c244d, roughness: 0.55, metalness: 0.08 }}),
          new THREE.MeshStandardMaterial({{ color: 0x163b78, roughness: 0.42, metalness: 0.12 }}),
          new THREE.MeshStandardMaterial({{ color: 0x08172f, roughness: 0.6, metalness: 0.04 }}),
          new THREE.MeshStandardMaterial({{ color: 0x0b1b35, roughness: 0.6, metalness: 0.04 }}),
          new THREE.MeshStandardMaterial({{ color: 0x12305f, roughness: 0.5, metalness: 0.08 }})
        ]
      );
      group.add(panel);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(panel.geometry),
        new THREE.LineBasicMaterial({{ color: 0xe9f3ff }})
      );
      group.add(edges);

      for (let i = -2; i <= 2; i++) {{
        const lineGeom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(i * width / 6, -height / 2, thickness / 2 + 0.002),
          new THREE.Vector3(i * width / 6, height / 2, thickness / 2 + 0.002)
        ]);
        group.add(new THREE.Line(lineGeom, new THREE.LineBasicMaterial({{ color: 0xcfe3ff }})));
      }}
      const tilt = THREE.MathUtils.degToRad(DATA.tiltDeg);
      const az = THREE.MathUtils.degToRad(DATA.azimuthDeg);
      // Consistent with pvlib surface_azimuth: 0=north, 90=east, 180=south,
      // 270=west; this vector denotes the horizontal projection direction of
      // the panel normal.
      const normal = new THREE.Vector3(
        Math.sin(tilt) * Math.sin(az),
        Math.sin(tilt) * Math.cos(az),
        Math.cos(tilt)
      ).normalize();
      const defaultNormal = new THREE.Vector3(0, 0, 1);
      group.quaternion.setFromUnitVectors(defaultNormal, normal);
      group.position.set(0, 0, 0.16);

      const arrow = new THREE.ArrowHelper(normal, new THREE.Vector3(0, 0, 0.24), 0.38, 0xffdd66, 0.08, 0.04);
      group.add(arrow);
      return group;
    }}
    scene.add(buildPanel());
    document.getElementById("opacity").addEventListener("input", (event) => {{
      skyMaterial.opacity = Number(event.target.value);
    }});

    document.getElementById("reset").addEventListener("click", () => {{
      camera.position.copy(defaultCameraPosition);
      controls.target.set(0, 0, 0.45);
      controls.update();
    }});

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
