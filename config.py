# -*- coding: utf-8 -*-
"""Centralized configuration module.

Usage principles:
1. This file is the highest-priority parameter entry point of the refactored
   pipeline; prefer editing here for daily use.
2. Parameters in the legacy modules are kept as module-internal defaults or
   fine-tuning knobs.
3. Runtime precedence: CENTRAL_OVERRIDES > dataclass configuration in this
   file > legacy script defaults.

To add "tilt/azimuth orientation optimization" in the future, only add an
optimization section and an optimizer module; do not modify the base
computation logic of steps 01/02/03.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =========================================================
# Highest-priority override section
# =========================================================
# For quick parameter trials, edit only this section. Keys use the
# "section.field" convention, e.g.:
# CENTRAL_OVERRIDES = {
#     "run.single_date": "2026-05-08",
#     "site.lat": 38.8825,
#     "surface.tilt_deg": 30.0,
#     "surface.azimuth_deg": 180.0,
#     "pv_panel.panel_count": 2,
# }
CENTRAL_OVERRIDES: dict[str, Any] = {}


# =========================================================
# Path configuration
# =========================================================
@dataclass
class PathConfig:
    # Project root: location of legacy scripts, models, inputs, and results.
    project_dir: Path = Path(r"D:\projects\image_recog")

    # Input image directory. The pipeline currently takes the first image in
    # the directory as the measurement point name.
    input_dir: Path = Path(r"D:\projects\image_recog\in")

    # Output root keeps the legacy directory layout:
    # results/<image_name>/segmentation|sunpath|radiation.
    output_root_dir: Path = Path(r"D:\projects\image_recog\results")

    # Annual TMY per-minute data. The optimization module prefers the
    # GHI/DNI/DHI columns from this file.
    tmy_minute_csv: Path = Path(r"D:\projects\image_recog\radiation_out\TMY_Baoding_minute.csv")

    # Model paths.
    ade_model_dir: Path = Path(r"D:\projects\image_recog\models\nvidia_segformer_b4_ade")
    binary_base_model_dir: Path = Path(r"D:\projects\image_recog\models\nvidia_segformer_b4_ade")
    binary_ckpt_path: Path = Path(r"D:\projects\image_recog\models\ckpts\best_segformer_b4_binary.pt")


# =========================================================
# Run configuration
# =========================================================
@dataclass
class RunConfig:
    # Date for the single-day computation; used by both steps 02 and 03.
    single_date: str = "2026-05-08"

    # If True, delete previous results under results/<image_name> before
    # regenerating them.
    overwrite: bool = True

    # Time step of the solar occlusion CSV and the drawing sampling interval
    # of the trajectory plots.
    time_step_min: int = 1
    draw_freq_min: int = 5


# =========================================================
# Geographic and image orientation configuration
# =========================================================
@dataclass
class SiteConfig:
    tz: str = "Asia/Shanghai"
    lat: float = 38.921346
    lon: float = 115.502179
    alt_m: float = 25.0


@dataclass
class ImageDirectionConfig:
    # Azimuth actually pointed to by the top edge of the fisheye image:
    # 0=north, 90=east, 180=south, 270=west.
    top_azimuth_deg: float = 0.0

    # Whether to mirror the azimuth when projecting the solar trajectory.
    # Keeps the legacy script default.
    mirror_azimuth: bool = True


# =========================================================
# Segmentation module configuration
# =========================================================
@dataclass
class SegmentationConfig:
    # Inference precision and speed parameters; keeps the legacy default
    # computation mode.
    fp16: bool = True
    tile: int = 896
    overlap: int = 192
    scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    use_tta_hflip: bool = True
    bin_infer_size: int = 768

    # Fisheye circle detection and rotation correction.
    enable_rotation: bool = True
    inner_margin: float = 0.01
    force_center: bool = True
    center_bias_x: float = 0.0
    center_bias_y: float = 0.0

    # Save soft probability maps for downstream analysis.
    save_soft_npz: bool = True

    # False merges other into building and suppresses standalone other outputs.
    keep_other_class: bool = False


# =========================================================
# Solar trajectory and occlusion module configuration
# =========================================================
@dataclass
class SunpathConfig:
    projection_model: str = "equisolid"
    min_altitude_deg: float = 3.0
    max_jump_px: int = 80

    # Plotting switches.
    save_csv: bool = True
    save_image: bool = True
    cross_enabled: bool = True
    nesw_enabled: bool = True
    legend_enabled: bool = True


# =========================================================
# Radiation module configuration
# =========================================================
@dataclass
class SurfaceConfig:
    # Currently only used as single-shot computation parameters; the future
    # optimization module will sweep these two values in batch.
    tilt_deg: float = 0.0
    azimuth_deg: float = 180.0
    albedo: float = 0.20


@dataclass
class RadiationConfig:
    climate_type: str = "temperate"
    link_turbidity: float = 5.5
    clearsky_bias: float = 0.85
    enable_low_sun_correction: bool = True
    low_sun_factor: float = 0.20


# =========================================================
# PV panel parameter configuration
# =========================================================
@dataclass
class PVPanelConfig:
    # Whether to further convert POA irradiance into PV yield.
    enabled: bool = True

    # The following defaults come from the PV module parameter section of the
    # legacy facade PV potential script (see the legacy Chinese-named script
    # in the project root).
    pv_tech: str = "Monocrystalline_Si"
    module_width_mm: float = 1000.0
    module_height_mm: float = 750.0
    module_p_stc_w: float = 165.0
    pr_total: float = 0.75

    # The refactored pipeline computes effective area as
    # single panel area * panel_count * installable_ratio.
    installable_ratio: float = 1.0
    panel_count: int = 1


# =========================================================
# Reserved: tilt/azimuth orientation optimization configuration
# =========================================================
@dataclass
class OptimizationConfig:
    # When enabled, the pipeline runs the annual fixed tilt/azimuth
    # orientation optimization after the three-step computation.
    enabled: bool = True

    # Candidate angle ranges: (start, end, step). Azimuth convention follows
    # pvlib: 0=north, 90=east, 180=south, 270=west.
    tilt_range_deg: tuple[float, float, float] = (0.0, 90.0, 5.0)
    azimuth_range_deg: tuple[float, float, float] = (0.0, 360.0, 5.0)

    # Fine search step around the best point of the coarse scan. Set to 0 to
    # disable the fine search.
    refine_radius_deg: float = 5.0
    refine_step_deg: float = 1.0

    # Sampling interval of the optimization score. The coarse/fine searches
    # use sampled data for speed; the final output still uses the complete
    # annual per-minute data.
    search_sample_minutes: int = 15

    # After sampled optimization, re-evaluate the top N candidate angle pairs
    # with the full per-minute data.
    final_full_resolution_top_n: int = 10

    # Primary objective: paper equation
    # J = E_ann / E_ann^max - lambda(N_aut) * D_P10^2.
    objective: str = "annual_p10_soft_constraint"
    p10_requirement_ratio: float = 0.90
    storage_autonomy_days: float = 1.0

    # Optional manual path to the annual occlusion CSV; when empty, the
    # fullyear/sun_occlusion CSV is located automatically from the results
    # directory and common legacy output directories.
    occlusion_csv: Path | None = None

    # Export the complete annual per-minute table. The file is large but
    # convenient for later date-based filtering.
    export_minute_power: bool = True

    # Generate the heatmap and a simplified 3D panel schematic on the fisheye
    # image.
    export_visuals: bool = False




@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    run: RunConfig = field(default_factory=RunConfig)
    site: SiteConfig = field(default_factory=SiteConfig)
    image: ImageDirectionConfig = field(default_factory=ImageDirectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    sunpath: SunpathConfig = field(default_factory=SunpathConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    pv_panel: PVPanelConfig = field(default_factory=PVPanelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def build_config() -> AppConfig:
    """Build the final configuration and apply the highest-priority overrides."""
    cfg = AppConfig()
    apply_overrides(cfg, CENTRAL_OVERRIDES)
    return cfg


def apply_overrides(cfg: AppConfig, overrides: dict[str, Any]) -> None:
    """Write CENTRAL_OVERRIDES into the configuration object.

    Keys must use the "section.field" format, e.g. "surface.tilt_deg".
    Invalid keys raise immediately, so silently ineffective parameters are
    avoided.
    """
    for key, value in overrides.items():
        section_name, sep, field_name = key.partition(".")
        if not sep:
            raise ValueError(f"Override key is missing a section: {key}")
        section = getattr(cfg, section_name, None)
        if section is None:
            raise ValueError(f"Unknown configuration section: {section_name}")
        if not hasattr(section, field_name):
            raise ValueError(f"Unknown configuration field: {key}")
        setattr(section, field_name, value)
