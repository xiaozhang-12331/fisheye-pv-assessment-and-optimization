# -*- coding: utf-8 -*-
"""Refactored master pipeline.

Functional sections:
1. Discover the input image and prepare output directories.
2. Load the algorithm modules of the refactored pipeline.
3. Inject the centralized configuration into each functional module.
4. Execute sequentially: semantic segmentation -> solar trajectory
   occlusion -> radiation computation.

This file does not alter the base computation modes; it only performs
orchestration and parameter management.
"""

from __future__ import annotations

import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from config import AppConfig, build_config


# =========================================================
# Runtime path objects
# =========================================================
@dataclass
class RunPaths:
    image_name: str
    result_root: Path
    segmentation_dir: Path
    sunpath_dir: Path
    radiation_dir: Path
    optimization_dir: Path
    pv_best_panel_dir: Path


# =========================================================
# General utilities
# =========================================================
def print_title(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def format_seconds(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def find_first_image(input_dir: Path) -> Path:
    """Find the first image in the input directory.

    This keeps the legacy master-pipeline behavior: the measurement point
    name is taken from the stem of the first image. If batch processing of
    multiple measurement points is needed later, extend from here without
    touching steps 01/02/03.
    """
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
            "*.bmp", "*.BMP", "*.tif", "*.TIF", "*.tiff", "*.TIFF")
    images: list[Path] = []
    for pat in exts:
        images.extend(input_dir.glob(pat))
    if not images:
        raise FileNotFoundError(f"No image found in the input directory: {input_dir}")
    return sorted(images)[0]


def prepare_run_paths(cfg: AppConfig) -> RunPaths:
    """Build output paths under the legacy directory layout from the image name.

    Only paths are computed here; no directory is created or deleted. This
    keeps lightweight actions such as import tests or optimization
    pre-checks free of filesystem side effects.
    """
    image_path = find_first_image(cfg.paths.input_dir)
    image_name = image_path.stem

    result_root = cfg.paths.output_root_dir / image_name
    return RunPaths(
        image_name=image_name,
        result_root=result_root,
        segmentation_dir=result_root / "segmentation",
        sunpath_dir=result_root / "sunpath",
        radiation_dir=result_root / "radiation",
        optimization_dir=result_root / "optimization",
        pv_best_panel_dir=result_root / "pv_best_panel",
    )


def prepare_output_dirs(cfg: AppConfig, paths: RunPaths) -> None:
    """Prepare output directories according to the run configuration.

    Deletion of previous results happens only inside the formal main() call,
    preventing accidental removal from plain imports or path computations.
    """
    if cfg.run.overwrite and paths.result_root.exists():
        print(f"[overwrite] Removing previous results: {paths.result_root}")
        shutil.rmtree(paths.result_root)

    paths.segmentation_dir.mkdir(parents=True, exist_ok=True)
    paths.sunpath_dir.mkdir(parents=True, exist_ok=True)
    paths.radiation_dir.mkdir(parents=True, exist_ok=True)
    paths.pv_best_panel_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def isolated_argv(script_path: Path):
    """Prevent the segmentation module's argparse from misreading pipeline CLI args."""
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)]
    try:
        yield
    finally:
        sys.argv = old_argv


# =========================================================
# Configuration injection: 01 image recognition
# =========================================================
def configure_segmentation_module(seg, cfg: AppConfig, paths: RunPaths) -> None:
    seg.IN_DIR = str(cfg.paths.input_dir)
    seg.OUT_DIR = str(paths.segmentation_dir)
    seg.ADE_MODEL_DIR = str(cfg.paths.ade_model_dir)
    seg.BIN_BASE_MODEL_DIR = str(cfg.paths.binary_base_model_dir)
    seg.BIN_CKPT_PATH = str(cfg.paths.binary_ckpt_path)

    seg.FP16 = cfg.segmentation.fp16
    seg.TILE = cfg.segmentation.tile
    seg.OVERLAP = cfg.segmentation.overlap
    seg.SCALES = cfg.segmentation.scales
    seg.USE_TTA_HFLIP = cfg.segmentation.use_tta_hflip
    seg.BIN_INFER_SIZE = cfg.segmentation.bin_infer_size

    seg.ENABLE_ROTATION = cfg.segmentation.enable_rotation
    seg.IMAGE_TOP_AZIMUTH = cfg.image.top_azimuth_deg
    seg.INNER_MARGIN = cfg.segmentation.inner_margin
    seg.FORCE_CENTER = cfg.segmentation.force_center
    seg.CENTER_BIAS_X = cfg.segmentation.center_bias_x
    seg.CENTER_BIAS_Y = cfg.segmentation.center_bias_y
    seg.SAVE_SOFT_NPZ = cfg.segmentation.save_soft_npz
    seg.KEEP_OTHER_CLASS = cfg.segmentation.keep_other_class


# =========================================================
# Configuration injection: 02 trajectory overlay and occlusion detection
# =========================================================
def configure_sunpath_module(sun, cfg: AppConfig, paths: RunPaths) -> None:
    sun.CFG.POINT_DIR = paths.segmentation_dir
    sun.CFG.OUTPUT_ROOT = paths.sunpath_dir
    sun.CFG.IMAGE_NAME = paths.image_name
    sun.CFG.SEG_SUBDIR = ""

    sun.CFG.LAT = cfg.site.lat
    sun.CFG.LON = cfg.site.lon
    sun.CFG.ALT_M = cfg.site.alt_m
    sun.CFG.TZ = cfg.site.tz
    sun.CFG.SINGLE_DATE = cfg.run.single_date
    sun.CFG.TIME_STEP_MIN = cfg.run.time_step_min
    sun.CFG.DRAW_FREQ_MIN = cfg.run.draw_freq_min

    sun.CFG.PROJECTION_MODEL = cfg.sunpath.projection_model
    sun.CFG.IMG_TOP_AZIMUTH_DEG = cfg.image.top_azimuth_deg
    sun.CFG.MIRROR_AZIMUTH = cfg.image.mirror_azimuth
    sun.CFG.MIN_ALTITUDE_DEG = cfg.sunpath.min_altitude_deg
    sun.CFG.MAX_JUMP_PX = cfg.sunpath.max_jump_px
    sun.CFG.KEEP_OTHER_CLASS = cfg.segmentation.keep_other_class
    sun.CFG.OCCLUDED_CLASSES = (
        {"building", "vegetation", "other"}
        if cfg.segmentation.keep_other_class
        else {"building", "vegetation"}
    )

    sun.CFG.SAVE_CSV = cfg.sunpath.save_csv
    sun.CFG.SAVE_IMAGE = cfg.sunpath.save_image
    sun.CFG.CROSS_ENABLED = cfg.sunpath.cross_enabled
    sun.CFG.NESW_ENABLED = cfg.sunpath.nesw_enabled
    sun.CFG.LEGEND_ENABLED = cfg.sunpath.legend_enabled


# =========================================================
# Configuration injection: 03 radiation computation
# =========================================================
def configure_radiation_module(rad, cfg: AppConfig, paths: RunPaths) -> None:
    rad.CFG.OCCLUSION_DIR = paths.sunpath_dir / paths.image_name
    rad.CFG.OUTPUT_ROOT = paths.radiation_dir
    rad.CFG.SEG_MASK_DIR = paths.segmentation_dir

    rad.CFG.SINGLE_DATE = cfg.run.single_date
    rad.CFG.LAT = cfg.site.lat
    rad.CFG.LON = cfg.site.lon
    rad.CFG.ALT_M = cfg.site.alt_m
    rad.CFG.TZ = cfg.site.tz

    rad.CFG.SURF_TILT_DEG = cfg.surface.tilt_deg
    rad.CFG.SURF_AZIMUTH_DEG = cfg.surface.azimuth_deg
    rad.CFG.ALBEDO = cfg.surface.albedo

    rad.CFG.CLIMATE_TYPE = cfg.radiation.climate_type
    rad.CFG.LINK_TURBIDITY = cfg.radiation.link_turbidity
    rad.CFG.CLEARSKY_BIAS = cfg.radiation.clearsky_bias
    rad.CFG.ENABLE_LOW_SUN_CORRECTION = cfg.radiation.enable_low_sun_correction
    rad.CFG.LOW_SUN_FACTOR = cfg.radiation.low_sun_factor
    rad.CFG.KEEP_OTHER_CLASS = cfg.segmentation.keep_other_class


# =========================================================
# Three-step pipeline
# =========================================================
def run_segmentation(cfg: AppConfig, paths: RunPaths) -> None:
    print_title("1/3 Semantic segmentation")
    from modules import segmentation as seg

    script_path = Path(seg.__file__)
    configure_segmentation_module(seg, cfg, paths)
    with isolated_argv(script_path):
        seg.main()


def run_sunpath(cfg: AppConfig, paths: RunPaths) -> None:
    print_title("2/3 Solar trajectory and occlusion detection")
    from modules import sunpath_occlusion as sun

    configure_sunpath_module(sun, cfg, paths)
    sun.main()


def run_radiation(cfg: AppConfig, paths: RunPaths) -> None:
    print_title("3/3 Radiation computation")
    from modules import radiation as rad

    configure_radiation_module(rad, cfg, paths)
    rad.main()


def run_optimization(cfg: AppConfig, paths: RunPaths) -> None:
    print_title("4/4 Annual fixed tilt/azimuth orientation optimization")
    from modules import optimizer

    optimizer.run_optimization(cfg, paths)


def main() -> None:
    start = time.time()
    cfg = build_config()

    print_title("PV sky view factor analysis system (refactored pipeline)")
    paths = prepare_run_paths(cfg)
    prepare_output_dirs(cfg, paths)

    print(f"[Input image] {paths.image_name}")
    print(f"[Computation date] {cfg.run.single_date}")
    print(f"[Results directory] {paths.result_root}")

    run_segmentation(cfg, paths)
    run_sunpath(cfg, paths)
    run_radiation(cfg, paths)
    if cfg.optimization.enabled:
        run_optimization(cfg, paths)

    print_title("All tasks completed")
    print(f"Total elapsed time: {format_seconds(time.time() - start)}")
    print(f"Output directory: {paths.result_root}")


if __name__ == "__main__":
    main()
