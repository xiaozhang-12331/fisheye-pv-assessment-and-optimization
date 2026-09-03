# -*- coding: utf-8 -*-
"""Annual fixed tilt/azimuth orientation optimization module.

Optimization principles:
- The image recognition, occlusion classification, and radiative transfer models are left unchanged.
- Reuses the annual occlusion CSV and the TMY per-minute radiation data.
- The primary objective is the P10 of daily PV yield across the whole year, suitable for
  a "worst-day-first" (daily lower-bound priority) strategy.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from modules import pv_yield, visualization_3d
except ImportError:  # Allow same-directory imports when this file is run directly
    import pv_yield  # type: ignore
    import visualization_3d  # type: ignore


POA_COL = "POA_Total(W/m²)"

_TMY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_YEARLY_SOLAR_CACHE: dict[tuple[int, str, int, float, float], pd.DataFrame] = {}


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"missing required column, tried: {candidates}")


def get_radiation_module():
    """Lazily import radiation so that importing optimizer alone does not require cv2/pvlib."""
    try:
        from modules import radiation
    except ImportError:
        import radiation  # type: ignore
    return radiation


def get_pvlib_module():
    """Lazily import pvlib so that non-radiation parts can be tested without pvlib."""
    import pvlib
    return pvlib


def cfg_flag(cfg: Any, name: str, default: bool) -> bool:
    return bool(getattr(getattr(cfg, "optimization", None), name, default))


# =========================================================
# Input data
# =========================================================
def find_occlusion_csv(cfg: Any, paths: Any) -> Path:
    """Locate the annual occlusion CSV.

    Prefers config.optimization.occlusion_csv; otherwise searches the current
    result directory and common legacy output directories for a CSV whose
    filename contains "fullyear" and that belongs to the current image.
    If no fullyear file exists, falls back to the plain sun_occlusion CSV of
    the current image. Never uses a CSV belonging to another measurement point,
    to avoid mismatching the occlusion environment.
    """
    explicit = cfg.optimization.occlusion_csv
    if explicit:
        explicit = Path(explicit)
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Specified annual occlusion CSV does not exist: {explicit}")

    roots = [
        paths.sunpath_dir,
        paths.result_root,
        cfg.paths.project_dir / "sunpath_out",
        cfg.paths.project_dir / "app4",
        cfg.paths.project_dir / "batch_facade_year_v5" / "sunpath",
    ]
    candidates: list[Path] = []
    for root in roots:
        if Path(root).exists():
            candidates.extend(Path(root).rglob("*.csv"))

    image_name = paths.image_name
    def belongs_to_current_image(p: Path) -> bool:
        stem = p.stem.lower()
        name = image_name.lower()
        parent_names = {part.lower() for part in p.parts}
        return (
            stem == name
            or stem.startswith(f"{name}_")
            or stem.startswith(f"sun_occlusion_{name}_")
            or stem.startswith(f"{name}_sunpath")
            or name in parent_names
        )

    pool = [p for p in candidates if belongs_to_current_image(p)]
    fullyear = [p for p in pool if "fullyear" in p.name.lower()]
    if fullyear:
        return max(fullyear, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"No annual occlusion CSV found for the current image {image_name}. "
        "The program will try to generate one automatically; to use a specific annual file, "
        "set optimization.occlusion_csv"
    )


def build_full_year_occlusion_from_segmentation(cfg: Any, paths: Any, year: int) -> pd.DataFrame:
    """Automatically build a per-minute occlusion series for all 365 days from the current segmentation.

    This allows the orientation optimization to satisfy the "typical meteorological year,
    365-day annual PV yield" requirement even when only a single-day pipeline has been run.
    The function reuses the projection and occlusion classification of sunpath_occlusion
    without altering the occlusion algorithm.
    """
    try:
        from modules import sunpath_occlusion as sun
    except ImportError:
        import sunpath_occlusion as sun  # type: ignore

    sun.CFG.POINT_DIR = paths.segmentation_dir
    sun.CFG.OUTPUT_ROOT = paths.sunpath_dir
    sun.CFG.IMAGE_NAME = paths.image_name
    sun.CFG.SEG_SUBDIR = ""
    sun.CFG.LAT = cfg.site.lat
    sun.CFG.LON = cfg.site.lon
    sun.CFG.ALT_M = cfg.site.alt_m
    sun.CFG.TZ = cfg.site.tz
    sun.CFG.TIME_STEP_MIN = cfg.run.time_step_min
    sun.CFG.PROJECTION_MODEL = cfg.sunpath.projection_model
    sun.CFG.IMG_TOP_AZIMUTH_DEG = cfg.image.top_azimuth_deg
    sun.CFG.MIRROR_AZIMUTH = cfg.image.mirror_azimuth
    sun.CFG.KEEP_OTHER_CLASS = cfg.segmentation.keep_other_class

    overlay, (cx, cy), radius = sun.load_overlay_and_geometry(paths.segmentation_dir, paths.image_name, "")
    masks = sun.load_masks(paths.segmentation_dir, paths.image_name)

    solar = get_yearly_solar_cache(cfg, year)
    h, w = masks["sky"].shape
    u, v = sun.sun_to_pixel(
        solar["altitude_deg"].to_numpy(dtype=float),
        solar["azimuth_deg"].to_numpy(dtype=float),
        cx,
        cy,
        radius,
        cfg.image.top_azimuth_deg,
        cfg.image.mirror_azimuth,
    )
    u = u + float(getattr(cfg.image, "sunpath_x_offset_px", 0.0))
    v = v + float(getattr(cfg.image, "sunpath_y_offset_px", 0.0))
    occ = sun.classify_occlusion(masks, u, v, h, w)
    full = solar.copy()
    full["occlusion"] = occ
    full["u_px"] = u
    full["v_py"] = v
    full.attrs["occlusion_source"] = f"in_memory:{paths.image_name}:{year}"

    if cfg_flag(cfg, "export_occlusion_csv", True):
        out_dir = paths.sunpath_dir / paths.image_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"sun_occlusion_{paths.image_name}_{year}_fullyear.csv"
        write_csv_with_excel_copy(full, out_csv)
        full.attrs["occlusion_source"] = str(out_csv)
        print(f"[optimization] Annual occlusion CSV generated automatically: {out_csv}")
        return load_occlusion_csv(out_csv, cfg.site.tz)

    print(f"[optimization] Annual occlusion series generated: in-memory ({len(full)} rows)")
    result = full[["time", "altitude_deg", "azimuth_deg", "occlusion"]].reset_index(drop=True)
    result.attrs["occlusion_source"] = full.attrs["occlusion_source"]
    return result


def get_yearly_solar_cache(cfg: Any, year: int) -> pd.DataFrame:
    key = (
        int(year),
        str(cfg.site.tz),
        int(cfg.run.time_step_min),
        round(float(cfg.site.lat), 8),
        round(float(cfg.site.lon), 8),
    )
    cached = _YEARLY_SOLAR_CACHE.get(key)
    if cached is not None:
        return cached

    pvlib = get_pvlib_module()
    start = pd.Timestamp(f"{year}-01-01").tz_localize(cfg.site.tz)
    end = pd.Timestamp(f"{year}-12-31").tz_localize(cfg.site.tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=cfg.run.time_step_min)
    times = pd.date_range(start, end, freq=f"{cfg.run.time_step_min}min", tz=cfg.site.tz)
    solpos = pvlib.solarposition.get_solarposition(times, cfg.site.lat, cfg.site.lon)
    solar = pd.DataFrame({
        "date": times.strftime("%Y-%m-%d"),
        "time": times,
        "altitude_deg": solpos["apparent_elevation"].to_numpy(dtype=float),
        "zenith_deg": 90.0 - solpos["apparent_elevation"].to_numpy(dtype=float),
        "azimuth_deg": solpos["azimuth"].to_numpy(dtype=float) % 360.0,
    })
    _YEARLY_SOLAR_CACHE[key] = solar
    print(f"[optimization] Annual solar position cached: {year}, {len(solar)} rows")
    return solar


def load_occlusion_csv(path: Path, tz: str) -> pd.DataFrame:
    """Read an occlusion CSV and normalize it to a table with time/altitude_deg/azimuth_deg/occlusion."""
    df = pd.read_csv(path)
    required = {"altitude_deg", "azimuth_deg", "occlusion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Occlusion CSV is missing columns: {sorted(missing)}")

    out = df.copy()
    if "time" not in out.columns:
        raise ValueError("Occlusion CSV is missing the time column")

    if "date" in out.columns:
        dt = out["date"].astype(str) + " " + out["time"].astype(str)
        out["time"] = pd.to_datetime(dt, errors="coerce")
    else:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")

    if out["time"].dt.tz is None:
        out["time"] = out["time"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        out["time"] = out["time"].dt.tz_convert(tz)

    out = out.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    return out[["time", "altitude_deg", "azimuth_deg", "occlusion"]].reset_index(drop=True)


def normalize_other_to_building(df: pd.DataFrame, keep_other_class: bool) -> pd.DataFrame:
    """Treat legacy other occlusion as building when running in three-class mode."""
    if keep_other_class or "occlusion" not in df.columns:
        return df
    out = df.copy()
    out["occlusion"] = out["occlusion"].astype(str).str.lower().replace({"other": "building"})
    return out


def load_tmy_minute_csv(path: Path, tz: str) -> pd.DataFrame:
    """Read the TMY per-minute data, which must contain at least date/time/GHI/DNI/DHI."""
    path = Path(path)
    cache_key = (str(path.resolve()), str(tz))
    cached = _TMY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        raise FileNotFoundError(f"TMY per-minute file does not exist: {path}")
    df = pd.read_csv(path)
    required = {"date", "time", "GHI", "DNI", "DHI"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TMY file is missing columns: {sorted(missing)}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["date"].astype(str) + " " + out["time"].astype(str), errors="coerce")
    out["time"] = out["time"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    for col in ["GHI", "DNI", "DHI"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    result = out[["time", "GHI", "DNI", "DHI"]].reset_index(drop=True)
    _TMY_CACHE[cache_key] = result
    return result


def infer_tmy_year(weather: pd.DataFrame) -> int:
    """Infer the year from the TMY time column, typically REFERENCE_YEAR_FOR_CALENDAR."""
    years = pd.to_datetime(weather["time"]).dt.year
    return int(years.mode().iloc[0])


def align_weather_and_occlusion(occ: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge occlusion and TMY radiation data on month-day-hour-minute.

    TMY is a typical meteorological year whose calendar year may differ from
    the year of the annual occlusion CSV; therefore merging on the full
    timestamp alone would yield an empty table when the years differ.
    """
    occ2 = occ.copy()
    weather2 = weather.copy()
    occ2["_mdhm"] = pd.to_datetime(occ2["time"]).dt.strftime("%m-%d %H:%M")
    weather2["_mdhm"] = pd.to_datetime(weather2["time"]).dt.strftime("%m-%d %H:%M")
    merged = occ2.merge(
        weather2[["_mdhm", "GHI", "DNI", "DHI"]],
        on="_mdhm",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.drop(columns=["_mdhm"])
    if merged.empty:
        raise ValueError("Merging the occlusion CSV with the TMY data on month-day-hour-minute produced an empty table; please check the time formats")
    return merged.sort_values("time").reset_index(drop=True)


def estimate_dt_hours(times: pd.Series) -> float:
    """Estimate the time step, used by default for per-minute/hourly integration."""
    t = pd.to_datetime(times).sort_values()
    diffs = t.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1.0 / 60.0
    return float(diffs.median() / 3600.0)


def sample_for_search(base: pd.DataFrame, sample_minutes: int) -> pd.DataFrame:
    """Subsample for the coarse/fine searches; the final output still uses the full per-minute data."""
    if sample_minutes <= 1:
        return base
    work = base.copy()
    idx = pd.DatetimeIndex(work["time"])
    mask = (idx.minute % int(sample_minutes) == 0)
    sampled = work.loc[mask].copy()
    if sampled.empty:
        return base
    return sampled.reset_index(drop=True)


# =========================================================
# Radiation candidate computation
# =========================================================
def configure_radiation_for_optimizer(cfg: Any) -> None:
    """Inject the centralized configuration into radiation.CFG without changing the radiation core formulas."""
    radiation = get_radiation_module()
    radiation.CFG.LAT = cfg.site.lat
    radiation.CFG.LON = cfg.site.lon
    radiation.CFG.ALT_M = cfg.site.alt_m
    radiation.CFG.TZ = cfg.site.tz
    radiation.CFG.ALBEDO = cfg.surface.albedo
    radiation.CFG.CLIMATE_TYPE = cfg.radiation.climate_type
    radiation.CFG.LINK_TURBIDITY = cfg.radiation.link_turbidity
    radiation.CFG.CLEARSKY_BIAS = cfg.radiation.clearsky_bias
    radiation.CFG.ENABLE_LOW_SUN_CORRECTION = cfg.radiation.enable_low_sun_correction
    radiation.CFG.LOW_SUN_FACTOR = cfg.radiation.low_sun_factor
    radiation.CFG.KEEP_OTHER_CLASS = cfg.segmentation.keep_other_class


def compute_base_radiation(df: pd.DataFrame, svf: float) -> pd.DataFrame:
    """Run the existing radiation model once to obtain base columns such as occluded DNI_eff and diffuse."""
    radiation = get_radiation_module()
    old_tilt = radiation.CFG.SURF_TILT_DEG
    old_azimuth = radiation.CFG.SURF_AZIMUTH_DEG
    try:
        radiation.CFG.SURF_TILT_DEG = 0.0
        radiation.CFG.SURF_AZIMUTH_DEG = 180.0
        return radiation.compute_radiation(df, svf)
    finally:
        radiation.CFG.SURF_TILT_DEG = old_tilt
        radiation.CFG.SURF_AZIMUTH_DEG = old_azimuth


class DynamicSkyDiffuseModel:
    """Direction-aware dynamic diffuse model used by panel optimization.

    Photo luminance is treated as relative angular information only. The
    absolute diffuse energy still comes from each minute's DHI, so exposure and
    camera response cannot create extra annual energy.
    """

    SKY_GRID_ALT_STEP_DEG = 5.0
    SKY_GRID_AZ_STEP_DEG = 5.0

    def __init__(self, cfg: Any, paths: Any):
        self.cfg = cfg
        self.paths = paths
        self._load_fit_scales()
        self._load_image_data()
        self._build_direction_grid()

    def _load_fit_scales(self) -> None:
        radiation_cfg = getattr(self.cfg, "radiation", None)
        self.sky_diffuse_scale = float(getattr(radiation_cfg, "sky_diffuse_scale", 1.0))
        self.vegetation_diffuse_scale = float(getattr(radiation_cfg, "vegetation_diffuse_scale", 1.0))
        self.building_diffuse_scale = float(getattr(radiation_cfg, "building_diffuse_scale", 1.0))
        self.aerosol_diffuse_scale = float(getattr(radiation_cfg, "aerosol_diffuse_scale", 1.0))

    @classmethod
    def try_create(cls, cfg: Any, paths: Any) -> "DynamicSkyDiffuseModel | None":
        try:
            model = cls(cfg, paths)
            print(f"[optimization] Dynamic anisotropic diffuse model enabled: {len(model.alt_deg)} sky directions")
            return model
        except Exception as exc:
            print(f"[optimization] Dynamic diffuse model unavailable, falling back to the original diffuse algorithm: {exc}")
            return None

    def _load_image_data(self) -> None:
        import cv2

        seg_dir = Path(self.paths.segmentation_dir)
        image_name = self.paths.image_name
        square_path = seg_dir / f"{image_name}_square.png"
        if not square_path.exists():
            square_path = seg_dir / f"{image_name}_overlay.png"
        bgr = cv2.imread(str(square_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(square_path)
        self.h, self.w = bgr.shape[:2]

        masks = {}
        for short, name in [("sky", "sky"), ("veg", "vegetation"), ("bld", "building"), ("oth", "other")]:
            path = seg_dir / f"{image_name}_mask_{short}.png"
            if short == "oth" and not path.exists():
                ref = next((m for m in (masks.get("sky"), masks.get("vegetation"), masks.get("building")) if m is not None), None)
                if ref is None:
                    raise FileNotFoundError(path)
                masks[name] = np.zeros_like(ref, dtype=bool)
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            masks[name] = img > 127
        if not bool(getattr(getattr(self.cfg, "segmentation", None), "keep_other_class", False)):
            masks["building"] = masks["building"] | masks["other"]
            masks["other"] = np.zeros_like(masks["other"], dtype=bool)
        self.masks = masks

        meta_path = seg_dir / f"{image_name}_segmentation_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.cx, self.cy = [float(v) for v in meta["circle_center_square_xy"]]
            self.radius = float(meta["circle_radius_square_px"])
        else:
            self.cx = (self.w - 1) / 2.0
            self.cy = (self.h - 1) / 2.0
            self.radius = min(self.h, self.w) / 2.0

        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        lum = 0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
        sky_lum = lum[self.masks["sky"]]
        baseline = float(np.nanmedian(sky_lum)) if sky_lum.size else float(np.nanmedian(lum))
        if not np.isfinite(baseline) or baseline <= 1e-6:
            baseline = 1.0
        self.relative_luminance = np.clip(lum / baseline, 0.25, 3.0)

        obstacle = self.masks["vegetation"] | self.masks["building"] | self.masks["other"]
        sky_u8 = self.masks["sky"].astype(np.uint8)
        obs_u8 = obstacle.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        boundary = (cv2.dilate(sky_u8, kernel) > 0) & (cv2.dilate(obs_u8, kernel) > 0)
        dist = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
        self.edge_weight = np.exp(-(dist / 10.0) ** 2).astype(np.float32)

    def _project_to_pixel(self, alt_deg: np.ndarray, az_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.radians(90.0 - alt_deg)
        model = str(self.cfg.sunpath.projection_model).lower()
        if model == "equidistant":
            rho = self.radius * theta / (math.pi / 2.0)
        elif model == "equisolid":
            f = self.radius / math.sqrt(2.0)
            rho = 2.0 * f * np.sin(theta / 2.0)
        else:
            raise ValueError(f"unknown projection model: {model}")

        az_rot = az_deg - float(self.cfg.image.top_azimuth_deg)
        if bool(self.cfg.image.mirror_azimuth):
            az_rot = -az_rot
        phi = np.radians(az_rot % 360.0)
        u = self.cx + rho * np.sin(phi)
        v = self.cy - rho * np.cos(phi)
        return u, v

    def _build_direction_grid(self) -> None:
        alt_vals = np.arange(self.SKY_GRID_ALT_STEP_DEG / 2.0, 90.0, self.SKY_GRID_ALT_STEP_DEG)
        az_vals = np.arange(0.0, 360.0, self.SKY_GRID_AZ_STEP_DEG)
        alt, az = np.meshgrid(alt_vals, az_vals, indexing="ij")
        alt = alt.ravel()
        az = az.ravel()
        u, v = self._project_to_pixel(alt, az)
        valid = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        ui = np.clip(u.astype(int), 0, self.w - 1)[valid]
        vi = np.clip(v.astype(int), 0, self.h - 1)[valid]

        self.alt_deg = alt[valid]
        self.az_deg = az[valid]
        alt_rad = np.radians(self.alt_deg)
        az_rad = np.radians(self.az_deg)
        self.dir_vectors = np.column_stack([
            np.sin(az_rad) * np.cos(alt_rad),
            np.cos(az_rad) * np.cos(alt_rad),
            np.sin(alt_rad),
        ])
        d_alt = math.radians(self.SKY_GRID_ALT_STEP_DEG)
        d_az = math.radians(self.SKY_GRID_AZ_STEP_DEG)
        self.solid_angle = np.cos(alt_rad) * d_alt * d_az
        self.horizon_weight = np.clip((15.0 - self.alt_deg) / 15.0, 0.0, 1.0)
        horizontal_incidence = np.maximum(self.dir_vectors[:, 2], 0.0)
        self.horizontal_reference = float(np.sum(horizontal_incidence * self.solid_angle))
        self.horizontal_horizon_reference = float(
            np.sum(horizontal_incidence * self.solid_angle * self.horizon_weight)
        )

        sky = self.masks["sky"][vi, ui]
        veg = self.masks["vegetation"][vi, ui]
        bld = self.masks["building"][vi, ui]
        oth = self.masks["other"][vi, ui]
        photo = self.relative_luminance[vi, ui].astype(float)
        edge = self.edge_weight[vi, ui].astype(float)
        self.direction_luminance = np.clip(photo + 0.25 * edge, 0.2, 3.5)
        self.direction_class = np.full(len(self.alt_deg), "outside", dtype=object)
        self.direction_class[sky] = "sky"
        self.direction_class[veg] = "vegetation"
        self.direction_class[bld] = "building"
        self.direction_class[oth] = "other"

    @staticmethod
    def _panel_normal(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        tilt = math.radians(float(tilt_deg))
        az = math.radians(float(azimuth_deg) % 360.0)
        return np.array([
            math.sin(az) * math.sin(tilt),
            math.cos(az) * math.sin(tilt),
            math.cos(tilt),
        ], dtype=float)

    @staticmethod
    def _vegetation_diffuse_factor_for_month(month: int) -> float:
        if month in (6, 7, 8):
            return 0.40
        if month in (12, 1, 2):
            return 0.70
        return 0.525

    def _class_diffuse_factors_for_month(self, month: int) -> dict[str, float]:
        keep_other_class = bool(getattr(getattr(self.cfg, "segmentation", None), "keep_other_class", False))
        return {
            "sky": 1.0 * self.sky_diffuse_scale,
            "vegetation": self._vegetation_diffuse_factor_for_month(int(month)) * self.vegetation_diffuse_scale,
            "building": 0.02 * self.building_diffuse_scale,
            "other": (0.04 * self.aerosol_diffuse_scale if keep_other_class else 0.02 * self.building_diffuse_scale),
            "outside": 0.0,
        }

    def _direction_visibility_for_month(self, month: int) -> np.ndarray:
        factors = self._class_diffuse_factors_for_month(month)
        trans = np.zeros(len(self.direction_class), dtype=float)
        for name, factor in factors.items():
            trans[self.direction_class == name] = factor
        return trans * self.direction_luminance

    def _occlusion_to_circumsolar_factor(self, occ: pd.Series, months: np.ndarray) -> np.ndarray:
        out = np.zeros(len(occ), dtype=float)
        labels = occ.astype(str).str.lower().to_numpy()
        for month in np.unique(months):
            factors = self._class_diffuse_factors_for_month(int(month))
            mask = months == month
            if np.any(mask):
                out[mask] = pd.Series(labels[mask]).map(factors).fillna(0.0).to_numpy(dtype=float)
        return out

    def effective_dhi(self, base: pd.DataFrame, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        """Return an equivalent horizontal DHI for legacy Perez-based callers."""
        poa_diffuse = self.effective_poa_diffuse(base, tilt_deg, azimuth_deg)
        sky_view = max((1.0 + math.cos(math.radians(float(tilt_deg)))) / 2.0, 1e-6)
        return poa_diffuse / sky_view

    def effective_poa_diffuse(self, base: pd.DataFrame, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
        """Compute panel-plane sky diffuse directly from visible sky directions.

        The old dynamic model normalized by the candidate panel's visible
        hemisphere, which preserved average sky brightness but removed most of
        the sky-view effect. Here the directional integral is normalized to a
        horizontal open-sky reference, so a vertical panel sees roughly half of
        the isotropic sky diffuse while a horizontal panel sees the full dome.
        """
        normal = self._panel_normal(tilt_deg, azimuth_deg)
        incidence = np.maximum(self.dir_vectors @ normal, 0.0)
        if self.horizontal_reference <= 1e-9:
            return np.zeros(len(base), dtype=float)

        dni = np.maximum(base["DNI"].to_numpy(dtype=float), 0.0)
        dhi = np.maximum(base["DHI"].to_numpy(dtype=float), 0.0)
        ghi = np.maximum(base["GHI"].to_numpy(dtype=float), 0.0)
        months = pd.DatetimeIndex(base["time"]).month.to_numpy()

        weighted = incidence * self.solid_angle
        background = np.zeros(len(base), dtype=float)
        horizon = np.zeros(len(base), dtype=float)
        for month in np.unique(months):
            visibility = self._direction_visibility_for_month(int(month))
            background_m = float(np.sum(weighted * visibility) / self.horizontal_reference)
            if self.horizontal_horizon_reference > 1e-9:
                horizon_m = float(
                    np.sum(weighted * self.horizon_weight * visibility)
                    / self.horizontal_horizon_reference
                )
            else:
                horizon_m = background_m
            month_mask = months == month
            background[month_mask] = np.clip(background_m, 0.0, 2.0)
            horizon[month_mask] = np.clip(horizon_m, 0.0, 2.5)

        zenith = np.clip(90.0 - base["altitude_deg"].to_numpy(dtype=float), 0.0, 90.0)
        sun_alt = np.radians(np.clip(base["altitude_deg"].to_numpy(dtype=float), 0.0, 90.0))
        sun_az = np.radians(base["azimuth_deg"].to_numpy(dtype=float) % 360.0)
        sun_vec = np.column_stack([
            np.sin(sun_az) * np.cos(sun_alt),
            np.cos(sun_az) * np.cos(sun_alt),
            np.sin(sun_alt),
        ])
        cos_aoi = np.maximum(sun_vec @ normal, 0.0)
        cos_zenith = np.maximum(np.cos(np.radians(zenith)), 1e-6)
        sun_on_panel = np.clip(cos_aoi / cos_zenith, 0.0, 3.0)

        clearness = dni / np.maximum(dni + dhi, 1e-6)
        diffuse_ratio = dhi / np.maximum(ghi, 1e-6)

        # The sky dome is not uniform. The solar corridor is much brighter than
        # ordinary sky, so panels facing the sun path should receive more
        # diffuse even when direct beam is blocked by vegetation/buildings.
        circumsolar_share = np.clip(0.18 + 0.62 * clearness, 0.18, 0.78)
        horizon_share = np.clip(0.04 + 0.12 * diffuse_ratio, 0.04, 0.18)
        background_share = np.clip(1.0 - circumsolar_share - horizon_share, 0.08, 0.70)
        share_sum = np.maximum(circumsolar_share + horizon_share + background_share, 1e-6)
        circumsolar_share = circumsolar_share / share_sum
        horizon_share = horizon_share / share_sum
        background_share = background_share / share_sum

        circumsolar = self._occlusion_to_circumsolar_factor(base["occlusion"], months) * sun_on_panel
        factor = background_share * background + horizon_share * horizon + circumsolar_share * circumsolar
        return dhi * np.clip(factor, 0.0, 2.5)


def compute_poa_for_orientation(base: pd.DataFrame, tilt_deg: float, azimuth_deg: float,
                                albedo: float,
                                sky_model: DynamicSkyDiffuseModel | None = None) -> pd.DataFrame:
    """Compute POA for one fixed panel orientation."""
    pvlib = get_pvlib_module()
    times = pd.DatetimeIndex(base["time"])
    zenith = np.clip(90.0 - base["altitude_deg"].to_numpy(dtype=float), 0, 90)
    solar_azimuth = base["azimuth_deg"].to_numpy(dtype=float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    direct_col = first_existing_column(base, ["Direct(W/m²)", "Direct(W/m㎡)", "Direct(W/m虏)"])
    diffuse_col = first_existing_column(base, ["Diffuse(W/m²)", "Diffuse(W/m㎡)", "Diffuse(W/m虏)"])
    total_col = first_existing_column(base, ["Total_Horizontal(W/m²)", "Total_Horizontal(W/m㎡)", "Total_Horizontal(W/m虏)"])

    if sky_model is not None:
        poa_diffuse = sky_model.effective_poa_diffuse(base, tilt_deg, azimuth_deg)
        aoi = pvlib.irradiance.aoi(
            surface_tilt=float(tilt_deg),
            surface_azimuth=float(azimuth_deg),
            solar_zenith=zenith,
            solar_azimuth=solar_azimuth,
        )
        poa_direct = base["DNI_eff"].to_numpy(dtype=float) * np.maximum(np.cos(np.radians(aoi)), 0.0)
        # In the fisheye-photo workflow the available irradiance is modeled
        # from the upper hemisphere visible in the image. Do not add a generic
        # pvlib ground-reflection bonus, otherwise steep panels can be rewarded
        # for a lower hemisphere that the image does not actually describe.
        poa_ground = np.zeros(len(base), dtype=float)
        return pd.DataFrame({
            "POA_Direct(W/m²)": np.nan_to_num(poa_direct, nan=0.0),
            "POA_Diffuse(W/m²)": np.nan_to_num(poa_diffuse, nan=0.0),
            "POA_Ground(W/m²)": np.nan_to_num(poa_ground, nan=0.0),
            POA_COL: np.nan_to_num(poa_direct + poa_diffuse + poa_ground, nan=0.0),
        })
    else:
        dhi_for_panel = base[diffuse_col].to_numpy(dtype=float)
        ghi_for_panel = base[total_col].to_numpy(dtype=float)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt_deg),
        surface_azimuth=float(azimuth_deg),
        dni=base["DNI_eff"].to_numpy(dtype=float),
        ghi=ghi_for_panel,
        dhi=dhi_for_panel,
        solar_zenith=zenith,
        solar_azimuth=solar_azimuth,
        dni_extra=dni_extra,
        model="perez",
        albedo=float(albedo),
    )
    return pd.DataFrame({
        "POA_Direct(W/m²)": np.nan_to_num(poa["poa_direct"], nan=0.0),
        "POA_Diffuse(W/m²)": np.nan_to_num(poa["poa_diffuse"], nan=0.0),
        "POA_Ground(W/m²)": np.nan_to_num(poa["poa_ground_diffuse"], nan=0.0),
        POA_COL: np.nan_to_num(poa["poa_global"], nan=0.0),
    })


def compute_poa_for_orientation(base: pd.DataFrame, tilt_deg: float, azimuth_deg: float,
                                albedo: float,
                                sky_model: DynamicSkyDiffuseModel | None = None) -> pd.DataFrame:
    """Compute POA for one fixed panel orientation."""
    pvlib = get_pvlib_module()
    times = pd.DatetimeIndex(base["time"])
    zenith = np.clip(90.0 - base["altitude_deg"].to_numpy(dtype=float), 0, 90)
    solar_azimuth = base["azimuth_deg"].to_numpy(dtype=float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    direct_col = first_existing_column(base, ["Direct(W/m²)", "Direct(W/m㎡)", "Direct(W/m虏)"])
    diffuse_col = first_existing_column(base, ["Diffuse(W/m²)", "Diffuse(W/m㎡)", "Diffuse(W/m虏)"])
    total_col = first_existing_column(base, ["Total_Horizontal(W/m²)", "Total_Horizontal(W/m㎡)", "Total_Horizontal(W/m虏)"])

    if sky_model is not None:
        poa_diffuse = sky_model.effective_poa_diffuse(base, tilt_deg, azimuth_deg)
        aoi = pvlib.irradiance.aoi(
            surface_tilt=float(tilt_deg),
            surface_azimuth=float(azimuth_deg),
            solar_zenith=zenith,
            solar_azimuth=solar_azimuth,
        )
        poa_direct = base["DNI_eff"].to_numpy(dtype=float) * np.maximum(np.cos(np.radians(aoi)), 0.0)
        # In the fisheye-photo workflow the available irradiance is modeled
        # from the upper hemisphere visible in the image. Do not add a generic
        # pvlib ground-reflection bonus, otherwise steep panels can be rewarded
        # for a lower hemisphere that the image does not actually describe.
        poa_ground = np.zeros(len(base), dtype=float)
        return pd.DataFrame({
            "POA_Direct(W/m²)": np.nan_to_num(poa_direct, nan=0.0),
            "POA_Diffuse(W/m²)": np.nan_to_num(poa_diffuse, nan=0.0),
            "POA_Ground(W/m²)": np.nan_to_num(poa_ground, nan=0.0),
            POA_COL: np.nan_to_num(poa_direct + poa_diffuse + poa_ground, nan=0.0),
        })
    else:
        dhi_for_panel = base[diffuse_col].to_numpy(dtype=float)
        ghi_for_panel = base[total_col].to_numpy(dtype=float)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt_deg),
        surface_azimuth=float(azimuth_deg),
        dni=base["DNI_eff"].to_numpy(dtype=float),
        ghi=ghi_for_panel,
        dhi=dhi_for_panel,
        solar_zenith=zenith,
        solar_azimuth=solar_azimuth,
        dni_extra=dni_extra,
        model="perez",
        albedo=float(albedo),
    )
    return pd.DataFrame({
        "POA_Direct(W/m²)": np.nan_to_num(poa["poa_direct"], nan=0.0),
        "POA_Diffuse(W/m²)": np.nan_to_num(poa["poa_diffuse"], nan=0.0),
        "POA_Ground(W/m²)": np.nan_to_num(poa["poa_ground_diffuse"], nan=0.0),
        POA_COL: np.nan_to_num(poa["poa_global"], nan=0.0),
    })


def compute_poa_for_orientation(base: pd.DataFrame, tilt_deg: float, azimuth_deg: float,
                                albedo: float,
                                sky_model: DynamicSkyDiffuseModel | None = None) -> pd.DataFrame:
    """Compute POA for one fixed panel orientation."""
    pvlib = get_pvlib_module()
    times = pd.DatetimeIndex(base["time"])
    zenith = np.clip(90.0 - base["altitude_deg"].to_numpy(dtype=float), 0, 90)
    solar_azimuth = base["azimuth_deg"].to_numpy(dtype=float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    direct_col = first_existing_column(base, ["Direct(W/m²)", "Direct(W/m㎡)", "Direct(W/m虏)"])
    diffuse_col = first_existing_column(base, ["Diffuse(W/m²)", "Diffuse(W/m㎡)", "Diffuse(W/m虏)"])
    total_col = first_existing_column(base, ["Total_Horizontal(W/m²)", "Total_Horizontal(W/m㎡)", "Total_Horizontal(W/m虏)"])

    if sky_model is not None:
        poa_diffuse = sky_model.effective_poa_diffuse(base, tilt_deg, azimuth_deg)
        aoi = pvlib.irradiance.aoi(
            surface_tilt=float(tilt_deg),
            surface_azimuth=float(azimuth_deg),
            solar_zenith=zenith,
            solar_azimuth=solar_azimuth,
        )
        poa_direct = base["DNI_eff"].to_numpy(dtype=float) * np.maximum(np.cos(np.radians(aoi)), 0.0)
        # In the fisheye-photo workflow the available irradiance is modeled
        # from the upper hemisphere visible in the image. Do not add a generic
        # pvlib ground-reflection bonus, otherwise steep panels can be rewarded
        # for a lower hemisphere that the image does not actually describe.
        poa_ground = np.zeros(len(base), dtype=float)
        return pd.DataFrame({
            "POA_Direct(W/m²)": np.nan_to_num(poa_direct, nan=0.0),
            "POA_Diffuse(W/m²)": np.nan_to_num(poa_diffuse, nan=0.0),
            "POA_Ground(W/m²)": np.nan_to_num(poa_ground, nan=0.0),
            POA_COL: np.nan_to_num(poa_direct + poa_diffuse + poa_ground, nan=0.0),
        })
    else:
        dhi_for_panel = base[diffuse_col].to_numpy(dtype=float)
        ghi_for_panel = base[total_col].to_numpy(dtype=float)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(tilt_deg),
        surface_azimuth=float(azimuth_deg),
        dni=base["DNI_eff"].to_numpy(dtype=float),
        ghi=ghi_for_panel,
        dhi=dhi_for_panel,
        solar_zenith=zenith,
        solar_azimuth=solar_azimuth,
        dni_extra=dni_extra,
        model="perez",
        albedo=float(albedo),
    )
    return pd.DataFrame({
        "POA_Direct(W/m²)": np.nan_to_num(poa["poa_direct"], nan=0.0),
        "POA_Diffuse(W/m²)": np.nan_to_num(poa["poa_diffuse"], nan=0.0),
        "POA_Ground(W/m²)": np.nan_to_num(poa["poa_ground_diffuse"], nan=0.0),
        POA_COL: np.nan_to_num(poa["poa_global"], nan=0.0),
    })


def angle_values(bounds: tuple[float, float, float]) -> np.ndarray:
    """Generate an angle sequence that includes the endpoint."""
    start, stop, step = [float(v) for v in bounds]
    if step <= 0:
        raise ValueError("The angle step must be greater than 0")
    vals = np.arange(start, stop + step * 0.5, step)
    return vals[(vals >= min(start, stop) - 1e-9) & (vals <= max(start, stop) + 1e-9)]


def candidate_pairs(tilts: np.ndarray, azimuths: np.ndarray) -> list[tuple[float, float]]:
    return [(float(t), float(a % 360.0)) for t in tilts for a in azimuths]


def refine_pairs(best_tilt: float, best_azimuth: float, radius: float, step: float) -> list[tuple[float, float]]:
    """Generate fine-search candidates around the best coarse-search point."""
    if radius <= 0 or step <= 0:
        return []
    tilts = np.arange(max(0.0, best_tilt - radius), min(90.0, best_tilt + radius) + step * 0.5, step)
    azimuths = np.arange(best_azimuth - radius, best_azimuth + radius + step * 0.5, step)
    return candidate_pairs(tilts, azimuths)


def daily_scores(minute: pd.DataFrame, spec: pv_yield.PVPanelSpec) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aggregate by day and return P10, totals, and other scores."""
    work = minute.copy()
    work["date"] = pd.to_datetime(work["time"]).dt.date.astype(str)

    if spec.enabled:
        daily = work.groupby("date", as_index=False)["energy_kwh_per_step"].sum()
        daily = daily.rename(columns={"energy_kwh_per_step": "daily_energy_kwh"})
        score_col = "daily_energy_kwh"
    else:
        daily = work.groupby("date", as_index=False)["energy_kwh_m2_per_step"].sum()
        daily = daily.rename(columns={"energy_kwh_m2_per_step": "daily_energy_kwh_m2"})
        score_col = "daily_energy_kwh_m2"

    vals = daily[score_col].to_numpy(dtype=float)
    metrics = {
        "daily_p10": float(np.percentile(vals, 10)) if len(vals) else 0.0,
        "daily_p50": float(np.percentile(vals, 50)) if len(vals) else 0.0,
        "daily_p90": float(np.percentile(vals, 90)) if len(vals) else 0.0,
        "daily_min": float(np.min(vals)) if len(vals) else 0.0,
        "daily_mean": float(np.mean(vals)) if len(vals) else 0.0,
        "annual_total": float(np.sum(vals)) if len(vals) else 0.0,
        "day_count": float(len(vals)),
    }
    return daily, metrics


def evaluate_orientation(base: pd.DataFrame, tilt: float, azimuth: float, cfg: Any,
                         spec: pv_yield.PVPanelSpec, dt_hours: float,
                         sky_model: DynamicSkyDiffuseModel | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute the per-minute results and scores for one candidate tilt/azimuth."""
    minute = base.copy()
    poa = compute_poa_for_orientation(base, tilt, azimuth, cfg.surface.albedo, sky_model)
    for col in ["POA_Direct(W/m²)", "POA_Diffuse(W/m²)", "POA_Ground(W/m²)", "POA_Total(W/m²)"]:
        minute[col] = poa[col].to_numpy(dtype=float)
    minute["panel_tilt_deg"] = float(tilt)
    minute["panel_azimuth_deg"] = float(azimuth % 360.0)
    minute = pv_yield.add_yield_columns(minute, spec, dt_hours)
    _, metrics = daily_scores(minute, spec)
    metrics.update({
        "tilt_deg": float(tilt),
        "azimuth_deg": float(azimuth % 360.0),
    })
    return minute, metrics


def add_objective_scores(grid: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Add the paper objective J to a candidate grid.

    J(beta,gamma) = E_ann / E_ann^max - lambda(N_aut) * D_P10^2
    D_P10 = max(0, (E_P10^req - E_P10) / E_P10^ref)
    """
    out = grid.copy()
    if out.empty:
        return out

    annual = pd.to_numeric(out["annual_total"], errors="coerce").fillna(0.0)
    p10 = pd.to_numeric(out["daily_p10"], errors="coerce").fillna(0.0)

    annual_max = float(annual.max())
    p10_ref = float(p10.max())
    p10_requirement_ratio = float(getattr(cfg.optimization, "p10_requirement_ratio", 0.90))
    storage_autonomy_days = float(getattr(cfg.optimization, "storage_autonomy_days", 1.0))
    if storage_autonomy_days <= 0:
        raise ValueError("optimization.storage_autonomy_days must be greater than 0")

    p10_req = p10_requirement_ratio * p10_ref
    annual_norm = annual / annual_max if annual_max > 0 else annual * 0.0
    if p10_ref > 0:
        p10_gap = np.maximum(0.0, (p10_req - p10) / p10_ref)
    else:
        p10_gap = p10 * 0.0
    p10_lambda = 1.0 / storage_autonomy_days

    out["annual_norm"] = annual_norm.astype(float)
    out["p10_ref"] = p10_ref
    out["p10_req"] = p10_req
    out["p10_gap"] = p10_gap.astype(float)
    out["p10_lambda"] = p10_lambda
    out["storage_autonomy_days"] = storage_autonomy_days
    out["p10_requirement_ratio"] = p10_requirement_ratio
    out["objective_score"] = out["annual_norm"] - p10_lambda * np.square(out["p10_gap"])
    return out


# =========================================================
# Output and visualization
# =========================================================
def write_heatmap(grid: pd.DataFrame, out_path: Path, score_col: str) -> None:
    """Export the tilt/azimuth score heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = grid.pivot_table(index="tilt_deg", columns="azimuth_deg", values=score_col, aggfunc="max")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    ax.set_xlabel("Azimuth deg")
    ax.set_ylabel("Tilt deg")
    ax.set_title(f"Optimization score: {score_col}")
    fig.colorbar(im, ax=ax, label=score_col)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def find_fisheye_background(paths: Any) -> Path | None:
    """Prefer segmentation output images as the static illustrative background."""
    candidates = [
        paths.segmentation_dir / f"{paths.image_name}_overlay.png",
        paths.segmentation_dir / f"{paths.image_name}_square.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def render_panel_visual_3d(paths: Any, out_path: Path, best: dict[str, float],
                           projection_model: str) -> None:
    """Export a true three-dimensional fisheye hemisphere with the panel pose."""
    bg_path = find_fisheye_background(paths)
    if bg_path is None:
        raise FileNotFoundError("No fisheye background image available for 3D visualization")
    visualization_3d.render_best_panel_scene(
        background_path=bg_path,
        out_path=out_path,
        best=best,
        projection_model=projection_model,
    )


def render_interactive_sky_dome(paths: Any, html_path: Path, best: dict[str, float],
                                projection_model: str,
                                sun_path: list[dict[str, float]] | None = None) -> None:
    """Export an interactive Three.js 3D sky dome with drag support."""
    bg_path = find_fisheye_background(paths)
    if bg_path is None:
        raise FileNotFoundError("No fisheye background image available for interactive 3D visualization")
    visualization_3d.render_interactive_sky_dome_html(
        background_path=bg_path,
        html_path=html_path,
        best=best,
        projection_model=projection_model,
        sun_path=sun_path,
    )


SUMMARY_CN_COLUMNS = {
    "image_name": "image_name",
    "occlusion_csv": "occlusion_csv_path",
    "tmy_minute_csv": "tmy_minute_csv_path",
    "dt_hours": "dt_hours",
    "svf": "sky_view_factor_svf",
    "score_basis": "score_basis",
    "fixed_south_tilt_deg": "fixed_south_tilt_deg",
    "fixed_south_azimuth_deg": "fixed_south_azimuth_deg",
    "fixed_south_daily_p10": "fixed_south_p10_daily_yield_kwh",
    "fixed_south_annual_total": "fixed_south_annual_yield_kwh",
    "tilt_deg": "optimal_tilt_deg",
    "azimuth_deg": "optimal_azimuth_deg",
    "daily_p10": "p10_daily_yield",
    "daily_p50": "p50_daily_yield",
    "daily_p90": "p90_daily_yield",
    "daily_min": "min_daily_yield",
    "daily_mean": "mean_daily_yield",
    "annual_total": "annual_yield_total",
    "improvement_kwh": "improvement_kwh",
    "improvement_ratio": "improvement_ratio",
    "objective_score": "objective_j",
    "annual_norm": "annual_total_normalized",
    "p10_ref": "p10_reference",
    "p10_req": "p10_target",
    "p10_gap": "p10_normalized_gap",
    "p10_lambda": "p10_penalty_weight",
    "storage_autonomy_days": "storage_autonomy_days",
    "p10_requirement_ratio": "p10_retention_ratio",
    "day_count": "day_count",
    "pv_enabled": "pv_panel_enabled",
    "pv_tech": "pv_module_technology",
    "module_width_mm": "module_width_mm",
    "module_height_mm": "module_height_mm",
    "module_p_stc_w": "module_rated_power_w",
    "module_area_m2": "module_area_m2",
    "eta_stc": "stc_efficiency",
    "installable_ratio": "installable_ratio",
    "panel_count": "module_count",
    "active_area_m2": "active_area_m2",
    "installed_kwp": "installed_capacity_kwp",
    "pr_total": "performance_ratio_pr",
    "approx_module_count": "equivalent_module_count",
}


GRID_CN_COLUMNS = {
    "tilt_deg": "tilt_deg",
    "azimuth_deg": "azimuth_deg",
    "daily_p10": "p10_daily_yield",
    "daily_p50": "p50_daily_yield",
    "daily_p90": "p90_daily_yield",
    "daily_min": "min_daily_yield",
    "daily_mean": "mean_daily_yield",
    "annual_total": "annual_yield_total",
    "objective_score": "objective_j",
    "annual_norm": "annual_total_normalized",
    "p10_ref": "p10_reference",
    "p10_req": "p10_target",
    "p10_gap": "p10_normalized_gap",
    "p10_lambda": "p10_penalty_weight",
    "storage_autonomy_days": "storage_autonomy_days",
    "p10_requirement_ratio": "p10_retention_ratio",
    "day_count": "day_count",
}


MINUTE_CN_COLUMNS = {
    "time": "time",
    "altitude_deg": "solar_altitude_deg",
    "azimuth_deg": "solar_azimuth_deg",
    "occlusion": "occlusion_type",
    "GHI": "ghi_w_m2",
    "DNI": "dni_w_m2",
    "DHI": "dhi_w_m2",
    "DNI_eff": "dni_effective_w_m2",
    "Direct(W/m²)": "direct_horizontal_w_m2",
    "Diffuse(W/m²)": "diffuse_total_w_m2",
    "Sky_Diffuse(W/m²)": "sky_diffuse_w_m2",
    "Vegetation_Diffuse(W/m²)": "vegetation_diffuse_w_m2",
    "Building_Diffuse(W/m²)": "building_diffuse_w_m2",
    "Aerosol_Diffuse(W/m²)": "aerosol_diffuse_w_m2",
    "Ground_Reflect(W/m²)": "ground_reflected_w_m2",
    "Total_Horizontal(W/m²)": "total_horizontal_after_occlusion_w_m2",
    "POA_Direct(W/m²)": "poa_direct_w_m2",
    "POA_Diffuse(W/m²)": "poa_diffuse_w_m2",
    "POA_Ground(W/m²)": "poa_ground_reflected_w_m2",
    POA_COL: "poa_total_w_m2",
    "Period": "period",
    "panel_tilt_deg": "panel_tilt_deg",
    "panel_azimuth_deg": "panel_azimuth_deg",
    "energy_kwh_m2_per_step": "energy_kwh_m2_per_step",
    "power_w": "power_w",
    "energy_kwh_per_step": "energy_kwh_per_step",
}


DAILY_CN_COLUMNS = {
    "orientation": "orientation",
    "date": "date",
    "season": "season",
    "daily_energy_kwh": "daily_yield_kwh",
    "daily_energy_kwh_m2": "daily_energy_kwh_m2",
}


SEASON_CN_COLUMNS = {
    "orientation": "orientation",
    "season": "season",
    "season_energy_kwh": "season_yield_kwh",
    "season_energy_kwh_m2": "season_energy_kwh_m2",
    "day_count": "day_count",
}


SINGLE_DAY_POWER_COLUMNS = [
    "time",
    "solar_altitude_deg",
    "solar_azimuth_deg",
    "occlusion_type",
    "panel_tilt_deg",
    "panel_azimuth_deg",
    "poa_total_w_m2",
    "power_w",
    "energy_kwh_per_step",
]


def to_chinese_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns for user-facing CSV output without affecting internal English-column computation."""
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def write_csv_with_excel_copy(df: pd.DataFrame, path: Path) -> None:
    """Write both a UTF-8-SIG copy and a GBK copy of the CSV.

    The UTF-8-SIG file suits programmatic processing; the GBK copy opens
    directly in Chinese Windows Excel by double-clicking.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    gbk_path = path.with_name(path.stem + "_excel.csv")
    df.to_csv(gbk_path, index=False, encoding="gbk", errors="replace")


def add_season_column(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    month = dates.dt.month
    out["season"] = np.select(
        [
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        ["spring", "summer", "autumn"],
        default="winter",
    )
    return out


def season_scores(daily: pd.DataFrame, spec: pv_yield.PVPanelSpec) -> pd.DataFrame:
    work = add_season_column(daily)
    if spec.enabled:
        value_col = "daily_energy_kwh"
        out_col = "season_energy_kwh"
    else:
        value_col = "daily_energy_kwh_m2"
        out_col = "season_energy_kwh_m2"
    grouped = work.groupby("season", as_index=False).agg(
        **{out_col: (value_col, "sum")},
        day_count=("date", "count"),
    )
    return grouped


def select_single_date_minutes(best_minute_cn: pd.DataFrame, single_date: str) -> pd.DataFrame:
    """Select the per-minute yield table for the configured date.

    TMY and the annual occlusion may differ in calendar year, so the configured
    date is matched by month-day.
    """
    target = pd.to_datetime(single_date).strftime("%m-%d")
    work = best_minute_cn.copy()
    time_col = "time"
    t = pd.to_datetime(work[time_col], errors="coerce")
    selected = work.loc[t.dt.strftime("%m-%d") == target].copy()
    if selected.empty:
        raise ValueError(f"No per-minute data found in the annual results for the configured date {single_date}")
    keep = [c for c in SINGLE_DAY_POWER_COLUMNS if c in selected.columns]
    return selected[keep]


def build_single_day_sun_path(best_minute: pd.DataFrame, single_date: str, step_min: int = 5) -> list[dict[str, float]]:
    """Extract solar trajectory points for the specified date from the optimal per-minute results, for the interactive 3D sky dome."""
    target = pd.to_datetime(single_date).strftime("%m-%d")
    work = best_minute.copy()
    t = pd.to_datetime(work["time"], errors="coerce")
    selected = work.loc[t.dt.strftime("%m-%d") == target].copy()
    selected = selected[pd.to_numeric(selected["altitude_deg"], errors="coerce") > 0]
    if step_min > 1:
        tt = pd.to_datetime(selected["time"], errors="coerce")
        selected = selected.loc[tt.dt.minute % step_min == 0]
    return [
        {
            "altitude_deg": float(row["altitude_deg"]),
            "azimuth_deg": float(row["azimuth_deg"]),
        }
        for _, row in selected.iterrows()
    ]


def select_best(grid: pd.DataFrame, cfg: Any) -> dict[str, float]:
    """Select the best candidate by P10, annual total, and proximity to conventional tilt angles."""
    work = add_objective_scores(grid, cfg)
    work["tilt_penalty"] = (work["tilt_deg"] - 35.0).abs()
    work = work.sort_values(
        ["objective_score", "annual_total", "daily_p10", "tilt_penalty"],
        ascending=[False, False, False, True],
    )
    return work.iloc[0].to_dict()


def evaluate_many(pairs: list[tuple[float, float]], base: pd.DataFrame, cfg: Any,
                  spec: pv_yield.PVPanelSpec, dt_hours: float, desc: str,
                  sky_model: DynamicSkyDiffuseModel | None = None) -> list[dict[str, float]]:
    """Evaluate a batch of candidate angles with a progress bar."""
    rows = []
    for tilt, azimuth in tqdm(pairs, desc=desc, ncols=90):
        _, metrics = evaluate_orientation(base, tilt, azimuth, cfg, spec, dt_hours, sky_model)
        rows.append(metrics)
    return rows


def fixed_south_orientation(cfg: Any) -> tuple[float, float]:
    """Return the fixed-south baseline orientation."""
    tilt = getattr(getattr(cfg, "optimization", None), "fixed_south_tilt_deg", None)
    if tilt is None:
        tilt = cfg.site.lat
    return float(tilt), 180.0


# =========================================================
# Main entry
# =========================================================
def run_optimization(cfg: Any, paths: Any) -> dict[str, Any]:
    """Run the annual fixed tilt/azimuth orientation optimization and write the results."""
    out_dir = getattr(paths, "pv_best_panel_dir", paths.optimization_dir)
    visuals_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg_flag(cfg, "export_visuals", False):
        visuals_dir.mkdir(parents=True, exist_ok=True)

    configure_radiation_for_optimizer(cfg)
    radiation = get_radiation_module()
    spec = pv_yield.spec_from_config(cfg.pv_panel)

    print(f"[optimization] TMY per-minute CSV: {cfg.paths.tmy_minute_csv}")

    weather = load_tmy_minute_csv(cfg.paths.tmy_minute_csv, cfg.site.tz)
    tmy_year = infer_tmy_year(weather)
    try:
        occ_path = find_occlusion_csv(cfg, paths)
        print(f"[optimization] Occlusion CSV: {occ_path}")
        occ = load_occlusion_csv(occ_path, cfg.site.tz)
    except FileNotFoundError:
        print(f"[optimization] No annual occlusion CSV found; generating the annual occlusion series automatically for year {tmy_year}...")
        occ = build_full_year_occlusion_from_segmentation(cfg, paths, tmy_year)
        occ_path = occ.attrs.get(
            "occlusion_source",
            f"in_memory:{paths.image_name}:{tmy_year}",
        )
    occ = normalize_other_to_building(occ, cfg.segmentation.keep_other_class)
    merged = align_weather_and_occlusion(occ, weather)
    dt_hours = estimate_dt_hours(merged["time"])
    full_row_count = len(merged)
    daylight = pd.to_numeric(merged["altitude_deg"], errors="coerce").fillna(-90.0) > 0.0
    irradiance = merged[["GHI", "DNI", "DHI"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    merged = merged.loc[daylight & irradiance.gt(0.0).any(axis=1)].reset_index(drop=True)
    if merged.empty:
        raise ValueError("No daylight irradiance rows remain after filtering")
    print(f"[optimization] daylight filter: {len(merged)} / {full_row_count} rows retained")

    svf = radiation.compute_svf(paths.segmentation_dir, paths.image_name)
    base = compute_base_radiation(merged, svf)
    sky_model = DynamicSkyDiffuseModel.try_create(cfg, paths)
    fixed_tilt, fixed_azimuth = fixed_south_orientation(cfg)
    fixed_minute, fixed_metrics = evaluate_orientation(
        base, fixed_tilt, fixed_azimuth, cfg, spec, dt_hours, sky_model
    )
    fixed_daily, _ = daily_scores(fixed_minute, spec)
    base_search = sample_for_search(base, cfg.optimization.search_sample_minutes)
    search_dt_hours = estimate_dt_hours(base_search["time"])
    print(
        f"[optimization] Search stage uses {cfg.optimization.search_sample_minutes}-minute sampling: "
        f"{len(base_search)} / {len(base)} rows"
    )

    rough_pairs = candidate_pairs(
        angle_values(cfg.optimization.tilt_range_deg),
        angle_values(cfg.optimization.azimuth_range_deg),
    )
    print(f"[optimization] Coarse-search candidate count: {len(rough_pairs)}")
    rough_rows = evaluate_many(rough_pairs, base_search, cfg, spec, search_dt_hours, "coarse search", sky_model)
    grid = add_objective_scores(pd.DataFrame(rough_rows), cfg)

    best = select_best(grid, cfg)
    refined = refine_pairs(
        best["tilt_deg"],
        best["azimuth_deg"],
        cfg.optimization.refine_radius_deg,
        cfg.optimization.refine_step_deg,
    )
    if refined:
        seen = {(round(r["tilt_deg"], 6), round(r["azimuth_deg"], 6)) for _, r in grid.iterrows()}
        refine_rows = []
        refine_eval_pairs = []
        for tilt, azimuth in refined:
            key = (round(tilt, 6), round(azimuth, 6))
            if key in seen:
                continue
            refine_eval_pairs.append((tilt, azimuth))
        if refine_eval_pairs:
            print(f"[optimization] Fine-search candidate count: {len(refine_eval_pairs)}")
            refine_rows = evaluate_many(refine_eval_pairs, base_search, cfg, spec, search_dt_hours, "fine search", sky_model)
        if refine_rows:
            grid = add_objective_scores(pd.concat([grid, pd.DataFrame(refine_rows)], ignore_index=True), cfg)

    top_n = max(1, int(cfg.optimization.final_full_resolution_top_n))
    top_candidates_df = grid.sort_values(
        ["objective_score", "annual_total", "daily_p10"],
        ascending=[False, False, False],
    ).head(top_n)
    top_pairs = [(float(r["tilt_deg"]), float(r["azimuth_deg"])) for _, r in top_candidates_df.iterrows()]
    print(f"[optimization] Full per-minute re-evaluation of the top {len(top_pairs)} candidates")
    final_rows = evaluate_many(top_pairs, base, cfg, spec, dt_hours, "top candidates re-check", sky_model)
    final_grid = add_objective_scores(pd.DataFrame(final_rows), cfg)

    best = select_best(final_grid, cfg)
    best_minute, _ = evaluate_orientation(base, best["tilt_deg"], best["azimuth_deg"], cfg, spec, dt_hours, sky_model)
    best_daily, daily_metrics = daily_scores(best_minute, spec)
    best.update(daily_metrics)

    fixed_annual = float(fixed_metrics["annual_total"])
    optimized_annual = float(best["annual_total"])
    improvement_kwh = optimized_annual - fixed_annual
    improvement_ratio = improvement_kwh / fixed_annual if fixed_annual > 0 else 0.0

    meta = pv_yield.yield_metadata(spec)
    summary = {
        "image_name": paths.image_name,
        "occlusion_csv": str(occ_path),
        "tmy_minute_csv": str(cfg.paths.tmy_minute_csv),
        "dt_hours": dt_hours,
        "svf": svf,
        "score_basis": "pv_yield_kwh" if spec.enabled else "poa_kwh_m2",
        "fixed_south_tilt_deg": fixed_tilt,
        "fixed_south_azimuth_deg": fixed_azimuth,
        "fixed_south_daily_p10": fixed_metrics["daily_p10"],
        "fixed_south_annual_total": fixed_annual,
        "improvement_kwh": improvement_kwh,
        "improvement_ratio": improvement_ratio,
        **best,
        **meta,
    }

    if cfg_flag(cfg, "export_search_grid", True):
        write_csv_with_excel_copy(to_chinese_columns(grid, GRID_CN_COLUMNS), out_dir / "optimization_search_grid.csv")
    if cfg_flag(cfg, "export_final_candidates", True):
        write_csv_with_excel_copy(to_chinese_columns(final_grid, GRID_CN_COLUMNS), out_dir / "optimization_final_top_candidates.csv")
    write_csv_with_excel_copy(
        to_chinese_columns(pd.DataFrame([summary]), SUMMARY_CN_COLUMNS),
        out_dir / "optimization_summary.csv",
    )

    if cfg.optimization.export_minute_power:
        fixed_minute_cn = to_chinese_columns(fixed_minute, MINUTE_CN_COLUMNS)
        best_minute_cn = to_chinese_columns(best_minute, MINUTE_CN_COLUMNS)
        write_csv_with_excel_copy(fixed_minute_cn, out_dir / "fixed_south_minute_power.csv")
        write_csv_with_excel_copy(best_minute_cn, out_dir / "optimized_minute_power.csv")
    else:
        best_minute_cn = None

    fixed_daily_out = fixed_daily.copy()
    fixed_daily_out.insert(0, "orientation", "fixed_south")
    best_daily_out = best_daily.copy()
    best_daily_out.insert(0, "orientation", "optimized")
    combined_daily = add_season_column(pd.concat([fixed_daily_out, best_daily_out], ignore_index=True))
    if cfg_flag(cfg, "export_daily_power", True):
        best_daily_cn = to_chinese_columns(best_daily, DAILY_CN_COLUMNS)
        write_csv_with_excel_copy(best_daily_cn, out_dir / "annual_daily_power.csv")
        write_csv_with_excel_copy(to_chinese_columns(combined_daily, DAILY_CN_COLUMNS), out_dir / "daily_power.csv")

    fixed_season = season_scores(fixed_daily, spec)
    fixed_season.insert(0, "orientation", "fixed_south")
    best_season = season_scores(best_daily, spec)
    best_season.insert(0, "orientation", "optimized")
    combined_season = pd.concat([fixed_season, best_season], ignore_index=True)
    write_csv_with_excel_copy(to_chinese_columns(combined_season, SEASON_CN_COLUMNS), out_dir / "season_power.csv")

    sun_path_points: list[dict[str, float]] = []
    if cfg.optimization.export_minute_power:
        assert best_minute_cn is not None
        single_day_cn = select_single_date_minutes(best_minute_cn, cfg.run.single_date)
        write_csv_with_excel_copy(single_day_cn, out_dir / "date_minute_power.csv")
        sun_path_points = build_single_day_sun_path(best_minute, cfg.run.single_date, step_min=5)

    if cfg.optimization.export_visuals:
        write_heatmap(grid, visuals_dir / "optimization_heatmap.png", "objective_score")
        render_panel_visual_3d(
            paths,
            visuals_dir / "best_panel_on_fisheye.png",
            summary,
            cfg.sunpath.projection_model,
        )
        render_interactive_sky_dome(
            paths,
            visuals_dir / "interactive_sky_dome.html",
            summary,
            cfg.sunpath.projection_model,
            sun_path_points,
        )

    print(f"[optimization] Optimal tilt angle: {summary['tilt_deg']:.2f} deg")
    print(f"[optimization] Optimal azimuth angle: {summary['azimuth_deg']:.2f} deg")
    print(f"[optimization] Objective function J: {summary['objective_score']:.6f}")
    print(f"[optimization] P10: {summary['daily_p10']:.4f}")
    print(f"[optimization] P10 gap: {summary['p10_gap']:.6f}")
    print(f"[optimization] Fixed-south annual total: {summary['fixed_south_annual_total']:.4f}")
    print(f"[optimization] Annual total: {summary['annual_total']:.4f}")
    print(f"[optimization] Improvement: {summary['improvement_kwh']:.4f}")
    print(f"[optimization] Output directory: {out_dir}")
    return summary


if __name__ == "__main__":
    from config import build_config
    from pipeline import prepare_run_paths

    app_cfg = build_config()
    run_paths = prepare_run_paths(app_cfg)
    run_optimization(app_cfg, run_paths)
