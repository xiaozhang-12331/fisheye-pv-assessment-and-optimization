# -*- coding: utf-8 -*-
"""
Radiation computation module (integrated version).
- Direct (beam): based on occlusion type and seasonal transmittance
- Diffuse: dynamic scattering model (sky / vegetation / building / aerosol / ground) with climate correction
- POA: Perez model transposition to the tilted plane
- Automatically reads the occlusion CSV and outputs a complete radiation CSV

New-architecture notes:
- This module corresponds to step 3 of the pipeline: it reads the occlusion CSV and
  the segmentation masks, then outputs the radiation result CSV.
- Daily parameters should preferably be adjusted in ../config.py; pipeline.py injects
  them into CFG before running.
- For later tilt/azimuth optimization, prefer reusing compute_radiation(); do not
  modify the base computation formulas.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pvlib
import cv2
import glob
import re
from tqdm import tqdm

# =========================================================
# All tunable parameters (edit here only)
# =========================================================
class CFG:
    # ---- Paths ----
    OCCLUSION_DIR = Path(r"D:\projects\image_recog\results\2423\sunpath\2423")
    OUTPUT_ROOT   = Path(r"D:\projects\image_recog\results\2423\radiation")
    SEG_MASK_DIR  = Path(r"D:\projects\image_recog\results\2423\segmentation")
    KEEP_OTHER_CLASS = False

    # ---- Date ----
    SINGLE_DATE = "2026-04-15"

    # ---- Geography ----
    LAT = 38.8825
    LON = 115.5556
    ALT_M = 14.0
    TZ = "Asia/Shanghai"

    # ---- Receiving surface ----
    SURF_TILT_DEG = 0.0
    SURF_AZIMUTH_DEG = 180.0
    ALBEDO = 0.2

    # ---- Clear-sky model ----
    LINK_TURBIDITY = 5.5
    CLEARSKY_BIAS = 0.85          # Clear-sky bias correction
    ENABLE_LOW_SUN_CORRECTION = True
    LOW_SUN_FACTOR = 0.20

    # ---- Climate correction ----
    CLIMATE_TYPE = "temperate"     # temperate / subtropical / tropical / arid / plateau

    # ---- Season mapping ----
    SEASON_MAP = {
        "Spring": (3, 4, 5),
        "Summer": (6, 7, 8),
        "Autumn": (9, 10, 11),
        "Winter": (12, 1, 2)
    }

    # ---- Direct-beam transmittance ----
    DIRECT_TRANSMITTANCE = {
        "Spring": {"sky": 1.0, "building": 0.0, "vegetation": 0.525, "other": 0.0, "outside": 0.0},
        "Summer": {"sky": 1.0, "building": 0.0, "vegetation": 0.40, "other": 0.0, "outside": 0.0},
        "Autumn": {"sky": 1.0, "building": 0.0, "vegetation": 0.525, "other": 0.0, "outside": 0.0},
        "Winter": {"sky": 1.0, "building": 0.0, "vegetation": 0.70, "other": 0.0, "outside": 0.0},
    }
    DEFAULT_TAU = 0.0

    # ---- Dynamic scattering parameters ----
    SKY_FACTOR = {          # Base coefficient of sky-diffuse scattering
        "morning": 0.90, "noon": 1.00, "afternoon": 0.95, "evening": 0.75, "night": 0.0
    }
    VEGETATION_BASE = {     # Base vegetation scattering values (by season)
        "spring": 0.22, "summer": 0.30, "autumn": 0.18, "winter": 0.10
    }
    CIRCUMSOLAR_FACTOR = {  # Circumsolar scattering enhancement coefficient
        "morning": 0.10, "noon": 0.18, "afternoon": 0.14, "evening": 0.06, "night": 0.0
    }
    BUILDING_BASE = {       # Base coefficient of building scattering
        "morning": 0.08, "noon": 0.12, "afternoon": 0.10, "evening": 0.05, "night": 0.0
    }
    BUILDING_EDGE_BOOST = { # Enhancement for bright building edges
        "morning": 0.08, "noon": 0.20, "afternoon": 0.15, "evening": 0.05, "night": 0.0
    }
    OTHER_SCATTER = {       # Aerosol scattering coefficient
        "morning": 0.04, "noon": 0.03, "afternoon": 0.035, "evening": 0.025, "night": 0.0
    }
    GROUND_REFLECTANCE = {  # Ground-reflected coefficient
        "morning": 0.10, "noon": 0.15, "afternoon": 0.13, "evening": 0.08, "night": 0.0
    }

# =========================================================
# Utility functions
# =========================================================
def auto_find_occlusion_csv():
    """Automatically locate the most recent occlusion CSV."""
    pattern = str(CFG.OCCLUSION_DIR / "**" / "sun_occlusion_*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError("No sun_occlusion CSV found")
    return Path(max(files, key=lambda p: Path(p).stat().st_mtime))

def extract_image_name(path):
    """Extract the image name from the occlusion CSV filename."""
    m = re.match(r"sun_occlusion_(.+?)_(\d{4}-\d{2}-\d{2})$", path.stem)
    if not m:
        raise ValueError("Failed to parse the image name")
    return m.group(1)

def get_season(month):
    """Map a month to the season name (spring/summer/autumn/winter)."""
    if month in [3, 4, 5]: return "spring"
    if month in [6, 7, 8]: return "summer"
    if month in [9, 10, 11]: return "autumn"
    return "winter"

def climate_factor():
    """Climate-type correction coefficient."""
    table = {
        "temperate": 1.00, "subtropical": 1.10,
        "tropical": 1.20, "arid": 0.85, "plateau": 1.15
    }
    return table.get(CFG.CLIMATE_TYPE, 1.0)

def get_time_period(times):
    """Classify each timestamp as morning / noon / afternoon / evening / night."""
    loc = pvlib.location.Location(CFG.LAT, CFG.LON, CFG.TZ, CFG.ALT_M)
    solpos = loc.get_solarposition(times)
    elevation = solpos["apparent_elevation"].values
    rs = loc.get_sun_rise_set_transit(times)
    sunrise = rs["sunrise"].dt.tz_convert(CFG.TZ)
    sunset = rs["sunset"].dt.tz_convert(CFG.TZ)

    result = []
    for t, sr, ss, ele in zip(times, sunrise, sunset, elevation):
        if ele <= 0:
            result.append("night")
            continue
        progress = (t - sr).total_seconds() / (ss - sr).total_seconds()
        if progress < 0.25:
            result.append("morning")
        elif progress < 0.65:
            result.append("noon")
        elif progress < 0.9:
            result.append("afternoon")
        else:
            result.append("evening")
    return np.array(result)

def compute_svf(mask_dir, image_name):
    """Compute the sky view factor (SVF)."""
    masks = {}
    for name in ["sky", "veg", "bld", "oth"]:
        path = mask_dir / f"{image_name}_mask_{name}.png"
        if name == "oth" and not path.exists():
            ref = next((m for m in (masks.get("sky"), masks.get("veg"), masks.get("bld")) if m is not None), None)
            if ref is None:
                raise FileNotFoundError(path)
            masks[name] = np.zeros_like(ref, dtype=bool)
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        masks[name] = img > 127
    if not CFG.KEEP_OTHER_CLASS:
        masks["bld"] = masks["bld"] | masks["oth"]
        masks["oth"] = np.zeros_like(masks["oth"], dtype=bool)

    valid = masks["sky"] | masks["veg"] | masks["bld"] | masks["oth"]
    h, w = masks["sky"].shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rmax = min(h, w) / 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rmax
    theta = (np.pi / 2) * np.clip(rr, 0, 1)
    weight = np.cos(theta)
    weight[~valid] = 0
    svf = (masks["sky"].astype(float) * weight).sum() / weight.sum()
    return svf

def build_clearsky(times):
    """Ineichen clear-sky model -> DNI, GHI, DHI."""
    loc = pvlib.location.Location(CFG.LAT, CFG.LON, CFG.TZ, CFG.ALT_M)
    solpos = loc.get_solarposition(times)
    elevation = solpos["apparent_elevation"].values
    zenith = np.clip(90.0 - elevation, 0.0, 89.9)

    am_rel = pvlib.atmosphere.get_relative_airmass(zenith)
    pressure = pvlib.atmosphere.alt2pres(CFG.ALT_M)
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, pressure)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    tl = pd.Series(CFG.LINK_TURBIDITY, index=times)

    cs = pvlib.clearsky.ineichen(
        apparent_zenith=zenith,
        airmass_absolute=am_abs,
        linke_turbidity=tl,
        altitude=CFG.ALT_M,
        dni_extra=dni_extra
    )
    dni = cs["dni"].values * CFG.CLEARSKY_BIAS
    ghi = cs["ghi"].values * CFG.CLEARSKY_BIAS
    dhi = cs["dhi"].values * CFG.CLEARSKY_BIAS

    if CFG.ENABLE_LOW_SUN_CORRECTION:
        scale = np.sin(np.radians(np.clip(elevation, 0, 90)))
        scale = 1.0 - CFG.LOW_SUN_FACTOR * (1 - scale)
        dni *= scale
        ghi *= scale

    return pd.DataFrame({"time": times, "GHI": ghi, "DNI": dni, "DHI": dhi})

def get_tau(month, occlusion):
    """Direct-beam transmittance."""
    season = "Spring"
    for s, months in CFG.SEASON_MAP.items():
        if month in months:
            season = s
            break
    spec = CFG.DIRECT_TRANSMITTANCE.get(season, {}).get(occlusion, CFG.DEFAULT_TAU)
    if isinstance(spec, (list, tuple)):
        return (spec[0] + spec[1]) / 2.0
    return float(spec)

# =========================================================
# Main computation function
# =========================================================
def compute_radiation(df, svf):
    """
    df: DataFrame containing time, altitude_deg, azimuth_deg, occlusion, DNI, GHI, DHI.
    svf: Sky view factor.
    Returns: DataFrame with the radiation components added as new columns.
    """
    df = df.copy()
    if not CFG.KEEP_OTHER_CLASS:
        df["occlusion"] = df["occlusion"].astype(str).str.lower().replace({"other": "building"})
    times = pd.DatetimeIndex(df["time"])
    zenith = np.clip(90.0 - df["altitude_deg"].values, 0, 90)
    azimuth = df["azimuth_deg"].values
    occ = df["occlusion"].astype(str).values
    months = times.month.values

    # ---- 1. Direct (beam) ----
    tau = np.array([get_tau(m, o) for m, o in zip(months, occ)])
    dni_eff = df["DNI"].values * tau
    ghi_direct = dni_eff * np.cos(np.radians(zenith))

    # ---- 2. Dynamic scattering ----
    dhi = df["DHI"].values
    periods = get_time_period(times)
    climate_scale = climate_factor()

    sky_diff = np.zeros(len(df))
    veg_diff = np.zeros(len(df))
    bld_diff = np.zeros(len(df))
    aero_diff = np.zeros(len(df))
    ground_diff = np.zeros(len(df))
    diffuse = np.zeros(len(df))

    for i in tqdm(range(len(df)), desc="Computing diffuse components", ncols=80):
        if periods[i] == "night":
            continue

        base = dhi[i]
        season = get_season(months[i])
        p = periods[i]
        occ_i = occ[i]

        # --- Sky diffuse (scaled by SVF) ---
        sky = base * CFG.SKY_FACTOR[p] * svf
        if occ_i == "sky":
            sky += base * 0.18

        # --- Vegetation diffuse ---
        veg = base * CFG.VEGETATION_BASE[season] * (0.8 + CFG.CIRCUMSOLAR_FACTOR[p])
        if occ_i == "vegetation":
            veg += base * CFG.CIRCUMSOLAR_FACTOR[p] * 2.0
        else:
            veg += base * 0.03

        # --- Building diffuse ---
        bld = base * CFG.BUILDING_BASE[p]
        if occ_i == "building":
            bld += base * CFG.BUILDING_EDGE_BOOST[p] * 1.8
        else:
            bld += base * 0.02

        # --- Aerosol diffuse ---
        aero = base * CFG.LINK_TURBIDITY * CFG.OTHER_SCATTER[p]

        # --- Ground-reflected ---
        ground = base * CFG.GROUND_REFLECTANCE[p]

        # Climate correction
        sky *= climate_scale
        veg *= climate_scale
        bld *= climate_scale
        aero *= climate_scale
        ground *= climate_scale

        sky_diff[i] = sky
        veg_diff[i] = veg
        bld_diff[i] = bld
        aero_diff[i] = aero
        ground_diff[i] = ground
        diffuse[i] = sky + veg + bld + aero + ground

    # ---- 3. POA (plane-of-array) irradiance ----
    dni_extra = pvlib.irradiance.get_extra_radiation(times.dayofyear)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=CFG.SURF_TILT_DEG,
        surface_azimuth=CFG.SURF_AZIMUTH_DEG,
        dni=dni_eff,
        ghi=ghi_direct + diffuse,
        dhi=diffuse,
        solar_zenith=zenith,
        solar_azimuth=azimuth,
        dni_extra=dni_extra,
        model="perez",
        albedo=CFG.ALBEDO
    )

    # ---- 4. Assign results to the DataFrame ----
    df["DNI_eff"] = dni_eff
    df["Direct(W/m²)"] = ghi_direct
    df["Diffuse(W/m²)"] = diffuse
    df["Sky_Diffuse(W/m²)"] = sky_diff
    df["Vegetation_Diffuse(W/m²)"] = veg_diff
    df["Building_Diffuse(W/m²)"] = bld_diff
    df["Aerosol_Diffuse(W/m²)"] = aero_diff
    df["Ground_Reflect(W/m²)"] = ground_diff
    df["Total_Horizontal(W/m²)"] = ghi_direct + diffuse
    df["POA_Direct(W/m²)"] = np.nan_to_num(poa["poa_direct"])
    df["POA_Diffuse(W/m²)"] = np.nan_to_num(poa["poa_diffuse"])
    df["POA_Ground(W/m²)"] = np.nan_to_num(poa["poa_ground_diffuse"])
    df["POA_Total(W/m²)"] = np.nan_to_num(poa["poa_global"])
    df["Period"] = periods

    return df

# =========================================================
# Main entry point
# =========================================================
def main():
    print("=" * 60)
    print("Solar radiation computation (integrated version)")
    print("=" * 60)

    occ_path = auto_find_occlusion_csv()
    image_name = extract_image_name(occ_path)
    print(f"[Image] {image_name}")

    out_dir = CFG.OUTPUT_ROOT / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the occlusion data
    df = pd.read_csv(occ_path)
    dt = df["date"].astype(str) + " " + df["time"].astype(str)
    df["time"] = pd.to_datetime(dt).dt.tz_localize(CFG.TZ)
    df = df.drop_duplicates("time").sort_values("time")

    # SVF
    svf = compute_svf(CFG.SEG_MASK_DIR, image_name)
    print(f"[SVF] {svf:.4f}")

    # Clear-sky irradiance
    clearsky = build_clearsky(pd.DatetimeIndex(df["time"]))
    merged = df[["time", "altitude_deg", "azimuth_deg", "occlusion"]].merge(
        clearsky, on="time", how="left"
    )

    # Main computation
    result = compute_radiation(merged, svf)

    # Export CSV
    out_csv = out_dir / f"{image_name}_radiation.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Daily irradiation
    dt_hr = 1 / 60
    total = result["POA_Total(W/m²)"].sum() * dt_hr / 1000.0
    print(f"\nTotal irradiation: {total:.3f} kWh/m²")
    print(f"[Saved] {out_csv}")
    print("\n[DONE] Finished")

if __name__ == "__main__":
    main()
