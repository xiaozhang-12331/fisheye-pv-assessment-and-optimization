# -*- coding: utf-8 -*-
"""PV panel yield conversion utilities.

Formula provenance:
- `compute_pv_yield_from_radiation()` in the legacy script
  `building_facade_radiation_pv_potential_full.py`.

This module only performs the simplified STC conversion from POA irradiance to
PV power/energy:
- No additional models such as temperature correction, inverter curves, soiling,
  or module degradation are introduced.
- The legacy-code approach is retained: module STC efficiency * effective area * PR_TOTAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PVPanelSpec:
    """PV panel specification.

    Attributes:
        enabled: When False, no actual PV yield is computed and only the
            irradiance in kWh/m2 is retained.
        pv_tech: Module technology identifier, used for output records only.
        module_width_mm/module_height_mm: Dimensions of a single module.
        module_p_stc_w: Rated STC power of a single module, in W.
        pr_total: System performance ratio (PR), including combined derating
            for the inverter, wiring losses, etc.
        installable_ratio: Installable/effective area ratio.
        panel_count: Number of modules.
    """

    enabled: bool = True
    pv_tech: str = "Monocrystalline_Si"
    module_width_mm: float = 1200.0
    module_height_mm: float = 600.0
    module_p_stc_w: float = 158.4
    pr_total: float = 0.75
    installable_ratio: float = 1.0
    panel_count: int = 1


def spec_from_config(panel_cfg: Any) -> PVPanelSpec:
    """Build a lightweight spec from config.PVPanelConfig to avoid exposing dataclass details to the optimizer."""
    return PVPanelSpec(
        enabled=bool(panel_cfg.enabled),
        pv_tech=str(panel_cfg.pv_tech),
        module_width_mm=float(panel_cfg.module_width_mm),
        module_height_mm=float(panel_cfg.module_height_mm),
        module_p_stc_w=float(panel_cfg.module_p_stc_w),
        pr_total=float(panel_cfg.pr_total),
        installable_ratio=float(panel_cfg.installable_ratio),
        panel_count=int(panel_cfg.panel_count),
    )


def module_area_m2(spec: PVPanelSpec) -> float:
    """Compute the area of a single module, in m2."""
    width_m = spec.module_width_mm / 1000.0
    height_m = spec.module_height_mm / 1000.0
    area = width_m * height_m
    if area <= 0:
        raise ValueError("Invalid module area; check module_width_mm/module_height_mm")
    return float(area)


def stc_efficiency(spec: PVPanelSpec) -> float:
    """Compute the STC module efficiency following the legacy-code formula.

    eta_stc = module_p_stc_w / (1000 W/m2 * module_area_m2)
    e.g. 120 W / (1000 * 1.2 m * 0.6 m) = 0.1667.
    """
    area = module_area_m2(spec)
    if spec.module_p_stc_w <= 0:
        raise ValueError("Invalid module rated power; check module_p_stc_w")
    return float(spec.module_p_stc_w / (1000.0 * area))


def active_area_m2(spec: PVPanelSpec) -> float:
    """Compute the total effective light-receiving area.

    In the new architecture this is computed as the single-module area *
    panel count * installable ratio.
    """
    if spec.panel_count <= 0:
        raise ValueError("Invalid panel count; check panel_count")
    if spec.installable_ratio < 0:
        raise ValueError("The installable ratio must not be negative")
    return float(module_area_m2(spec) * spec.panel_count * spec.installable_ratio)


def installed_kwp(spec: PVPanelSpec) -> float:
    """Estimate the installed capacity, in kWp.

    Following the legacy code: active_area_m2 * eta_stc.
    Since eta_stc corresponds to a reference irradiance of 1 kW/m2, this product
    is numerically the capacity in kWp.
    """
    return float(active_area_m2(spec) * stc_efficiency(spec))


def yield_metadata(spec: PVPanelSpec) -> dict[str, float | int | str | bool]:
    """Return module parameters and derived metrics for writing to the summary."""
    area = module_area_m2(spec)
    active_area = active_area_m2(spec)
    eta = stc_efficiency(spec)
    return {
        "pv_enabled": spec.enabled,
        "pv_tech": spec.pv_tech,
        "module_width_mm": spec.module_width_mm,
        "module_height_mm": spec.module_height_mm,
        "module_p_stc_w": spec.module_p_stc_w,
        "module_area_m2": area,
        "eta_stc": eta,
        "installable_ratio": spec.installable_ratio,
        "panel_count": spec.panel_count,
        "active_area_m2": active_area,
        "installed_kwp": active_area * eta,
        "pr_total": spec.pr_total,
        "approx_module_count": active_area / area,
    }


def poa_to_power_w(poa_w_m2: np.ndarray | pd.Series, spec: PVPanelSpec) -> np.ndarray:
    """Convert POA_Total(W/m2) into an estimated AC-side power in W.

    power_w = POA_Total * active_area_m2 * eta_stc * pr_total
    """
    poa = np.asarray(poa_w_m2, dtype=float)
    if not spec.enabled:
        return np.zeros_like(poa, dtype=float)
    return np.nan_to_num(poa, nan=0.0) * active_area_m2(spec) * stc_efficiency(spec) * spec.pr_total


def power_to_energy_kwh(power_w: np.ndarray | pd.Series, dt_hours: float) -> np.ndarray:
    """Integrate power in W over the time step to obtain kWh."""
    if dt_hours <= 0:
        raise ValueError("The time step must be greater than 0")
    power = np.asarray(power_w, dtype=float)
    return np.nan_to_num(power, nan=0.0) * dt_hours / 1000.0


def poa_to_energy_kwh_m2(poa_w_m2: np.ndarray | pd.Series, dt_hours: float) -> np.ndarray:
    """Integrate POA_Total(W/m2) into per-area irradiance in kWh/m2."""
    if dt_hours <= 0:
        raise ValueError("The time step must be greater than 0")
    poa = np.asarray(poa_w_m2, dtype=float)
    return np.nan_to_num(poa, nan=0.0) * dt_hours / 1000.0


def add_yield_columns(df: pd.DataFrame, spec: PVPanelSpec, dt_hours: float) -> pd.DataFrame:
    """Add irradiance, power, and PV yield columns to the sub-hourly/hourly result table.

    When spec.enabled=False, power_w and energy_kwh_per_step are zero, while
    energy_kwh_m2_per_step is retained for irradiance-based optimization.
    """
    if "POA_Total(W/m²)" not in df.columns:
        raise KeyError("Missing POA_Total(W/m²) column")

    out = df.copy()
    out["energy_kwh_m2_per_step"] = poa_to_energy_kwh_m2(out["POA_Total(W/m²)"], dt_hours)
    out["power_w"] = poa_to_power_w(out["POA_Total(W/m²)"], spec)
    out["energy_kwh_per_step"] = power_to_energy_kwh(out["power_w"], dt_hours)
    return out
