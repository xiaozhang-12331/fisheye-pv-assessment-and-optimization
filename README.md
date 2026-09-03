# PV Sky-View Analysis System (Refactored Pipeline Documentation)

This document describes the code structure, inputs and outputs, parameter locations, computation principles, optimization algorithm, and interactive 3-D visualization of `D:\projects\image_recog\新架构`.

## 1. Overall Functionality

The refactored pipeline is organized into four stages:

1. Semantic segmentation of fisheye images: identify sky, vegetation, building, and other regions.
2. Solar trajectory and occlusion determination: project the solar position onto the fisheye image and decide, for every minute, whether the sun is obstructed.
3. Radiation computation: compute the irradiance components based on the occlusion type, clear-sky or meteorological data, and the diffuse model.
4. PV panel orientation optimization: under the occlusion-modulated radiation environment, find the optimal fixed tilt and azimuth angles for the whole year and compute the PV yield.

Ordinary radiation computation and best-panel computation are kept separate:

- `radiation/`: pure radiation computation results.
- `pv_best_panel/`: PV yield and visualization results for the optimal tilt/azimuth panel.

## 2. Main Code Locations

- `config.py`: centralized configuration entry. Adjust parameters here first.
- `pipeline.py`: master entry point; runs segmentation, occlusion, radiation, and optimization in sequence.
- `modules/segmentation.py`: semantic segmentation of fisheye images.
- `modules/sunpath_occlusion.py`: solar trajectory overlay and occlusion determination.
- `modules/radiation.py`: radiation computation.
- `modules/pv_yield.py`: PV panel parameters and irradiance-to-yield conversion.
- `modules/optimizer.py`: annual fixed tilt/azimuth orientation optimization.
- `modules/visualization_3d.py`: static 3-D rendering and the interactive Three.js sky dome.

## 3. Input Locations

### 3.1 Fisheye image input

Configured in `config.py -> PathConfig.input_dir`:

```python
input_dir = Path(r"D:\projects\image_recog\in")
```

Notes:

- The pipeline reads the first image found in this directory.
- The image stem becomes the result directory name; e.g. `4.jpg` maps to `results/4/`.

Supported formats: jpg, jpeg, png, bmp, tif, tiff.

### 3.2 Model paths

Configured in `config.py -> PathConfig`:

```python
ade_model_dir = Path(r"D:\projects\image_recog\models\nvidia_segformer_b4_ade")
binary_base_model_dir = Path(r"D:\projects\image_recog\models\nvidia_segformer_b4_ade")
binary_ckpt_path = Path(r"D:\projects\image_recog\models\ckpts\best_segformer_b4_binary.pt")
```

Notes:

- `ade_model_dir` is the base model for the semantic segmentation.
- `binary_ckpt_path` is the auxiliary binary building-segmentation checkpoint.
- The segmentation module runs offline by default and never downloads models from the network.

### 3.3 TMY minute-level meteorological data

Configured in `config.py -> PathConfig.tmy_minute_csv`:

```python
tmy_minute_csv = Path(r"D:\projects\image_recog\radiation_out\TMY_Baoding_minute.csv")
```

Required columns: `date`, `time`, `GHI`, `DNI`, `DHI`.

A copy of the Baoding TMY file used in the paper is bundled in this repository at
`data/TMY_Baoding_minute.csv`. Point `tmy_minute_csv` to this file (or copy it to
the location shown above) to reproduce the published results with the exact
input data.

Notes:

- This is a typical meteorological year (TMY) dataset.
- During optimization it is merged with the annual occlusion data by month-day-hour-minute.
- The TMY year and the occlusion CSV year do not need to match.

### 3.4 Annual occlusion data

Configured in `config.py -> OptimizationConfig.occlusion_csv`:

```python
CENTRAL_OVERRIDES = {
    "optimization.enabled": True,
    "optimization.occlusion_csv": r"D:\path\to\your_fullyear_sunpath.csv",
}
```

Required columns: `date`, `time`, `altitude_deg`, `azimuth_deg`, `occlusion`.

Notes:

- `occlusion` comes from the fisheye-based occlusion determination and is the core input of the optimization.
- If not set explicitly, the program first looks for the annual occlusion CSV that matches the current image name.
- If no annual occlusion CSV is found, the program reuses the fisheye segmentation result of the current measurement point and automatically computes per-minute solar positions and occlusion for all 365 days of the TMY.
- The annual occlusion sequence does not re-identify obstacles throughout the year; it applies the sky/tree/building environment observed in the single fisheye image to the annual solar trajectory.
- Seasonal variations such as vegetation transmittance are not encoded in the occlusion CSV; they are handled by the seasonal transmittance parameters in the radiation module.
- The program never uses the occlusion CSV of a different measurement point, to avoid mismatched obstruction environments.

## 4. Output Locations

Assume the input image is `4.jpg`; the output root directory is:

```text
D:\projects\image_recog\results\4
```

### 4.1 Segmentation outputs

Directory:

```text
results/4/segmentation/
```

Main files:

- `4_square.png`: square-cropped, rotation-corrected fisheye image.
- `4_overlay.png`: segmentation overlay.
- `4_mask_sky.png`: sky mask.
- `4_mask_veg.png`: vegetation mask.
- `4_mask_bld.png`: building mask.
- `4_mask_oth.png`: other-class mask (only written when `keep_other_class=True`).
- `4_soft4.npz`: four-channel soft probability map.

If `config.py -> SegmentationConfig.keep_other_class=False` (the default), the output uses three-class semantics:
`sky / vegetation / building`. The `other` class is merged into `building` at the probability level;
`4_mask_oth.png` is then removed (any stale copy is deleted);
`4_soft4.npz` keeps the four-channel format, but the other channel is merged into the building channel and zeroed.

### 4.2 Solar trajectory and occlusion outputs

Directory:

```text
results/4/sunpath/4/
```

Main files:

- `sun_occlusion_4_YYYY-MM-DD.csv`: per-minute solar occlusion results for the specified date.
- `4_sunpath.png`: solar trajectory overlay image.

### 4.3 Ordinary radiation outputs

Directory:

```text
results/4/radiation/4/
```

Main files:

- `4_radiation.csv`

Notes:

- This is the pure radiation computation result.
- It does not represent the optimized best-panel result.
- It uses the single tilt/azimuth configured in `config.py -> surface.tilt_deg` and `surface.azimuth_deg`.

### 4.4 Best-panel outputs

Directory:

```text
results/4/pv_best_panel/
```

Two main tables are written:

1. `annual_daily_power.csv`

   Contents:

   - date
   - daily_energy_kwh

   Notes:

   - Uses the TMY minute-level data.
   - Uses the annual occlusion CSV.
   - Uses the optimal fixed tilt/azimuth found by the optimization.

2. `date_minute_power.csv`

   Contents:

   - time
   - solar altitude angle
   - solar azimuth angle
   - occlusion type
   - panel tilt angle
   - panel azimuth angle
   - total POA irradiance
   - power
   - per-step energy

   Notes:

   - The date comes from `config.py -> RunConfig.single_date`.
   - TMY and occlusion data are matched by month-day; years need not agree.

### 4.5 Visualization outputs

Directory:

```text
results/4/pv_best_panel/visuals/
```

Main files:

- `best_panel_on_fisheye.png`: static 3-D hemisphere preview.
- `interactive_sky_dome.html`: draggable interactive 3-D sky dome.
- `fisheye_texture.png`: backup copy of the raw fisheye texture.
- `fisheye_sky_texture.png`: sky-only texture; the opacity slider controls this layer.
- `fisheye_obstacle_texture.png`: non-sky texture (vegetation/building/other); always opaque.
- `optimization_heatmap.png`: tilt/azimuth score heatmap.

Interactive HTML controls:

- Left-drag: rotate the view.
- Scroll wheel: zoom.
- Right-drag: pan.
- "Reset view" button: restore the default view.

Caveats:

- The HTML loads Three.js and OrbitControls from a CDN.
- Without network access the HTML may fail to load Three.js.
- For offline use, place the Three.js files in a local `vendor/` directory.

## 5. Parameters

All recommended parameters are adjusted in `config.py`.

### 5.1 Highest-priority override block

Location:

```python
CENTRAL_OVERRIDES: dict[str, Any] = {}
```

Example:

```python
CENTRAL_OVERRIDES = {
    "optimization.enabled": True,
    "optimization.occlusion_csv": r"D:\path\to\your_fullyear_sunpath.csv",
    "run.single_date": "2026-05-08",
    "pv_panel.panel_count": 2,
}
```

Priority order:

```text
CENTRAL_OVERRIDES > config.py defaults > module-internal defaults
```

### 5.2 Run parameters (RunConfig)

Location: `config.py -> RunConfig`

- `single_date`: the single date to analyze; affects the single-day occlusion, the single-day radiation, and the `date_minute_power.csv` output.
- `overwrite`: whether to delete the previous result directory.
- `time_step_min`: time step of the occlusion computation, default 1 minute.
- `draw_freq_min`: drawing interval of the trajectory overlay.

### 5.3 Geographic parameters (SiteConfig)

Location: `config.py -> SiteConfig`

- `tz`: time zone, e.g. `Asia/Shanghai`.
- `lat`: latitude.
- `lon`: longitude.
- `alt_m`: elevation in meters.

These parameters feed the solar position computation, the clear-sky model, and the radiation computation.

### 5.4 Image orientation parameters (ImageDirectionConfig)

Location: `config.py -> ImageDirectionConfig`

- `top_azimuth_deg`: azimuth the top edge of the fisheye image actually points to. 0=north, 90=east, 180=south, 270=west.
- `mirror_azimuth`: whether to apply an east-west mirror correction when projecting the solar trajectory.

### 5.5 Segmentation parameters (SegmentationConfig)

Location: `config.py -> SegmentationConfig`

- `fp16`: whether to run half-precision inference.
- `tile`: sliding-window size.
- `overlap`: sliding-window overlap in pixels.
- `scales`: multi-scale inference ratios.
- `use_tta_hflip`: whether to use horizontal-flip TTA.
- `bin_infer_size`: inference size of the binary building model.
- `enable_rotation`: whether to correct rotation according to the image orientation.
- `inner_margin`: shrink ratio of the fisheye circle.
- `save_soft_npz`: whether to save the four-class probability map.
- `keep_other_class`: whether to keep the `other` class. Default `False`: the other class is merged into building, and segmentation, occlusion, radiation, and optimization all treat it as building. Set `True` to keep four-class semantics.

### 5.6 Solar trajectory parameters (SunpathConfig)

Location: `config.py -> SunpathConfig`

- `projection_model`: fisheye projection model, default `equisolid`.
- `min_altitude_deg`: trajectory points below this solar elevation are not treated as valid visible positions.
- `max_jump_px`: breakage threshold for trajectory segments.

- `save_csv`: whether to save the occlusion CSV.
- `save_image`: whether to save the trajectory image.

### 5.7 Ordinary radiation receiver parameters (SurfaceConfig)

Location: `config.py -> SurfaceConfig`

- `tilt_deg`: receiver tilt used by the ordinary radiation computation.
- `azimuth_deg`: receiver azimuth used by the ordinary radiation computation.
- `albedo`: ground-surface albedo.

Notes:

- These values are used for the ordinary radiation output in `radiation/`.
- The optimization module sweeps tilt and azimuth over ranges; it does not use only these single values.

### 5.8 Radiation parameters (RadiationConfig)

Location: `config.py -> RadiationConfig`

- `climate_type`: climate-type correction.
- `link_turbidity`: Linke turbidity.
- `clearsky_bias`: clear-sky model correction factor.
- `enable_low_sun_correction`: whether to enable the low-sun correction.
- `low_sun_factor`: strength of the low-sun correction.

### 5.9 PV panel parameters (PVPanelConfig)

Location: `config.py -> PVPanelConfig`

- `enabled`: whether to convert irradiance into PV yield.
- `pv_tech`: module technology name.
- `module_width_mm`: module width in mm.
- `module_height_mm`: module height in mm.
- `module_p_stc_w`: module rated power at STC in W.
- `pr_total`: total system performance ratio, including inverter, wiring, and other losses.
- `installable_ratio`: effective installation ratio.
- `panel_count`: number of modules.

Yield conversion formulas:

```text
module_area_m2 = module_width_m * module_height_m
eta_stc = module_p_stc_w / (1000 * module_area_m2)
active_area_m2 = module_area_m2 * panel_count * installable_ratio
power_w = POA_Total(W/m²) * active_area_m2 * eta_stc * pr_total
energy_kwh = power_w * dt_hours / 1000
```

The default values originate from the legacy facade PV potential script in the project history.

### 5.10 Optimization parameters (OptimizationConfig)

Location: `config.py -> OptimizationConfig`

- `enabled`: whether optimization is enabled.
- `tilt_range_deg`: coarse-search tilt range `(start, end, step)`.
- `azimuth_range_deg`: coarse-search azimuth range `(start, end, step)`.
- `refine_radius_deg`: refinement search radius.
- `refine_step_deg`: refinement search step.
- `objective`: the current primary objective is `annual_p10_soft_constraint`.
- `occlusion_csv`: explicit annual occlusion CSV.
- `export_minute_power`: whether to export the per-minute power table.
- `export_visuals`: whether to export visualizations.

## 6. Computation Principles

### 6.1 Image segmentation

The segmentation module uses a SegFormer semantic segmentation model to classify the fisheye image into:

- sky
- vegetation
- building
- other

The resulting masks are the basis of the subsequent occlusion determination.

### 6.2 Fisheye geometry

The fisheye image is treated as a hemispherical sky projection.

The default projection model is `equisolid`. The solar trajectory projection uses:

```text
theta = 90° - altitude
rho = sqrt(2) * R * sin(theta / 2)
u = cx + rho * sin(azimuth)
v = cy - rho * cos(azimuth)
```

where:

- `altitude` is the solar elevation angle.
- `azimuth` is the solar azimuth angle.
- `R` is the fisheye circle radius.
- `(cx, cy)` is the fisheye circle center.

### 6.3 Occlusion determination

The solar elevation and azimuth are computed for every minute.

The solar position is then projected into fisheye image coordinates.

The occlusion type is decided by the mask class at that pixel:

- `sky`: unobstructed.
- `vegetation`: obstructed by vegetation.
- `building`: obstructed by a building.
- `other`: obstructed by other objects.
- `outside`: the sun is outside the fisheye circle, or it is night.

The occlusion type modulates the direct-beam transmittance and the diffuse computation.

### 6.4 Radiation computation

The radiation module computes:

- `GHI`: global horizontal irradiance.
- `DNI`: direct normal irradiance.
- `DHI`: diffuse horizontal irradiance.
- `DNI_eff`: effective direct irradiance after occlusion transmittance.
- `Diffuse`: total dynamic diffuse irradiance.
- `POA_Total`: total plane-of-array irradiance.

POA uses the Perez model from pvlib:

```python
pvlib.irradiance.get_total_irradiance(..., model="perez")
```

### 6.5 Optimization algorithm

The optimization module searches for the best fixed installation:

- tilt angle `tilt_deg`
- azimuth angle `azimuth_deg`

Procedure:

1. Read the annual occlusion CSV.
2. Read the TMY minute-level meteorological data.
3. Merge occlusion and TMY by month-day-hour-minute.
4. Run one base radiation computation to obtain the occlusion-modulated direct and diffuse components.
5. For each candidate tilt/azimuth pair, recompute the POA.
6. Convert POA into PV power and per-step energy.
7. Aggregate daily energy per day.
8. Compute the P10 of the annual daily-energy series.
9. Select the angle pair with the highest P10.
10. Tie-break by higher annual total PV yield.

Why P10:

- The target application is standalone PV devices whose daily load must be covered by same-day generation.
- Maximizing only the annual total may favor summer or a few high-yield days.
- P10 emphasizes the weaker days of the year and is better suited to daily minimum-reliability requirements.

## 7. Interactive 3-D Sky Dome

File:

```text
results/<image_name>/pv_best_panel/visuals/interactive_sky_dome.html
```

Implementation:

- Uses Three.js.
- Builds a 3-D hemisphere mesh.
- Maps the fisheye image as a texture onto the inner hemisphere surface.
- Places a 3-D PV panel model at the center.
- Rotates the panel according to the optimal tilt and azimuth.
- Uses OrbitControls for dragging, zooming, and panning.

Coordinate conventions:

- X: east.
- Y: north.
- Z: zenith.
- Azimuth: 0=north, 90=east, 180=south, 270=west.
- The panel azimuth follows the `pvlib.surface_azimuth` convention: the horizontal projection direction of the panel normal, i.e. the direction the panel faces; 190° means 10° west of south, not north.
- In the interactive view a yellow arrow indicates the panel normal/facing direction, for verifying the optimal azimuth.

Caveats:

- This does not reconstruct real 3-D building depth from a single image.
- The fisheye image is only used as a sky-dome texture.
- The actual occlusion determination still comes from the original fisheye segmentation and the solar trajectory analysis.

## 8. Usage

### 8.1 Standard single-day run

Run:

```powershell
python D:\projects\image_recog\新架构\pipeline.py
```

This executes in sequence:

1. Semantic segmentation.
2. Single-day solar trajectory occlusion.
3. Single-day radiation computation.
4. Annual optimization, if `optimization.enabled=True`.

### 8.2 Enabling annual optimization

Set at the top of `config.py`:

```python
CENTRAL_OVERRIDES = {
    "optimization.enabled": True,
    "optimization.occlusion_csv": r"D:\path\to\your_fullyear_sunpath.csv",
}
```

### 8.3 Choosing the date for the per-minute power output

Modify:

```python
CENTRAL_OVERRIDES = {
    "run.single_date": "2026-05-08",
}
```

Output file:

```text
results/<image_name>/pv_best_panel/date_minute_power.csv
```

## 9. FAQ

### 9.1 Why is there no optimization result?

Check:

```python
optimization.enabled
```

It must be `True`.

If no annual occlusion CSV matches the current measurement point, the program automatically generates the annual occlusion series from the current fisheye segmentation result.

### 9.2 Why does the HTML fail to open or show a blank page?

The HTML loads Three.js from a CDN.

If the machine is offline, the browser may fail to load Three.js.

Solutions:

- Open it with network access.
- Or switch to local `vendor/three.module.js` and `OrbitControls.js`.

### 9.3 Why is the 3-D sky dome not a real 3-D building model?

A single fisheye image carries no depth information and cannot reliably reconstruct real building geometry.

The current 3-D view maps the fisheye image onto the sky hemisphere to display the obstruction environment and the best panel pose.

The actual occlusion is still decided by the pixel-level occlusion determination of the original algorithm.

### 9.4 Why can TMY and occlusion data with different years be merged?

TMY is a typical meteorological year; its year label is only representative.

Optimization matches by month-day-hour-minute:

```text
01-01 00:00
01-01 00:01
...
```

This allows any annual solar occlusion series to be combined with the TMY radiation data.

## 10. Interpreting the Outputs

### 10.1 `annual_daily_power.csv`

Shows how much energy is generated on each day of the year.

Low-yield dates usually result from:

- Low TMY irradiance on that day.
- Low solar elevation.
- Severe obstruction at the current measurement point.
- Residual obstruction that even the optimal orientation cannot avoid.

### 10.2 `date_minute_power.csv`

Shows the minute-by-minute generation process of a single day.

Key columns:

- `occlusion_type`: which region the sun falls in during that minute.
- `poa_total_w_m2`: irradiance received by the optimal panel pose in that minute.
- `power_w`: instantaneous power after module conversion.
- `energy_kwh_per_step`: energy contributed in that minute.

### 10.3 `interactive_sky_dome.html`

Shows intuitively:

- Where the fisheye obstruction environment sits on the 3-D sky dome.
- The optimal PV panel tilt and facing direction.
- The angular relation between the panel and the dome.

## 11. Important Limitations

- The current model does not apply module temperature corrections.
- The current model ignores soiling, degradation, and inverter efficiency curves.
- The current model does not reconstruct real 3-D near-field obstructions.
- PV panel parameters only scale the yield; if area, efficiency, and PR are constants, they usually do not change the optimal angles.
- The optimal orientation is entirely determined by the fisheye occlusion determination of the current measurement point and the TMY data.
