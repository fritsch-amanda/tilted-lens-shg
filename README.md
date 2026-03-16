# Tilted BK7 Lens — ASM + SHG + Phase Visualizer

An interactive GUI for simulating the propagation of structured laser beams through a tilted BK7 plano-convex lens, including second-harmonic generation (SHG) and full phase visualization. Built with NumPy, Matplotlib, and SciPy.

---

## Features

- **Angular Spectrum Method (ASM)** propagation of fundamental and SHG fields
- **Multiple input beam modes**: TEM₀₀, HG(1,0)@45°, HG(2,0)@45°, LG(p=0,ℓ=1), LG(p=0,ℓ=2)
- **Tilted lens astigmatism**: tangential and sagittal focal lengths computed via Snell's law through BK7 (Sellmeier dispersion)
- **SHG field** modelled as `E²` propagated at `λ/2` via ASM
- **High-quality rendering** at 1024×1024 grid resolution with bicubic interpolation — no contour staircase artefacts
- **Phase maps** rendered as full-brightness HSV hue wheels
- **Adaptive windowing** crops each panel to 99% encircled power for a tight, unpadded view
- **Interactive sliders and radio buttons** for real-time parameter sweeps

---

## Fixed Parameters

| Parameter | Value |
|---|---|
| Fundamental wavelength λ | 1064 nm |
| SHG wavelength | 532 nm |
| Input beam waist w₀ | 900 µm |
| Rayleigh range zᴿ | ~2.39 m |
| Lens target focal length f | 200 mm |
| Lens glass | BK7 (Sellmeier dispersion) |

---

## Interactive Controls

| Control | Range | Default |
|---|---|---|
| Tilt angle θ (°) | 0 – 55° | 22° |
| Propagation window (mm) | 50 – 800 mm | 450 mm |
| Crystal / convergence position (mm) | 1 – 750 mm | 180 mm |
| Propagation after SHG (mm) | 0 – 600 mm | 0 mm |
| Input mode | 5 options | TEM₀₀ |

---

## Layout

```
┌─────────────────────────────────────────────────────────┐
│  Axial beam profile  (tangential + sagittal w_eff)       │
├──────────────┬───────────────┬───────────────┬──────────┤
│ Input  |E|²  │ Fund. ω  |E|² │ SHG 2ω  |E|² │  Info   │
├──────────────┼───────────────┼───────────────┤  panel  │
│ Input phase  │ Fund. phase   │ SHG phase     │         │
└──────────────┴───────────────┴───────────────┴──────────┘
```

Intensity panels use the **magma** colormap (fundamental) and **hot** colormap (SHG). Phase panels use a full-brightness **HSV** hue wheel with a small polar legend.

---

## Requirements

```
numpy
matplotlib
scipy
```

Install with:

```bash
pip install numpy matplotlib scipy
```

---

## Usage

```bash
python tilted_lens_sim.py
```

The interactive window opens immediately. Drag any slider or click a radio button to update all panels in real time.

---

## Physics Notes

- **BK7 refractive index** is computed from the Sellmeier equation at λ = 1064 nm.
- **Tilted lens focal lengths** for the tangential (`fₜ`) and sagittal (`fₛ`) planes are derived by applying Snell's law at the curved surface, yielding two distinct foci — the source of astigmatism.
- **ABCD / q-parameter** tracing gives the analytical waist positions and sizes used in the axial profile plot.
- **ASM propagation** applies a band-limited angular spectrum transfer function; evanescent components (k² > k²) are zeroed.
- **SHG** is approximated as `E_SHG ∝ E²_fund` (undepleted pump, plane-wave phase-matching assumed), then propagated at λ/2 via ASM.
- The **effective mode radius** for higher-order modes is scaled by `√(N+1)` where N is the mode order, so the grid and window always enclose the full beam.

---

## Key Quality Parameters

| Parameter | Value | Effect |
|---|---|---|
| `GRID_N` | 1024 | 4× more pixels vs. 512; finer spatial sampling |
| `FILL_FAC` | 0.72 | Tighter adaptive crop (99% encircled power) |
| `interpolation` | `'bicubic'` | Sub-pixel smooth rendering in imshow |
| `figure.dpi` | 120 | Sharper display on high-DPI screens |
