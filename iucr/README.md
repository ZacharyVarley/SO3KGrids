# IUCR panel exports

Scripts here produce **submission-ready figures** with content-based file names (see `stems.py`). Split panels omit `(a)` / `(b)` letters; combined 3D figures omit panel letters too (add them in LaTeX if needed).

## Requirements

- Existing data files only for curves/CDFs (nothing recomputed):
  - `figure2_data.json`
  - `figure3_data_witness_col.npz`, `figure3_data_self_nn_col.npz`
  - `figure4_data.npz`
  - `figure5_data.npz`
- Figure 1 builds one matched rejection/KR dataset on the fly (needs **pykeops**; same as `figure1_col.py`).
- Figure 6 builds matched primitive/FCC KR grids on the fly (same as `figure6.py`).
- Saved layout JSON from the interactive tools:
  - `figure1_col_settings.json`
  - `figure2_col_settings.json`
  - `figure3_col_settings.json` (under a `"view"` key)
  - `figure4_col_settings.json`
  - `figure5_col_settings.json`
  - `figure6_settings.json`
- LaTeX on `PATH` (figures use `text.usetex`)
- Poppler **`pdftops`** on `PATH` for EPS (`conda install -c conda-forge poppler` or `apt install poppler-utils`)

## Run

From the repository root:

```bash
python -m iucr.export_panels
```

Optional: `--only 1 4 6` or `--out-dir /path/to/output`. Numbers map to content groups:

| `--only` | Content |
|----------|---------|
| 1 | SO(3)/T rejection vs KR (combined) |
| 2 | $E_3/E_3^*$ and $\mathrm{cu}_z^*$ panels |
| 3 | Witness and self NN CDFs |
| 4 | Grid-method $E_3$, CR, legend, stacked reference |
| 5 | Thomson $E_3$ and covering radius vs iteration |
| 6 | Laue O primitive vs FCC KR (full-width 2×3) |

**Preserved from JSON:** axis ranges, log scales, series visibility, arrows, fits, figure-2 cu_z options, figure-5 `plot_cr_ratio`, figure-4 legend layout, 3D views for figures 1 and 6.

**Standardized for submission** ([IUCr artwork guide](https://journals.iucr.org/j/services/help/artwork/guide.html)):

- Single-column width **88.5 mm** for 2D panels
- Full-page width **171 mm** for `01_so3t_rejection_vs_kr_fz` (side-by-side, like `figure1.py`) and `12_laue_o_pc_vs_fcc_kr_grids`, with **~2.5 mm** label typography at native width
- PNG at **600 d.p.i.**, plus **EPS** (vector, PDF→Poppler) and PDF (usetex)
- Line weights **1.0 pt** (0.35–1.5 pt band); 2D markers **4 pt**
- Font sizes in the **1.5–3 mm** band at column width
- Figure 4: Fibonacci curves subsampled to match other series marker density

## Output

Default directory: `iucr_panels/`

Each stem is saved as `.png`, `.eps`, and `.pdf`. Names are **ordered by prefix** (`01_` … `12_`) and describe content, not journal figure numbers:

| Stem | Content |
|------|---------|
| `01_so3t_rejection_vs_kr_fz` | SO(3)/T: rejection vs KR in the FZ (side-by-side, full page) |
| `02_e3_ratio_vs_ns3` | $E_3/E_3^*$ vs $N_{S^3}$ (cubochoric anisotropy) |
| `03_cuz_star_vs_cuxy` | $\mathrm{cu}_z^*$ vs $\mathrm{cu}_{xy}$ |
| `04_witness_nn_cdf` | Witness-set NN CDF |
| `05_self_nn_cdf` | Self-NN CDF |
| `06_e3_ratio_grid_methods` | $E_3/E_3^*$ vs $N_{S^3}$ (grid methods) |
| `07_cr_ratio_grid_methods` | Covering-radius ratio vs $N_{S^3}$ |
| `08_grid_methods_legend` | Shared 4-column legend |
| `09_grid_methods_stacked` | E3 + CR + legend stacked (layout reference) |
| `10_thomson_e3_ratio_vs_iter` | Thomson $E_3/E_3^*$ vs iteration |
| `11_covering_radius_vs_iter` | Covering radius vs iteration |
| `12_laue_o_pc_vs_fcc_kr_grids` | Laue O: primitive vs FCC KR (CU / HO / FZ), full page |

### EPS vs PDF

Matplotlib rasterizes direct EPS when `text.usetex=True`. Submission EPS is built from the usetex **PDF** via `pdftops -eps`. Check `head -3 file.eps` for `poppler pdftops` and `PDF Creator: Matplotlib`.

## Typography

Adjust defaults in `figure_ui_common.iucr_font_sizes()` or tune interactively then re-export. Stacked/combined layouts in the repo root still use `figure*_manual.py` / `figure*_col.py`.
