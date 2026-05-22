# IUCR panel exports

`python -m iucr.export_panels` produces the maintained submission set with ordered, content-descriptive stems from `iucr/stems.py`.

## Requirements

- Cached inputs already present in the release tree:
  - `figures/data/figure2_anisotropy.json`
  - `figures/data/figure3_witness_nn.npz`, `figures/data/figure3_self_nn.npz`
  - `figures/data/figure4_grid_methods.npz`
  - `figures/data/figure5_thomson_relaxation.npz`
- Layout JSON in `figures/settings/`
- LaTeX on `PATH` (`text.usetex=True`)
- Poppler `pdftops` on `PATH` for EPS conversion
- Figure 1 uses `figures.so3t_rejection_vs_kr` and benefits from a working `pykeops` install
- Figure 6 uses `figures.laue_o_pc_vs_fcc` and is built directly from code

## Run

From the repository root:

```bash
python -m iucr.export_panels
```

Optional: `--only 1 4 6` or `--out-dir /path/to/output`.

| `--only` | Content |
|---|---|
| `1` | SO(3)/T rejection vs KR |
| `2` | Cubochoric anisotropy panels |
| `3` | Witness and self-NN CDFs |
| `4` | Grid-method comparison panels |
| `5` | Thomson relaxation panels |
| `6` | Laue O primitive-vs-FCC KR panel |

The exporter preserves axis limits, log scales, legend placement, annotations, and 3D views from `figures/settings/*.json`.

## Submission standard

- Single-column width: **88.5 mm**
- Full-page width: **171 mm** for the two 3D combined panels
- PNG: **600 d.p.i.**
- EPS: vector, generated from the TeX PDF via `pdftops -eps`
- PDF: TeX-rendered reference output
- Line and text sizes constrained to the IUCr artwork guide

## Output stems

Default output directory: `iucr_panels/`

| Stem | Content |
|---|---|
| `01_so3t_rejection_vs_kr_fz` | SO(3)/T rejection vs KR |
| `02_e3_ratio_vs_ns3` | $E_3/E_3^*$ vs $N_{S^3}$ |
| `03_cuz_star_vs_cuxy` | $\mathrm{cu}_z^*$ vs $\mathrm{cu}_{xy}$ |
| `04_witness_nn_cdf` | Witness-set NN CDF |
| `05_self_nn_cdf` | Self-NN CDF |
| `06_e3_ratio_grid_methods` | Grid-method $E_3/E_3^*$ |
| `07_cr_ratio_grid_methods` | Grid-method covering-radius ratio |
| `08_grid_methods_legend` | Shared legend |
| `09_grid_methods_stacked` | Stacked reference export |
| `10_thomson_e3_ratio_vs_iter` | Thomson $E_3/E_3^*$ vs iteration |
| `11_covering_radius_vs_iter` | Covering radius vs iteration |
| `12_laue_o_pc_vs_fcc_kr_grids` | Laue O primitive vs FCC KR |

## EPS note

The EPS files are true vector exports produced from the TeX PDF through Poppler. They are intentionally verbose PostScript, but not raster wrappers.

## Typography

Adjust shared IUCr typography defaults in `figures/common.py`, or tune the saved layouts in `figures/settings/` and rerun the exporter.
