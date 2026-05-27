<h1 align="center">Uniform Grids on SO(3)/K</h1>

<p align="center">
  <em>Reference implementation for</em><br/>
  <strong>Approximately Uniform Grids over Crystal Orientations</strong><br/>
  Z. T. Varley &middot; M. De Graef
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="KeOps" src="https://img.shields.io/badge/KeOps-symbolic%20kernels-6f42c1">
</p>

<p align="center">
  <img src="paper_figures/01_so3t_rejection_vs_kr_fz.png" alt="Cubochoric rejection versus KR transport in the SO(3)/T fundamental zone." width="780" />
</p>

---

## The short version

Sampling crystal orientations means sampling the quotient $\mathrm{SO}(3)/K$, where $K$ is the rotational point group of the crystal. The textbook recipe (build a cubochoric grid, then throw away every point outside the fundamental zone) is wasteful for low-symmetry groups and produces uneven sampling near zone boundaries.

This repo takes a different route. For each crystallographic point group it constructs a **constant-Jacobian Knothe-Rosenblatt transport** that pushes the homochoric ball *directly* onto the fundamental zone. Feed in any structured point set on the cube (Sobol, Halton, a regular SC/FCC/BCC lattice) to the orientation quotient.

## What's in the box

The five $K$-specific transports live in [mappings/](mappings/) as drop-in `ho2hoFZ_*` functions:

- **$C_k$** and **$D_k$**: fully analytic, derived from per-axis CDF inverses.
- **$T$** and **$O$**: piecewise analytic over the cubic FZ; the octahedral map ships in both standard cubochoric-aligned and FCC-aligned variants.
- **$I$**: fitted azimuthal/polar CDFs with a post-rotation correction for the icosahedral FZ.

Numerical backbone in [src/](src/): orientation conversions, Laue-group operators, $\mathrm{SO}(3)$ baselines (super-Fibonacci, Marsaglia, Shoemake, cubochoric), Riesz energy, covering radius, and a quotient-metric Thomson relaxer for benchmark baselines.

Plotting and the paper figure pipeline live in [figures/](figures/); the camera-ready PNG / PDF / EPS bundles are checked in under [paper_figures/](paper_figures/).

## Stereogram

An octahedral KR grid lifted to all of $\mathrm{SO}(3)$ by the 24 elements of $O$, drawn in homochoric coordinates.

<details>
<summary>Show stereogram</summary>
<br/>
<p align="center">
  <img src="assets/stereogram_so3_o.gif" alt="Stereogram: octahedral KR grid on SO(3) in homochoric coordinates." width="600" />
</p>
</details>

> **Free-fuse.** ~50 cm from the screen, two spheres about 10 cm apart, cross your eyes until they overlap.

## Reproduce the paper

```bash
pip install -r requirements.txt
python -m publication.export_figures            # all twelve panels
python -m publication.export_figures --only 1 4 # a subset
```

Outputs land in [paper_figures/](paper_figures/) as PNG, PDF, and true vector EPS (rendered from the TeX PDF via Poppler `pdftops`, not raster-wrapped). Layout details are in [publication/README.md](publication/README.md).

Cached data behind the figures can be rebuilt from scratch:

```bash
python -m figures.generate_cubochoric_anisotropy
python -m figures.generate_nn_cdf_data
python -m figures.generate_grid_method_metrics
python -m figures.generate_thomson_relaxation
```

Any panel module under [figures/](figures/) is also runnable on its own, which is handy for tweaking limits or colour mappings interactively:

```bash
python -m figures.cubochoric_anisotropy
```

Figure 1 benefits from a working `pykeops` install for the kernel evaluations.

## Citation

```text
coming soon...
```

## License

MIT. See [LICENSE](LICENSE).
