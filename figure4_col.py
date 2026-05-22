#!/usr/bin/env python3
"""
figure4.py — Riesz energy ratios (E1, E2, E3) and covering-radius ratio vs N_eff.

Compares KR grids (Laue 2–12), cubochoric rejection (Laue 1), and Fibonacci
sampling baselines across a range of effective grid sizes.

Layout: 2×2 panel grid with a hand-crafted structured legend. Interactive mode
gives a clean control panel in a separate window with per-panel axis-scale
toggles, grid control, and full typography tuning. Static mode exports
publication-ready PNG/PDF.

Methods plotted:
  • KR grids for Laue 2–12 (C2 … I/532), one curve per group
  • Cubochoric rejection grid (black)
  • Fibonacci sampling — reject w<0 (solid grey)
  • Fibonacci sampling — all + canonicalize (dashed grey)

Usage:
    python figure4.py              # generate data (if needed) + interactive plot
    python figure4.py --force      # force-regenerate all data
    python figure4.py --plot-only  # plot from existing .npz
    python figure4.py --static     # non-interactive PNG/PDF export
    python figure4.py --quick      # fewer Fibonacci samples (for testing)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline

import torch
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.usetex"] = True

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import (
    FuncFormatter,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
)
from matplotlib.widgets import Button, CheckButtons, Slider

from figure_ui_common import (
    ControlWindow,
    FIBONACCI_SERIES_KEYS,
    IUCR_COL_W,
    IUCR_DPI,
    IUCR_FONT,
    IUCR_MARGINS,
    IUCR_MIN_LW,
    add_panel_label,
    apply_plot_rcparams,
    bind_typography,
    build_typography_section,
    export_figure_with_tex,
    figure4_mean_non_fib_marker_count,
    markevery_stride,
    push_typography,
    restyle_axes,
    sync_typography,
)

from grid_FZ import cu_kr_grid, cu_rej_grid, kr_sample_laue
from grid_SO3 import so3_super_fibonacci
from orientation_ops import cu2qu, qu_std, qu_norm
from riesz_energy import optimal_constants_S3, riesz_energies_fused
from covering_radius import covering_radius, covering_radius_star_deg

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LAUE_ID_C1 = 1
CARD_C1 = 1

KR_GROUPS: List[dict] = [
    {"name": "C2", "laue_id": 2, "card": 2},
    {"name": "C3", "laue_id": 6, "card": 3},
    {"name": "C4", "laue_id": 4, "card": 4},
    {"name": "C6", "laue_id": 8, "card": 6},
    {"name": "D2", "laue_id": 3, "card": 4},
    {"name": "D3", "laue_id": 7, "card": 6},
    {"name": "D4", "laue_id": 5, "card": 8},
    {"name": "D6", "laue_id": 9, "card": 12},
    {"name": "T", "laue_id": 10, "card": 12},
    {"name": "O", "laue_id": 11, "card": 24},
    {"name": "I", "laue_id": 12, "card": 60},
]

LOGQUAD_COEFFS: Dict[str, Tuple[float, float, float]] = {
    "C2": (+0.595700, +0.063365, -0.008238),
    "C3": (+0.505407, +0.068467, -0.008634),
    "C4": (+0.426731, +0.094131, -0.012107),
    "C6": (+0.401063, +0.092710, -0.012194),
    "D3": (+0.594580, +0.065238, -0.008549),
    "D4": (+0.489191, +0.026999, -0.000234),
    "D6": (+0.664528, -0.185023, +0.026121),
}

DATA_FILE = "figure4_data.npz"
SETTINGS_FILE = "figure4_col_settings.json"
DEFAULT_STEM = "figure4_col"

# Typography spec — (widget_name, label, settings_attr [, fmt])
_TYPO_ROWS = [
    [("text_size", "Font", "label_size"), ("title_size", "Title", "title_size")],
    [("subtitle_size", "Sub", "subtitle_size"), ("panel_label_size", "Panel", "panel_label_size")],
    [("panel_label_x", "Lbl x", "panel_label_x", "{:.3f}"), ("panel_label_y", "Lbl y", "panel_label_y", "{:.3f}")],
    [("legend_size", "Leg", "legend_size")],
    [("grid_alpha", "Grid \u03b1", "grid_alpha", "{:.2f}")],
]

N_EFF_MIN = 10_000
N_EFF_MAX = 1_000_000
FIB_N_STEP = 100_000
DPI = IUCR_DPI
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUAL THEME                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PALETTE = {
    # KR groups (match figure3/figure4 tab20 ordering)
    "C2": "#1f77b4",
    "C3": "#aec7e8",
    "C4": "#ff7f0e",
    "C6": "#ffbb78",
    "D2": "#2ca02c",
    "D3": "#98df8a",
    "D4": "#d62728",
    "D6": "#ff9896",
    "T": "#9467bd",
    "O": "#c5b0d5",
    "I": "#8c564b",
    # Baselines
    "rej": "#1a1a1a",
    "fcc": "#1a1a1a",
    "fib_rej": "#5d6d7e",
    "fib_all": "#95a5a6",
}

LINESTYLES = {k: "-" for k in PALETTE}
LINESTYLES["fib_all"] = "-"
LINESTYLES["fib_rej"] = "-"
LINESTYLES["fcc"] = "--"
LINESTYLES["kr_fcc_T"] = "--"
LINESTYLES["kr_fcc_O"] = "--"
LINESTYLES["kr_fcc_I"] = "--"

# Markers: match figure2 — C2–C6 thin_diamond, triangle_up, diamond, hexagon;
# D2–D6 same shapes rotated (D2 90°, D3 90°, D4 45°, D6 90°).
MARKERS: Dict[str, Union[str, tuple]] = {
    "C2": "d",  # thin_diamond
    "C3": "^",  # triangle_up
    "C4": "D",  # diamond
    "C6": "h",  # hexagon
    "D2": ("d", 90),  # thin_diamond rotated 90°
    "D3": ("^", 90),  # triangle_up rotated 90°
    "D4": ("D", 45),  # diamond rotated 45°
    "D6": ("h", 90),  # hexagon rotated 90°
    "T": "H",  # tetrahedral: hexagon (3-fold axes)
    "O": "o",  # octahedral: circle
    "I": "p",  # icosahedral: pentagon (5-fold)
    "rej": "o",
    "fcc": "o",
    "fib_rej": "o",
    "fib_all": "o",
}

KR_NAMES = [g["name"] for g in KR_GROUPS]


# Paint order: KR underneath, baselines on top
PLOT_ORDER = (
    [f"kr_{n}" for n in KR_NAMES]
    + ["kr_fcc_T", "kr_fcc_O", "kr_fcc_I"]
    + ["fib_all", "fib_rej", "fcc", "rej"]
)

# Panel short keys (used in settings dicts for xlog/ylog)
PANEL_KEYS = ["E3", "CR"]

# Column layout: E3 panel, theta-excess panel, legend row.
PANELS_COL = [
    ("E3_ratio", "E3", False),
    ("cr_ratio", "CR", True),
]

# Legend: 4 columns for horizontal layout below panels
LEGEND_COLUMNS = [
    ("Baselines", ["rej", "fcc", "fib_rej", "fib_all"]),
    ("Cyclic", ["kr_C2", "kr_C3", "kr_C4", "kr_C6"]),
    ("Dihedral", ["kr_D2", "kr_D3", "kr_D4", "kr_D6"]),
    ("Cubic / Ico", ["kr_T", "kr_O", "kr_I"]),
]

LEGEND_LABELS = {
    "rej": "$\\textrm{Cubochoric (PC)}$",
    "fcc": "$\\textrm{Cubochoric (FCC)}$",
    "fib_rej": "$\\textrm{Fibonacci (reject)}$",
    "fib_all": "$\\textrm{Fibonacci (canon.)}$",
    "kr_C2": "$C_{2}$",
    "kr_C3": "$C_{3}$",
    "kr_C4": "$C_{4}$",
    "kr_C6": "$C_{6}$",
    "kr_D2": "$D_{2}$",
    "kr_D3": "$D_{3}$",
    "kr_D4": "$D_{4}$",
    "kr_D6": "$D_{6}$",
    "kr_T": "$T$",
    "kr_O": "$O$",
    "kr_I": "$I$",
}


def _series_name(key: str) -> str:
    if key.startswith("kr_fcc_"):
        return key[7:]
    if key.startswith("kr_"):
        return key[3:]
    return key


def _color(key: str) -> str:
    return PALETTE.get(_series_name(key), "#666")


def _ls(key: str) -> str:
    if key in LINESTYLES:
        return LINESTYLES[key]
    return LINESTYLES.get(_series_name(key), "-")


def _marker(key: str):
    """Return marker for key: str or MarkerStyle (for rotated markers)."""
    m = MARKERS.get(_series_name(key), "o")
    if isinstance(m, tuple):
        sym, rot = m
        ms = MarkerStyle(sym)
        ms._transform = ms.get_transform().rotate_deg(rot)
        return ms
    return m


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GRID BUILDERS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _rhat_star(name: str, cu_xy: int) -> float:
    if name in LOGQUAD_COEFFS:
        b0, b1, b2 = LOGQUAD_COEFFS[name]
        lx = math.log(max(cu_xy, 1))
        return b0 + b1 * lx + b2 * lx * lx
    return 1.0


def _kr_grid_n_base(cu_xy: int, cu_z: int) -> int:
    return (2 * cu_xy + 1) ** 2 * (2 * cu_z + 1)


def _kr_grid_n_fcc(cu_xy: int, cu_z: int) -> int:
    n = _kr_grid_n_base(cu_xy, cu_z)
    return (n + 1) // 2


def _enumerate_kr_sizes(name: str, card: int) -> List[Tuple[int, int]]:
    out = []
    lo = (N_EFF_MIN + 1) // (2 * card)
    hi = N_EFF_MAX // (2 * card)
    for cu_xy in range(1, 501):
        cu_z = max(1, round(_rhat_star(name, cu_xy) * cu_xy))
        nb = _kr_grid_n_base(cu_xy, cu_z)
        ne = 2 * card * nb
        if nb > hi:
            break
        if lo <= nb and N_EFF_MIN <= ne <= N_EFF_MAX:
            out.append((cu_xy, cu_z))
    return out


def _enumerate_kr_sizes_fcc(name: str, card: int) -> List[Tuple[int, int]]:
    out = []
    lo = (N_EFF_MIN + 1) // (2 * card)
    hi = N_EFF_MAX // (2 * card)
    for cu_xy in range(1, 501):
        cu_z = max(1, round(_rhat_star(name, cu_xy) * cu_xy))
        nb = _kr_grid_n_fcc(cu_xy, cu_z)
        ne = 2 * card * nb
        if nb > hi:
            break
        if lo <= nb and N_EFF_MIN <= ne <= N_EFF_MAX:
            out.append((cu_xy, cu_z))
    return out


def _build_kr(cu_xy, cu_z, laue_id, card, dev):
    q = cu_kr_grid(cu_xy, laue_id, dev, z_semi_edge_length=cu_z)
    return q, 2 * card * int(q.shape[0])


def _fcc_cubochoric_grid(
    semi_edge_length: int,
    device: torch.device,
    z_semi_edge_length: int = 0,
) -> torch.Tensor:
    cu_max = 0.5 * torch.pi ** (2.0 / 3.0)
    cu_xy = torch.linspace(-cu_max, cu_max, 2 * semi_edge_length + 2, device=device)
    cu_xy = cu_xy[:-1]
    cu_xy = cu_xy + 0.5 * (cu_xy[1] - cu_xy[0])

    if z_semi_edge_length == 0:
        z_semi_edge_length = semi_edge_length
    cu_z = torch.linspace(-cu_max, cu_max, 2 * z_semi_edge_length + 2, device=device)
    cu_z = cu_z[:-1]
    cu_z = cu_z + 0.5 * (cu_z[1] - cu_z[0])

    X, Y, Z = torch.meshgrid(cu_xy, cu_xy, cu_z, indexing="ij")
    ix = torch.arange(-semi_edge_length, semi_edge_length + 1, device=device)
    iz = torch.arange(-z_semi_edge_length, z_semi_edge_length + 1, device=device)
    I, J, K = torch.meshgrid(ix, ix, iz, indexing="ij")
    mask = ((I + J + K) % 2) == 0
    return torch.stack([X[mask], Y[mask], Z[mask]], dim=-1)


def _build_kr_fcc(cu_xy, cu_z, laue_id, card, dev):
    cu = _fcc_cubochoric_grid(cu_xy, dev, z_semi_edge_length=cu_z)
    q0 = qu_norm(qu_std(cu2qu(cu)))
    q = kr_sample_laue(q0, laue_id)
    return q, 2 * card * int(q.shape[0])


def _enumerate_rej_semi() -> List[int]:
    out = []
    for s in range(1, 201):
        ne = 2 * CARD_C1 * (2 * s + 1) ** 3
        if ne > N_EFF_MAX:
            break
        if N_EFF_MIN <= ne <= N_EFF_MAX:
            out.append(s)
    return out


def _enumerate_fcc_semi() -> List[int]:
    out = []
    for s in range(1, 501):
        ne = 2 * CARD_C1 * _kr_grid_n_fcc(s, s)
        if ne > N_EFF_MAX:
            break
        if N_EFF_MIN <= ne <= N_EFF_MAX:
            out.append(s)
    return out


def _build_rej(semi, dev):
    q = cu_rej_grid(semi, LAUE_ID_C1, dev)
    return q, 2 * CARD_C1 * int(q.shape[0])


def _build_fcc(semi, dev):
    cu = _fcc_cubochoric_grid(semi, dev)
    q = qu_norm(qu_std(cu2qu(cu)))
    return q, 2 * CARD_C1 * int(q.shape[0])


def _build_fib_reject(n_eff_target, dev):
    q = so3_super_fibonacci(max(2, int(n_eff_target)), dev, reject_invalid=True)
    q = qu_norm(qu_std(q))
    return q, 2 * CARD_C1 * int(q.shape[0])


def _build_fib_all(n_eff_target, dev):
    q = so3_super_fibonacci(max(1, n_eff_target // 2), dev, reject_invalid=False)
    q = qu_norm(qu_std(q))
    return q, 2 * CARD_C1 * int(q.shape[0])


def _metrics(q, laue_id, card, dev, quiet=True):
    n_base = int(q.shape[0])
    n_eff = 2 * card * n_base
    E1, E2, E3 = riesz_energies_fused(q, laue_id)
    E1o, E2o, E3o = optimal_constants_S3(n_eff)
    elc = min(20, max(5, n_base // 500))
    if quiet:
        with redirect_stdout(StringIO()):
            cr = covering_radius(
                q,
                laue_id,
                edge_length_check=elc,
                grid_eps_factor=2.0,
                qhull_options="QJ",
            )
    else:
        cr = covering_radius(
            q, laue_id, edge_length_check=elc, grid_eps_factor=2.0, qhull_options="QJ"
        )
    cr_deg = float(cr) * 180.0 / math.pi
    cr_star = covering_radius_star_deg(n_eff)
    return {
        "N_eff": n_eff,
        "E1_ratio": E1 / E1o if E1o else float("nan"),
        "E2_ratio": E2 / E2o if E2o else float("nan"),
        "E3_ratio": E3 / E3o if E3o else float("nan"),
        "cr_ratio": (cr_deg / cr_star) if cr_star else float("nan"),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA GENERATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def generate_data(dev, force=False, quick=False) -> dict:
    METRICS = ("n_eff", "E1_ratio", "E2_ratio", "E3_ratio", "cr_ratio")
    all_keys = [f"kr_{g['name']}" for g in KR_GROUPS] + [
        "kr_fcc_T",
        "kr_fcc_O",
        "kr_fcc_I",
        "rej",
        "fcc",
        "fib_rej",
        "fib_all",
    ]

    existing = {}
    if os.path.exists(DATA_FILE) and not force:
        print(f"[data] Found {DATA_FILE}; regenerating only missing entries")
        existing = _load_data()

    results = {k: {m: [] for m in METRICS} for k in all_keys}
    for k in all_keys:
        if k in existing:
            for m in METRICS:
                arr = np.asarray(existing[k].get(m, np.array([])))
                results[k][m] = list(arr.tolist())

    def _has_data(key: str) -> bool:
        return len(results[key]["n_eff"]) > 0

    def _store(key, m):
        for mk in METRICS:
            results[key][mk].append(m["N_eff"] if mk == "n_eff" else m[mk])

    for g in KR_GROUPS:
        key = f"kr_{g['name']}"
        if _has_data(key):
            print(f"[KR {g['name']}] already present; skipping")
            continue
        sizes = _enumerate_kr_sizes(g["name"], g["card"])
        print(f"[KR {g['name']}] {len(sizes)} sizes")
        for i, (cx, cz) in enumerate(sizes):
            t0 = time.time()
            try:
                q, _ = _build_kr(cx, cz, g["laue_id"], g["card"], dev)
                if q.shape[0] < 10:
                    continue
                m = _metrics(q, g["laue_id"], g["card"], dev)
                _store(key, m)
                if (i + 1) % 5 == 0 or i == 0 or i == len(sizes) - 1:
                    print(
                        f"  [{i+1}/{len(sizes)}] cx={cx} cz={cz} "
                        f"Neff={m['N_eff']:,} E3r={m['E3_ratio']:.4f} "
                        f"({time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                print(f"  [{i+1}/{len(sizes)}] cx={cx} cz={cz} FAILED: {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    semis = _enumerate_rej_semi()
    if _has_data("rej"):
        print("[Rej Laue-1] already present; skipping")
    else:
        print(f"[Rej Laue-1] {len(semis)} sizes")
        for i, s in enumerate(semis):
            t0 = time.time()
            try:
                q, _ = _build_rej(s, dev)
                if q.shape[0] < 10:
                    continue
                m = _metrics(q, LAUE_ID_C1, CARD_C1, dev)
                _store("rej", m)
                print(
                    f"  [{i+1}/{len(semis)}] semi={s} Neff={m['N_eff']:,} "
                    f"E3r={m['E3_ratio']:.4f} ({time.time()-t0:.1f}s)"
                )
            except Exception as e:
                print(f"  [{i+1}/{len(semis)}] semi={s} FAILED: {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    fcc_semis = _enumerate_fcc_semi()
    if _has_data("fcc"):
        print("[FCC Laue-1] already present; skipping")
    else:
        print(f"[FCC Laue-1] {len(fcc_semis)} sizes")
        for i, s in enumerate(fcc_semis):
            t0 = time.time()
            try:
                q, _ = _build_fcc(s, dev)
                if q.shape[0] < 10:
                    continue
                m = _metrics(q, LAUE_ID_C1, CARD_C1, dev)
                _store("fcc", m)
                if (i + 1) % 5 == 0 or i == 0 or i == len(fcc_semis) - 1:
                    print(
                        f"  [{i+1}/{len(fcc_semis)}] semi={s} Neff={m['N_eff']:,} "
                        f"E3r={m['E3_ratio']:.4f} ({time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                print(f"  [{i+1}/{len(fcc_semis)}] semi={s} FAILED: {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    for g in KR_GROUPS:
        if g["name"] not in ("T", "O", "I"):
            continue
        key = f"kr_fcc_{g['name']}"
        if _has_data(key):
            print(f"[KR-FCC {g['name']}] already present; skipping")
            continue
        sizes = _enumerate_kr_sizes_fcc(g["name"], g["card"])
        print(f"[KR-FCC {g['name']}] {len(sizes)} sizes")
        for i, (cx, cz) in enumerate(sizes):
            t0 = time.time()
            try:
                q, _ = _build_kr_fcc(cx, cz, g["laue_id"], g["card"], dev)
                if q.shape[0] < 10:
                    continue
                m = _metrics(q, g["laue_id"], g["card"], dev)
                _store(key, m)
                if (i + 1) % 5 == 0 or i == 0 or i == len(sizes) - 1:
                    print(
                        f"  [{i+1}/{len(sizes)}] cx={cx} cz={cz} "
                        f"Neff={m['N_eff']:,} E3r={m['E3_ratio']:.4f} "
                        f"({time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                print(f"  [{i+1}/{len(sizes)}] cx={cx} cz={cz} FAILED: {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    ns = list(range(N_EFF_MIN, N_EFF_MAX + 1, FIB_N_STEP))
    if quick:
        ns = ns[:: max(1, len(ns) // 5)]
    if _has_data("fib_rej"):
        print("[Fib reject] already present; skipping")
    else:
        print(f"[Fib reject] {len(ns)} samples")
        for i, n in enumerate(ns):
            t0 = time.time()
            try:
                q, _ = _build_fib_reject(n, dev)
                if q.shape[0] < 10:
                    continue
                m = _metrics(q, LAUE_ID_C1, CARD_C1, dev)
                _store("fib_rej", m)
                if (i + 1) % 10 == 0 or i == 0 or i == len(ns) - 1:
                    print(
                        f"  [{i+1}/{len(ns)}] n={n} Neff={m['N_eff']:,} "
                        f"E3r={m['E3_ratio']:.4f} ({time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                print(f"  [{i+1}/{len(ns)}] n={n} FAILED: {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    ns2 = list(range(N_EFF_MIN // 2, N_EFF_MAX // 2 + 1, FIB_N_STEP // 2))
    if quick:
        ns2 = ns2[:: max(1, len(ns2) // 5)]
    if _has_data("fib_all"):
        print("[Fib canon] already present; skipping")
    else:
        print(f"[Fib canon] {len(ns2)} samples")
        for i, n in enumerate(ns2):
            t0 = time.time()
            try:
                q, _ = _build_fib_all(2 * n, dev)
                if q.shape[0] < 10:
                    continue
                m = _metrics(q, LAUE_ID_C1, CARD_C1, dev)
                _store("fib_all", m)
                if (i + 1) % 10 == 0 or i == 0 or i == len(ns2) - 1:
                    print(
                        f"  [{i+1}/{len(ns2)}] n={n} Neff={m['N_eff']:,} "
                        f"E3r={m['E3_ratio']:.4f} ({time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                print(f"  [{i+1}/{len(ns2)}] n={n} FAILED: {e}")
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    save = {}
    for key in results:
        for mk, v in results[key].items():
            save[f"{key}_{mk}"] = np.array(v)
    np.savez(DATA_FILE, **save)
    print(f"[data] Saved → {DATA_FILE}")
    return results


def _load_data() -> dict:
    raw = dict(np.load(DATA_FILE, allow_pickle=True))
    keys = {k.rsplit("_n_eff", 1)[0] for k in raw if k.endswith("_n_eff")}
    results = {}
    for key in keys:
        results[key] = {}
        for mk in ("n_eff", "E1_ratio", "E2_ratio", "E3_ratio", "cr_ratio"):
            arr = raw.get(f"{key}_{mk}", np.array([]))
            if arr.ndim == 0:
                arr = np.atleast_1d(arr.item())
            results[key][mk] = arr
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SETTINGS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _default_visible() -> Dict[str, bool]:
    d = {"rej": True, "fcc": True, "fib_rej": True, "fib_all": True}
    for n in KR_NAMES:
        d[f"kr_{n}"] = True
    d["kr_fcc_T"] = True
    d["kr_fcc_O"] = True
    d["kr_fcc_I"] = True
    return d


def _default_xlog() -> Dict[str, bool]:
    return {pk: True for pk in PANEL_KEYS}


def _default_ylog() -> Dict[str, bool]:
    return {pk: False for pk in PANEL_KEYS}


@dataclass
class Settings:
    # Typography
    label_size: float = IUCR_FONT["label"]
    tick_size: float = IUCR_FONT["tick"]
    minor_tick_size: float = 5.0
    title_size: float = IUCR_FONT["title"]
    subtitle_size: float = IUCR_FONT["subtitle"]
    panel_label_size: float = IUCR_FONT["panel_label"]
    panel_label_x: float = 0.02
    panel_label_y: float = 0.98
    legend_size: float = IUCR_FONT["legend"]
    legend_title_size: float = IUCR_FONT["legend_title"]
    # Lines
    lw_kr: float = 1.0
    lw_baseline: float = 1.0
    marker_size: float = 2.5
    marker_every: int = 5
    # Axis range
    x_min: float = float(N_EFF_MIN)
    x_max: float = float(N_EFF_MAX)
    # Grid lines
    show_grid: bool = True
    grid_alpha: float = 0.20
    show_minor_tick_labels: bool = False
    # Per-panel axis scales  (keys: E1, E2, E3, CR)
    xlog: Dict[str, bool] = field(default_factory=_default_xlog)
    ylog: Dict[str, bool] = field(default_factory=_default_ylog)
    # Visibility
    show: Dict[str, bool] = field(default_factory=_default_visible)

    # ── Diagnostic / readability toggles ───────────────────────────────
    # optional family linestyles (helps when curves nearly overlap)
    use_family_linestyles: bool = False

    # optional per-panel y-zoom (applied after data transform)
    # Provide None to disable, or tuple(ymin, ymax).
    yzoom_E1: Tuple[float, float] | None = None
    yzoom_E2: Tuple[float, float] | None = None


def _load_settings(path: str) -> Settings:
    s = Settings()
    if not os.path.exists(path):
        return s
    try:
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if k == "show" and isinstance(v, dict):
                s.show = {**_default_visible(), **v}
            elif k == "xlog" and isinstance(v, dict):
                s.xlog = {**_default_xlog(), **v}
            elif k == "ylog" and isinstance(v, dict):
                s.ylog = {**_default_ylog(), **v}
            elif hasattr(s, k):
                setattr(s, k, v)

        # Back-compat guards for older JSON files
        if not hasattr(s, "show_minor_tick_labels"):
            s.show_minor_tick_labels = False
        print(f"[settings] Loaded {path}")
    except Exception as e:
        print(f"[settings] Warning: {e}")
    return s


def _save_settings(s: Settings, path: str):
    with open(path, "w") as f:
        json.dump(asdict(s), f, indent=2)
    print(f"[settings] Saved → {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HAND-CRAFTED STRUCTURED LEGEND                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  4 columns, each with a bold header and entry rows:
#
#    Baselines          Cyclic       Dihedral       Cubic / Ico
#    ── Cubochoric      ── C2        ── D2          ── T
#    ── Fib (reject)    ── C3        ── D3          ── O
#    ── Fib (canon.)    ── C4        ── D4          ── I
#                       ── C6        ── D6
#
#  Hidden series are shown with grey text and no swatch.

_LEGEND_TAG = "_figure4_legend"


def _clear_legend(fig):
    """Remove all legend artists we previously drew."""
    fig.lines = [ln for ln in fig.lines if getattr(ln, _LEGEND_TAG, False) is False]
    for txt in list(fig.texts):
        if getattr(txt, _LEGEND_TAG, False):
            txt.remove()


def _draw_legend_in_axes(ax, S: Settings, visible_set: set):
    """Draw compact fixed-order legend block using native matplotlib layout."""
    ax.set_axis_off()
    dashed_visible = any(k in visible_set for k in ("kr_fcc_T", "kr_fcc_O", "kr_fcc_I"))
    # Matplotlib legend fills column-first for ncol>1.
    # Feed in explicit column-major order to get desired visual columns:
    # col1: C2 C3 C4 C6
    # col2: D2 D3 D4 D6
    # col3: T O I KR-FCC
    # col4: Cubochoric Cubochoric FCC Fib reject Fib canon
    col1 = ["kr_C2", "kr_C3", "kr_C4", "kr_C6"]
    col2 = ["kr_D2", "kr_D3", "kr_D4", "kr_D6"]
    col3 = ["kr_T", "kr_O", "kr_I", "_kr_fcc_note" if dashed_visible else "_blank"]
    col4 = ["rej", "fcc", "fib_rej", "fib_all"]
    legend_items = col1 + col2 + col3 + col4

    handles = []
    labels = []

    for key in legend_items:
        if key == "_blank":
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
            continue

        if key == "_kr_fcc_note":
            handles.append(Line2D([0], [0], color="#444", ls="--", lw=S.lw_kr))
            # labels.append("FCC")
            labels.append("$\\textrm{FCC}$")
            continue

        if key not in visible_set:
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
            continue

        ms = S.marker_size
        mk = _marker(key) if ms > 0 else None
        handles.append(
            Line2D(
                [0],
                [0],
                color=_color(key),
                ls=_ls(key),
                lw=S.lw_kr,
                marker=mk,
                markersize=ms * 1.2 if ms > 0 else 0,
                markeredgewidth=0.35,
                markeredgecolor="white",
            )
        )
        labels.append(LEGEND_LABELS.get(key, key))

    leg = ax.legend(
        handles,
        labels,
        ncol=4,
        fontsize=S.legend_size,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#ccc",
        loc="center",
        borderpad=0.45,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    for h in leg.legend_handles:
        setattr(h, _LEGEND_TAG, True)


def _draw_legend(fig, S: Settings, visible_set: set, legend_ax=None):
    """Draw the legend. If legend_ax provided, draw there; else at bottom of fig."""
    if legend_ax is not None:
        _draw_legend_in_axes(legend_ax, S, visible_set)
        return
    fs, tfs = S.legend_size, S.legend_title_size
    col_x = [0.08, 0.31, 0.54, 0.77]
    y, row_dy, hdr_gap, sw_len, txt_gap = 0.12, 0.02, 0.005, 0.04, 0.008
    for ci, (title, keys) in enumerate(LEGEND_COLUMNS):
        x0 = col_x[ci]
        t = fig.text(
            x0,
            y,
            title,
            fontsize=tfs,
            fontweight="bold",
            color="#333",
            va="top",
            ha="left",
        )
        setattr(t, _LEGEND_TAG, True)
        y -= row_dy + hdr_gap
        for key in keys:
            label = LEGEND_LABELS.get(key, key)
            vis = key in visible_set
            if vis:
                lw = S.lw_kr
                ms = S.marker_size
                mk = _marker(key) if ms > 0 else None
                ln = Line2D(
                    [x0, x0 + sw_len],
                    [y - 0.004, y - 0.004],
                    color=_color(key),
                    ls=_ls(key),
                    lw=lw,
                    marker=mk,
                    markersize=ms * 1.2,
                    markeredgewidth=0.35,
                    markeredgecolor="white",
                    transform=fig.transFigure,
                    figure=fig,
                    clip_on=False,
                )
                setattr(ln, _LEGEND_TAG, True)
                fig.lines.append(ln)
                t = fig.text(
                    x0 + sw_len + txt_gap,
                    y,
                    label,
                    fontsize=fs,
                    va="top",
                    ha="left",
                    color=_color(key),
                )
            else:
                t = fig.text(
                    x0 + sw_len + txt_gap,
                    y,
                    label,
                    fontsize=fs,
                    va="top",
                    ha="left",
                    color="#bbb",
                )
            setattr(t, _LEGEND_TAG, True)
            y -= row_dy
        y = 0.12


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CORE DRAWING LOGIC  (shared by static + interactive)                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _panel_ylabel(pk: str, S: Settings) -> str:
    """Return y-axis label for panel pk."""
    if pk == "E1":
        return "$E_1/E_1^*$"
    if pk == "E2":
        return "$E_2/E_2^*$"
    if pk == "E3":
        return "$E_3/E_3^*$"
    if pk == "E1_rel":
        return "$E_1/E_1^* \\textrm{ ratio to best}$"
    if pk == "E2_rel":
        return "$E_2/E_2^* \\textrm{ ratio to best}$"
    if pk == "E3_rel":
        return "$E_3/E_3^* \\textrm{ ratio to best}$"
    return "$\\theta_{\\mathrm{S^3}}/\\theta_{\\mathrm{S^3}}^*-1$"


def _yzoom_for_panel(pk: str, S: Settings):
    if pk in ("E1", "E1_rel"):
        return S.yzoom_E1
    if pk in ("E2", "E2_rel"):
        return S.yzoom_E2
    return None


def _draw_rel2best_panel(ax, metric: str, pk: str, visible, results, S: Settings):
    """Draw one rel2best panel using cubic spline interpolation."""
    use_xlog = True
    series_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for key in visible:
        r = results.get(key, {})
        x = np.asarray(r.get("n_eff", []))
        yy = np.asarray(r.get(metric, []))
        if x.size == 0 or yy.size == 0:
            continue
        idx_sort = np.argsort(x)
        x, yy = x[idx_sort], yy[idx_sort]
        ok = np.isfinite(yy) & (x > 0)
        if not ok.any():
            continue
        x, yy = x[ok], yy[ok]
        if x.size < 2:
            continue
        series_data.append((key, x, yy))

    if not series_data:
        return

    def _collapse_duplicate_x(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse duplicate x-values by averaging y, preserving sorted x."""
        if x.size == 0:
            return x, y
        ux, inv = np.unique(x, return_inverse=True)
        if ux.size == x.size:
            return x, y
        y_sum = np.zeros_like(ux, dtype=np.float64)
        y_cnt = np.zeros_like(ux, dtype=np.float64)
        np.add.at(y_sum, inv, y.astype(np.float64, copy=False))
        np.add.at(y_cnt, inv, 1.0)
        y_avg = y_sum / np.clip(y_cnt, 1.0, None)
        return ux, y_avg

    N_GRID = 400
    x_grid = np.logspace(
        np.log10(max(S.x_min, 1e-10)),
        np.log10(min(S.x_max, 1e20)),
        N_GRID,
    )
    spline_entries: List[Tuple[str, CubicSpline, np.ndarray]] = []
    splines_at_grid = []
    for key, x, yy in series_data:
        try:
            x, yy = _collapse_duplicate_x(x, yy)
            if x.size < 2:
                continue
            cs = CubicSpline(np.log10(np.maximum(x, 1e-300)), yy)
            yy_interp = cs(np.log10(x_grid))
            spline_entries.append((key, cs, x))
            splines_at_grid.append(yy_interp)
        except Exception:
            continue

    if not splines_at_grid or not spline_entries:
        return

    stack = np.stack(splines_at_grid)
    min_at_grid = np.nanmin(stack, axis=0)
    min_at_grid = np.maximum(min_at_grid, 1e-300)

    splines = [entry[1] for entry in spline_entries]
    for i, (key, _cs, x_orig) in enumerate(spline_entries):
        yy_rel = splines_at_grid[i] / min_at_grid
        yy_rel[~np.isfinite(yy_rel)] = np.nan
        is_bl = not key.startswith("kr_")
        lw = S.lw_kr
        ms = S.marker_size
        ls = _ls(key)
        ax.plot(
            x_grid,
            yy_rel,
            color=_color(key),
            ls=ls,
            lw=lw,
            zorder=10 if is_bl else 5,
        )

        if ms > 0:
            x_mark = x_orig[(x_orig >= S.x_min) & (x_orig <= S.x_max)]
            if x_mark.size > 0:
                logx_mark = np.log10(np.maximum(x_mark, 1e-300))
                y_num = splines[i](logx_mark)
                y_all = np.vstack([sp(logx_mark) for sp in splines])
                y_den = np.nanmin(y_all, axis=0)
                y_den = np.maximum(y_den, 1e-300)
                y_mark = y_num / y_den
                y_mark[~np.isfinite(y_mark)] = np.nan
                ax.plot(
                    x_mark,
                    y_mark,
                    linestyle="None",
                    marker=_marker(key),
                    markersize=ms,
                    markeredgewidth=0.35,
                    markeredgecolor="white",
                    color=_color(key),
                    zorder=11 if is_bl else 6,
                )

    ax.set_xlabel("$N_{\\mathrm{S^3}}$", fontsize=S.label_size, labelpad=10)
    ax.set_ylabel(_panel_ylabel(pk, S), fontsize=S.label_size, labelpad=20)
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlim(S.x_min, S.x_max)
    ax.axhline(1.0, lw=0.5, color="#888", alpha=0.5, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(which="major", labelsize=S.tick_size)
    ax.minorticks_on()
    ax.tick_params(which="minor", labelsize=S.minor_tick_size, length=2)
    if not S.show_minor_tick_labels:
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if S.show_grid:
        ax.grid(True, which="major", alpha=S.grid_alpha, lw=0.35)
        ax.grid(True, which="minor", alpha=S.grid_alpha * 0.35, lw=0.2)
    else:
        ax.grid(False, which="both")


def _draw_standard_panel(ax, metric: str, pk: str, visible, results, S: Settings):
    """Draw one standard (raw/residual) panel."""
    use_xlog = S.xlog.get(pk, True)
    default_ylog = {"E1": False, "E2": False, "E3": False, "CR": True}.get(pk, False)
    use_ylog = S.ylog.get(pk, default_ylog)

    fib_target = None
    if any(k in FIBONACCI_SERIES_KEYS for k in visible):
        fib_target = figure4_mean_non_fib_marker_count(
            visible, results, metric,
            x_min=S.x_min, x_max=S.x_max, xlog=S.xlog, ylog=S.ylog, pk=pk,
        )

    for key in visible:
        r = results.get(key, {})
        x = np.asarray(r.get("n_eff", []))
        yy = np.asarray(r.get(metric, []))
        if x.size == 0 or yy.size == 0:
            continue
        idx_sort = np.argsort(x)
        x, yy = x[idx_sort], yy[idx_sort]

        ok = np.isfinite(yy)
        if use_ylog:
            ok &= yy > 0
        if use_xlog:
            ok &= x > 0
        if not ok.any():
            continue
        x, yy = x[ok], yy[ok]
        ok2 = (x >= S.x_min) & (x <= S.x_max)
        x, yy = x[ok2], yy[ok2]

        is_bl = not key.startswith("kr_")
        lw = S.lw_kr
        ms = S.marker_size
        if ms > 0:
            if key in FIBONACCI_SERIES_KEYS and fib_target is not None:
                me = markevery_stride(len(x), fib_target)
            else:
                me = 1
        else:
            me = None
        ls = _ls(key)

        ax.plot(
            x,
            # subtract 1 if covering radius ratio
            yy - 1 if pk == "CR" else yy,
            color=_color(key),
            ls=ls,
            lw=lw,
            marker=_marker(key) if ms > 0 else None,
            markersize=ms,
            markevery=me,
            markeredgewidth=0.35,
            markeredgecolor="white",
            zorder=10 if is_bl else 5,
        )

    ax.set_xlabel("$N_{\\mathrm{S^3}}$", fontsize=S.label_size, labelpad=10)
    ax.set_ylabel(_panel_ylabel(pk, S), fontsize=S.label_size, labelpad=20)
    ax.set_xscale("log" if use_xlog else "linear")
    ax.set_yscale("log" if use_ylog else "linear")
    ax.set_xlim(S.x_min, S.x_max)

    yz = _yzoom_for_panel(pk, S)
    if yz is not None:
        ax.set_ylim(yz[0], yz[1])

    ax.tick_params(which="major", labelsize=S.tick_size)
    ax.minorticks_on()
    ax.tick_params(which="minor", labelsize=S.minor_tick_size, length=2)
    if not S.show_minor_tick_labels:
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if S.show_grid:
        ax.grid(True, which="major", alpha=S.grid_alpha, lw=0.35)
        ax.grid(True, which="minor", alpha=S.grid_alpha * 0.35, lw=0.2)
    else:
        ax.grid(False, which="both")


def _draw_panels(axes, results, S: Settings):
    """
    Draw column panels: E3, theta-excess, then legend.
    axes is a length-3 sequence: [E3_ax, CR_ax, legend_ax].
    """
    visible = [k for k in PLOT_ORDER if k in results and S.show.get(k, True)]
    visible_set = set(visible)

    for idx, (metric, pk, default_ylog) in enumerate(PANELS_COL):
        ax = axes[idx]
        ax.cla()
        _draw_standard_panel(ax, metric, pk, visible, results, S)
        tag = f"({chr(97 + idx)})"
        if getattr(S, "_iucr_skip_panel_labels", False):
            tag = None
        if tag:
            add_panel_label(
                ax,
                tag,
                x=S.panel_label_x,
                y=S.panel_label_y,
                fontsize=S.panel_label_size,
                use_tex=True,
            )

    leg_ax = axes[2]
    leg_ax.cla()
    _draw_legend_in_axes(leg_ax, S, visible_set)

    return visible_set


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATIC EXPORT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def plot_static(results: dict, s: Settings, stem: str = DEFAULT_STEM):
    """Export publication-ready vertical column figure with legend."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )

    fig = plt.figure(figsize=(IUCR_COL_W, 5.5), dpi=150)
    gs = GridSpec(
        3,
        1,
        left=IUCR_MARGINS["left"],
        right=IUCR_MARGINS["right"],
        bottom=IUCR_MARGINS["bottom"],
        top=IUCR_MARGINS["top"],
        hspace=0.34,
        height_ratios=[1, 1, 0.28],
    )
    axes = np.empty((3,), dtype=object)
    axes[0] = fig.add_subplot(gs[0, 0])
    axes[1] = fig.add_subplot(gs[1, 0])
    axes[2] = fig.add_subplot(gs[2, 0])

    _draw_panels(axes, results, s)

    for fmt in ("png", "pdf"):
        path = f"{stem}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"[export] {path}")
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INTERACTIVE FIGURE                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class InteractiveFigure:
    """
    Two-window explorer: 2×2 data panels + legend, with controls for styling.
    """

    def __init__(self, results: dict, settings: Settings):
        self.R = results
        self.S = settings
        self._updating = False

        apply_plot_rcparams()

        self.fig = plt.figure(
            figsize=(IUCR_COL_W, 5.5),
            num="Figure 4 — Energy & Covering-Radius Ratios",
        )
        self._build_main_layout()

        self.ctrl_fig = plt.figure(figsize=(11.0, 7.0), num="Figure 4 — Controls")
        self.ctrl_fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.01)
        self._widgets: dict = {}
        self._series_keys: list[str] = []
        self._xlog_keys: list[str] = []
        self._ylog_keys: list[str] = []
        self._build_controls()
        self._connect_events()
        self.redraw()

    def _build_main_layout(self):
        """Build the column figure grid + legend."""
        self.fig.clear()
        gs = GridSpec(
            3,
            1,
            left=IUCR_MARGINS["left"],
            right=IUCR_MARGINS["right"],
            bottom=IUCR_MARGINS["bottom"],
            top=IUCR_MARGINS["top"],
            hspace=IUCR_MARGINS["hspace"],
            height_ratios=[1, 1, 0.28],
        )
        self.axes = np.empty((3,), dtype=object)
        self.axes[0] = self.fig.add_subplot(gs[0, 0])
        self.axes[1] = self.fig.add_subplot(gs[1, 0])
        self.axes[2] = self.fig.add_subplot(gs[2, 0])

    # ================================================================== #
    #  BUILD CONTROLS                                                     #
    # ================================================================== #

    def _build_controls(self):
        """Build two-column controls: shared global controls + figure-specific series controls."""
        S = self.S
        fig = self.ctrl_fig
        fig.clear()

        self._series_keys = (
            ["rej", "fib_rej", "fib_all"]
            + [f"kr_{n}" for n in KR_NAMES]
            + ["kr_fcc_T", "kr_fcc_O", "kr_fcc_I"]
        )
        self._series_keys = (
            ["rej", "fcc", "fib_rej", "fib_all"]
            + [f"kr_{n}" for n in KR_NAMES]
            + ["kr_fcc_T", "kr_fcc_O", "kr_fcc_I"]
        )
        self._xlog_keys = ["E3", "CR"]
        self._ylog_keys = ["E3", "CR"]
        cw = ControlWindow(fig, left=0.04, right=0.92, top=0.96, col_gap=0.05, col1_frac=0.42)
        self._cw = cw

        # Column 1: shared controls (text entries — fire on Enter only)
        cw.section(1, "Typography")
        cw.number_row(1, [("text_size", "Font", S.label_size), ("title_size", "Title", S.title_size)])
        cw.number_row(1, [("subtitle_size", "Sub", S.subtitle_size), ("panel_label_size", "Panel", S.panel_label_size)])
        cw.number_row(1, [("panel_label_x", "Lbl x", S.panel_label_x, "{:.3f}"), ("panel_label_y", "Lbl y", S.panel_label_y, "{:.3f}")])
        cw.number_row(1, [("legend_size", "Leg", S.legend_size)])
        cw.number_row(1, [("grid_alpha", "Grid α", S.grid_alpha, "{:.2f}")])

        cw.section(1, "Style")
        cw.slider(1, "line_width", "Line width", 0.5, 8.0, S.lw_kr, step=0.1)
        cw.slider(1, "marker_size", "Marker size", 0, 8, S.marker_size, step=0.5)

        cw.section(1, "Display")
        d_checks = cw.checkbox_grid(
            1,
            "_display_checks",
            [
                ("show_grid", "Grid", S.show_grid),
                ("show_minor_tick_labels", "Minor ticks", S.show_minor_tick_labels),
            ],
            n_cols=1,
            row_h=0.040,
            label_size=7.0,
        )
        self._widgets["show_grid"] = d_checks[0]
        self._widgets["show_minor_tick_labels"] = d_checks[1]

        cw.section(1, "Axis scales")
        axis_items = []
        for pk in self._xlog_keys:
            axis_items.append((f"xlog_{pk}", f"x:{pk}", S.xlog.get(pk, True)))
        for pk in self._ylog_keys:
            axis_items.append((f"ylog_{pk}", f"y:{pk}", S.ylog.get(pk, False)))
        axis_checks = cw.checkbox_grid(1, "_axis_checks", axis_items, n_cols=2, row_h=0.040, label_size=6.8)
        for i, item in enumerate(axis_items):
            self._widgets[item[0]] = axis_checks[i]

        cw.section(1, "Export")
        cw.button_row(1, [("save_png", "Save PNG"), ("save_pdf", "Save PDF")])
        cw.button_row(1, [("save_cfg", "Save settings"), ("load_cfg", "Load settings")])

        # Column 2: custom controls
        cw.section(2, "Series visibility")
        series_items = []
        color_map = {}
        for key in self._series_keys:
            short = key.replace("kr_", "").replace("fib_", "fib ")
            series_items.append((key, short, S.show.get(key, True)))
            color_map[key] = _color(key)
        series_checks = cw.checkbox_grid(
            2,
            "_series_checks",
            series_items,
            n_cols=4,
            row_h=0.040,
            label_size=6.7,
            color_map=color_map,
        )
        for i, key in enumerate(self._series_keys):
            self._widgets[f"show_{key}"] = series_checks[i]

        cw.connect_scroll()

        # Merge all cw.widgets into self._widgets so _connect_events can find them
        self._widgets.update(cw.widgets)

    # ================================================================== #
    #  CONNECT EVENTS                                                     #
    # ================================================================== #

    def _connect_events(self):
        W = self._widgets
        # Text entries (number_row) — fire on Enter, lightweight restyle
        bind_typography(self._cw, _TYPO_ROWS, self._on_restyle)
        # grid_alpha text entry — visual only
        W["grid_alpha"].on_submit(lambda _, _k="grid_alpha": self._on_ui_change())
        # Sliders — visual only
        for k in ("line_width", "marker_size"):
            W[k].on_changed(lambda _, _k=k: self._on_ui_change())
        for k in ("show_grid", "show_minor_tick_labels"):
            W[k].on_clicked(lambda _: self._on_ui_change())
        for pk in self._xlog_keys:
            W[f"xlog_{pk}"].on_clicked(lambda _: self._on_ui_change())
        for pk in self._ylog_keys:
            W[f"ylog_{pk}"].on_clicked(lambda _: self._on_ui_change())
        # Series visibility — needs full redraw (adds/removes line artists)
        for key in self._series_keys:
            W[f"show_{key}"].on_clicked(lambda _: self._on_visibility_change())
        W["save_png"].on_clicked(lambda _: self._export("png"))
        W["save_pdf"].on_clicked(lambda _: self._export("pdf"))
        W["save_cfg"].on_clicked(lambda _: _save_settings(self.S, SETTINGS_FILE))
        W["load_cfg"].on_clicked(lambda _: self._reload_settings())

    # ================================================================== #
    #  SYNC  widgets ↔ Settings                                           #
    # ================================================================== #

    def _sync_from_widgets(self):
        W = self._widgets
        cw = self._cw
        sync_typography(cw, self.S, _TYPO_ROWS)
        self.S.tick_size = self.S.label_size * 0.85
        self.S.minor_tick_size = self.S.label_size * 0.65
        self.S.legend_title_size = self.S.legend_size
        self.S.lw_kr = float(W["line_width"].val)
        self.S.lw_baseline = self.S.lw_kr
        self.S.marker_size = float(W["marker_size"].val)
        self.S.marker_every = 1
        self.S.grid_alpha = cw.get_val("grid_alpha", self.S.grid_alpha)
        self.S.show_minor_tick_labels = bool(
            W["show_minor_tick_labels"].get_status()[0]
        )
        self.S.show_grid = bool(W["show_grid"].get_status()[0])
        self.S.use_family_linestyles = False
        for pk in self._xlog_keys:
            self.S.xlog[pk] = bool(W[f"xlog_{pk}"].get_status()[0])
        for pk in self._ylog_keys:
            self.S.ylog[pk] = bool(W[f"ylog_{pk}"].get_status()[0])
        for key in self._series_keys:
            self.S.show[key] = bool(W[f"show_{key}"].get_status()[0])

    def _push_settings_to_widgets(self):
        """Push current self.S state into all widgets (for load-settings)."""
        W = self._widgets
        cw = self._cw
        push_typography(cw, self.S, _TYPO_ROWS)
        W["line_width"].set_val(self.S.lw_kr)
        W["marker_size"].set_val(self.S.marker_size)
        W["line_width"].set_val(self.S.lw_kr)
        W["marker_size"].set_val(self.S.marker_size)
        for name in ("show_grid", "show_minor_tick_labels"):
            if bool(W[name].get_status()[0]) != getattr(self.S, name):
                W[name].set_active(0)
        for pk in self._xlog_keys:
            if bool(W[f"xlog_{pk}"].get_status()[0]) != self.S.xlog.get(pk, True):
                W[f"xlog_{pk}"].set_active(0)
        for pk in self._ylog_keys:
            if bool(W[f"ylog_{pk}"].get_status()[0]) != self.S.ylog.get(pk, False):
                W[f"ylog_{pk}"].set_active(0)
        for key in self._series_keys:
            if bool(W[f"show_{key}"].get_status()[0]) != self.S.show.get(key, True):
                W[f"show_{key}"].set_active(0)

    # ================================================================== #
    #  CALLBACKS                                                          #
    # ================================================================== #

    def _on_ui_change(self):
        """Fast path: visual-only changes (line width, markers, grid, scales)."""
        if self._updating:
            return
        self._sync_from_widgets()
        self._apply_visual()

    def _on_visibility_change(self):
        """Series visibility toggled — needs full redraw to add/remove artists."""
        if self._updating:
            return
        self._sync_from_widgets()
        self.redraw()

    def _on_restyle(self):
        """Lightweight: update typography without recomputing data."""
        if self._updating:
            return
        self._sync_from_widgets()
        self._restyle()

    def _restyle(self):
        S = self.S
        restyle_axes(
            self.fig,
            [(self.axes[i], f"({chr(97 + i)})") for i in range(2)],
            S,
            use_tex=True,
            font_attr="label_size",
            minor_tick_attr="minor_tick_size",
        )

    def _reload_settings(self):
        self.S = _load_settings(SETTINGS_FILE)
        self._updating = True
        self._push_settings_to_widgets()
        self._updating = False
        self.redraw()

    def _export(self, fmt):
        self._sync_from_widgets()
        _save_settings(self.S, SETTINGS_FILE)
        path = f"{DEFAULT_STEM}.{fmt}"
        export_figure_with_tex(
            self.fig,
            path,
            redraw_callback=lambda: self.redraw(),
            dpi=DPI,
        )
        print(f"[export] {path}")

    # ================================================================== #
    #  REDRAW                                                             #
    # ================================================================== #

    def redraw(self):
        _clear_legend(self.fig)
        _draw_panels(self.axes, self.R, self.S)
        self.fig.canvas.draw_idle()

    def _apply_visual(self):
        """Fast path: update line widths, marker sizes, grid, scales
        on existing artists without clearing / replotting."""
        S = self.S

        for idx, (metric, pk, default_ylog) in enumerate(PANELS_COL):
            ax = self.axes[idx]

            # Update all line artists
            for line in ax.lines:
                line.set_linewidth(S.lw_kr)
                line.set_markersize(S.marker_size)

            # Axis scales
            use_xlog = S.xlog.get(pk, True)
            use_ylog = S.ylog.get(pk, {"E1": False, "E2": False, "E3": False, "CR": True}.get(pk, False))
            ax.set_xscale("log" if use_xlog else "linear")
            ax.set_yscale("log" if use_ylog else "linear")
            ax.set_xlim(S.x_min, S.x_max)

            # Grid
            if S.show_grid:
                ax.grid(True, which="major", alpha=S.grid_alpha, lw=0.35)
                ax.grid(True, which="minor", alpha=S.grid_alpha * 0.35, lw=0.2)
            else:
                ax.grid(False, which="both")

            # Minor ticks
            ax.minorticks_on()
            ax.tick_params(which="major", labelsize=S.tick_size)
            ax.tick_params(which="minor", labelsize=S.minor_tick_size, length=2)
            if not S.show_minor_tick_labels:
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())

        self._restyle()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def main():
    ap = argparse.ArgumentParser(
        description="Figure 4: Figure 4 + FCC KR (T/O/I) dashed overlays"
    )
    ap.add_argument("--force", action="store_true", help="Regenerate data")
    ap.add_argument("--plot-only", action="store_true", help="Plot from existing data")
    ap.add_argument("--quick", action="store_true", help="Fewer Fibonacci points")
    ap.add_argument(
        "--static", action="store_true", help="Non-interactive export (PNG + PDF)"
    )
    ap.add_argument("--stem", default=DEFAULT_STEM, help="Output filename stem")
    args = ap.parse_args()

    if args.plot_only:
        if not os.path.exists(DATA_FILE):
            sys.exit(f"{DATA_FILE} not found — run without --plot-only first.")
        results = _load_data()
    else:
        results = generate_data(DEVICE, force=args.force, quick=args.quick)

    settings = _load_settings(SETTINGS_FILE)

    if args.static:
        plot_static(results, settings, stem=args.stem)
    else:
        _app = InteractiveFigure(results, settings)
        plt.show()


if __name__ == "__main__":
    main()
