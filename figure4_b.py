#!/usr/bin/env python3
"""
figure4_b.py — Riesz energy ratios (E1, E2, E3) and covering-radius ratio vs N_eff.

Compares KR grids (Laue 2–12), cubochoric FCC grid (Laue 1), and Fibonacci
sampling baselines across a range of effective grid sizes.

Layout: 2×2 panel grid with a hand-crafted structured legend. Interactive mode
gives a clean control panel in a separate window with per-panel axis-scale
toggles, grid control, and full typography tuning. Static mode exports
publication-ready PNG/PDF.

Methods plotted:
    • KR grids for Laue 2–12 (C2 … I/532), all built from FCC cubochoric fill
    • Cubochoric FCC grid (black)
  • Fibonacci sampling — reject w<0 (solid grey)
  • Fibonacci sampling — all + canonicalize (dashed grey)

Usage:
    python figure4_b.py              # generate data (if needed) + interactive plot
    python figure4_b.py --force      # force-regenerate all data
    python figure4_b.py --plot-only  # plot from existing .npz
    python figure4_b.py --static     # non-interactive PNG/PDF export
    python figure4_b.py --quick      # fewer Fibonacci samples (for testing)
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

from grid_FZ import kr_sample_laue
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

DATA_FILE = "figure4_b_data.npz"
SETTINGS_FILE = "figure4_b_settings.json"
DEFAULT_STEM = "figure4_b"
N_EFF_MIN = 100_000
N_EFF_MAX = 1_000_000
FIB_N_STEP = 100_000
DPI = 300
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
    "fcc": "#1a1a1a",
    "fib_rej": "#5d6d7e",
    "fib_all": "#95a5a6",
}

LINESTYLES = {k: "-" for k in PALETTE}
LINESTYLES["fib_all"] = "-"
LINESTYLES["fib_rej"] = "-"
LINESTYLES["fcc"] = "-"

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
    "fcc": "o",
    "fib_rej": "o",
    "fib_all": "o",
}

KR_NAMES = [g["name"] for g in KR_GROUPS]


# Paint order: KR underneath, baselines on top
PLOT_ORDER = (
    [f"kr_{n}" for n in KR_NAMES]
    + ["fib_all", "fib_rej", "fcc"]
)

# Panel short keys (used in settings dicts for xlog/ylog)
PANEL_KEYS = ["E1", "E2", "E3", "CR"]

# 2×2 layout: (E1, E2) row 0, (E3, θ_cov) row 1, legend row 2
PANELS_2X2 = [
    ("E1_ratio", "E1", False),
    ("E2_ratio", "E2", False),
    ("E3_ratio", "E3", False),
    ("cr_ratio", "CR", True),
]

# Legend: 4 columns for horizontal layout below panels
LEGEND_COLUMNS = [
    ("Baselines", ["fcc", "fib_rej", "fib_all"]),
    ("Cyclic", ["kr_C2", "kr_C3", "kr_C4", "kr_C6"]),
    ("Dihedral", ["kr_D2", "kr_D3", "kr_D4", "kr_D6"]),
    ("Cubic / Ico", ["kr_T", "kr_O", "kr_I"]),
]

LEGEND_LABELS = {
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


def _enumerate_fcc_semi() -> List[int]:
    out = []
    for s in range(1, 501):
        ne = 2 * CARD_C1 * _kr_grid_n_fcc(s, s)
        if ne > N_EFF_MAX:
            break
        if N_EFF_MIN <= ne <= N_EFF_MAX:
            out.append(s)
    return out


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
    all_keys = [f"kr_{g['name']}" for g in KR_GROUPS] + ["fcc", "fib_rej", "fib_all"]

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
    d = {"fcc": True, "fib_rej": True, "fib_all": True}
    for n in KR_NAMES:
        d[f"kr_{n}"] = True
    return d


def _default_xlog() -> Dict[str, bool]:
    return {pk: True for pk in PANEL_KEYS}


def _default_ylog() -> Dict[str, bool]:
    return {pk: (pk == "CR") for pk in PANEL_KEYS}


@dataclass
class Settings:
    # Typography
    label_size: float = 10.0
    tick_size: float = 8.5
    minor_tick_size: float = 6.5
    title_size: float = 11.0
    legend_size: float = 7.5
    legend_title_size: float = 7.5
    # Lines
    lw_kr: float = 1.2
    lw_baseline: float = 1.6
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
    # Matplotlib legend fills column-first for ncol>1.
    # Feed in explicit column-major order to get desired visual columns:
    # col1: C2 C3 C4 C6
    # col2: D2 D3 D4 D6
    # col3: T O I
    # col4: Cubochoric FCC Fib reject Fib canon
    col1 = ["kr_C2", "kr_C3", "kr_C4", "kr_C6"]
    col2 = ["kr_D2", "kr_D3", "kr_D4", "kr_D6"]
    col3 = ["kr_T", "kr_O", "kr_I", "_blank"]
    col4 = ["fcc", "fib_rej", "fib_all", "_blank"]
    legend_items = col1 + col2 + col3 + col4

    handles = []
    labels = []

    for key in legend_items:
        if key == "_blank":
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
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
        me = 1 if ms > 0 else None
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
    Draw 2×2 panels: (E1, E2) row 0, (E3, θ_cov) row 1, legend row 2.
    axes is (3, 2): axes[0:2,0:2] for panels, axes[2,0] for legend (spans both cols).
    """
    visible = [k for k in PLOT_ORDER if k in results and S.show.get(k, True)]
    visible_set = set(visible)
    use_rel2best_E12 = len(visible) >= 2  # E1/E2: ratio-to-best spreads bundled lines

    for idx, (metric, pk, default_ylog) in enumerate(PANELS_2X2):
        ax = axes[idx // 2, idx % 2]
        ax.cla()
        if pk in ("E1", "E2") and use_rel2best_E12:
            _draw_rel2best_panel(ax, metric, f"{pk}_rel", visible, results, S)
        else:
            _draw_standard_panel(ax, metric, pk, visible, results, S)
        ax.text(
            0.02,
            0.98,
            f"$\\mathbf{{({chr(97 + idx)})}}$",
            transform=ax.transAxes,
            fontsize=S.title_size,
            fontweight="bold",
            va="top",
        )

    leg_ax = axes[2, 0]
    leg_ax.cla()
    _draw_legend_in_axes(leg_ax, S, visible_set)

    return visible_set


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STATIC EXPORT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def plot_static(results: dict, s: Settings, stem: str = DEFAULT_STEM):
    """Export publication-ready 2×2 figure with legend."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.7,
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

    fig = plt.figure(figsize=(10.5, 6.5), dpi=150)
    gs = GridSpec(
        3,
        2,
        left=0.08,
        right=0.97,
        bottom=0.08,
        top=0.94,
        wspace=0.28,
        hspace=0.32,
        height_ratios=[1, 1, 0.28],
    )
    axes = np.empty((3, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])
    axes[2, 0] = fig.add_subplot(gs[2, :])
    axes[2, 1] = axes[2, 0]

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

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
                "mathtext.fontset": "cm",
                "axes.linewidth": 0.6,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
            }
        )

        self.fig = plt.figure(
            figsize=(10.5, 6.5),
            num="Figure 4b — Energy & Covering-Radius Ratios",
        )
        self._build_main_layout()

        self.ctrl_fig = plt.figure(figsize=(6.2, 8.8), num="Figure 4b — Controls")
        self.ctrl_fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.01)
        self._widgets: dict = {}
        self._series_keys: list[str] = []
        self._xlog_keys: list[str] = []
        self._ylog_keys: list[str] = []
        self._build_controls()
        self._connect_events()
        self.redraw()

    def _build_main_layout(self):
        """Build the 2×2 figure grid + legend."""
        self.fig.clear()
        gs = GridSpec(
            3,
            2,
            left=0.07,
            right=0.97,
            bottom=0.06,
            top=0.94,
            wspace=0.28,
            hspace=0.30,
            height_ratios=[1, 1, 0.28],
        )
        self.axes = np.empty((3, 2), dtype=object)
        for r in range(2):
            for c in range(2):
                self.axes[r, c] = self.fig.add_subplot(gs[r, c])
        self.axes[2, 0] = self.fig.add_subplot(gs[2, :])
        self.axes[2, 1] = self.axes[2, 0]

    # ================================================================== #
    #  BUILD CONTROLS                                                     #
    # ================================================================== #

    def _build_controls(self):
        """Build grouped, compact control panel with explicit style and scale controls."""
        S = self.S
        fig = self.ctrl_fig
        fig.clear()

        text_base = S.label_size
        y = 0.97
        pad = 0.026
        sh, bh = 0.020, 0.028
        ch = 0.016

        def section(title):
            nonlocal y
            y -= 0.012
            ax = fig.add_axes([0.04, y - 0.012, 0.92, 0.012])
            ax.set_axis_off()
            ax.text(
                0, 0, title, fontsize=8, fontweight="bold", color="#444", va="bottom"
            )
            y -= 0.021

        def slider(name, label, lo, hi, val, step=None):
            nonlocal y
            ax = fig.add_axes([0.12, y - sh, 0.80, sh])
            sl = Slider(ax, label, lo, hi, valinit=val, valstep=step, color="#4a90d9")
            sl.label.set_fontsize(7)
            sl.valtext.set_fontsize(7)
            self._widgets[name] = sl
            y -= pad
            return sl

        def checkbox_at(name, label, checked, x, y_pos, w):
            ax = fig.add_axes([x, y_pos - ch, w, ch])
            chk = CheckButtons(ax, [label], [checked])
            for t in chk.labels:
                t.set_fontsize(6.5)
            self._widgets[name] = chk
            return chk

        def btn(name, label, x, w):
            nonlocal y
            ax = fig.add_axes([x, y - bh, w, bh])
            b = Button(ax, label, color="#e8eef4", hovercolor="#d0dce8")
            b.label.set_fontsize(7)
            self._widgets[name] = b
            return b

        self._series_keys = (
            ["fcc", "fib_rej", "fib_all"] + [f"kr_{n}" for n in KR_NAMES]
        )
        self._xlog_keys = ["E1", "E2", "E3", "CR"]
        self._ylog_keys = ["E1", "E2", "E3", "CR"]

        # Export
        section("Export")
        btn("save_png", "Save PNG", 0.04, 0.22)
        btn("save_pdf", "Save PDF", 0.28, 0.22)
        btn("save_cfg", "Save settings", 0.52, 0.22)
        btn("load_cfg", "Load settings", 0.76, 0.22)
        y -= bh + 0.02

        # Typography
        section("Typography")
        slider("text_size", "Text size", 7, 24, text_base, step=0.5)
        slider("legend_size", "Legend size", 6, 24, S.legend_size, step=0.5)

        # Display and style
        section("Display")
        y0 = y
        checkbox_at("show_grid", "Grid", S.show_grid, 0.04, y0, 0.12)
        checkbox_at(
            "show_minor_tick_labels",
            "Minor ticks",
            S.show_minor_tick_labels,
            0.20,
            y0,
            0.14,
        )
        y -= ch + pad
        slider("grid_alpha", "Grid opacity", 0.0, 1.0, S.grid_alpha)

        section("Plot style")
        slider("line_width", "Line width", 0.5, 8.0, S.lw_kr, step=0.1)
        slider("marker_size", "Marker size", 0, 8, S.marker_size, step=0.5)

        # Axis scales
        section("Axis scales")
        y0 = y
        xw = 0.10
        labels = ["E1", "E2", "E3", "CR"]
        for i, pk in enumerate(labels):
            checkbox_at(
                f"xlog_{pk}", f"x:{pk}", S.xlog.get(pk, True), 0.04 + i * 0.23, y0, xw
            )
        y -= ch + 0.008
        y1 = y
        for i, pk in enumerate(labels):
            checkbox_at(
                f"ylog_{pk}", f"y:{pk}", S.ylog.get(pk, False), 0.04 + i * 0.23, y1, xw
            )
        y -= ch + pad

        # Series visibility
        section("Series visibility")
        series_names = self._series_keys
        n_cols = 5
        cw = 0.92 / n_cols
        for i, key in enumerate(series_names):
            short = key.replace("kr_", "").replace("fib_", "fib ")
            row, col = i // n_cols, i % n_cols
            y_row = y - row * (ch + 0.006)
            chk = checkbox_at(
                f"show_{key}",
                short,
                S.show.get(key, True),
                0.04 + col * cw,
                y_row,
                cw - 0.02,
            )
            for t in chk.labels:
                t.set_color(_color(key))
        n_rows = (len(series_names) + n_cols - 1) // n_cols
        y -= n_rows * (ch + 0.006) + 0.01

    # ================================================================== #
    #  CONNECT EVENTS                                                     #
    # ================================================================== #

    def _connect_events(self):
        W = self._widgets
        for k in (
            "text_size",
            "legend_size",
            "grid_alpha",
            "line_width",
            "marker_size",
        ):
            W[k].on_changed(lambda _, _k=k: self._on_ui_change())
        for k in ("show_grid", "show_minor_tick_labels"):
            W[k].on_clicked(lambda _: self._on_ui_change())
        for pk in self._xlog_keys:
            W[f"xlog_{pk}"].on_clicked(lambda _: self._on_ui_change())
        for pk in self._ylog_keys:
            W[f"ylog_{pk}"].on_clicked(lambda _: self._on_ui_change())
        for key in self._series_keys:
            W[f"show_{key}"].on_clicked(lambda _: self._on_ui_change())
        W["save_png"].on_clicked(lambda _: self._export("png"))
        W["save_pdf"].on_clicked(lambda _: self._export("pdf"))
        W["save_cfg"].on_clicked(lambda _: _save_settings(self.S, SETTINGS_FILE))
        W["load_cfg"].on_clicked(lambda _: self._reload_settings())

    # ================================================================== #
    #  SYNC  widgets ↔ Settings                                           #
    # ================================================================== #

    def _sync_from_widgets(self):
        W = self._widgets
        base = float(W["text_size"].val)
        self.S.label_size = base
        self.S.tick_size = base * 0.85
        self.S.minor_tick_size = base * 0.65
        self.S.title_size = base * 1.1
        self.S.legend_size = float(W["legend_size"].val)
        self.S.legend_title_size = float(W["legend_size"].val)
        self.S.lw_kr = float(W["line_width"].val)
        self.S.lw_baseline = self.S.lw_kr
        self.S.marker_size = float(W["marker_size"].val)
        self.S.marker_every = 1
        self.S.grid_alpha = float(W["grid_alpha"].val)
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
        W["text_size"].set_val(self.S.label_size)
        W["legend_size"].set_val(self.S.legend_size)
        W["grid_alpha"].set_val(self.S.grid_alpha)
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
        if self._updating:
            return
        self._sync_from_widgets()
        self.redraw()

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
        self.fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"[export] {path}")

    # ================================================================== #
    #  REDRAW                                                             #
    # ================================================================== #

    def redraw(self):
        _clear_legend(self.fig)
        _draw_panels(self.axes, self.R, self.S)
        self.fig.canvas.draw_idle()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def main():
    ap = argparse.ArgumentParser(
        description="Figure 4b: FCC cubochoric fill for all KR groups + FCC baseline"
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
