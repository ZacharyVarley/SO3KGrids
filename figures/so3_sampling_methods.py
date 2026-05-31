#!/usr/bin/env python
"""SO(3) sampler benchmark: E3/E3* and covering-radius excess vs N_S3 (panels 13–14)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from src.so3_sampling.benchmark_cache import method_slug

_HERE = Path(__file__).resolve().parent
DATA_FILE = str(_HERE / "data" / "figure13_so3_sampling_methods.npz")
SETTINGS_JSON = str(_HERE / "settings" / "so3_sampling_methods.json")

DPI = 600

LABEL_SIZE = 8.0
TICK_SIZE = 7.0
MINOR_TICK_SIZE = 5.0
LEGEND_SIZE = 6.5

LW = 1.0
MARKER_SIZE = 4.0
MARKER_EDGE_LW = 0.35

SHOW_GRID = True
GRID_ALPHA = 0.20
SHOW_MINOR_TICK_LABELS = False

# Internal cache key for octahedral KR + FCC cubochoric fill (avoid "FCC-KR" in labels).
METHOD_FCC_FILL_KR = "FCC KR"

METHOD_ORDER = [
    "Hopf",
    "HArDiSh",
    "Cubochoric",
    "super-Fibonacci",
    METHOD_FCC_FILL_KR,
]

# Grayscale + linestyle (print-safe; no color required).
SERIES_STYLE = {
    "Hopf": {"color": "#111111", "ls": "-", "lw": LW},
    "HArDiSh": {"color": "#444444", "ls": "--", "lw": LW},
    "Cubochoric": {"color": "#666666", "ls": "-.", "lw": LW},
    "super-Fibonacci": {"color": "#888888", "ls": ":", "lw": LW},
    METHOD_FCC_FILL_KR: {"color": "#000000", "ls": (0, (5, 1)), "lw": LW * 1.15},
}

MARKERS = {
    "Hopf": "o",
    "HArDiSh": "s",
    "Cubochoric": "^",
    "super-Fibonacci": "D",
    METHOD_FCC_FILL_KR: "p",
}

LEGEND_LABELS = {
    "Hopf": r"$\mathrm{Hopf}$",
    "HArDiSh": r"$\mathrm{HArDiSh}$",
    "Cubochoric": r"$\mathrm{Cubochoric}$",
    "super-Fibonacci": r"$\mathrm{super\mbox{-}Fibonacci}$",
    METHOD_FCC_FILL_KR: r"$\mathrm{KR}_{O}\;(\mathrm{FCC\ lattice})$",
}

SHOW = {name: True for name in METHOD_ORDER}


def load_col_settings(path: str = SETTINGS_JSON) -> None:
    """Apply column settings from JSON to module globals."""
    global LABEL_SIZE, TICK_SIZE, MINOR_TICK_SIZE, LEGEND_SIZE, LW, MARKER_SIZE
    global SHOW_GRID, GRID_ALPHA, SHOW_MINOR_TICK_LABELS, SHOW
    if not os.path.exists(path):
        print(f"[settings] {path} not found — using script defaults")
        return
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for key, global_name in (
        ("label_size", "LABEL_SIZE"),
        ("tick_size", "TICK_SIZE"),
        ("minor_tick_size", "MINOR_TICK_SIZE"),
        ("legend_size", "LEGEND_SIZE"),
        ("lw", "LW"),
        ("marker_size", "MARKER_SIZE"),
        ("grid_alpha", "GRID_ALPHA"),
    ):
        if key in d:
            globals()[global_name] = d[key]
    if "show_grid" in d:
        SHOW_GRID = d["show_grid"]
    if "show_minor_tick_labels" in d:
        SHOW_MINOR_TICK_LABELS = d["show_minor_tick_labels"]
    if "show" in d:
        SHOW.update(d["show"])
    print(f"[settings] Loaded {path}")


def load_data() -> dict[str, dict[str, object]]:
    if not os.path.exists(DATA_FILE):
        print(
            f"Error: '{DATA_FILE}' not found. Run "
            "`python -m generators.so3_sampling_benchmark --pack-only` first."
        )
        sys.exit(1)
    raw = dict(np.load(DATA_FILE, allow_pickle=True))
    results: dict[str, dict[str, object]] = {}
    for name in METHOD_ORDER:
        slug = method_slug(name)
        n_eff = raw.get(f"{slug}_n_eff")
        if n_eff is None:
            continue
        results[name] = {
            "n_eff": np.asarray(n_eff),
            "E3_ratio": np.asarray(raw[f"{slug}_e3_ratio"]),
            "cr_excess": np.asarray(raw[f"{slug}_cr_excess"]),
        }
    return results


def _style_axis(ax, *, ylabel: str) -> None:
    ax.set_xlabel(r"$N_{\mathrm{S^3}}$", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xscale("log")
    ax.autoscale(axis="x", tight=False)
    ax.tick_params(which="major", labelsize=TICK_SIZE)
    ax.minorticks_on()
    ax.tick_params(which="minor", labelsize=MINOR_TICK_SIZE, length=2)
    if not SHOW_MINOR_TICK_LABELS:
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if SHOW_GRID:
        ax.grid(True, which="major", alpha=GRID_ALPHA, lw=0.35)
        ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.35, lw=0.2)
    else:
        ax.grid(False)


def _plot_series(
    ax,
    results: dict[str, dict[str, object]],
    visible: list[str],
    *,
    metric: str,
    require_positive_y: bool,
    with_legend: bool,
) -> None:
    for name in visible:
        r = results.get(name, {})
        x = np.asarray(r.get("n_eff", []))
        y = np.asarray(r.get(metric, []))
        ok = np.isfinite(y) & (x > 0)
        if require_positive_y:
            ok &= y > 0
        if not ok.any():
            continue
        order = np.argsort(x[ok])
        st = SERIES_STYLE[name]
        ax.plot(
            x[ok][order],
            y[ok][order],
            color=st["color"],
            linestyle=st["ls"],
            lw=st["lw"],
            marker=MARKERS[name],
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_LW,
            markerfacecolor=st["color"],
            markeredgecolor="white",
            label=LEGEND_LABELS[name] if with_legend else None,
        )


def _add_legend(ax, *, loc: str = "lower right") -> None:
    ax.legend(
        loc=loc,
        fontsize=LEGEND_SIZE,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#ccc",
        handlelength=2.0,
        labelspacing=0.35,
    )


def draw_e3_panel(ax, results: dict[str, dict[str, object]], visible: list[str]) -> None:
    _plot_series(
        ax,
        results,
        visible,
        metric="E3_ratio",
        require_positive_y=True,
        with_legend=True,
    )
    _style_axis(ax, ylabel=r"$E_3/E_3^*$")
    _add_legend(ax, loc="lower right")


def draw_cr_panel(ax, results: dict[str, dict[str, object]], visible: list[str]) -> None:
    _plot_series(
        ax,
        results,
        visible,
        metric="cr_excess",
        require_positive_y=False,
        with_legend=True,
    )
    ax.axhline(0.0, color="k", linestyle=":", lw=0.8, alpha=0.7, zorder=0)
    _style_axis(ax, ylabel=r"$\theta/\theta^*-1$")
    _add_legend(ax, loc="lower left")
