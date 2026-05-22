#!/usr/bin/env python
"""Figure 4 — E₃ ratio + covering radius vs N_eff (manual / non-interactive)."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import NullFormatter
from figure_ui_common import (
    FIBONACCI_SERIES_KEYS,
    add_panel_label,
    figure4_mean_non_fib_marker_count,
    markevery_stride,
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS — edit these directly                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Output
STEM = "figure4_col"
DPI = 600

# Data
DATA_FILE = "figure4_data.npz"

# Figure size (IUCr single column)
FIG_W = 3.25
FIG_H = 5.5

# Margins
MARGIN_LEFT = 0.18
MARGIN_RIGHT = 0.97
MARGIN_TOP = 0.97
MARGIN_BOTTOM = 0.09
HSPACE = 0.34

# Typography
LABEL_SIZE = 8.0
TICK_SIZE = 7.0
MINOR_TICK_SIZE = 5.0
TITLE_SIZE = 9.0
PANEL_LABEL_SIZE = 9.0
PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
LEGEND_SIZE = 7.0

# Lines
LW_KR = 1.0
MARKER_SIZE = 4.0
MARKER_EDGE_LW = 0.35

# Axis range
X_MIN = 10_000.0
X_MAX = 1_000_000.0

# Grid
SHOW_GRID = True
GRID_ALPHA = 0.20
SHOW_MINOR_TICK_LABELS = False

# Per-panel axis scales
XLOG = {"E3": True, "CR": True}
YLOG = {"E3": False, "CR": False}

# Series visibility — set False to hide
SHOW = {
    "rej": True, "fcc": True, "fib_rej": True, "fib_all": True,
    "kr_C2": True, "kr_C3": True, "kr_C4": True, "kr_C6": True,
    "kr_D2": True, "kr_D3": True, "kr_D4": True, "kr_D6": True,
    "kr_T": True, "kr_O": True, "kr_I": True,
    "kr_fcc_T": True, "kr_fcc_O": True, "kr_fcc_I": True,
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  THEME                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

KR_NAMES = ["C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6", "T", "O", "I"]

PLOT_ORDER = (
    [f"kr_{n}" for n in KR_NAMES]
    + ["kr_fcc_T", "kr_fcc_O", "kr_fcc_I"]
    + ["fib_all", "fib_rej", "fcc", "rej"]
)

PANELS_COL = [
    ("E3_ratio", "E3", False),
    ("cr_ratio", "CR", True),
]

PALETTE = {
    "C2": "#1f77b4", "C3": "#aec7e8", "C4": "#ff7f0e", "C6": "#ffbb78",
    "D2": "#2ca02c", "D3": "#98df8a", "D4": "#d62728", "D6": "#ff9896",
    "T": "#9467bd", "O": "#c5b0d5", "I": "#8c564b",
    "rej": "#1a1a1a", "fcc": "#1a1a1a", "fib_rej": "#5d6d7e", "fib_all": "#95a5a6",
}

LINESTYLES = {k: "-" for k in PALETTE}
LINESTYLES["fcc"] = "--"
LINESTYLES["kr_fcc_T"] = "--"
LINESTYLES["kr_fcc_O"] = "--"
LINESTYLES["kr_fcc_I"] = "--"

MARKERS = {
    "C2": "d", "C3": "^", "C4": "D", "C6": "h",
    "D2": ("d", 90), "D3": ("^", 90), "D4": ("D", 45), "D6": ("h", 90),
    "T": "H", "O": "o", "I": "p",
    "rej": "o", "fcc": "o", "fib_rej": "o", "fib_all": "o",
}

LEGEND_COLUMNS = [
    ("Baselines", ["rej", "fcc", "fib_rej", "fib_all"]),
    ("Cyclic", ["kr_C2", "kr_C3", "kr_C4", "kr_C6"]),
    ("Dihedral", ["kr_D2", "kr_D3", "kr_D4", "kr_D6"]),
    ("Cubic / Ico", ["kr_T", "kr_O", "kr_I"]),
]

LEGEND_LABELS = {
    "rej": r"$\textrm{Cubochoric (PC)}$",
    "fcc": r"$\textrm{Cubochoric (FCC)}$",
    "fib_rej": r"$\textrm{Fibonacci (reject)}$",
    "fib_all": r"$\textrm{Fibonacci (canon.)}$",
    "kr_C2": r"$C_{2}$", "kr_C3": r"$C_{3}$", "kr_C4": r"$C_{4}$", "kr_C6": r"$C_{6}$",
    "kr_D2": r"$D_{2}$", "kr_D3": r"$D_{3}$", "kr_D4": r"$D_{4}$", "kr_D6": r"$D_{6}$",
    "kr_T": r"$T$", "kr_O": r"$O$", "kr_I": r"$I$",
}


def _series_name(key):
    if key.startswith("kr_fcc_"):
        return key[7:]
    if key.startswith("kr_"):
        return key[3:]
    return key


def _color(key):
    return PALETTE.get(_series_name(key), "#666")


def _ls(key):
    if key in LINESTYLES:
        return LINESTYLES[key]
    return LINESTYLES.get(_series_name(key), "-")


def _marker(key):
    m = MARKERS.get(_series_name(key), "o")
    if isinstance(m, tuple):
        sym, rot = m
        ms = MarkerStyle(sym)
        ms._transform = ms.get_transform().rotate_deg(rot)
        return ms
    return m


def _panel_ylabel(pk):
    if pk == "E3":
        return r"$E_3/E_3^*$"
    return r"$\theta_{\mathrm{S^3}}/\theta_{\mathrm{S^3}}^*-1$"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SETTINGS_JSON = "figure4_col_settings.json"


def load_col_settings(path: str = SETTINGS_JSON) -> None:
    """Apply interactive column settings from JSON to module globals."""
    global LABEL_SIZE, TICK_SIZE, MINOR_TICK_SIZE, LEGEND_SIZE, LW_KR, MARKER_SIZE
    global X_MIN, X_MAX, SHOW_GRID, GRID_ALPHA, SHOW_MINOR_TICK_LABELS, XLOG, YLOG, SHOW
    if not os.path.exists(path):
        print(f"[settings] {path} not found — using script defaults")
        return
    with open(path) as f:
        d = json.load(f)
    if "label_size" in d:
        LABEL_SIZE = d["label_size"]
    if "tick_size" in d:
        TICK_SIZE = d["tick_size"]
    if "minor_tick_size" in d:
        MINOR_TICK_SIZE = d["minor_tick_size"]
    if "legend_size" in d:
        LEGEND_SIZE = d["legend_size"]
    if "lw_kr" in d:
        LW_KR = d["lw_kr"]
    if "marker_size" in d:
        MARKER_SIZE = d["marker_size"]
    if "x_min" in d:
        X_MIN = d["x_min"]
    if "x_max" in d:
        X_MAX = d["x_max"]
    if "show_grid" in d:
        SHOW_GRID = d["show_grid"]
    if "grid_alpha" in d:
        GRID_ALPHA = d["grid_alpha"]
    if "show_minor_tick_labels" in d:
        SHOW_MINOR_TICK_LABELS = d["show_minor_tick_labels"]
    if "xlog" in d:
        XLOG.update(d["xlog"])
    if "ylog" in d:
        YLOG.update(d["ylog"])
    if "show" in d:
        SHOW.update(d["show"])
    print(f"[settings] Loaded {path}")


def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Run figure4_b.py --generate first.")
        sys.exit(1)
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
# ║  PLOTTING                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def draw_standard_panel(ax, metric, pk, visible, results):
    use_xlog = XLOG.get(pk, True)
    use_ylog = YLOG.get(pk, False)
    ms = float(MARKER_SIZE)
    lw = float(LW_KR)
    mew = float(MARKER_EDGE_LW)

    fib_target = None
    if any(k in FIBONACCI_SERIES_KEYS for k in visible):
        fib_target = figure4_mean_non_fib_marker_count(
            visible, results, metric,
            x_min=X_MIN, x_max=X_MAX, xlog=XLOG, ylog=YLOG, pk=pk,
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
        ok2 = (x >= X_MIN) & (x <= X_MAX)
        x, yy = x[ok2], yy[ok2]

        is_bl = not key.startswith("kr_")
        if key in FIBONACCI_SERIES_KEYS and fib_target is not None and ms > 0:
            me = markevery_stride(len(x), fib_target)
        else:
            me = 1
        ax.plot(
            x, yy - 1 if pk == "CR" else yy,
            color=_color(key), ls=_ls(key), lw=lw,
            marker=_marker(key) if ms > 0 else None, markersize=ms,
            markevery=me, markeredgewidth=mew, markeredgecolor="white",
            zorder=10 if is_bl else 5,
        )

    ax.set_xlabel(r"$N_{\mathrm{S^3}}$", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel(_panel_ylabel(pk), fontsize=LABEL_SIZE, labelpad=20)
    ax.set_xscale("log" if use_xlog else "linear")
    ax.set_yscale("log" if use_ylog else "linear")
    ax.set_xlim(X_MIN, X_MAX)
    ax.tick_params(which="major", labelsize=TICK_SIZE)
    ax.minorticks_on()
    ax.tick_params(which="minor", labelsize=MINOR_TICK_SIZE, length=2)
    if not SHOW_MINOR_TICK_LABELS:
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if SHOW_GRID:
        glw = float(MARKER_EDGE_LW)
        ax.grid(True, which="major", alpha=GRID_ALPHA, lw=glw)
        ax.grid(True, which="minor", alpha=GRID_ALPHA * 0.35, lw=glw * 0.6)


def draw_legend(ax, visible_set):
    ax.set_axis_off()
    dashed_visible = any(k in visible_set for k in ("kr_fcc_T", "kr_fcc_O", "kr_fcc_I"))
    col1 = ["kr_C2", "kr_C3", "kr_C4", "kr_C6"]
    col2 = ["kr_D2", "kr_D3", "kr_D4", "kr_D6"]
    col3 = ["kr_T", "kr_O", "kr_I", "_kr_fcc_note" if dashed_visible else "_blank"]
    col4 = ["rej", "fcc", "fib_rej", "fib_all"]
    legend_items = col1 + col2 + col3 + col4
    handles, labels = [], []
    for key in legend_items:
        if key == "_blank":
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
        elif key == "_kr_fcc_note":
            handles.append(Line2D([0], [0], color="#444", ls="--", lw=float(LW_KR)))
            labels.append(r"$\textrm{FCC}$")
        elif key not in visible_set:
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
        else:
            ms = MARKER_SIZE
            mk = _marker(key) if ms > 0 else None
            handles.append(Line2D(
                [0], [0], color=_color(key), ls=_ls(key), lw=float(LW_KR),
                marker=mk, markersize=ms * 1.2 if ms > 0 else 0,
                markeredgewidth=float(MARKER_EDGE_LW), markeredgecolor="white",
            ))
            labels.append(LEGEND_LABELS.get(key, key))
    ax.legend(handles, labels, ncol=4, fontsize=LEGEND_SIZE,
              frameon=True, fancybox=False, framealpha=1.0, edgecolor="#ccc",
              loc="center", borderpad=0.45, columnspacing=1.0,
              handlelength=1.8, handletextpad=0.45, labelspacing=0.35)


def main():
    load_col_settings()
    results = load_data()
    visible = [k for k in PLOT_ORDER if k in results and SHOW.get(k, True)]
    visible_set = set(visible)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
    gs = GridSpec(3, 1, left=MARGIN_LEFT, right=MARGIN_RIGHT,
                  bottom=MARGIN_BOTTOM, top=MARGIN_TOP,
                  hspace=HSPACE, height_ratios=[1, 1, 0.28])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for idx, (metric, pk, _) in enumerate(PANELS_COL):
        ax = axes[idx]
        draw_standard_panel(ax, metric, pk, visible, results)
        add_panel_label(ax, f"({chr(97 + idx)})", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)

    draw_legend(axes[2], visible_set)

    for ext in ("png", "pdf"):
        path = f"{STEM}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"[export] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
