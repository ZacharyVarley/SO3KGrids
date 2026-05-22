#!/usr/bin/env python
"""Figure 2 — E₃/E₃* ratio + optimal cu_z* (manual / non-interactive)."""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import NullFormatter
from figure_ui_common import add_panel_label

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS — edit these directly                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Output
STEM = "figure2_col"
DPI = 600

# Data
DATA_FILE = "figure2_data.json"

# Figure size (IUCr single column)
FIG_W = 3.25
FIG_H = 5.5

# Margins
MARGIN_LEFT = 0.18
MARGIN_RIGHT = 0.97
MARGIN_TOP = 0.97
MARGIN_BOTTOM = 0.09
HSPACE = 0.30

# Typography
FONT_SIZE = 8.0
TITLE_SIZE = 9.0
PANEL_LABEL_SIZE = 9.0
PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
LEGEND_SIZE = 7.0

# Scatter / lines
MARKER_SIZE = 4.0
LINE_WIDTH = 1.0

# Data range
CU_XY_LO = 3
CU_XY_HI = 25

# Group visibility
SHOW_GROUPS = {"C2": True, "C3": True, "C4": True, "C6": True,
               "D2": True, "D3": True, "D4": True, "D6": True}

# Axis options
LOG_X_RATIO = True
LOG_Y_RATIO = False
SHOW_GRID = True
GRID_ALPHA = 0.20
SHOW_MINOR_TICK_LABELS = False

# Overlays
SHOW_FITS_CUZ = True
SHOW_FITS_LOGQUAD = False
SHOW_FITS_RATIO = False
SHOW_MISMATCHES = False
SHOW_R_CONT = False
SHOW_LEGEND_R = True

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  THEME                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

GROUP_COLORS = {
    "C2": "#1f77b4", "C3": "#aec7e8", "C4": "#ff7f0e", "C6": "#ffbb78",
    "D2": "#2ca02c", "D3": "#98df8a", "D4": "#d62728", "D6": "#ff9896",
}
GROUP_MARKERS = {
    "C2": "d", "C3": "^", "C4": "D", "C6": "h",
    "D2": ("d", 90), "D3": ("^", 90), "D4": ("D", 45), "D6": ("h", 90),
}
_GROUP_NAMES = ["C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6"]


def _latex_group_name(name):
    m = re.match(r"^([CD])(\d+)$", name)
    if m:
        return rf"${m.group(1)}_{{{m.group(2)}}}$"
    return name


def _get_marker(gn):
    m = GROUP_MARKERS.get(gn, "o")
    if isinstance(m, tuple):
        sym, rot = m
        ms = MarkerStyle(sym)
        ms._transform = ms.get_transform().rotate_deg(rot)
        return ms
    return m


def _read_cu_z(rec):
    v = rec.get("cu_z_best")
    if v is None:
        v = rec.get("hz_best")
    return v


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_grouped():
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Run figure2_gen.py first.")
        sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    grouped = {}
    for r in data["results"]:
        grouped.setdefault(r["group"], []).append(r)
    for g in grouped:
        grouped[g].sort(key=lambda x: x["cu_xy"])
    visible = [g for g in _GROUP_NAMES if SHOW_GROUPS.get(g, True) and g in grouped]
    return grouped, visible


def plot_e3_ratio_panel(ax, grouped, visible):
    """Panel: E₃/E₃* vs N_{S³} (no panel letter)."""
    fs = int(round(FONT_SIZE))
    ms = float(MARKER_SIZE)
    lw = float(LINE_WIDTH)
    for gn in visible:
        recs = [r for r in grouped[gn] if CU_XY_LO <= r["cu_xy"] <= CU_XY_HI]
        if not recs:
            continue
        x = np.array([r["N_eff"] for r in recs], dtype=float)
        y = np.array([r["E3_ratio_best"] for r in recs], dtype=float)
        col = GROUP_COLORS.get(gn, "gray")
        ax.scatter(x, y, marker=_get_marker(gn), color=col, s=ms**2,
                   edgecolors="0.2", linewidths=0.3,
                   label=_latex_group_name(gn), zorder=5)
        if SHOW_FITS_RATIO and len(x) >= 2:
            log_x = np.log(x)
            coeffs = np.polyfit(log_x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            ys = np.polyval(coeffs, np.log(xs))
            ax.plot(xs, ys, color=col, linewidth=lw * 0.7, linestyle="--", alpha=0.6)

    ax.set_xscale("log" if LOG_X_RATIO else "linear")
    ax.set_yscale("log" if LOG_Y_RATIO else "linear")
    ax.set_xlabel(r"$N_{S^3}$", fontsize=fs)
    ax.set_ylabel(r"$E_3/E_3^*$", fontsize=fs)
    ax.tick_params(labelsize=max(6, fs - 1))
    ax.minorticks_on()
    if not SHOW_MINOR_TICK_LABELS:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=max(5, LEGEND_SIZE), loc="best", framealpha=0.8)
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)


def plot_cuz_panel(ax, grouped, visible):
    """Panel: cu_z* vs cu_xy (no panel letter)."""
    fs = int(round(FONT_SIZE))
    ms = float(MARKER_SIZE)
    lw = float(LINE_WIDTH)
    r_space = SHOW_FITS_LOGQUAD
    slope_dict = {}
    logquad_dict = {}
    for gn in visible:
        recs = [r for r in grouped[gn] if CU_XY_LO <= r["cu_xy"] <= CU_XY_HI]
        if not recs:
            continue
        x = np.array([r["cu_xy"] for r in recs], dtype=float)
        cuz = np.array([_read_cu_z(r) for r in recs], dtype=float)
        if SHOW_FITS_CUZ and len(x) >= 2:
            slope_dict[gn] = float(np.sum(x * cuz) / np.sum(x * x))
        if SHOW_FITS_LOGQUAD and len(x) >= 3:
            rhat = np.array([r.get("r_cont", _read_cu_z(r) / r["cu_xy"]) for r in recs], dtype=float)
            xl = np.log(x)
            A = np.vstack([np.ones_like(xl), xl, xl**2]).T
            beta, *_ = np.linalg.lstsq(A, rhat, rcond=None)
            logquad_dict[gn] = beta

    for gn in visible:
        recs = [r for r in grouped[gn] if CU_XY_LO <= r["cu_xy"] <= CU_XY_HI]
        if not recs:
            continue
        x = np.array([r["cu_xy"] for r in recs], dtype=float)
        cuz = np.array([_read_cu_z(r) for r in recs], dtype=float)
        col = GROUP_COLORS.get(gn, "gray")
        if r_space:
            y_scatter = np.array([r.get("r_cont", _read_cu_z(r) / r["cu_xy"]) for r in recs], dtype=float)
        else:
            y_scatter = cuz
        ax.scatter(x, y_scatter, marker=_get_marker(gn), color=col, s=ms**2,
                   edgecolors="0.2", linewidths=0.3, zorder=5)

        x_fit = np.linspace(float(x.min()), float(x.max()), 200)
        if SHOW_R_CONT and not r_space:
            cx_vals, rc_vals = [], []
            for r_rec in recs:
                rc = r_rec.get("r_cont")
                if rc is not None:
                    cx_vals.append(r_rec["cu_xy"])
                    rc_vals.append(rc * r_rec["cu_xy"])
            if cx_vals:
                ax.plot(cx_vals, rc_vals, color=col, linewidth=lw * 0.4, linestyle="-", alpha=0.5)
        if gn in slope_dict:
            slope = slope_dict[gn]
            if r_space:
                ax.axhline(slope, color=col, linewidth=lw * 0.5, linestyle="-", alpha=0.5)
            else:
                ax.plot(x_fit, slope * x_fit, color=col, linewidth=lw * 0.7, linestyle="-", alpha=0.6)
        if gn in logquad_dict:
            beta = logquad_dict[gn]
            lxf = np.log(x_fit)
            rhat_fit = beta[0] + beta[1] * lxf + beta[2] * lxf**2
            if r_space:
                ax.plot(x_fit, rhat_fit, color=col, linewidth=lw * 0.7, linestyle="-", alpha=0.7)
            else:
                ax.plot(x_fit, rhat_fit * x_fit, color=col, linewidth=lw * 0.7, linestyle="-", alpha=0.7)

    handles, labels = [], []
    for gn in visible:
        recs = [r for r in grouped[gn] if CU_XY_LO <= r["cu_xy"] <= CU_XY_HI]
        if not recs:
            continue
        col = GROUP_COLORS.get(gn, "gray")
        h = Line2D([], [], color=col, marker=_get_marker(gn), linestyle="None",
                   markersize=ms, markeredgewidth=0.3, markeredgecolor="0.2")
        lbl = _latex_group_name(gn)
        if gn in slope_dict:
            lbl = rf"{lbl}  $r^*\!={slope_dict[gn]:.3f}$"
        handles.append(h)
        labels.append(lbl)

    ax.set_xlabel(r"$\mathrm{cu}_{xy}$", fontsize=fs)
    if r_space:
        ax.set_ylabel(r"$\hat{r}$", fontsize=fs)
    else:
        ax.set_ylabel(r"$\mathrm{cu}_z^{\,*}$", fontsize=fs)
    ax.tick_params(labelsize=max(6, fs - 1))
    ax.minorticks_on()
    if not SHOW_MINOR_TICK_LABELS:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)
    if handles and SHOW_LEGEND_R:
        ax.legend(handles, labels, fontsize=max(5, LEGEND_SIZE),
                  loc="upper left" if not r_space else "best", framealpha=0.8)


def main():
    grouped, visible = _load_grouped()

    with plt.rc_context({"text.usetex": True}):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = GridSpec(2, 1, hspace=HSPACE,
                      left=MARGIN_LEFT, right=MARGIN_RIGHT,
                      top=MARGIN_TOP, bottom=MARGIN_BOTTOM)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[1, 0])
        plot_e3_ratio_panel(axL, grouped, visible)
        plot_cuz_panel(axR, grouped, visible)
        add_panel_label(axL, "(a)", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)
        add_panel_label(axR, "(b)", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)

        for ext in ("png", "pdf"):
            path = f"{STEM}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"[export] {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
