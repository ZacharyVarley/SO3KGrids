#!/usr/bin/env python3
"""
figure2_plot.py  —  Interactive Figure 2: axial anisotropy in cubochoric KR grids

Loads pre-computed data from ``figure2_data.json`` (produced by figure2_gen.py)
and provides an interactive matplotlib interface with sliders, toggles, and
save/load functionality.  **No numerical recomputation is performed.**

Panels
------
  (a) E₃/E₃* ratio  vs  N_{S³}   — quality of the optimised grid
  (b) cu_z*        vs  cu_xy    — optimal z semi-edge as a function of x/y semi-edge

Controls (separate scrollable control window)
---------------------------------------------
  Steppers : marker size (+/- 0.5), line width (+/- 0.25)
  Sliders  : font size, title font, cu_xy lo/hi bounds
  Toggles  : per-group visibility (C2, C3, C4, C6, D2, D3, D4, D6)
  Options  : log-x, log-y, fits on (b), fits on (a), mismatches, r-hat line
  File I/O : load / save settings (JSON), export PNG / PDF

Dependencies: matplotlib, numpy, json (stdlib).
"""

import json
import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Union

from matplotlib.markers import MarkerStyle

import numpy as np
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from figure_ui_common import (
    ControlWindow,
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
    load_dataclass_settings,
    push_typography,
    restyle_axes,
    save_dataclass_settings,
    sync_typography,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA_FILE = "figure2_data.json"
DEFAULT_STEM = "figure2_col"
DEFAULT_SETTINGS_JSON = "figure2_col_settings.json"
DPI = IUCR_DPI

# Typography spec — (widget_name, label, settings_attr [, fmt])
_TYPO_ROWS = [
    [("font_size", "Font", "font_size"), ("title_size", "Title", "title_size")],
    [("subtitle_size", "Sub", "subtitle_size"), ("panel_label_size", "Panel", "panel_label_size")],
    [("panel_label_x", "Lbl x", "panel_label_x", "{:.3f}"), ("panel_label_y", "Lbl y", "panel_label_y", "{:.3f}")],
    [("legend_size", "Leg", "legend_size")],
]

# Per-group colours and markers (match figure3/figure4 tab20 ordering)
GROUP_COLORS: Dict[str, str] = {
    "C2": "#1f77b4",
    "C3": "#aec7e8",
    "C4": "#ff7f0e",
    "C6": "#ffbb78",
    "D2": "#2ca02c",
    "D3": "#98df8a",
    "D4": "#d62728",
    "D6": "#ff9896",
}

# C2–C6: thin_diamond, triangle_up, diamond, hexagon. D2–D6: same shapes rotated 90°.
GROUP_MARKERS: Dict[str, Union[str, tuple]] = {
    "C2": "d",  # thin_diamond
    "C3": "^",  # triangle_up
    "C4": "D",  # diamond
    "C6": "h",  # hexagon
    "D2": ("d", 90),  # thin_diamond rotated 90°
    "D3": ("^", 90),  # triangle_up rotated 90°
    "D4": ("D", 45),  # diamond rotated 45°
    "D6": ("h", 90),  # hexagon rotated 90°
}


def _latex_group_name(name: str) -> str:
    """Return LaTeX label for group name (e.g. C2 -> $C_{2}$)."""
    m = re.match(r"^([CD])(\d+)$", name)
    if m:
        return rf"${m.group(1)}_{{{m.group(2)}}}$"
    if name in ("T", "O", "I"):
        return rf"${name}$"
    return name


def _get_marker(gn: str):
    """Return marker for group: string or MarkerStyle (for rotated markers)."""
    m = GROUP_MARKERS.get(gn, "o")
    if isinstance(m, tuple):
        sym, rot = m
        ms = MarkerStyle(sym)
        ms._transform = ms.get_transform().rotate_deg(rot)
        return ms
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Settings dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Settings:
    # Typography — IUCr defaults (legible at single-column width)
    font_size: float = IUCR_FONT["label"]
    title_size: float = IUCR_FONT["title"]
    subtitle_size: float = IUCR_FONT["subtitle"]
    panel_label_size: float = IUCR_FONT["panel_label"]
    panel_label_x: float = 0.02
    panel_label_y: float = 0.98
    legend_size: float = IUCR_FONT["legend"]
    marker_size: float = 4.0
    line_width: float = 1.0

    # cu_xy bounds (affects both plotting AND fits)
    cu_xy_lo: int = 3
    cu_xy_hi: int = 25

    # Group visibility
    show_C2: bool = True
    show_C3: bool = True
    show_C4: bool = True
    show_C6: bool = True
    show_D2: bool = True
    show_D3: bool = True
    show_D4: bool = True
    show_D6: bool = True

    # Axis options
    log_x_ratio: bool = True
    log_y_ratio: bool = False

    # Display
    show_grid: bool = True
    grid_alpha: float = 0.20
    show_minor_tick_labels: bool = False

    # Overlays
    show_fits_cuz: bool = True
    show_fits_logquad: bool = False
    show_fits_ratio: bool = False
    show_mismatches: bool = False
    show_r_cont: bool = False
    show_legend_R: bool = True

    # Figure size — IUCr single-column
    fig_width: float = IUCR_COL_W
    fig_height: float = 5.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get(rec: dict, key: str, fallback: str = ""):
    """Read a field with backward-compatible fallback key."""
    v = rec.get(key)
    if v is None and fallback:
        v = rec.get(fallback)
    return v


def load_data(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Error: data file '{path}' not found.  Run figure2_gen.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def group_results(data: dict) -> Dict[str, List[dict]]:
    """Group result records by group name, sorted by cu_xy."""
    groups: Dict[str, List[dict]] = {}
    for r in data["results"]:
        g = r["group"]
        groups.setdefault(g, []).append(r)
    for g in groups:
        groups[g].sort(key=lambda x: x["cu_xy"])
    return groups


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Panel drawing (shared by interactive UI and IUCR export)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def visible_groups(grouped: dict, s: Settings) -> List[str]:
    return [
        g for g in _GROUP_NAMES
        if getattr(s, f"show_{g}", True) and g in grouped
    ]


def filter_recs_by_cu_xy(recs: List[dict], s: Settings) -> List[dict]:
    lo, hi = s.cu_xy_lo, s.cu_xy_hi
    return [r for r in recs if lo <= r["cu_xy"] <= hi]


def draw_e3_ratio_panel(
    ax,
    grouped: dict,
    s: Settings,
    *,
    use_tex: bool = False,
    panel_tag: str | None = "(a)",
) -> None:
    """E₃/E₃* vs N_{S³}."""
    fs = int(round(s.font_size))
    ms = float(s.marker_size)
    lw = float(s.line_width)
    visible = visible_groups(grouped, s)

    ax.cla()
    for gn in visible:
        recs = filter_recs_by_cu_xy(grouped[gn], s)
        if not recs:
            continue
        x = np.array([r["N_eff"] for r in recs], dtype=float)
        y = np.array([r["E3_ratio_best"] for r in recs], dtype=float)
        col = GROUP_COLORS.get(gn, "gray")
        mrk = _get_marker(gn)
        ax.scatter(
            x, y, marker=mrk, color=col, s=ms**2,
            edgecolors="0.2", linewidths=0.3,
            label=_latex_group_name(gn), zorder=5,
        )
        if s.show_fits_ratio and len(x) >= 2:
            log_x = np.log(x)
            coeffs = np.polyfit(log_x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            ys = np.polyval(coeffs, np.log(xs))
            ax.plot(xs, ys, color=col, linewidth=lw * 0.7, linestyle="--", alpha=0.6)

    ax.set_xscale("log" if s.log_x_ratio else "linear")
    ax.set_yscale("log" if s.log_y_ratio else "linear")
    ax.set_xlabel(r"$N_{S^3}$", fontsize=fs)
    ax.set_ylabel(r"$E_3/E_3^*$", fontsize=fs)
    if panel_tag:
        add_panel_label(
            ax, panel_tag, x=s.panel_label_x, y=s.panel_label_y,
            fontsize=s.panel_label_size, use_tex=use_tex,
        )
    ax.tick_params(labelsize=max(6, fs - 1))
    ax.minorticks_on()
    if not s.show_minor_tick_labels:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=max(5, s.legend_size), loc="best", framealpha=0.8)
    ax.grid(s.show_grid, alpha=s.grid_alpha)


def draw_cuz_panel(
    ax,
    grouped: dict,
    s: Settings,
    *,
    use_tex: bool = False,
    panel_tag: str | None = "(b)",
) -> None:
    """cu_z* or r̂ vs cu_xy."""
    fs = int(round(s.font_size))
    ms = float(s.marker_size)
    lw = float(s.line_width)
    visible = visible_groups(grouped, s)
    r_space = s.show_fits_logquad

    ax.cla()
    slope_dict: Dict[str, float] = {}
    logquad_dict: Dict[str, np.ndarray] = {}
    for gn in visible:
        recs = filter_recs_by_cu_xy(grouped[gn], s)
        if not recs:
            continue
        x = np.array([r["cu_xy"] for r in recs], dtype=float)
        cuz = np.array([Figure2Interactive._read_cu_z(r) for r in recs], dtype=float)
        if s.show_fits_cuz and len(x) >= 2:
            slope_dict[gn] = float(np.sum(x * cuz) / np.sum(x * x))
        if s.show_fits_logquad and len(x) >= 3:
            rhat = np.array(
                [r.get("r_cont", Figure2Interactive._read_cu_z(r) / r["cu_xy"]) for r in recs],
                dtype=float,
            )
            xl = np.log(x)
            A = np.vstack([np.ones_like(xl), xl, xl**2]).T
            beta, *_ = np.linalg.lstsq(A, rhat, rcond=None)
            logquad_dict[gn] = beta

    for gn in visible:
        recs = filter_recs_by_cu_xy(grouped[gn], s)
        if not recs:
            continue
        x = np.array([r["cu_xy"] for r in recs], dtype=float)
        cuz = np.array([Figure2Interactive._read_cu_z(r) for r in recs], dtype=float)
        col = GROUP_COLORS.get(gn, "gray")
        mrk = _get_marker(gn)
        if r_space:
            y_scatter = np.array(
                [r.get("r_cont", Figure2Interactive._read_cu_z(r) / r["cu_xy"]) for r in recs],
                dtype=float,
            )
        else:
            y_scatter = cuz
        ax.scatter(
            x, y_scatter, marker=mrk, color=col, s=ms**2,
            edgecolors="0.2", linewidths=0.3, zorder=5,
        )
        x_fit = np.linspace(float(x.min()), float(x.max()), 200)
        if s.show_r_cont and not r_space:
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
                if s.show_mismatches:
                    cuz_pred = np.round(slope * x).astype(int)
                    mismatch = cuz_pred != cuz.astype(int)
                    if np.any(mismatch):
                        ax.scatter(
                            x[mismatch], cuz[mismatch], s=ms**2 * 6,
                            facecolors="none", edgecolors=col, linewidths=1.5, zorder=10,
                        )
        if gn in logquad_dict:
            beta = logquad_dict[gn]
            lxf = np.log(x_fit)
            rhat_fit = beta[0] + beta[1] * lxf + beta[2] * lxf**2
            if r_space:
                ax.plot(x_fit, rhat_fit, color=col, linewidth=lw * 0.7, linestyle="-", alpha=0.7)
                if s.show_mismatches:
                    lx_d = np.log(x)
                    rhat_at_data = beta[0] + beta[1] * lx_d + beta[2] * lx_d**2
                    cuz_fit = np.round(rhat_at_data * x).astype(int)
                    cuz_obs = np.round(y_scatter * x).astype(int)
                    mm_mask = cuz_fit != cuz_obs
                    if np.any(mm_mask):
                        ax.scatter(
                            x[mm_mask], y_scatter[mm_mask], s=ms**2 * 6,
                            facecolors="none", edgecolors=col, linewidths=1.5, zorder=10,
                        )
            else:
                ax.plot(x_fit, rhat_fit * x_fit, color=col, linewidth=lw * 0.7, linestyle="-", alpha=0.7)

    handles, labels = [], []
    for gn in visible:
        recs = filter_recs_by_cu_xy(grouped[gn], s)
        if not recs:
            continue
        col = GROUP_COLORS.get(gn, "gray")
        mrk = _get_marker(gn)
        h = Line2D(
            [], [], color=col, marker=mrk, linestyle="None",
            markersize=ms, markeredgewidth=0.3, markeredgecolor="0.2",
        )
        lbl = _latex_group_name(gn)
        if gn in slope_dict:
            lbl = rf"{lbl}  $r^*\!={slope_dict[gn]:.3f}$"
        handles.append(h)
        labels.append(lbl)

    ax.set_xlabel(r"$\mathrm{cu}_{xy}$", fontsize=fs)
    ax.set_ylabel(r"$\hat{r}$" if r_space else r"$\mathrm{cu}_z^{\,*}$", fontsize=fs)
    if panel_tag:
        add_panel_label(
            ax, panel_tag, x=s.panel_label_x, y=s.panel_label_y,
            fontsize=s.panel_label_size, use_tex=use_tex,
        )
    ax.tick_params(labelsize=max(6, fs - 1))
    ax.minorticks_on()
    if not s.show_minor_tick_labels:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(s.show_grid, alpha=s.grid_alpha)
    if handles and s.show_legend_R:
        ax.legend(
            handles, labels, fontsize=max(5, s.legend_size),
            loc="upper left" if not r_space else "best", framealpha=0.8,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Interactive Figure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Canonical ordering
_GROUP_NAMES = ["C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6"]
_OPT_LABELS = [
    "Log X",
    "Log Y",
    "Linear fit",
    "Log-quad fit",
    "Fits (ratio)",
    "Mismatches",
    "r\u0302 line",
    "Legend (b)",
]
_OPT_ATTRS = [
    "log_x_ratio",
    "log_y_ratio",
    "show_fits_cuz",
    "show_fits_logquad",
    "show_fits_ratio",
    "show_mismatches",
    "show_r_cont",
    "show_legend_R",
]


class Figure2Interactive:
    """Matplotlib-based interactive viewer for figure2 data."""

    # ── construction ─────────────────────────────────────────────────
    def __init__(self, data: dict, settings: Settings):
        self.data = data
        self.grouped = group_results(data)
        self.s = settings
        self._loading = False
        self._exporting_tex = False

        # Determine data range for cu_xy bounds
        all_cu_xy = sorted(set(r["cu_xy"] for r in data["results"]))
        self._cu_xy_data_min = min(all_cu_xy) if all_cu_xy else 3
        self._cu_xy_data_max = max(all_cu_xy) if all_cu_xy else 25
        # Clamp settings to data range
        self.s.cu_xy_lo = max(self.s.cu_xy_lo, self._cu_xy_data_min)
        self.s.cu_xy_hi = min(self.s.cu_xy_hi, self._cu_xy_data_max)

        # Figure layout: [plot_L | plot_R]
        apply_plot_rcparams()
        self.fig = plt.figure(figsize=(s.fig_width, s.fig_height), dpi=110)
        self.fig.canvas.manager.set_window_title("Figure 2")
        gs = GridSpec(
            2, 1,
            hspace=IUCR_MARGINS["hspace"],
            left=IUCR_MARGINS["left"],
            right=IUCR_MARGINS["right"],
            top=IUCR_MARGINS["top"],
            bottom=IUCR_MARGINS["bottom"],
        )
        self.axL = self.fig.add_subplot(gs[0, 0])
        self.axR = self.fig.add_subplot(gs[1, 0])

        self.ctrl = plt.figure(figsize=(11.0, 7.0), dpi=110)
        self.ctrl.canvas.manager.set_window_title("Figure 2 Controls")

        # Widget bookkeeping
        self.widgets: dict = {}

        self._build_ui()
        self._bind_callbacks()

        # First paint
        self.redraw()

    # ── UI construction ──────────────────────────────────────────────
    def _build_ui(self):
        s = self.s

        cw = ControlWindow(self.ctrl, left=0.05, right=0.92, top=0.96, col_gap=0.04, col1_frac=0.42)
        self._cw = cw
        self.widgets = cw.widgets

        # Column 1: shared controls (text entries — fire on Enter only)
        build_typography_section(cw, 1, s, _TYPO_ROWS)

        cw.section(1, "Style")
        cw.slider(1, "marker_size", "Marker size", 1.0, 20.0, s.marker_size, step=0.5)
        cw.slider(1, "line_width", "Line width", 0.25, 5.0, s.line_width, step=0.25)

        cw.section(1, "Display")
        cw.checkbox_grid(
            1,
            "display_toggles",
            [("show_grid", "Grid", s.show_grid)],
            n_cols=1,
            row_h=0.038,
            label_size=7.0,
        )
        cw.slider(1, "grid_alpha", "Grid opacity", 0.05, 0.5, s.grid_alpha)

        cw.section(1, "Axes")
        cw.checkbox_grid(
            1,
            "axis_toggles",
            [
                ("log_x_ratio", "Log X", s.log_x_ratio),
                ("log_y_ratio", "Log Y", s.log_y_ratio),
                ("show_legend_R", "Legend (b)", s.show_legend_R),
            ],
            n_cols=1,
            row_h=0.038,
            label_size=7.0,
        )

        cw.section(1, "File I/O")
        cw.textbox(1, "settings_path", "", DEFAULT_SETTINGS_JSON)
        cw.button_row(1, [("load", "Load"), ("save", "Save")])
        cw.textbox(1, "stem", "", DEFAULT_STEM)
        cw.button_row(1, [("save_png", "PNG"), ("save_pdf", "PDF")])

        # Column 2: figure-specific controls
        cw.section(2, "Data Range")
        cw.slider(
            2,
            "cu_xy_lo",
            "cu_xy lo",
            self._cu_xy_data_min,
            self._cu_xy_data_max,
            float(s.cu_xy_lo),
            step=1,
        )
        cw.slider(
            2,
            "cu_xy_hi",
            "cu_xy hi",
            self._cu_xy_data_min,
            self._cu_xy_data_max,
            float(s.cu_xy_hi),
            step=1,
        )

        cw.section(2, "Groups")
        group_items = [(g, g, getattr(s, f"show_{g}")) for g in _GROUP_NAMES]
        cw.checkbox_grid(
            2,
            "groups",
            group_items,
            n_cols=4,
            row_h=0.040,
            label_size=7.0,
            color_map=GROUP_COLORS,
        )

        cw.section(2, "Fits / Overlays")
        fit_items = [
            ("show_fits_cuz", "Linear fit", s.show_fits_cuz),
            ("show_fits_logquad", "Log-quad fit", s.show_fits_logquad),
            ("show_fits_ratio", "Fits (ratio)", s.show_fits_ratio),
            ("show_mismatches", "Mismatches", s.show_mismatches),
            ("show_r_cont", "r-hat line", s.show_r_cont),
        ]
        cw.checkbox_grid(
            2,
            "fit_opts",
            fit_items,
            n_cols=2,
            row_h=0.040,
            label_size=6.8,
        )

        cw.connect_scroll()

    # ── callbacks ────────────────────────────────────────────────────
    def _bind_callbacks(self):
        w = self.widgets
        # Text entries (number_row) — fire on Enter only (lightweight restyle)
        cw = self._cw
        bind_typography(cw, _TYPO_ROWS, self._on_restyle)
        # Visual-only sliders (no data recompute)
        for k in ("marker_size", "line_width", "grid_alpha"):
            w[k].on_changed(lambda _: self._on_visual_change())
        # Display toggles that only affect rendering, not data
        for chk in w["display_toggles"]:
            chk.on_clicked(lambda _: self._on_visual_change())
        for chk in w["axis_toggles"]:
            chk.on_clicked(lambda _: self._on_visual_change())
        # Data range — needs full redraw
        for k in ("cu_xy_lo", "cu_xy_hi"):
            w[k].on_changed(lambda _: self._on_data_change())
        # Group/fit toggles — needs full redraw
        for chk in w["groups"]:
            chk.on_clicked(lambda _: self._on_data_change())
        for chk in w["fit_opts"]:
            chk.on_clicked(lambda _: self._on_data_change())

        w["load"].on_clicked(lambda _: self.load_settings(w["settings_path"].text))
        w["save"].on_clicked(lambda _: self.save_settings(w["settings_path"].text))
        w["save_png"].on_clicked(lambda _: self.save_figure(w["stem"].text, "png"))
        w["save_pdf"].on_clicked(lambda _: self.save_figure(w["stem"].text, "pdf"))

    def _on_data_change(self):
        """Full redraw: data range, group visibility, or fit toggles changed."""
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self.redraw()

    def _on_visual_change(self):
        """Fast path: update stored artists for marker/line/grid/scale changes."""
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self._apply_visual()

    def _on_restyle(self):
        """Lightweight: update typography without recomputing data."""
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self._restyle()

    def _restyle(self):
        restyle_axes(
            self.fig,
            [(self.axL, "(a)"), (self.axR, "(b)")],
            self.s,
            use_tex=self._exporting_tex,
        )

    def _apply_visual(self):
        """Update marker sizes, line widths, grid, axis scales, legend on stored artists."""
        s = self.s
        ms = float(s.marker_size)
        lw = float(s.line_width)

        # Update scatter sizes and line widths on all axes
        for ax in (self.axL, self.axR):
            for coll in ax.collections:
                coll.set_sizes([ms**2])
            for line in ax.lines:
                line.set_linewidth(lw * 0.7)

            # Axis scales
            ax.minorticks_on()
            if not s.show_minor_tick_labels:
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())

            # Grid
            if s.show_grid:
                ax.grid(True, alpha=s.grid_alpha)
            else:
                ax.grid(False)

        # Panel (a) axis scales
        self.axL.set_xscale("log" if s.log_x_ratio else "linear")
        self.axL.set_yscale("log" if s.log_y_ratio else "linear")

        # Panel (a) legend
        leg_L = self.axL.get_legend()
        if leg_L is not None:
            for h in leg_L.legend_handles:
                if hasattr(h, 'set_markersize'):
                    h.set_markersize(ms)

        # Panel (b) legend visibility
        leg_R = self.axR.get_legend()
        if leg_R is not None:
            leg_R.set_visible(bool(s.show_legend_R))
            for h in leg_R.legend_handles:
                if hasattr(h, 'set_markersize'):
                    h.set_markersize(ms)

        self._restyle()

    # ── settings <-> widgets ─────────────────────────────────────────
    def _sync_widgets_to_settings(self):
        w, s = self.widgets, self.s
        cw = self._cw
        sync_typography(cw, s, _TYPO_ROWS)
        s.marker_size = float(w["marker_size"].val)
        s.line_width = float(w["line_width"].val)
        s.grid_alpha = float(w["grid_alpha"].val)
        s.show_grid = bool(w["display_toggles"][0].get_status()[0])
        s.log_x_ratio = bool(w["axis_toggles"][0].get_status()[0])
        s.log_y_ratio = bool(w["axis_toggles"][1].get_status()[0])
        s.show_legend_R = bool(w["axis_toggles"][2].get_status()[0])
        s.show_minor_tick_labels = False
        s.cu_xy_lo = int(round(w["cu_xy_lo"].val))
        s.cu_xy_hi = int(round(w["cu_xy_hi"].val))
        # Enforce lo <= hi
        if s.cu_xy_lo > s.cu_xy_hi:
            s.cu_xy_lo, s.cu_xy_hi = s.cu_xy_hi, s.cu_xy_lo

        for i, gn in enumerate(_GROUP_NAMES):
            setattr(s, f"show_{gn}", bool(w["groups"][i].get_status()[0]))
        fit_attrs = [
            "show_fits_cuz",
            "show_fits_logquad",
            "show_fits_ratio",
            "show_mismatches",
            "show_r_cont",
        ]
        for i, attr in enumerate(fit_attrs):
            if i < len(w["fit_opts"]):
                setattr(s, attr, bool(w["fit_opts"][i].get_status()[0]))

    def _push_settings_to_widgets(self):
        w, s = self.widgets, self.s
        cw = self._cw
        push_typography(cw, s, _TYPO_ROWS)
        w["marker_size"].set_val(s.marker_size)
        w["line_width"].set_val(s.line_width)
        w["grid_alpha"].set_val(s.grid_alpha)
        if bool(w["display_toggles"][0].get_status()[0]) != s.show_grid:
            w["display_toggles"][0].set_active(0)
        axis_desired = [s.log_x_ratio, s.log_y_ratio, s.show_legend_R]
        for i, desired in enumerate(axis_desired):
            if bool(w["axis_toggles"][i].get_status()[0]) != bool(desired):
                w["axis_toggles"][i].set_active(0)
        # cu_xy bounds
        w["cu_xy_lo"].set_val(float(s.cu_xy_lo))
        w["cu_xy_hi"].set_val(float(s.cu_xy_hi))

        for i, gn in enumerate(_GROUP_NAMES):
            desired = getattr(s, f"show_{gn}")
            if bool(w["groups"][i].get_status()[0]) != desired:
                w["groups"][i].set_active(0)
        fit_attrs = [
            "show_fits_cuz",
            "show_fits_logquad",
            "show_fits_ratio",
            "show_mismatches",
            "show_r_cont",
        ]
        for i, attr in enumerate(fit_attrs):
            if i < len(w["fit_opts"]):
                desired = getattr(s, attr)
                if bool(w["fit_opts"][i].get_status()[0]) != desired:
                    w["fit_opts"][i].set_active(0)

    # ── settings I/O ─────────────────────────────────────────────────
    def save_settings(self, path: str):
        self._sync_widgets_to_settings()
        fw, fh = self.fig.get_size_inches()
        self.s.fig_width = float(fw)
        self.s.fig_height = float(fh)
        path = (path or "").strip() or DEFAULT_SETTINGS_JSON
        save_dataclass_settings(path, self.s)
        print(f"Saved settings -> {path}")

    def load_settings(self, path: str):
        path = (path or "").strip() or DEFAULT_SETTINGS_JSON
        if not load_dataclass_settings(path, self.s):
            print(f"Settings file '{path}' not found.")
            return
        # Clamp cu_xy bounds to data range
        self.s.cu_xy_lo = max(self.s.cu_xy_lo, self._cu_xy_data_min)
        self.s.cu_xy_hi = min(self.s.cu_xy_hi, self._cu_xy_data_max)
        self.fig.set_size_inches(self.s.fig_width, self.s.fig_height, forward=True)
        self._loading = True
        self._push_settings_to_widgets()
        self._loading = False
        self.redraw()

    # ── save figure ──────────────────────────────────────────────────
    def save_figure(self, stem: str, ext: str):
        self._sync_widgets_to_settings()
        stem = (stem or "").strip() or DEFAULT_STEM
        ext = (ext or "").strip().lower()
        out_path = f"{stem}.{ext}"
        save_dataclass_settings(f"{stem}_settings.json", self.s)
        self._exporting_tex = True
        try:
            export_figure_with_tex(
                self.fig,
                out_path,
                redraw_callback=lambda: self.redraw(),
                dpi=DPI,
            )
            print(f"Saved {out_path} (TeX export)")
        except Exception as exc:
            print(f"Export failed for {out_path}: {exc}")
            raise
        finally:
            self._exporting_tex = False
            self.redraw()

    # ── helpers ──────────────────────────────────────────────────────
    def _visible_groups(self) -> List[str]:
        return [
            g
            for g in _GROUP_NAMES
            if getattr(self.s, f"show_{g}", True) and g in self.grouped
        ]

    def _filter_recs(self, recs: List[dict]) -> List[dict]:
        """Filter records by cu_xy bounds."""
        lo, hi = self.s.cu_xy_lo, self.s.cu_xy_hi
        return [r for r in recs if lo <= r["cu_xy"] <= hi]

    @staticmethod
    def _read_cu_z(rec: dict):
        """Read cu_z_best with backward compat for old hz_best key."""
        v = rec.get("cu_z_best")
        if v is None:
            v = rec.get("hz_best")
        return v

    # ── main redraw ──────────────────────────────────────────────────
    def redraw(self):
        s = self.s
        draw_e3_ratio_panel(
            self.axL, self.grouped, s,
            use_tex=self._exporting_tex, panel_tag="(a)",
        )
        draw_cuz_panel(
            self.axR, self.grouped, s,
            use_tex=self._exporting_tex, panel_tag="(b)",
        )
        self.fig.canvas.draw_idle()



def main():
    parser = argparse.ArgumentParser(
        description="Figure 2 column layout (interactive or headless export)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export PNG and PDF using current settings, then exit",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem for --export",
    )
    parser.add_argument(
        "--settings",
        default=DEFAULT_SETTINGS_JSON,
        help="Settings JSON path",
    )
    args = parser.parse_args()

    data = load_data(DATA_FILE)
    meta = data.get("metadata", {})

    print(f"Loaded {len(data['results'])} records from '{DATA_FILE}'")
    print(f"  Version : {meta.get('version', '?')}")
    print(f"  Groups  : {meta.get('groups', '?')}")
    print(f"  cu_xy   : {meta.get('cu_xy_range', '?')}")
    print()

    s = Settings()

    # Try loading saved settings
    settings_path = (args.settings or "").strip() or DEFAULT_SETTINGS_JSON
    if os.path.exists(settings_path):
        try:
            with open(settings_path) as f:
                d = json.load(f)
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            print(f"  Loaded settings from '{settings_path}'")
        except Exception as e:
            print(f"  Warning: could not load settings: {e}")

    app = Figure2Interactive(data, s)
    if args.export:
        stem = (args.stem or "").strip() or DEFAULT_STEM
        app.save_figure(stem, "png")
        app.save_figure(stem, "pdf")
        plt.close("all")
        return

    plt.show()


if __name__ == "__main__":
    main()
