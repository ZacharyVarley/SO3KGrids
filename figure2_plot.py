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
import os
import re
import sys
from dataclasses import asdict, dataclass, field
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
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA_FILE = "figure2_data.json"
DEFAULT_STEM = "figure2"
DEFAULT_SETTINGS_JSON = "figure2_settings.json"
DPI = 300

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
    # Style
    font_size: float = 10.0
    title_size: float = 12.0
    marker_size: float = 5.0
    line_width: float = 1.5

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

    # Display (like figure5)
    show_grid: bool = True
    grid_alpha: float = 0.20
    show_minor_tick_labels: bool = False

    # Overlays
    show_fits_cuz: bool = True  # linear fit cu_z = slope·cu_xy through origin
    show_fits_logquad: bool = False  # log-quad: r̂ = β₀+β₁·ln(cu_xy)+β₂·ln²(cu_xy)
    show_fits_ratio: bool = False  # trend line on ratio panel
    show_mismatches: bool = False  # circle where fit ≠ cu_z_best
    show_r_cont: bool = False  # dashed line of continuous r̂·cu_xy
    show_legend_R: bool = True  # legend on second plot (b)

    # Figure size (inches)
    fig_width: float = 15.0
    fig_height: float = 6.5


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
        self.fig = plt.figure(figsize=(15.0, 6.5), dpi=110)
        self.fig.canvas.manager.set_window_title("Figure 2")
        self.fig.set_size_inches(self.s.fig_width, self.s.fig_height, forward=True)
        gs = GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2, left=0.06, right=0.98, top=0.92, bottom=0.10)
        self.axL = self.fig.add_subplot(gs[0, 0])
        self.axR = self.fig.add_subplot(gs[0, 1])

        self.ctrl = plt.figure(figsize=(4.8, 9.2), dpi=110)
        self.ctrl.canvas.manager.set_window_title("Figure 2 Controls")

        # Widget bookkeeping
        self.widgets: dict = {}
        self._ctrl_axes: list = []
        self._ctrl_rects: list = []

        self._build_ui()
        self._bind_callbacks()
        self._setup_scroll()

        # First paint
        self.redraw()

    # ── UI construction ──────────────────────────────────────────────
    def _build_ui(self):
        L = 0.08
        W = 0.84
        y = 0.96  # cursor starts near top
        SH = 0.018  # slider height
        DY = 0.030  # vertical step between sliders
        BH = 0.025  # button / textbox height
        GAP = 0.012  # gap before section header
        HDR_H = 0.014  # header height

        def _reg(ax, rect):
            self._ctrl_axes.append(ax)
            self._ctrl_rects.append((ax, rect))

        def _section(title):
            nonlocal y
            y -= GAP
            rect = (L, y - HDR_H, W, HDR_H)
            ax = self.ctrl.add_axes(list(rect))
            ax.set_axis_off()
            ax.text(
                0.0,
                0.1,
                title.upper(),
                fontsize=6.5,
                fontweight="bold",
                color="0.45",
                transform=ax.transAxes,
                va="bottom",
                family="sans-serif",
            )
            _reg(ax, rect)
            y -= HDR_H + 0.003

        def _slider(name, label, lo, hi, val, step=None):
            nonlocal y
            rect = (L, y, W, SH)
            ax = self.ctrl.add_axes(list(rect))
            _reg(ax, rect)
            ax.set_facecolor("0.97")
            sl = Slider(ax, label, lo, hi, valinit=val, valstep=step)
            for t in (sl.label, sl.valtext):
                t.set_fontsize(7)
                t.set_color("0.3")
            self.widgets[name] = sl
            y -= DY
            return sl

        def _stepper(name, label, val, step, lo, hi, fmt="{:.1f}"):
            """Label + value display + [-] [+] buttons."""
            nonlocal y
            lw = W * 0.38  # label width
            vw = W * 0.24  # value display width
            bw = W * 0.19  # each button width
            h = BH

            # Label
            rect = (L, y, lw, h)
            ax_l = self.ctrl.add_axes(list(rect))
            ax_l.set_axis_off()
            ax_l.text(
                0.95,
                0.5,
                label,
                fontsize=7,
                color="0.3",
                ha="right",
                va="center",
                transform=ax_l.transAxes,
            )
            _reg(ax_l, rect)

            # Value display
            rect = (L + lw, y, vw, h)
            ax_v = self.ctrl.add_axes(list(rect))
            ax_v.set_axis_off()
            vtxt = ax_v.text(
                0.5,
                0.5,
                fmt.format(val),
                fontsize=7,
                color="0.2",
                ha="center",
                va="center",
                transform=ax_v.transAxes,
                family="monospace",
            )
            _reg(ax_v, rect)

            # Minus button
            rect = (L + lw + vw, y, bw, h)
            ax_m = self.ctrl.add_axes(list(rect))
            _reg(ax_m, rect)
            btn_m = Button(ax_m, "\u2212", color="0.93", hovercolor="0.86")
            btn_m.label.set_fontsize(9)

            # Plus button
            rect = (L + lw + vw + bw, y, bw, h)
            ax_p = self.ctrl.add_axes(list(rect))
            _reg(ax_p, rect)
            btn_p = Button(ax_p, "+", color="0.93", hovercolor="0.86")
            btn_p.label.set_fontsize(9)

            info = {
                "val": val,
                "step": step,
                "lo": lo,
                "hi": hi,
                "text": vtxt,
                "fmt": fmt,
                "btn_m": btn_m,
                "btn_p": btn_p,  # prevent GC
            }
            self.widgets[name] = info

            def _make_cb(delta):
                def cb(_event):
                    v = info["val"] + delta
                    v = max(info["lo"], min(info["hi"], v))
                    # Snap to step grid
                    v = round(v / step) * step
                    v = max(info["lo"], min(info["hi"], v))
                    info["val"] = v
                    info["text"].set_text(info["fmt"].format(v))
                    self._on_change()

                return cb

            btn_m.on_clicked(_make_cb(-step))
            btn_p.on_clicked(_make_cb(step))
            y -= h + 0.006

        def _button_row(pairs):
            nonlocal y
            n = len(pairs)
            pad = 0.004
            bw = (W - pad * (n - 1)) / n
            for i, (bname, blabel) in enumerate(pairs):
                bx = L + i * (bw + pad)
                rect = (bx, y, bw, BH)
                ax = self.ctrl.add_axes(list(rect))
                _reg(ax, rect)
                b = Button(ax, blabel, color="0.93", hovercolor="0.86")
                b.label.set_fontsize(7)
                self.widgets[bname] = b
            y -= BH + 0.006

        def _textbox(name, label, initial):
            nonlocal y
            rect = (L, y, W, BH)
            ax = self.ctrl.add_axes(list(rect))
            _reg(ax, rect)
            tb = TextBox(ax, label, initial=initial)
            tb.label.set_fontsize(7)
            self.widgets[name] = tb
            y -= BH + 0.005

        def _check_row(items):
            """items: list of (name, label, initial)."""
            nonlocal y
            ch = 0.016
            n = len(items)
            pad = 0.008
            cw = (W - pad * (n - 1)) / n
            for i, (name, label, initial) in enumerate(items):
                rect = (L + i * (cw + pad), y - ch, cw - 0.002, ch)
                ax = self.ctrl.add_axes(list(rect))
                _reg(ax, rect)
                chk = CheckButtons(ax, [label], [initial])
                for t in chk.labels:
                    t.set_fontsize(6.5)
                self.widgets[name] = chk
            y -= ch + 0.006

        s = self.s

        # ── STYLE ──
        _section("Style")
        _slider("font_size", "Font", 6, 18, s.font_size)
        _slider("title_size", "Title", 6, 22, s.title_size)
        _stepper(
            "marker_size",
            "Marker",
            s.marker_size,
            step=0.5,
            lo=1.0,
            hi=20.0,
            fmt="{:.1f}",
        )
        _stepper(
            "line_width",
            "Line W",
            s.line_width,
            step=0.25,
            lo=0.25,
            hi=5.0,
            fmt="{:.2f}",
        )

        # ── DISPLAY ──
        _section("Display")
        _check_row(
            [
                ("show_grid", "Grid", s.show_grid),
                ("show_minor_tick_labels", "Minor ticks", s.show_minor_tick_labels),
            ]
        )
        _slider("grid_alpha", "Grid opacity", 0.05, 0.5, s.grid_alpha)

        # ── DATA RANGE ──
        _section("Data Range")
        _slider(
            "cu_xy_lo",
            "cu_xy lo",
            self._cu_xy_data_min,
            self._cu_xy_data_max,
            float(s.cu_xy_lo),
            step=1,
        )
        _slider(
            "cu_xy_hi",
            "cu_xy hi",
            self._cu_xy_data_min,
            self._cu_xy_data_max,
            float(s.cu_xy_hi),
            step=1,
        )

        # ── GROUP TOGGLES ──
        _section("Groups")
        COLS, ROWS = 4, 2
        gh = 0.042
        cw = W / COLS
        group_states = [getattr(s, f"show_{g}") for g in _GROUP_NAMES]
        g0 = y - gh * ROWS
        self.widgets["groups"] = []
        for idx, gn in enumerate(_GROUP_NAMES):
            r, c = idx // COLS, idx % COLS
            bx = L + c * cw
            by = g0 + (ROWS - 1 - r) * gh
            rect = (bx, by, cw - 0.002, gh - 0.003)
            ax_c = self.ctrl.add_axes(list(rect))
            _reg(ax_c, rect)
            chk = CheckButtons(
                ax_c,
                [gn],
                [group_states[idx] if idx < len(group_states) else True],
            )
            for txt in chk.labels:
                txt.set_fontsize(7)
                txt.set_color(GROUP_COLORS.get(gn, "black"))
            self.widgets["groups"].append(chk)
        y = g0 - 0.006

        # ── OPTIONS ──
        _section("Options")
        opt_states = [getattr(s, a) for a in _OPT_ATTRS]
        COLS_O = 3
        ROWS_O = -(-len(_OPT_LABELS) // COLS_O)  # ceil division
        gho = 0.042
        cwo = W / COLS_O
        g0o = y - gho * ROWS_O
        self.widgets["options"] = []
        for idx, lbl in enumerate(_OPT_LABELS):
            r, c = idx // COLS_O, idx % COLS_O
            bx = L + c * cwo
            by = g0o + (ROWS_O - 1 - r) * gho
            rect = (bx, by, cwo - 0.002, gho - 0.003)
            ax_c = self.ctrl.add_axes(list(rect))
            _reg(ax_c, rect)
            chk = CheckButtons(
                ax_c,
                [lbl],
                [opt_states[idx] if idx < len(opt_states) else False],
            )
            for txt in chk.labels:
                txt.set_fontsize(6.5)
            self.widgets["options"].append(chk)
        y = g0o - 0.006

        # ── FILE I/O ──
        _section("File I/O")
        _textbox("settings_path", "", DEFAULT_SETTINGS_JSON)
        _button_row([("load", "Load"), ("save", "Save")])
        _textbox("stem", "", DEFAULT_STEM)
        _button_row([("save_png", "PNG"), ("save_pdf", "PDF")])

    # ── callbacks ────────────────────────────────────────────────────
    def _bind_callbacks(self):
        w = self.widgets
        for k in ("font_size", "title_size", "cu_xy_lo", "cu_xy_hi", "grid_alpha"):
            w[k].on_changed(lambda _: self._on_change())
        for k in ("show_grid", "show_minor_tick_labels"):
            w[k].on_clicked(lambda _: self._on_change())
        # stepper callbacks are bound inside _stepper()
        for chk in w["groups"]:
            chk.on_clicked(lambda _: self._on_change())
        for chk in w["options"]:
            chk.on_clicked(lambda _: self._on_change())

        w["load"].on_clicked(lambda _: self.load_settings(w["settings_path"].text))
        w["save"].on_clicked(lambda _: self.save_settings(w["settings_path"].text))
        w["save_png"].on_clicked(lambda _: self.save_figure(w["stem"].text, "png"))
        w["save_pdf"].on_clicked(lambda _: self.save_figure(w["stem"].text, "pdf"))

    def _on_change(self):
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self.redraw()

    # ── scroll ───────────────────────────────────────────────────────
    def _setup_scroll(self):
        self._scroll_off = 0.0
        self._scroll_top = 0.98
        if self._ctrl_rects:
            self._scroll_max = max(
                0.0,
                self._scroll_top - min(r[1] for _, r in self._ctrl_rects),
            )
        else:
            self._scroll_max = 0.0

        def on_scroll(evt):
            if evt.inaxes not in self._ctrl_axes or evt.button not in ("up", "down"):
                return
            step = 0.08
            if evt.button == "up":
                self._scroll_off = max(0.0, self._scroll_off - step)
            else:
                self._scroll_off = min(self._scroll_max, self._scroll_off + step)
            self._apply_scroll()

        self.ctrl.canvas.mpl_connect("scroll_event", on_scroll)
        self._apply_scroll()

    def _apply_scroll(self):
        top = self._scroll_top
        for ax, (l, b, w, h) in self._ctrl_rects:
            nb = b + self._scroll_off
            ax.set_position([l, nb, w, h])
            ax.set_visible(0 < nb + h and nb < top)
        self.ctrl.canvas.draw_idle()

    # ── settings <-> widgets ─────────────────────────────────────────
    def _sync_widgets_to_settings(self):
        w, s = self.widgets, self.s
        s.font_size = float(w["font_size"].val)
        s.title_size = float(w["title_size"].val)
        s.marker_size = float(w["marker_size"]["val"])
        s.line_width = float(w["line_width"]["val"])
        s.grid_alpha = float(w["grid_alpha"].val)
        s.show_grid = bool(w["show_grid"].get_status()[0])
        s.show_minor_tick_labels = bool(w["show_minor_tick_labels"].get_status()[0])
        s.cu_xy_lo = int(round(w["cu_xy_lo"].val))
        s.cu_xy_hi = int(round(w["cu_xy_hi"].val))
        # Enforce lo <= hi
        if s.cu_xy_lo > s.cu_xy_hi:
            s.cu_xy_lo, s.cu_xy_hi = s.cu_xy_hi, s.cu_xy_lo

        for i, gn in enumerate(_GROUP_NAMES):
            setattr(s, f"show_{gn}", bool(w["groups"][i].get_status()[0]))
        for i, attr in enumerate(_OPT_ATTRS):
            if i < len(w["options"]):
                setattr(s, attr, bool(w["options"][i].get_status()[0]))

    def _push_settings_to_widgets(self):
        w, s = self.widgets, self.s
        w["font_size"].set_val(s.font_size)
        w["title_size"].set_val(s.title_size)
        w["grid_alpha"].set_val(s.grid_alpha)
        # Steppers
        for name in ("marker_size", "line_width"):
            info = w[name]
            val = getattr(s, name)
            info["val"] = val
            info["text"].set_text(info["fmt"].format(val))
        # Display checkboxes
        if bool(w["show_grid"].get_status()[0]) != s.show_grid:
            w["show_grid"].set_active(0)
        if (
            bool(w["show_minor_tick_labels"].get_status()[0])
            != s.show_minor_tick_labels
        ):
            w["show_minor_tick_labels"].set_active(0)
        # cu_xy bounds
        w["cu_xy_lo"].set_val(float(s.cu_xy_lo))
        w["cu_xy_hi"].set_val(float(s.cu_xy_hi))

        for i, gn in enumerate(_GROUP_NAMES):
            desired = getattr(s, f"show_{gn}")
            if bool(w["groups"][i].get_status()[0]) != desired:
                w["groups"][i].set_active(0)
        for i, attr in enumerate(_OPT_ATTRS):
            if i < len(w["options"]):
                desired = getattr(s, attr)
                if bool(w["options"][i].get_status()[0]) != desired:
                    w["options"][i].set_active(0)

    # ── settings I/O ─────────────────────────────────────────────────
    def save_settings(self, path: str):
        self._sync_widgets_to_settings()
        fw, fh = self.fig.get_size_inches()
        self.s.fig_width = float(fw)
        self.s.fig_height = float(fh)
        with open(path, "w") as f:
            json.dump(asdict(self.s), f, indent=2)

    def load_settings(self, path: str):
        try:
            with open(path) as f:
                d = json.load(f)
        except FileNotFoundError:
            print(f"Settings file '{path}' not found.")
            return
        for k, v in d.items():
            if hasattr(self.s, k):
                setattr(self.s, k, v)
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
        with open(f"{stem}_settings.json", "w") as f:
            json.dump(asdict(self.s), f, indent=2)
        prev_usetex = plt.rcParams.get("text.usetex", False)
        self._exporting_tex = True
        plt.rcParams["text.usetex"] = True
        self.redraw()
        try:
            self.fig.savefig(
                out_path,
                dpi=DPI,
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"Saved {out_path} (TeX export)")
        except Exception as exc:
            print(f"Export failed for {out_path}: {exc}")
            raise
        finally:
            self._exporting_tex = False
            plt.rcParams["text.usetex"] = prev_usetex
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
        fs = int(round(s.font_size))
        ts = int(round(s.title_size))
        ms = float(s.marker_size)
        lw = float(s.line_width)
        visible = self._visible_groups()

        # ────────── Panel (a): E₃/E₃* vs N_{S³} ──────────
        self.axL.cla()
        for gn in visible:
            recs = self._filter_recs(self.grouped[gn])
            if not recs:
                continue
            x = np.array([r["N_eff"] for r in recs], dtype=float)
            y = np.array([r["E3_ratio_best"] for r in recs], dtype=float)
            col = GROUP_COLORS.get(gn, "gray")
            mrk = _get_marker(gn)

            # Scatter only (no lines between points)
            self.axL.scatter(
                x,
                y,
                marker=mrk,
                color=col,
                s=ms**2,
                edgecolors="0.2",
                linewidths=0.3,
                label=_latex_group_name(gn),
                zorder=5,
            )

            if s.show_fits_ratio and len(x) >= 2:
                log_x = np.log(x)
                coeffs = np.polyfit(log_x, y, 1)
                xs = np.linspace(float(x.min()), float(x.max()), 200)
                ys = np.polyval(coeffs, np.log(xs))
                self.axL.plot(
                    xs,
                    ys,
                    color=col,
                    linewidth=lw * 0.7,
                    linestyle="--",
                    alpha=0.6,
                )

        self.axL.set_xscale("log" if s.log_x_ratio else "linear")
        self.axL.set_yscale("log" if s.log_y_ratio else "linear")

        self.axL.set_xlabel(r"$N_{S^3}$", fontsize=fs)
        self.axL.set_ylabel(r"$E_3/E_3^*$", fontsize=fs)
        # self.axL.set_title(
        #     r"$E_3/E_3^* \, \text{vs} \, N_{S^3}$",
        #     fontsize=ts,
        #     pad=6,
        # )
        self.axL.text(
            0.02,
            0.98,
            "(a)",
            transform=self.axL.transAxes,
            fontsize=ts,
            fontweight="bold",
            va="top",
        )
        self.axL.tick_params(labelsize=max(6, fs - 1))
        self.axL.minorticks_on()
        if not s.show_minor_tick_labels:
            self.axL.xaxis.set_minor_formatter(NullFormatter())
            self.axL.yaxis.set_minor_formatter(NullFormatter())
        self.axL.legend(fontsize=max(6, fs - 2), loc="best", framealpha=0.8)
        if s.show_grid:
            self.axL.grid(True, alpha=s.grid_alpha)
        else:
            self.axL.grid(False)

        # ────────── Panel (b) ──────────
        # When log-quad fit is active the panel switches to r̂-space
        # (y = r_cont);  otherwise shows cu_z*-space (y = cu_z_best).
        r_space = s.show_fits_logquad
        self.axR.cla()

        # First pass: compute fits so we can build unified legend
        slope_dict: Dict[str, float] = {}  # linear: cu_z = r*·cu_xy
        logquad_dict: Dict[str, np.ndarray] = {}  # β₀,β₁,β₂ for r̂(ln cu_xy)
        for gn in visible:
            recs = self._filter_recs(self.grouped[gn])
            if not recs:
                continue
            x = np.array([r["cu_xy"] for r in recs], dtype=float)
            cuz = np.array([self._read_cu_z(r) for r in recs], dtype=float)

            if s.show_fits_cuz and len(x) >= 2:
                slope_dict[gn] = float(np.sum(x * cuz) / np.sum(x * x))

            if s.show_fits_logquad and len(x) >= 3:
                rhat = np.array(
                    [r.get("r_cont", self._read_cu_z(r) / r["cu_xy"]) for r in recs],
                    dtype=float,
                )
                xl = np.log(x)
                A = np.vstack([np.ones_like(xl), xl, xl**2]).T
                beta, *_ = np.linalg.lstsq(A, rhat, rcond=None)
                logquad_dict[gn] = beta
                print(
                    f"  {gn}: r̂ = {beta[0]:+.6f} "
                    f"{beta[1]:+.6f}·ln(cu_xy) "
                    f"{beta[2]:+.6f}·ln²(cu_xy)"
                )
                # Report mismatches: round(r̂_fit·cu_xy) != round(r_cont·cu_xy)
                lx_data = np.log(x)
                rhat_pred = beta[0] + beta[1] * lx_data + beta[2] * lx_data**2
                cuz_from_fit = np.round(rhat_pred * x).astype(int)
                cuz_from_obs = np.round(rhat * x).astype(int)
                mm = np.where(cuz_from_fit != cuz_from_obs)[0]
                if len(mm) > 0:
                    parts = [
                        f"cu_xy={int(x[i]):d} "
                        f"(fit→{cuz_from_fit[i]}, obs→{cuz_from_obs[i]})"
                        for i in mm
                    ]
                    print(f"    mismatches: {', '.join(parts)}")
                else:
                    print(f"    no mismatches")

        # Second pass: plot scatter + fits
        for gn in visible:
            recs = self._filter_recs(self.grouped[gn])
            if not recs:
                continue
            x = np.array([r["cu_xy"] for r in recs], dtype=float)
            cuz = np.array([self._read_cu_z(r) for r in recs], dtype=float)
            col = GROUP_COLORS.get(gn, "gray")
            mrk = _get_marker(gn)

            # Choose scatter y-values depending on space
            if r_space:
                y_scatter = np.array(
                    [r.get("r_cont", self._read_cu_z(r) / r["cu_xy"]) for r in recs],
                    dtype=float,
                )
            else:
                y_scatter = cuz

            # Scatter (no lines between points)
            self.axR.scatter(
                x,
                y_scatter,
                marker=mrk,
                color=col,
                s=ms**2,
                edgecolors="0.2",
                linewidths=0.3,
                zorder=5,
            )

            x_fit = np.linspace(float(x.min()), float(x.max()), 200)

            # Continuous r̂ point-to-point line (cu_z-space only;
            # in r-space the scatter already IS r_cont)
            if s.show_r_cont and not r_space:
                cx_vals, rc_vals = [], []
                for r_rec in recs:
                    rc = r_rec.get("r_cont")
                    if rc is not None:
                        cx_vals.append(r_rec["cu_xy"])
                        rc_vals.append(rc * r_rec["cu_xy"])
                if cx_vals:
                    self.axR.plot(
                        cx_vals,
                        rc_vals,
                        color=col,
                        linewidth=lw * 0.4,
                        linestyle="-",
                        alpha=0.5,
                    )

            # Linear fit (solid lines)
            if gn in slope_dict:
                slope = slope_dict[gn]
                if r_space:
                    # In r̂-space the linear model is a horizontal line at r*
                    self.axR.axhline(
                        slope,
                        color=col,
                        linewidth=lw * 0.5,
                        linestyle="-",
                        alpha=0.5,
                    )
                else:
                    self.axR.plot(
                        x_fit,
                        slope * x_fit,
                        color=col,
                        linewidth=lw * 0.7,
                        linestyle="-",
                        alpha=0.6,
                    )
                    if s.show_mismatches:
                        cuz_pred = np.round(slope * x).astype(int)
                        mismatch = cuz_pred != cuz.astype(int)
                        if np.any(mismatch):
                            self.axR.scatter(
                                x[mismatch],
                                cuz[mismatch],
                                s=ms**2 * 6,
                                facecolors="none",
                                edgecolors=col,
                                linewidths=1.5,
                                zorder=10,
                            )

            # Log-quadratic fit curve
            if gn in logquad_dict:
                beta = logquad_dict[gn]
                lxf = np.log(x_fit)
                rhat_fit = beta[0] + beta[1] * lxf + beta[2] * lxf**2
                if r_space:
                    # Plot r̂(cu_xy) directly (solid)
                    self.axR.plot(
                        x_fit,
                        rhat_fit,
                        color=col,
                        linewidth=lw * 0.7,
                        linestyle="-",
                        alpha=0.7,
                    )
                    # Mismatches in r-space: round(r̂_fit·cu_xy) != round(r_obs·cu_xy)
                    if s.show_mismatches:
                        lx_d = np.log(x)
                        rhat_at_data = beta[0] + beta[1] * lx_d + beta[2] * lx_d**2
                        cuz_fit = np.round(rhat_at_data * x).astype(int)
                        cuz_obs = np.round(y_scatter * x).astype(int)
                        mm_mask = cuz_fit != cuz_obs
                        if np.any(mm_mask):
                            self.axR.scatter(
                                x[mm_mask],
                                y_scatter[mm_mask],
                                s=ms**2 * 6,
                                facecolors="none",
                                edgecolors=col,
                                linewidths=1.5,
                                zorder=10,
                            )
                else:
                    # Plot r̂(cu_xy) · cu_xy in cu_z-space (solid)
                    self.axR.plot(
                        x_fit,
                        rhat_fit * x_fit,
                        color=col,
                        linewidth=lw * 0.7,
                        linestyle="-",
                        alpha=0.7,
                    )

        # Build unified legend (one entry per group, slope appended)
        handles, labels = [], []
        for gn in visible:
            recs = self._filter_recs(self.grouped[gn])
            if not recs:
                continue
            col = GROUP_COLORS.get(gn, "gray")
            mrk = _get_marker(gn)
            h = Line2D(
                [],
                [],
                color=col,
                marker=mrk,
                linestyle="None",
                markersize=ms,
                markeredgewidth=0.3,
                markeredgecolor="0.2",
            )
            lbl = _latex_group_name(gn)
            if gn in slope_dict:
                lbl = rf"{lbl}  $r^*\!={slope_dict[gn]:.3f}$"
            handles.append(h)
            labels.append(lbl)

        self.axR.set_xlabel(r"$\mathrm{cu}_{xy}$", fontsize=fs)
        if r_space:
            self.axR.set_ylabel(r"$\hat{r}$", fontsize=fs)
            # self.axR.set_title(
            #     r"$\hat{r} \text{ vs } \mathrm{cu}_{xy}$",
            #     fontsize=ts,
            #     pad=6,
            # )
        else:
            self.axR.set_ylabel(r"$\mathrm{cu}_z^{\,*}$", fontsize=fs)
            # self.axR.set_title(
            #     r"Optimal $\mathrm{cu}_z \text{ vs } \mathrm{cu}_{xy}$",
            #     fontsize=ts,
            #     pad=6,
            # )
        self.axR.text(
            0.02,
            0.98,
            "(b)",
            transform=self.axR.transAxes,
            fontsize=ts,
            fontweight="bold",
            va="top",
        )
        self.axR.tick_params(labelsize=max(6, fs - 1))
        self.axR.minorticks_on()
        if not s.show_minor_tick_labels:
            self.axR.xaxis.set_minor_formatter(NullFormatter())
            self.axR.yaxis.set_minor_formatter(NullFormatter())
        if s.show_grid:
            self.axR.grid(True, alpha=s.grid_alpha)
        else:
            self.axR.grid(False)
        if handles and s.show_legend_R:
            self.axR.legend(
                handles,
                labels,
                fontsize=max(6, fs - 2),
                loc="upper left" if not r_space else "best",
                framealpha=0.8,
            )

        self.fig.canvas.draw_idle()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    data = load_data(DATA_FILE)
    meta = data.get("metadata", {})

    print(f"Loaded {len(data['results'])} records from '{DATA_FILE}'")
    print(f"  Version : {meta.get('version', '?')}")
    print(f"  Groups  : {meta.get('groups', '?')}")
    print(f"  cu_xy   : {meta.get('cu_xy_range', '?')}")
    print()

    s = Settings()

    # Try loading saved settings
    if os.path.exists(DEFAULT_SETTINGS_JSON):
        try:
            with open(DEFAULT_SETTINGS_JSON) as f:
                d = json.load(f)
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            print(f"  Loaded settings from '{DEFAULT_SETTINGS_JSON}'")
        except Exception as e:
            print(f"  Warning: could not load settings: {e}")

    _app = Figure2Interactive(data, s)
    plt.show()


if __name__ == "__main__":
    main()
