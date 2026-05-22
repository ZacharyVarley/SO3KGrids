#!/usr/bin/env python3
"""
Interactive Figure 3 CDF overlays.

Two-panel Figure 3 viewer:
- (a) witness->grid CDF overlay (left)
- (b) self-NN CDF overlay (right)

Controls window provides:
- Legend host subplot toggle (left/right)
- Legend corner selection
- Shared CDF bin count
- Shared symlog decades

Notes:
- This intentionally does NOT include Figure 3's "show pdf panel" option.
- Uses precomputed NPZ data files.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.widgets import TextBox

from figures.common import (
    ControlWindow,
    PAPER_COL_W,
    PAPER_DPI,
    PAPER_FONT,
    PAPER_MARGINS,
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


# ---------------------------------------------------------------------------
# CDF half-log scale (centered around p=0.5)
# ---------------------------------------------------------------------------


class _CDFHalfLogTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def __init__(self, decades: float):
        super().__init__()
        self.decades = float(decades)

    def transform_non_affine(self, a):
        a = np.asarray(a, dtype=float)
        a = np.clip(a, 1e-12, 1.0 - 1e-12)
        return np.where(a <= 0.5, np.log10(2.0 * a), -np.log10(2.0 * (1.0 - a)))

    def inverted(self):
        return _CDFHalfLogInverseTransform(self.decades)


class _CDFHalfLogInverseTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def __init__(self, decades: float):
        super().__init__()
        self.decades = float(decades)

    def transform_non_affine(self, a):
        a = np.asarray(a, dtype=float)
        return np.where(a <= 0.0, 10.0 ** a / 2.0, 1.0 - 10.0 ** (-a) / 2.0)

    def inverted(self):
        return _CDFHalfLogTransform(self.decades)


class CDFHalfLogScale(mscale.ScaleBase):
    name = "cdfhalflog"

    def __init__(self, axis, *, decades=3.0, **kwargs):
        super().__init__(axis)
        self.decades = float(decades)

    def get_transform(self):
        return _CDFHalfLogTransform(self.decades)

    def set_default_locators_and_formatters(self, axis):
        dmax = int(max(1, min(7, round(self.decades))))
        major = [0.5]
        for d in range(1, dmax + 1):
            major.extend([10.0 ** (-d), 1.0 - 10.0 ** (-d)])
        major.extend([0.1, 0.9])
        major = sorted({v for v in major if 1e-12 < v < 1 - 1e-12})

        minor = set()
        for d in range(1, dmax + 1):
            base = 10.0 ** (-d)
            for k in range(2, 10):
                v = k * base
                if 1e-12 < v < 0.5:
                    minor.add(v)
                u = 1.0 - k * base
                if 0.5 < u < 1 - 1e-12:
                    minor.add(u)
        for v in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
            minor.add(v)
        minor = sorted(v for v in minor if v not in major)

        axis.set_major_locator(ticker.FixedLocator(major))
        axis.set_minor_locator(ticker.FixedLocator(minor))

        def _fmt(v, _pos):
            if abs(v - 0.5) < 1e-9:
                return "0.5"
            if v >= 1.0 or v <= 0.0:
                return ""
            return f"{v:.6f}".rstrip("0").rstrip(".")

        axis.set_major_formatter(ticker.FuncFormatter(_fmt))


mscale.register_scale(CDFHalfLogScale)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


ORDERED_GROUPS = ["C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6", "T", "O", "I"]
_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_SETTINGS_DIR = _HERE / "settings"
DEFAULT_STEM = "nn_cdf_overlays"
SETTINGS_FILE = str(_SETTINGS_DIR / "nn_cdf_overlays.json")
DPI = PAPER_DPI

# Typography spec — (widget_name, label, settings_attr [, fmt])
_TYPO_ROWS = [
    [("font", "Font", "font_size"), ("title_size", "Title", "title_size")],
    [("subtitle_size", "Sub", "subtitle_size"), ("panel_label_size", "Panel", "panel_label_size")],
    [("panel_label_x", "Lbl x", "panel_label_x", "{:.3f}"), ("panel_label_y", "Lbl y", "panel_label_y", "{:.3f}")],
    [("legend_size", "Leg", "legend_size")],
]

TEX_LABEL = {
    "C2": r"$C_2$", "C3": r"$C_3$", "C4": r"$C_4$", "C6": r"$C_6$",
    "D2": r"$D_2$", "D3": r"$D_3$", "D4": r"$D_4$", "D6": r"$D_6$",
    "T": r"$T$", "O": r"$O$", "I": r"$I$",
}


def load_npz_overlay(path: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    names = [str(x) for x in raw["_group_names"]]
    out = {"metadata": {}, "groups": {}}
    if "_n_eff_target" in raw:
        out["metadata"]["n_eff_target"] = int(raw["_n_eff_target"])
    if "_witness_factor" in raw:
        out["metadata"]["witness_factor"] = int(raw["_witness_factor"])
    out["metadata"]["groups"] = names

    for name in names:
        g = {}
        for key in (
            "bin_edges", "counts_kr", "counts_rej", "N_kr", "N_rej",
            "N_witness", "mean_kr", "mean_rej"
        ):
            full = f"{name}_{key}"
            if full in raw:
                val = raw[full]
                g[key] = val.item() if getattr(val, "ndim", 0) == 0 else val
        out["groups"][name] = g
    return out


def rebin_histogram(fine_counts: np.ndarray, fine_edges: np.ndarray, coarse_edges: np.ndarray) -> np.ndarray:
    mids = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    idx = np.clip(np.digitize(mids, coarse_edges) - 1, 0, len(coarse_edges) - 2)
    out = np.bincount(idx, weights=fine_counts.astype(float), minlength=len(coarse_edges) - 1)
    return out.astype(float)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@dataclass
class ViewState:
    n_bins: int = 240
    symlog_decades: float = 3.0
    line_width: float = 1.0
    font_size: float = PAPER_FONT["label"]
    title_size: float = PAPER_FONT["title"]
    subtitle_size: float = PAPER_FONT["subtitle"]
    panel_label_size: float = PAPER_FONT["panel_label"]
    panel_label_x: float = 0.02
    panel_label_y: float = 0.98
    legend_size: float = PAPER_FONT["legend"]
    subplot_wspace: float = 0.18
    legend_on: str = "top"
    legend_loc: str = "lower right"
    export_stem: str = DEFAULT_STEM
    fig_width: float = PAPER_COL_W
    fig_height: float = 5.5
    show_arrow_a: bool = True
    arrow_a_text: str = "fewer gaps"
    arrow_a_x: float = 0.90
    arrow_a_y: float = 0.88
    arrow_a_angle_deg: float = 135.0
    arrow_a_len: float = 0.12
    arrow_a_text_dx: float = 0.02
    arrow_a_text_dy: float = 0.02
    arrow_a_text_size: float = 10.0
    arrow_a_arrow_size: float = 1.4
    show_arrow_b: bool = True
    arrow_b_text: str = "collision-free"
    arrow_b_x: float = 0.15
    arrow_b_y: float = 0.55
    arrow_b_angle_deg: float = -45.0
    arrow_b_len: float = 0.12
    arrow_b_text_dx: float = 0.02
    arrow_b_text_dy: float = 0.00
    arrow_b_text_size: float = 10.0
    arrow_b_arrow_size: float = 1.4


class Figure3PlotApp:
    def __init__(self, data3: dict, data4: dict):
        self.data3 = data3
        self.data4 = data4
        self.state = ViewState()
        if os.path.exists(SETTINGS_FILE):
            load_dataclass_settings(SETTINGS_FILE, self.state, wrapper_key="view")

        apply_plot_rcparams()

        groups3 = set(self.data3["groups"].keys())
        groups4 = set(self.data4["groups"].keys())
        self.groups = [g for g in ORDERED_GROUPS if g in groups3 and g in groups4]

        cmap = plt.get_cmap("tab20")
        self.group_colors = {name: cmap(i) for i, name in enumerate(ORDERED_GROUPS)}

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(
            2, 1, figsize=(self.state.fig_width, self.state.fig_height), dpi=110,
        )
        self.fig.canvas.manager.set_window_title("Figure 3")
        self._exporting_tex = False

        self.ctrl = plt.figure(figsize=(11.0, 7.0), dpi=110)
        self.ctrl.canvas.manager.set_window_title("Figure 3 Controls")
        self.widgets = {}
        self._loading = False
        self._build_controls()

        self.redraw()

    def _build_controls(self):
        self.ctrl.clf()
        cw = ControlWindow(self.ctrl, left=0.06, right=0.92, top=0.95, col_gap=0.05, col1_frac=0.44)
        self._cw = cw
        self.widgets = cw.widgets

        build_typography_section(cw, 1, self.state, _TYPO_ROWS)
        cw.number_row(1, [("wspace", "Spacing", self.state.subplot_wspace, "{:.2f}")])

        cw.section(1, "Plot")
        cw.number_row(1, [("bins", "Bins", self.state.n_bins, "{:.0f}"), ("dec", "Decades", self.state.symlog_decades)])
        cw.slider(1, "lw", "Line width", 1.0, 4.0, self.state.line_width, step=0.1)

        cw.section(1, "Legend")
        cw.radio_group(1, "legend_on", ["top", "bottom"], active=0, h=0.060)
        cw.radio_group(
            1,
            "legend_loc",
            ["lower right", "upper right", "upper left", "lower left"],
            active=0,
            h=0.105,
        )

        cw.section(1, "File I/O")
        cw.textbox(1, "settings_path", "Settings", SETTINGS_FILE)
        cw.button_row(1, [("load_settings", "Load"), ("save_settings", "Save")], h=0.038)
        cw.button_row(1, [("export", "PNG/PDF")], h=0.042)

        cw.section(2, "Annotations")
        cw.checkbox_grid(
            2,
            "show_arrows",
            [
                ("show_a", "Show arrow (a)", self.state.show_arrow_a),
                ("show_b", "Show arrow (b)", self.state.show_arrow_b),
            ],
            n_cols=1,
            row_h=0.042,
            label_size=8.0,
        )
        cw.radio_group(2, "arrow_target", ["(a)", "(b)"], active=0, h=0.055)
        cw.textbox(2, "arrow_text", "Text", self.state.arrow_a_text, h=0.040)
        cw.slider(2, "ann_x", "X", 0.00, 1.00, self.state.arrow_a_x, step=0.01)
        cw.slider(2, "ann_y", "Y", 0.00, 1.00, self.state.arrow_a_y, step=0.01)
        cw.slider(2, "ann_angle", "Angle", -180.0, 180.0, self.state.arrow_a_angle_deg, step=1.0)
        cw.slider(2, "ann_len", "Arrow length", 0.02, 0.40, self.state.arrow_a_len, step=0.01)
        cw.slider(2, "ann_text_size", "Text size", 6.0, 28.0, self.state.arrow_a_text_size, step=0.5)
        cw.connect_scroll()

        # Typography text entries — lightweight restyle only
        bind_typography(cw, _TYPO_ROWS, self._on_restyle)
        # Data text entries — full rebuild (rebins histograms)
        for key in ("bins", "dec"):
            self.widgets[key].on_submit(lambda _v: self._on_data_change())
        # Visual/layout — replot without rebinning
        self.widgets["wspace"].on_submit(lambda _v: self._on_visual_change())
        for key in ("lw", "ann_x", "ann_y", "ann_angle", "ann_len", "ann_text_size"):
            self.widgets[key].on_changed(lambda _v: self._on_visual_change())
        self.widgets["legend_on"].on_clicked(lambda _v: self._on_visual_change())
        self.widgets["legend_loc"].on_clicked(lambda _v: self._on_visual_change())
        for chk in self.widgets["show_arrows"]:
            chk.on_clicked(lambda _v: self._on_visual_change())
        self.widgets["arrow_target"].on_clicked(lambda _v: self._on_arrow_target_changed())
        self.widgets["arrow_text"].on_submit(lambda _v: self._on_visual_change())
        self.widgets["export"].on_clicked(self._save_png_pdf)
        self.widgets["load_settings"].on_clicked(lambda _evt: self.load_settings(self.widgets["settings_path"].text))
        self.widgets["save_settings"].on_clicked(lambda _evt: self.save_settings(self.widgets["settings_path"].text))

        self._push_active_annotation_state_to_controls()

    def _sync_visual_state(self):
        """Sync visual-only widget values into state (no bins/decades)."""
        cw = self._cw
        self.state.line_width = float(self.widgets["lw"].val)
        sync_typography(cw, self.state, _TYPO_ROWS)
        self.state.subplot_wspace = cw.get_val("wspace", self.state.subplot_wspace)
        self.state.legend_on = str(self.widgets["legend_on"].value_selected)
        self.state.legend_loc = str(self.widgets["legend_loc"].value_selected)
        show_a = self.widgets["show_arrows"][0].get_status()[0]
        show_b = self.widgets["show_arrows"][1].get_status()[0]
        self.state.show_arrow_a = bool(show_a)
        self.state.show_arrow_b = bool(show_b)
        self._read_active_annotation_controls_into_state()

    def _on_data_change(self):
        """Full rebuild: bins or decades changed — rebin histograms + replot."""
        if self._loading:
            return
        cw = self._cw
        self.state.n_bins = int(cw.get_val("bins", self.state.n_bins))
        self.state.symlog_decades = cw.get_val("dec", self.state.symlog_decades)
        self._sync_visual_state()
        self.redraw()

    def _on_visual_change(self):
        """Fast replot: reuse cached curves, only update line width / annotations / legend."""
        if self._loading:
            return
        self._sync_visual_state()
        self._replot()

    def _on_restyle(self):
        """Lightweight: update typography without recomputing data."""
        if self._loading:
            return
        sync_typography(self._cw, self.state, _TYPO_ROWS)
        self._restyle()

    def _restyle(self):
        """Update fonts/labels on existing axes without clearing."""
        restyle_axes(
            self.fig,
            list(zip(self.axes, ["(a)", "(b)"])),
            self.state,
            use_tex=self._exporting_tex,
            label_offset=1,
            min_tick=7,
            min_legend=6,
        )

    def _active_arrow_prefix(self) -> str:
        sel = str(self.widgets["arrow_target"].value_selected)
        return "arrow_b" if "b" in sel else "arrow_a"

    def _read_active_annotation_controls_into_state(self):
        p = self._active_arrow_prefix()
        setattr(self.state, f"{p}_text", self.widgets["arrow_text"].text)
        setattr(self.state, f"{p}_x", float(self.widgets["ann_x"].val))
        setattr(self.state, f"{p}_y", float(self.widgets["ann_y"].val))
        setattr(self.state, f"{p}_angle_deg", float(self.widgets["ann_angle"].val))
        setattr(self.state, f"{p}_len", float(self.widgets["ann_len"].val))
        setattr(self.state, f"{p}_text_size", float(self.widgets["ann_text_size"].val))

    def _push_active_annotation_state_to_controls(self):
        p = self._active_arrow_prefix()
        self._loading = True
        self.widgets["arrow_text"].set_val(str(getattr(self.state, f"{p}_text")))
        self.widgets["ann_x"].set_val(float(getattr(self.state, f"{p}_x")))
        self.widgets["ann_y"].set_val(float(getattr(self.state, f"{p}_y")))
        self.widgets["ann_angle"].set_val(float(getattr(self.state, f"{p}_angle_deg")))
        self.widgets["ann_len"].set_val(float(getattr(self.state, f"{p}_len")))
        self.widgets["ann_text_size"].set_val(float(getattr(self.state, f"{p}_text_size")))
        self._loading = False

    def _on_arrow_target_changed(self):
        if self._loading:
            return
        self._push_active_annotation_state_to_controls()

    def _draw_arrow_annotation(
        self,
        ax,
        show: bool,
        x: float,
        y: float,
        angle_deg: float,
        length: float,
        text: str,
        text_dx: float,
        text_dy: float,
        text_size: float,
        arrow_size: float,
        use_tex: bool,
    ):
        if not show:
            return
        theta = np.deg2rad(float(angle_deg))
        dx = float(length) * np.cos(theta)
        dy = float(length) * np.sin(theta)
        ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                lw=max(0.5, float(arrow_size)),
                color="black",
                mutation_scale=8.0 * max(0.5, float(arrow_size)),
            ),
        )
        ax.text(
            x + text_dx,
            y + text_dy,
            self._format_annotation_text(text, use_tex),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=max(6.0, float(text_size)),
        )

    @staticmethod
    def _format_annotation_text(text: str, use_tex: bool) -> str:
        txt = str(text)
        if not use_tex:
            return txt
        if "$" in txt:
            return txt
        safe = txt.replace("\\", r"\textbackslash ")
        safe = safe.replace("{", r"\{").replace("}", r"\}")
        return rf"$\textrm{{{safe}}}$"

    def _build_curves(self, groups_data: dict, xmax: float | None = None):
        if xmax is None:
            xmax = 1.0
        else:
            xmax = max(1.0, float(xmax))
        for name in self.groups:
            edges = groups_data[name]["bin_edges"]
            xmax = max(xmax, float(edges[-1]))
        bins = np.linspace(0.0, xmax, max(20, self.state.n_bins) + 1)

        curves = []
        for name in self.groups:
            g = groups_data[name]
            fine_edges = g["bin_edges"]
            c_kr = rebin_histogram(g["counts_kr"], fine_edges, bins)
            c_rej = rebin_histogram(g["counts_rej"], fine_edges, bins)
            curves.append((name, "KR", c_kr, bins))
            curves.append((name, "Rej", c_rej, bins))
        return curves, bins

    def _plot_panel(self, ax, curves, title: str, panel_tag: str, use_tex: bool):
        fs = int(round(self.state.font_size))
        ax.clear()

        for name, method, counts, edges in curves:
            total = float(np.sum(counts))
            if total <= 0:
                continue
            y = np.concatenate([[0.0], np.cumsum(counts) / total])
            color = self.group_colors[name]
            ls = "-" if method == "KR" else (0, (5, 3))
            ax.plot(edges, y, color=color, linestyle=ls, linewidth=self.state.line_width)

        decades = float(np.clip(self.state.symlog_decades, 1.0, 7.0))
        y_min = 10.0 ** (-decades)
        y_max = 1.0 - 10.0 ** (-decades)

        ax.set_yscale("cdfhalflog", decades=decades)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.35)
        ax.axhline(0.5, color="k", linewidth=1.0, alpha=0.9)

        if use_tex:
            xlabel = "$\\textrm{Disorientation angle } (^\\circ)$"
            ylabel = "$\\textrm{CDF}$"
        else:
            xlabel = r"Disorientation angle $(^\circ)$"
            ylabel = "CDF"
        ax.set_xlabel(xlabel, fontsize=fs + 1)
        ax.set_ylabel(ylabel, fontsize=fs + 1)
        # ax.set_title(title, fontsize=fs + 1)
        ax.tick_params(labelsize=max(7, fs - 1))
        if panel_tag:
            add_panel_label(
                ax,
                panel_tag,
                x=self.state.panel_label_x,
                y=self.state.panel_label_y,
                fontsize=self.state.panel_label_size,
                use_tex=use_tex,
            )

    def _legend_payload(self):
        col1 = ["C2", "C3", "C4", "C6"]
        col2 = ["D2", "D3", "D4", "D6"]
        col3 = ["T", "O", "I", "_blank"]
        col4 = ["_method_kr", "_method_rej", "_blank", "_blank"]
        legend_items = col1 + col2 + col3 + col4

        handles = []
        labels = []
        for key in legend_items:
            if key == "_blank":
                handles.append(Line2D([], [], alpha=0.0, lw=0.0))
                labels.append(" ")
                continue
            if key == "_method_kr":
                handles.append(Line2D([0], [0], color="k", lw=self.state.line_width, ls="-"))
                if self._exporting_tex:
                    labels.append("$\\textrm{KR}$")
                else:
                    labels.append("KR")
                continue
            if key == "_method_rej":
                handles.append(Line2D([0], [0], color="k", lw=self.state.line_width, ls=(0, (5, 3))))
                if self._exporting_tex:
                    labels.append("$\\textrm{Rej}$")
                else:
                    labels.append("Rej")
                continue
            if key not in self.groups:
                handles.append(Line2D([], [], alpha=0.0, lw=0.0))
                labels.append(" ")
                continue
            handles.append(Line2D([0], [0], color=self.group_colors[key], lw=self.state.line_width, ls="-"))
            labels.append(TEX_LABEL[key])
        return handles, labels

    def redraw(self):
        """Full rebuild: rebin histograms and replot everything."""
        global_xmax = 1.0
        for name in self.groups:
            global_xmax = max(global_xmax, float(self.data3["groups"][name]["bin_edges"][-1]))
            global_xmax = max(global_xmax, float(self.data4["groups"][name]["bin_edges"][-1]))

        self._cached_curves3, _ = self._build_curves(self.data3["groups"], xmax=global_xmax)
        self._cached_curves4, _ = self._build_curves(self.data4["groups"], xmax=global_xmax)
        self._replot()

    def _replot(self):
        """Replot from cached curves (no rebinning)."""
        use_tex = bool(self._exporting_tex)

        self._plot_panel(self.ax_left, self._cached_curves3, "Witness CDF", "(a)", use_tex)
        self._plot_panel(self.ax_right, self._cached_curves4, "Self-NN CDF", "(b)", use_tex)

        fs = int(round(self.state.font_size))
        self._draw_arrow_annotation(
            self.ax_left,
            self.state.show_arrow_a,
            self.state.arrow_a_x,
            self.state.arrow_a_y,
            self.state.arrow_a_angle_deg,
            self.state.arrow_a_len,
            self.state.arrow_a_text,
            self.state.arrow_a_text_dx,
            self.state.arrow_a_text_dy,
            self.state.arrow_a_text_size,
            self.state.arrow_a_arrow_size,
            use_tex,
        )
        self._draw_arrow_annotation(
            self.ax_right,
            self.state.show_arrow_b,
            self.state.arrow_b_x,
            self.state.arrow_b_y,
            self.state.arrow_b_angle_deg,
            self.state.arrow_b_len,
            self.state.arrow_b_text,
            self.state.arrow_b_text_dx,
            self.state.arrow_b_text_dy,
            self.state.arrow_b_text_size,
            self.state.arrow_b_arrow_size,
            use_tex,
        )

        handles, labels = self._legend_payload()
        for ax in (self.ax_left, self.ax_right):
            lg = ax.get_legend()
            if lg is not None:
                lg.remove()

        target = self.ax_left if self.state.legend_on == "top" else self.ax_right
        fs = int(round(self.state.font_size))
        if use_tex:
            title = "$\\mathbf{Group \\ | \\ Method}$"
        else:
            title = "Group / Method"
        target.legend(
            handles,
            labels,
            loc=self.state.legend_loc,
            fontsize=max(6, self.state.legend_size),
            framealpha=0.85,
            title=title,
            title_fontsize=max(6, self.state.legend_size + 1),
            ncol=4,
            handlelength=2.0,
            columnspacing=0.9,
            borderpad=0.6,
            handletextpad=0.45,
            labelspacing=0.35,
        )

        self.fig.subplots_adjust(
            left=PAPER_MARGINS["left"],
            right=PAPER_MARGINS["right"],
            top=PAPER_MARGINS["top"],
            bottom=PAPER_MARGINS["bottom"],
            hspace=max(0.02, min(0.80, float(self.state.subplot_wspace))),
        )
        self.fig.canvas.draw_idle()

    def _sync_state_to_widgets(self):
        self._loading = True
        cw = self._cw
        cw.set_val("bins", self.state.n_bins, "{:.0f}")
        cw.set_val("dec", self.state.symlog_decades, "{:.1f}")
        self.widgets["lw"].set_val(self.state.line_width)
        push_typography(cw, self.state, _TYPO_ROWS)
        cw.set_val("wspace", self.state.subplot_wspace, "{:.2f}")

        show_status = [
            bool(self.widgets["show_arrows"][0].get_status()[0]),
            bool(self.widgets["show_arrows"][1].get_status()[0]),
        ]
        if bool(show_status[0]) != bool(self.state.show_arrow_a):
            self.widgets["show_arrows"][0].set_active(0)
        if bool(show_status[1]) != bool(self.state.show_arrow_b):
            self.widgets["show_arrows"][1].set_active(0)

        leg_on_opts = ["top", "bottom"]
        if self.state.legend_on in leg_on_opts:
            cur = str(self.widgets["legend_on"].value_selected)
            if cur != self.state.legend_on:
                self.widgets["legend_on"].set_active(leg_on_opts.index(self.state.legend_on))

        leg_loc_opts = ["lower right", "upper right", "upper left", "lower left"]
        desired = self.state.legend_loc if self.state.legend_loc in leg_loc_opts else "lower right"
        cur = str(self.widgets["legend_loc"].value_selected)
        if cur != desired:
            self.widgets["legend_loc"].set_active(leg_loc_opts.index(desired))
        self._loading = False
        self._push_active_annotation_state_to_controls()

    def save_settings(self, path: str):
        path = (path or "").strip() or SETTINGS_FILE
        fig_w, fig_h = self.fig.get_size_inches()
        self.state.fig_width = float(fig_w)
        self.state.fig_height = float(fig_h)
        save_dataclass_settings(path, self.state, wrapper_key="view")
        print(f"Saved settings -> {path}")

    def load_settings(self, path: str):
        path = (path or "").strip() or SETTINGS_FILE
        if not load_dataclass_settings(path, self.state, wrapper_key="view"):
            print(f"Settings file '{path}' not found.")
            return

        try:
            fw = float(getattr(self.state, "fig_width", 3.48))
            fh = float(getattr(self.state, "fig_height", 7.2))
            self.fig.set_size_inches(fw, fh, forward=True)
        except Exception:
            pass

        self._sync_state_to_widgets()
        self.redraw()

    def _save_png_pdf(self, _evt=None):
        stem = self.state.export_stem or DEFAULT_STEM
        png_path = f"{stem}.png"
        pdf_path = f"{stem}.pdf"
        self._exporting_tex = True
        try:
            export_figure_with_tex(
                self.fig,
                png_path,
                redraw_callback=lambda: self.redraw(),
                dpi=DPI,
            )
            export_figure_with_tex(
                self.fig,
                pdf_path,
                redraw_callback=lambda: self.redraw(),
                dpi=DPI,
            )
        finally:
            self._exporting_tex = False
            self.redraw()
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


def _resolve_self_nn_default() -> str:
    return str(_DATA_DIR / "figure3_self_nn.npz")


def main():
    parser = argparse.ArgumentParser(description="Figure 3 combined witness (left) + self-NN (right) CDF overlay")
    parser.add_argument(
        "--witness-data",
        default=str(_DATA_DIR / "figure3_witness_nn.npz"),
        help="Path to witness-set NPZ data",
    )
    parser.add_argument(
        "--self-nn-data",
        default=None,
        help="Path to self-NN NPZ data (default: figures/data/figure3_self_nn.npz)",
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
    args = parser.parse_args()

    self_nn_path = args.self_nn_data or _resolve_self_nn_default()

    witness_path = args.witness_data

    if not os.path.exists(witness_path):
        raise FileNotFoundError(f"Witness data file not found: {args.witness_data}")
    if not os.path.exists(self_nn_path):
        raise FileNotFoundError(f"Self-NN data file not found: {self_nn_path}")

    data3 = load_npz_overlay(witness_path)
    data4 = load_npz_overlay(self_nn_path)

    app = Figure3PlotApp(data3, data4)
    app.state.export_stem = (args.stem or "").strip() or DEFAULT_STEM
    if args.export:
        app._save_png_pdf()
        plt.close("all")
        return

    plt.show()


if __name__ == "__main__":
    main()
