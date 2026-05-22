#!/usr/bin/env python3
"""
figure3.py

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
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox


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
DEFAULT_STEM = "figure3"
SETTINGS_FILE = "figure3_settings.json"
DPI = 300
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
    line_width: float = 1.4
    font_size: float = 10.0
    subplot_wspace: float = 0.16
    legend_on: str = "left"
    legend_loc: str = "lower right"
    export_stem: str = DEFAULT_STEM
    fig_width: float = 15.0
    fig_height: float = 7.0
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

        groups3 = set(self.data3["groups"].keys())
        groups4 = set(self.data4["groups"].keys())
        self.groups = [g for g in ORDERED_GROUPS if g in groups3 and g in groups4]

        cmap = plt.get_cmap("tab20")
        self.group_colors = {name: cmap(i) for i, name in enumerate(ORDERED_GROUPS)}

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(15.0, 7.0), dpi=110)
        self.fig.canvas.manager.set_window_title("Figure 3")
        self._exporting_tex = False

        self.ctrl = plt.figure(figsize=(7.2, 7.8), dpi=110)
        self.ctrl.canvas.manager.set_window_title("Figure 3 Controls")
        self.widgets = {}
        self._loading = False
        self._build_controls()

        self.redraw()

    def _build_controls(self):
        self.ctrl.clf()
        Lx, Rx = 0.06, 0.52
        W = 0.42

        def _section(x, y, title):
            ax = self.ctrl.add_axes([x, y, W, 0.028])
            ax.set_axis_off()
            ax.text(0.0, 0.5, title, fontsize=9, fontweight="bold", va="center")

        _section(Lx, 0.95, "Global")
        self.widgets["bins"] = self._slider(Lx, 0.90, W, "Bins", 40, 3000, self.state.n_bins, 10)
        self.widgets["dec"] = self._slider(Lx, 0.84, W, "Symlog decades", 1.0, 7.0, self.state.symlog_decades, 0.5)
        self.widgets["lw"] = self._slider(Lx, 0.78, W, "Line width", 0.2, 4.0, self.state.line_width, 0.1)
        self.widgets["font"] = self._slider(Lx, 0.72, W, "Font size", 7, 24, self.state.font_size, 1)
        self.widgets["wspace"] = self._slider(Lx, 0.66, W, "Plot spacing", 0.02, 0.60, self.state.subplot_wspace, 0.01)

        ax_export = self.ctrl.add_axes([Lx, 0.60, W, 0.042])
        self.widgets["export"] = Button(ax_export, "PNG/PDF")

        ax_settings = self.ctrl.add_axes([Lx, 0.545, W, 0.038])
        self.widgets["settings_path"] = TextBox(ax_settings, "Settings", initial=SETTINGS_FILE)
        self.widgets["settings_path"].label.set_fontsize(8)

        ax_load = self.ctrl.add_axes([Lx, 0.495, 0.20, 0.038])
        ax_save = self.ctrl.add_axes([Lx + 0.22, 0.495, 0.20, 0.038])
        self.widgets["load_settings"] = Button(ax_load, "Load")
        self.widgets["save_settings"] = Button(ax_save, "Save")

        _section(Lx, 0.45, "Legend")
        ax_leg_on = self.ctrl.add_axes([Lx, 0.385, 0.18, 0.060])
        self.widgets["legend_on"] = RadioButtons(ax_leg_on, ["left", "right"], active=0)
        for t in self.widgets["legend_on"].labels:
            t.set_fontsize(8)

        ax_leg_loc = self.ctrl.add_axes([Lx + 0.20, 0.34, 0.22, 0.105])
        self.widgets["legend_loc"] = RadioButtons(
            ax_leg_loc,
            ["lower right", "upper right", "upper left", "lower left"],
            active=0,
        )
        for t in self.widgets["legend_loc"].labels:
            t.set_fontsize(8)

        _section(Rx, 0.95, "Annotations")
        ax_show_arrows = self.ctrl.add_axes([Rx, 0.90, W, 0.05])
        self.widgets["show_arrows"] = CheckButtons(
            ax_show_arrows,
            ["Show arrow (a)", "Show arrow (b)"],
            [self.state.show_arrow_a, self.state.show_arrow_b],
        )
        for t in self.widgets["show_arrows"].labels:
            t.set_fontsize(8)

        ax_arrow_target = self.ctrl.add_axes([Rx, 0.835, 0.16, 0.055])
        self.widgets["arrow_target"] = RadioButtons(ax_arrow_target, ["(a)", "(b)"], active=0)
        for t in self.widgets["arrow_target"].labels:
            t.set_fontsize(8)

        ax_arrow_text = self.ctrl.add_axes([Rx + 0.18, 0.845, W - 0.18, 0.040])
        self.widgets["arrow_text"] = TextBox(ax_arrow_text, "Text", initial=self.state.arrow_a_text)
        self.widgets["arrow_text"].label.set_fontsize(8)

        self.widgets["ann_x"] = self._slider(Rx, 0.78, W, "X", 0.00, 1.00, self.state.arrow_a_x, 0.01)
        self.widgets["ann_y"] = self._slider(Rx, 0.72, W, "Y", 0.00, 1.00, self.state.arrow_a_y, 0.01)
        self.widgets["ann_angle"] = self._slider(Rx, 0.66, W, "Angle", -180.0, 180.0, self.state.arrow_a_angle_deg, 1.0)
        self.widgets["ann_len"] = self._slider(Rx, 0.60, W, "Arrow length", 0.02, 0.40, self.state.arrow_a_len, 0.01)
        self.widgets["ann_arrow_size"] = self._slider(Rx, 0.54, W, "Arrow size", 0.5, 4.0, self.state.arrow_a_arrow_size, 0.1)
        self.widgets["ann_text_dx"] = self._slider(Rx, 0.48, W, "Text dx", -0.50, 0.50, self.state.arrow_a_text_dx, 0.01)
        self.widgets["ann_text_dy"] = self._slider(Rx, 0.42, W, "Text dy", -0.50, 0.50, self.state.arrow_a_text_dy, 0.01)
        self.widgets["ann_text_size"] = self._slider(Rx, 0.36, W, "Text size", 6.0, 28.0, self.state.arrow_a_text_size, 0.5)

        for key in (
            "bins", "dec", "lw", "font", "wspace",
            "ann_x", "ann_y", "ann_angle", "ann_len",
            "ann_text_dx", "ann_text_dy", "ann_text_size", "ann_arrow_size",
        ):
            self.widgets[key].on_changed(lambda _v: self._on_change())
        self.widgets["legend_on"].on_clicked(lambda _v: self._on_change())
        self.widgets["legend_loc"].on_clicked(lambda _v: self._on_change())
        self.widgets["show_arrows"].on_clicked(lambda _v: self._on_change())
        self.widgets["arrow_target"].on_clicked(lambda _v: self._on_arrow_target_changed())
        self.widgets["arrow_text"].on_submit(lambda _v: self._on_change())
        self.widgets["export"].on_clicked(self._save_png_pdf)
        self.widgets["load_settings"].on_clicked(lambda _evt: self.load_settings(self.widgets["settings_path"].text))
        self.widgets["save_settings"].on_clicked(lambda _evt: self.save_settings(self.widgets["settings_path"].text))

        self._push_active_annotation_state_to_controls()

    def _slider(self, x, y, w, label, lo, hi, val, step):
        ax = self.ctrl.add_axes([x, y, w, 0.055])
        sl = Slider(ax, "", lo, hi, valinit=val, valstep=step)
        ax.text(0.0, 1.18, label, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
        sl.label.set_fontsize(8)
        sl.valtext.set_fontsize(8)
        return sl

    def _on_change(self):
        if self._loading:
            return
        self.state.n_bins = int(self.widgets["bins"].val)
        self.state.symlog_decades = float(self.widgets["dec"].val)
        self.state.line_width = float(self.widgets["lw"].val)
        self.state.font_size = float(self.widgets["font"].val)
        self.state.subplot_wspace = float(self.widgets["wspace"].val)
        self.state.legend_on = str(self.widgets["legend_on"].value_selected)
        self.state.legend_loc = str(self.widgets["legend_loc"].value_selected)
        show_a, show_b = self.widgets["show_arrows"].get_status()
        self.state.show_arrow_a = bool(show_a)
        self.state.show_arrow_b = bool(show_b)
        self._read_active_annotation_controls_into_state()
        self.redraw()

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
        setattr(self.state, f"{p}_text_dx", float(self.widgets["ann_text_dx"].val))
        setattr(self.state, f"{p}_text_dy", float(self.widgets["ann_text_dy"].val))
        setattr(self.state, f"{p}_text_size", float(self.widgets["ann_text_size"].val))
        setattr(self.state, f"{p}_arrow_size", float(self.widgets["ann_arrow_size"].val))

    def _push_active_annotation_state_to_controls(self):
        p = self._active_arrow_prefix()
        self._loading = True
        self.widgets["arrow_text"].set_val(str(getattr(self.state, f"{p}_text")))
        self.widgets["ann_x"].set_val(float(getattr(self.state, f"{p}_x")))
        self.widgets["ann_y"].set_val(float(getattr(self.state, f"{p}_y")))
        self.widgets["ann_angle"].set_val(float(getattr(self.state, f"{p}_angle_deg")))
        self.widgets["ann_len"].set_val(float(getattr(self.state, f"{p}_len")))
        self.widgets["ann_text_dx"].set_val(float(getattr(self.state, f"{p}_text_dx")))
        self.widgets["ann_text_dy"].set_val(float(getattr(self.state, f"{p}_text_dy")))
        self.widgets["ann_text_size"].set_val(float(getattr(self.state, f"{p}_text_size")))
        self.widgets["ann_arrow_size"].set_val(float(getattr(self.state, f"{p}_arrow_size")))
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
        tag_text = panel_tag if not use_tex else rf"$\mathbf{{{panel_tag}}}$"
        ax.text(0.02, 0.98, tag_text, transform=ax.transAxes, ha="left", va="top", fontsize=fs + 1, fontweight="bold")

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
        use_tex = bool(self._exporting_tex)
        global_xmax = 1.0
        for name in self.groups:
            global_xmax = max(global_xmax, float(self.data3["groups"][name]["bin_edges"][-1]))
            global_xmax = max(global_xmax, float(self.data4["groups"][name]["bin_edges"][-1]))

        curves3, _ = self._build_curves(self.data3["groups"], xmax=global_xmax)
        curves4, _ = self._build_curves(self.data4["groups"], xmax=global_xmax)

        self._plot_panel(self.ax_left, curves3, "Witness CDF", "(a)", use_tex)
        self._plot_panel(self.ax_right, curves4, "Self-NN CDF", "(b)", use_tex)

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

        target = self.ax_left if self.state.legend_on == "left" else self.ax_right
        fs = int(round(self.state.font_size))
        if use_tex:
            title = "$\\mathbf{Group \\ | \\ Method}$"
        else:
            title = "Group / Method"
        target.legend(
            handles,
            labels,
            loc=self.state.legend_loc,
            fontsize=max(7, fs - 2),
            framealpha=0.85,
            title=title,
            title_fontsize=max(8, fs - 1),
            ncol=4,
            handlelength=2.0,
            columnspacing=0.9,
            borderpad=0.6,
            handletextpad=0.45,
            labelspacing=0.35,
        )

        self.fig.subplots_adjust(
            left=0.1,
            right=0.98,
            top=0.9,
            bottom=0.1,
            wspace=max(0.02, min(0.60, float(self.state.subplot_wspace))),
        )
        self.fig.canvas.draw_idle()

    def _sync_state_to_widgets(self):
        self._loading = True
        self.widgets["bins"].set_val(self.state.n_bins)
        self.widgets["dec"].set_val(self.state.symlog_decades)
        self.widgets["lw"].set_val(self.state.line_width)
        self.widgets["font"].set_val(self.state.font_size)
        self.widgets["wspace"].set_val(self.state.subplot_wspace)

        show_status = self.widgets["show_arrows"].get_status()
        if bool(show_status[0]) != bool(self.state.show_arrow_a):
            self.widgets["show_arrows"].set_active(0)
        if bool(show_status[1]) != bool(self.state.show_arrow_b):
            self.widgets["show_arrows"].set_active(1)

        leg_on_opts = ["left", "right"]
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
        payload = {"view": asdict(self.state)}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved settings -> {path}")

    def load_settings(self, path: str):
        path = (path or "").strip() or SETTINGS_FILE
        try:
            with open(path) as f:
                payload = json.load(f)
        except FileNotFoundError:
            print(f"Settings file '{path}' not found.")
            return

        view = payload.get("view", {}) if isinstance(payload, dict) else {}
        for key, val in view.items():
            if hasattr(self.state, key):
                setattr(self.state, key, val)

        try:
            fw = float(getattr(self.state, "fig_width", 15.0))
            fh = float(getattr(self.state, "fig_height", 7.0))
            self.fig.set_size_inches(fw, fh, forward=True)
        except Exception:
            pass

        self._sync_state_to_widgets()
        self.redraw()

    def _save_png_pdf(self, _evt=None):
        stem = self.state.export_stem or DEFAULT_STEM
        png_path = f"{stem}.png"
        pdf_path = f"{stem}.pdf"
        prev_usetex = plt.rcParams.get("text.usetex", False)
        self._exporting_tex = True
        plt.rcParams["text.usetex"] = True
        self.redraw()
        self.fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        self.fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        self._exporting_tex = False
        plt.rcParams["text.usetex"] = prev_usetex
        self.redraw()
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


def _resolve_self_nn_default() -> str:
    candidates = ["figure3_data_self_nn.npz"]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return "figure3_data_self_nn.npz"


def main():
    parser = argparse.ArgumentParser(description="Figure 3 combined witness (left) + self-NN (right) CDF overlay")
    parser.add_argument("--witness-data", default="figure3_data_witness.npz", help="Path to witness-set NPZ data")
    parser.add_argument("--self-nn-data", default=None, help="Path to self-NN NPZ data (default: figure3_data_self_nn.npz)")
    args = parser.parse_args()

    self_nn_path = args.self_nn_data or _resolve_self_nn_default()

    if not os.path.exists(args.witness_data):
        raise FileNotFoundError(f"Witness data file not found: {args.witness_data}")
    if not os.path.exists(self_nn_path):
        raise FileNotFoundError(f"Self-NN data file not found: {self_nn_path}")

    data3 = load_npz_overlay(args.witness_data)
    data4 = load_npz_overlay(self_nn_path)

    app = Figure3PlotApp(data3, data4)
    _ = app
    plt.show()


if __name__ == "__main__":
    main()
