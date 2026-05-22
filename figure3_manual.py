#!/usr/bin/env python
"""Figure 3 — Witness + Self-NN CDF overlay (manual / non-interactive)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
from matplotlib import ticker
from matplotlib.lines import Line2D
from figure_ui_common import add_panel_label

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS — edit these directly                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Output
STEM = "figure3_col"
DPI = 600

# Data files
WITNESS_FILE = "figure3_data_witness_col.npz"
SELF_NN_FILE = "figure3_data_self_nn_col.npz"

# Figure size (IUCr single column)
FIG_W = 3.25
FIG_H = 5.5

# Margins
MARGIN_LEFT = 0.18
MARGIN_RIGHT = 0.97
MARGIN_TOP = 0.97
MARGIN_BOTTOM = 0.09
SUBPLOT_WSPACE = 0.18

# Typography
FONT_SIZE = 8.0
TITLE_SIZE = 9.0
PANEL_LABEL_SIZE = 9.0
PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
LEGEND_SIZE = 7.0
LINE_WIDTH = 1.0

# CDF scale
N_BINS = 240
SYMLOG_DECADES = 3.0

# Legend
LEGEND_ON = "top"      # "top" or "bottom" panel
LEGEND_LOC = "lower right"

# Arrow A (panel a — witness)
SHOW_ARROW_A = True
ARROW_A_TEXT = "fewer gaps"
ARROW_A_X = 0.90
ARROW_A_Y = 0.88
ARROW_A_ANGLE_DEG = 135.0
ARROW_A_LEN = 0.12
ARROW_A_TEXT_DX = 0.02
ARROW_A_TEXT_DY = 0.02
ARROW_A_TEXT_SIZE = 10.0
ARROW_A_ARROW_SIZE = 1.4

# Arrow B (panel b — self-NN)
SHOW_ARROW_B = True
ARROW_B_TEXT = "collision-free"
ARROW_B_X = 0.15
ARROW_B_Y = 0.55
ARROW_B_ANGLE_DEG = -45.0
ARROW_B_LEN = 0.12
ARROW_B_TEXT_DX = 0.02
ARROW_B_TEXT_DY = 0.00
ARROW_B_TEXT_SIZE = 10.0
ARROW_B_ARROW_SIZE = 1.4

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CDF HALF-LOG SCALE                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class _CDFHalfLogTransform(mtransforms.Transform):
    input_dims = output_dims = 1

    def __init__(self, decades):
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

    def __init__(self, decades):
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  THEME                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ORDERED_GROUPS = ["C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6", "T", "O", "I"]

TEX_LABEL = {
    "C2": r"$C_2$", "C3": r"$C_3$", "C4": r"$C_4$", "C6": r"$C_6$",
    "D2": r"$D_2$", "D3": r"$D_3$", "D4": r"$D_4$", "D6": r"$D_6$",
    "T": r"$T$", "O": r"$O$", "I": r"$I$",
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_npz_overlay(path):
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
        for key in ("bin_edges", "counts_kr", "counts_rej", "N_kr", "N_rej",
                     "N_witness", "mean_kr", "mean_rej"):
            full = f"{name}_{key}"
            if full in raw:
                val = raw[full]
                g[key] = val.item() if getattr(val, "ndim", 0) == 0 else val
        out["groups"][name] = g
    return out


def rebin_histogram(fine_counts, fine_edges, coarse_edges):
    mids = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    idx = np.clip(np.digitize(mids, coarse_edges) - 1, 0, len(coarse_edges) - 2)
    return np.bincount(idx, weights=fine_counts.astype(float), minlength=len(coarse_edges) - 1).astype(float)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_curves(groups_data, groups, xmax=None):
    if xmax is None:
        xmax = 1.0
    else:
        xmax = max(1.0, float(xmax))
    for name in groups:
        edges = groups_data[name]["bin_edges"]
        xmax = max(xmax, float(edges[-1]))
    bins = np.linspace(0.0, xmax, max(20, N_BINS) + 1)
    curves = []
    for name in groups:
        g = groups_data[name]
        fine_edges = g["bin_edges"]
        c_kr = rebin_histogram(g["counts_kr"], fine_edges, bins)
        c_rej = rebin_histogram(g["counts_rej"], fine_edges, bins)
        curves.append((name, "KR", c_kr, bins))
        curves.append((name, "Rej", c_rej, bins))
    return curves


def plot_panel(ax, curves, panel_tag=None, group_colors=None, use_tex=True):
    fs = int(round(FONT_SIZE))
    for name, method, counts, edges in curves:
        total = float(np.sum(counts))
        if total <= 0:
            continue
        y = np.concatenate([[0.0], np.cumsum(counts) / total])
        color = group_colors[name]
        ls = "-" if method == "KR" else (0, (5, 3))
        ax.plot(edges, y, color=color, linestyle=ls, linewidth=LINE_WIDTH)

    decades = float(np.clip(SYMLOG_DECADES, 1.0, 7.0))
    ax.set_yscale("cdfhalflog", decades=decades)
    ax.set_ylim(10.0 ** (-decades), 1.0 - 10.0 ** (-decades))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.35)
    ax.axhline(0.5, color="k", linewidth=1.0, alpha=0.9)

    if use_tex:
        ax.set_xlabel(r"$\textrm{Disorientation angle } (^\circ)$", fontsize=fs)
        ax.set_ylabel(r"$\textrm{CDF}$", fontsize=fs)
    else:
        ax.set_xlabel(r"Disorientation angle $(^\circ)$", fontsize=fs)
        ax.set_ylabel("CDF", fontsize=fs)
    ax.tick_params(labelsize=max(7, fs - 1))
    if panel_tag:
        add_panel_label(ax, panel_tag, x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=use_tex)


def draw_arrow(ax, show, x, y, angle_deg, length, text, text_dx, text_dy,
               text_size, arrow_size, use_tex=True):
    if not show:
        return
    theta = np.deg2rad(float(angle_deg))
    dx = float(length) * np.cos(theta)
    dy = float(length) * np.sin(theta)
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=max(0.5, float(arrow_size)),
                                color="black", mutation_scale=8.0 * max(0.5, float(arrow_size))))
    if use_tex:
        text = rf"$\textrm{{{text}}}$"
    ax.text(x + text_dx, y + text_dy, text, transform=ax.transAxes,
            ha="left", va="center", fontsize=max(6.0, float(text_size)))


def legend_payload(groups, group_colors, use_tex=True):
    col1 = ["C2", "C3", "C4", "C6"]
    col2 = ["D2", "D3", "D4", "D6"]
    col3 = ["T", "O", "I", "_blank"]
    col4 = ["_method_kr", "_method_rej", "_blank", "_blank"]
    legend_items = col1 + col2 + col3 + col4
    handles, labels = [], []
    for key in legend_items:
        if key == "_blank":
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
        elif key == "_method_kr":
            handles.append(Line2D([0], [0], color="k", lw=LINE_WIDTH, ls="-"))
            labels.append(r"$\textrm{KR}$" if use_tex else "KR")
        elif key == "_method_rej":
            handles.append(Line2D([0], [0], color="k", lw=LINE_WIDTH, ls=(0, (5, 3))))
            labels.append(r"$\textrm{Rej}$" if use_tex else "Rej")
        elif key not in groups:
            handles.append(Line2D([], [], alpha=0.0, lw=0.0))
            labels.append(" ")
        else:
            handles.append(Line2D([0], [0], color=group_colors[key], lw=LINE_WIDTH, ls="-"))
            labels.append(TEX_LABEL[key])
    return handles, labels


def main():
    data3 = load_npz_overlay(WITNESS_FILE)
    data4 = load_npz_overlay(SELF_NN_FILE)
    groups = [n for n in ORDERED_GROUPS if n in data3["groups"] and n in data4["groups"]]

    cmap = plt.get_cmap("tab20")
    group_colors = {name: cmap(i) for i, name in enumerate(ORDERED_GROUPS)}

    # Global xmax
    global_xmax = 1.0
    for name in groups:
        global_xmax = max(global_xmax, float(data3["groups"][name]["bin_edges"][-1]))
        global_xmax = max(global_xmax, float(data4["groups"][name]["bin_edges"][-1]))

    curves3 = build_curves(data3["groups"], groups, xmax=global_xmax)
    curves4 = build_curves(data4["groups"], groups, xmax=global_xmax)

    with plt.rc_context({"text.usetex": True}):
        fig, (ax_left, ax_right) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))

        plot_panel(ax_left, curves3, "(a)", group_colors)
        plot_panel(ax_right, curves4, "(b)", group_colors)

        draw_arrow(ax_left, SHOW_ARROW_A, ARROW_A_X, ARROW_A_Y, ARROW_A_ANGLE_DEG,
                   ARROW_A_LEN, ARROW_A_TEXT, ARROW_A_TEXT_DX, ARROW_A_TEXT_DY,
                   ARROW_A_TEXT_SIZE, ARROW_A_ARROW_SIZE)
        draw_arrow(ax_right, SHOW_ARROW_B, ARROW_B_X, ARROW_B_Y, ARROW_B_ANGLE_DEG,
                   ARROW_B_LEN, ARROW_B_TEXT, ARROW_B_TEXT_DX, ARROW_B_TEXT_DY,
                   ARROW_B_TEXT_SIZE, ARROW_B_ARROW_SIZE)

        handles, labels = legend_payload(groups, group_colors)
        target = ax_left if LEGEND_ON == "top" else ax_right
        target.legend(handles, labels, loc=LEGEND_LOC, fontsize=max(6, LEGEND_SIZE),
                      framealpha=0.85,
                      title=r"$\mathbf{Group \ | \ Method}$",
                      title_fontsize=max(6, LEGEND_SIZE + 1),
                      ncol=4, handlelength=2.0, columnspacing=0.9,
                      borderpad=0.6, handletextpad=0.45, labelspacing=0.35)

        fig.subplots_adjust(left=MARGIN_LEFT, right=MARGIN_RIGHT, top=MARGIN_TOP,
                            bottom=MARGIN_BOTTOM,
                            hspace=max(0.02, min(0.80, SUBPLOT_WSPACE)))

        for ext in ("png", "pdf"):
            path = f"{STEM}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"[export] {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
