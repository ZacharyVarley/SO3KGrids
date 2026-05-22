#!/usr/bin/env python3
"""Export IUCR submission figures with descriptive, ordered file names.

Uses saved layout JSON in ``figures/settings/``. Typography and line weights
follow the IUCr artwork guide. Does not regenerate numerical curve/grid data
except for the two 3D panels built directly from the maintained figure modules.

Usage:
    python -m iucr.export_panels
    python -m iucr.export_panels --only 2 4 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from figures.common import (
    IUCR_COL_W,
    IUCR_COL_W_MM,
    IUCR_DPI,
    IUCR_EXPORT_FORMATS,
    IUCR_FULL_W,
    IUCR_LINE_LW_STD,
    IUCR_MARGINS,
    IUCR_MARKER_SIZE_STD,
    IUCR_PANEL_H,
    IUCR_FULL_W_MM,
    IUCR_TARGET_TEXT_MM,
    apply_iucr_artwork_rcparams,
    apply_iucr_typography,
    apply_iucr_typography_full_page,
    clamp_iucr_line_lw,
    export_iucr_figure,
    iucr_artwork_style,
    iucr_font_sizes,
    iucr_font_sizes_full_page,
    pt_to_mm,
    scale_scatter_point_size,
    validate_iucr_fontsizes,
)
_ROOT_PATH = Path(_ROOT)
_FIGURES_DIR = _ROOT_PATH / "figures"
_FIGURE_SETTINGS_DIR = _FIGURES_DIR / "settings"
_FIGURE_DATA_DIR = _FIGURES_DIR / "data"
_IUCR_PANELS_DIR = _ROOT_PATH / "iucr_panels"

FIGURE1_SETTINGS = _FIGURE_SETTINGS_DIR / "so3t_rejection_vs_kr.json"
FIGURE2_SETTINGS = _FIGURE_SETTINGS_DIR / "cubochoric_anisotropy.json"
FIGURE3_SETTINGS = _FIGURE_SETTINGS_DIR / "nn_cdf_overlays.json"
FIGURE4_SETTINGS = _FIGURE_SETTINGS_DIR / "grid_method_comparison.json"
FIGURE5_SETTINGS = _FIGURE_SETTINGS_DIR / "thomson_relaxation.json"
FIGURE6_SETTINGS = _FIGURE_SETTINGS_DIR / "laue_o_pc_vs_fcc.json"
FIGURE3_WITNESS_DATA = _FIGURE_DATA_DIR / "figure3_witness_nn.npz"
FIGURE3_SELF_NN_DATA = _FIGURE_DATA_DIR / "figure3_self_nn.npz"

# Reference layouts for scaling 3D scatter sizes to full-page width.
_FIG1_REF_WIDTH_IN = 12.8
_FIG1_REF_HEIGHT_IN = 6.6
_FIG1_REF_POINT_SIZE = 28.0
_FIG6_REF_WIDTH_IN = 14.6
_FIG6_REF_HEIGHT_IN = 8.8
_FIG6_REF_POINT_SIZE = 7.0
from iucr import stems

_ARTWORK = iucr_artwork_style()

_DEFAULT_OUT_DIR = str(_IUCR_PANELS_DIR)

SETTINGS = {
    1: str(FIGURE1_SETTINGS),
    2: str(FIGURE2_SETTINGS),
    3: str(FIGURE3_SETTINGS),
    4: str(FIGURE4_SETTINGS),
    5: str(FIGURE5_SETTINGS),
    6: str(FIGURE6_SETTINGS),
}


def _ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _remove_legacy_stems(out_dir: str) -> None:
    """Drop pre-rename ``figureN_*`` assets so the output folder stays unambiguous."""
    import glob

    for path in glob.glob(os.path.join(out_dir, "figure*")):
        try:
            os.remove(path)
        except OSError:
            pass


def _stem(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name)


def _single_panel_axes(*, height: float = IUCR_PANEL_H):
    """One axes with the same margins as column interactive figures."""
    fig = plt.figure(figsize=(IUCR_COL_W, height))
    ax = fig.add_axes(
        [
            IUCR_MARGINS["left"],
            IUCR_MARGINS["bottom"],
            IUCR_MARGINS["right"] - IUCR_MARGINS["left"],
            IUCR_MARGINS["top"] - IUCR_MARGINS["bottom"],
        ]
    )
    return fig, ax


def _load_figure2_settings(path: str):
    from figures import cubochoric_anisotropy as f2

    s = f2.Settings()
    if os.path.exists(path):
        with open(path) as fh:
            d = json.load(fh)
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        print(f"[settings] Loaded {path}")
    apply_iucr_typography(s)
    s.marker_size = IUCR_MARKER_SIZE_STD
    s.line_width = IUCR_LINE_LW_STD
    return s


def export_so3t_rejection_kr(out_dir: str) -> list[str]:
    """SO(3)/T: rejection sampling vs KR (combined 3D panels)."""
    import numpy as np
    import torch

    from figures import so3t_rejection_vs_kr as f1

    torch.manual_seed(f1.SEED)
    np.random.seed(f1.SEED)
    s = f1.Settings()
    path = SETTINGS[1]
    if os.path.exists(path):
        with open(path) as fh:
            d = json.load(fh)
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        print(f"[settings] Loaded {path}")
    idx = int(np.clip(s.n_target_index, 0, len(f1.N_TARGET_FZ_OPTIONS) - 1))
    n_target = f1.N_TARGET_FZ_OPTIONS[idx]
    print(f"[figure1] Building dataset N_FZ={n_target:,} (index {idx})")
    datasets = [f1.build_dataset(n_target, seed=f1.SEED + 1000 * idx)]
    s.n_target_index = 0
    design_w = float(s.fig_width)
    s.panel_layout = "side_by_side"
    s.fig_width = IUCR_FULL_W
    s.fig_height = IUCR_FULL_W * (_FIG1_REF_HEIGHT_IN / _FIG1_REF_WIDTH_IN)
    s.margin_left = 0.04
    s.margin_right = 0.90
    s.margin_top = 0.94
    s.margin_bottom = 0.10
    s.gs_wspace = 0.06
    s.gs_hspace = 0.0
    s.cbar_ratio = 0.08
    s.title_pad = max(float(s.title_pad), 10.0)
    apply_iucr_typography_full_page(s, target_mm=IUCR_TARGET_TEXT_MM)
    s.point_size = scale_scatter_point_size(
        s.point_size,
        design_width_in=design_w,
        target_width_in=IUCR_FULL_W,
        reference_point_size=_FIG1_REF_POINT_SIZE,
        reference_width_in=_FIG1_REF_WIDTH_IN,
    )
    s.wire_thickness = clamp_iucr_line_lw(s.wire_thickness, name="wire_thickness")
    print(
        f"[figure1] side-by-side {s.fig_width:.2f}×{s.fig_height:.2f} in, "
        f"point_size={s.point_size:.2f}"
    )

    app = f1.FigureInteractive(datasets, s, create_controls=False)
    app._show_panel_labels = False
    stems_out: list[str] = []
    with plt.rc_context({"text.usetex": True}):
        app._exporting_tex = True
        app.redraw(full_colorbar=True, show_panel_labels=False)
        stem = _stem(out_dir, stems.SO3T_REJECTION_VS_KR_FZ)
        export_iucr_figure(app.fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        app._exporting_tex = False
        plt.close(app.fig)
    return stems_out


def export_cubochoric_anisotropy(out_dir: str) -> list[str]:
    from figures import cubochoric_anisotropy as f2

    data = f2.load_data(f2.DATA_FILE)
    grouped = f2.group_results(data)
    s = _load_figure2_settings(SETTINGS[2])
    stems_out = []

    with plt.rc_context({"text.usetex": True}):
        fig, ax = _single_panel_axes()
        f2.draw_e3_ratio_panel(ax, grouped, s, use_tex=True, panel_tag=None)
        stem = _stem(out_dir, stems.E3_RATIO_VS_NS3)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)

        fig, ax = _single_panel_axes()
        f2.draw_cuz_panel(ax, grouped, s, use_tex=True, panel_tag=None)
        stem = _stem(out_dir, stems.CUZ_STAR_VS_CUXY)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)
    return stems_out


def _figure3_add_legend(ax, app, use_tex: bool) -> None:
    handles, labels = app._legend_payload()
    fs = int(round(app.state.legend_size))
    title = "$\\mathbf{Group \\ | \\ Method}$" if use_tex else "Group / Method"
    ax.legend(
        handles,
        labels,
        loc=app.state.legend_loc,
        fontsize=max(6, fs),
        framealpha=1.0,
        title=title,
        title_fontsize=max(6, fs + 1),
        ncol=4,
        handlelength=2.0,
        columnspacing=0.9,
        borderpad=0.6,
        handletextpad=0.45,
        labelspacing=0.35,
    )


def export_nn_cdf_panels(out_dir: str) -> list[str]:
    from figures import nn_cdf_overlays as f3
    from figures.common import load_dataclass_settings

    witness_path = str(FIGURE3_WITNESS_DATA)
    self_nn_path = str(FIGURE3_SELF_NN_DATA)
    data3 = f3.load_npz_overlay(witness_path)
    data4 = f3.load_npz_overlay(self_nn_path)

    app = f3.Figure3PlotApp(data3, data4)
    if os.path.exists(SETTINGS[3]):
        load_dataclass_settings(SETTINGS[3], app.state, wrapper_key="view")
        print(f"[settings] Loaded {SETTINGS[3]}")
    apply_iucr_typography(app.state)
    app.state.line_width = IUCR_LINE_LW_STD

    app._exporting_tex = True
    global_xmax = 1.0
    for name in app.groups:
        global_xmax = max(global_xmax, float(app.data3["groups"][name]["bin_edges"][-1]))
        global_xmax = max(global_xmax, float(app.data4["groups"][name]["bin_edges"][-1]))
    app._cached_curves3, _ = app._build_curves(app.data3["groups"], xmax=global_xmax)
    app._cached_curves4, _ = app._build_curves(app.data4["groups"], xmax=global_xmax)

    stems_out = []
    with plt.rc_context({"text.usetex": True}):
        fig, ax = _single_panel_axes()
        app._plot_panel(ax, app._cached_curves3, "", None, True)
        app._draw_arrow_annotation(
            ax,
            app.state.show_arrow_a,
            app.state.arrow_a_x,
            app.state.arrow_a_y,
            app.state.arrow_a_angle_deg,
            app.state.arrow_a_len,
            app.state.arrow_a_text,
            app.state.arrow_a_text_dx,
            app.state.arrow_a_text_dy,
            app.state.arrow_a_text_size,
            app.state.arrow_a_arrow_size,
            True,
        )
        if app.state.legend_on == "top":
            _figure3_add_legend(ax, app, True)
        stem = _stem(out_dir, stems.WITNESS_NN_CDF)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)

        fig, ax = _single_panel_axes()
        app._plot_panel(ax, app._cached_curves4, "", None, True)
        app._draw_arrow_annotation(
            ax,
            app.state.show_arrow_b,
            app.state.arrow_b_x,
            app.state.arrow_b_y,
            app.state.arrow_b_angle_deg,
            app.state.arrow_b_len,
            app.state.arrow_b_text,
            app.state.arrow_b_text_dx,
            app.state.arrow_b_text_dy,
            app.state.arrow_b_text_size,
            app.state.arrow_b_arrow_size,
            True,
        )
        if app.state.legend_on != "top":
            _figure3_add_legend(ax, app, True)
        stem = _stem(out_dir, stems.SELF_NN_CDF)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)
    plt.close("all")
    return stems_out


def export_grid_method_comparison(out_dir: str) -> list[str]:
    from figures import grid_method_comparison as f4

    f4.load_col_settings(SETTINGS[4])
    fonts = iucr_font_sizes()
    f4.LABEL_SIZE = fonts["label"]
    f4.TICK_SIZE = fonts["tick"]
    f4.MINOR_TICK_SIZE = fonts["minor_tick"]
    f4.LEGEND_SIZE = fonts["legend"]
    f4.LW_KR = _ARTWORK["line_lw"]
    f4.MARKER_SIZE = _ARTWORK["marker_size"]
    f4.MARKER_EDGE_LW = _ARTWORK["marker_edgewidth"]

    results = f4.load_data()
    visible = [k for k in f4.PLOT_ORDER if k in results and f4.SHOW.get(k, True)]
    visible_set = set(visible)

    stems_out = []
    with plt.rc_context({"text.usetex": True}):
        for metric, pk, _ in f4.PANELS_COL:
            fig, ax = _single_panel_axes()
            f4.draw_standard_panel(ax, metric, pk, visible, results)
            stem = _stem(
                out_dir,
                stems.E3_RATIO_GRID_METHODS if pk == "E3" else stems.CR_RATIO_GRID_METHODS,
            )
            export_iucr_figure(fig, stem, dpi=IUCR_DPI)
            stems_out.append(stem)
            plt.close(fig)

        fig = plt.figure(figsize=(IUCR_COL_W, 1.05))
        leg_ax = fig.add_axes([0.05, 0.12, 0.90, 0.76])
        f4.draw_legend(leg_ax, visible_set)
        stem = _stem(out_dir, stems.GRID_METHODS_LEGEND)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)

        fig = plt.figure(figsize=(IUCR_COL_W, 5.5), dpi=150)
        gs = GridSpec(
            3, 1,
            left=IUCR_MARGINS["left"],
            right=IUCR_MARGINS["right"],
            bottom=IUCR_MARGINS["bottom"],
            top=IUCR_MARGINS["top"],
            hspace=0.34,
            height_ratios=[1, 1, 0.28],
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        for idx, (metric, pk, _) in enumerate(f4.PANELS_COL):
            f4.draw_standard_panel(axes[idx], metric, pk, visible, results)
        f4.draw_legend(axes[2], visible_set)
        stem = _stem(out_dir, stems.GRID_METHODS_STACKED)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)
    return stems_out


def export_thomson_relaxation(out_dir: str) -> list[str]:
    from figures import thomson_relaxation as f5

    f5.load_col_settings(SETTINGS[5])
    fonts = iucr_font_sizes()
    f5.LABEL_SIZE = fonts["label"]
    f5.TICK_SIZE = fonts["tick"]
    f5.LEGEND_SIZE = fonts["legend"]
    f5.LINE_LW = _ARTWORK["line_lw"]

    data = f5.load_data()
    visible = [k for k in f5._series_keys_from_data(data) if f5.SHOW.get(k, True)]
    max_x = f5._max_iteration(data, visible)

    stems_out = []
    with plt.rc_context({"text.usetex": True}):
        fig, ax = _single_panel_axes()
        f5.plot_e3_panel(ax, data, visible, max_x=max_x)
        stem = _stem(out_dir, stems.THOMSON_E3_RATIO_VS_ITER)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)

        fig, ax = _single_panel_axes()
        f5.plot_cr_panel(ax, data, visible, max_x=max_x)
        stem = _stem(out_dir, stems.COVERING_RADIUS_VS_ITER)
        export_iucr_figure(fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(fig)
    return stems_out


def export_laue_o_pc_vs_fcc_kr(out_dir: str) -> list[str]:
    """Primitive vs FCC KR grids for Laue O (2×3 combined figure)."""
    import torch

    from figures import laue_o_pc_vs_fcc as f6

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match = f6.find_best_match(h_min=1, h_max=30, n_s3_min=1_000, n_s3_max=200_000)
    print(
        f"[figure6] Matched N_S^3: primitive={match.n_s3_pc:,} (h={match.h_pc}), "
        f"fcc={match.n_s3_fcc:,} (h={match.h_fcc})"
    )
    cu_pc = f6.cubochoric_primitive_grid(match.h_pc, dev)
    cu_fcc = f6.cubochoric_fcc_grid(match.h_fcc, dev)
    row_top = f6.build_row(cu_pc, f6.LAUE_O)
    row_bot = f6.build_row(cu_fcc, f6.LAUE_O)

    s = f6.Settings()
    path = SETTINGS[6]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            s = f6.settings_from_raw(json.load(fh), s)
        print(f"[settings] Loaded {path}")
    apply_iucr_typography_full_page(s, field_map={"text_size": "label"})
    # Interactive UI uses large preview markers; export uses the maintained Figure 6 reference scale.
    s.point_size = scale_scatter_point_size(
        _FIG6_REF_POINT_SIZE,
        design_width_in=_FIG6_REF_WIDTH_IN,
        target_width_in=IUCR_FULL_W,
    )
    s.wire_thickness = clamp_iucr_line_lw(s.wire_thickness, name="wire_thickness")
    print(f"[figure6] point_size={s.point_size:.2f} (full-page scatter scale)")

    fig_h = IUCR_FULL_W * (_FIG6_REF_HEIGHT_IN / _FIG6_REF_WIDTH_IN)
    ui = f6.Figure7Interactive(
        row_top,
        row_bot,
        match,
        s,
        figsize=(IUCR_FULL_W, fig_h),
        dpi=150,
        create_controls=False,
        show_panel_labels=False,
    )

    stems_out = []
    with plt.rc_context({"text.usetex": True}):
        ui.redraw(full_cbar=True, use_tex=True)
        stem = _stem(out_dir, stems.LAUE_O_PC_VS_FCC_KR_GRIDS)
        export_iucr_figure(ui.fig, stem, dpi=IUCR_DPI)
        stems_out.append(stem)
        plt.close(ui.fig)
    return stems_out


_EXPORTERS = {
    1: ("SO(3)/T rejection vs KR", export_so3t_rejection_kr),
    2: ("Cubochoric axial anisotropy", export_cubochoric_anisotropy),
    3: ("NN CDF panels", export_nn_cdf_panels),
    4: ("Grid-method comparison", export_grid_method_comparison),
    5: ("Thomson relaxation", export_thomson_relaxation),
    6: ("Laue O PC vs FCC KR", export_laue_o_pc_vs_fcc_kr),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export IUCR figures with descriptive stems (see iucr/stems.py).",
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        choices=sorted(_EXPORTERS),
        metavar="N",
        help="Export subset: 1=SO(3)/T, 2=anisotropy, 3=CDFs, 4=grids, 5=Thomson, 6=PC/FCC",
    )
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    out_dir = _ensure_out_dir(os.path.abspath(args.out_dir))
    _remove_legacy_stems(out_dir)
    apply_iucr_artwork_rcparams()
    fonts = iucr_font_sizes()
    fonts_full = iucr_font_sizes_full_page()
    validate_iucr_fontsizes(**fonts)
    validate_iucr_fontsizes(**fonts_full)
    print(f"IUCr column width: {IUCR_COL_W_MM:.1f} mm ({IUCR_COL_W:.3f} in)")
    print(
        f"IUCr full-page width: {IUCR_FULL_W_MM:.1f} mm ({IUCR_FULL_W:.3f} in), "
        f"target label height {IUCR_TARGET_TEXT_MM:.1f} mm"
    )
    print(f"Export formats: {', '.join(IUCR_EXPORT_FORMATS)} @ {IUCR_DPI} d.p.i. (PNG)")
    print("IUCr typography (printed height at column width):")
    for name, pt in sorted(fonts.items()):
        print(f"  {name}: {pt:.1f} pt → {pt_to_mm(pt):.2f} mm")
    print(
        f"Standard artwork: line {IUCR_LINE_LW_STD} pt, "
        f"markers {IUCR_MARKER_SIZE_STD} pt, marker edge {_ARTWORK['marker_edgewidth']} pt"
    )
    print("Stems: content-based names in iucr/stems.py (ordered 01–12).")
    print("Layout from figures/settings/*.json.")

    which = args.only or sorted(_EXPORTERS)
    all_stems: list[str] = []
    for n in which:
        label, fn = _EXPORTERS[n]
        print(f"\n=== {n}: {label} ===")
        all_stems.extend(fn(out_dir))

    print(f"\nDone. {len(all_stems)} stem(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
