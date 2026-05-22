#!/usr/bin/env python3
"""
Interactive Figure 1 (SO(3)/T): rejection vs KR.

Controls (all in a scrollable right-side panel):
  Sliders:  font size, dataset index, elevation, azimuth, zoom, zoom-centre-z,
            cut-plane angle, point size, boundary threshold, face alpha,
            face-colour hue + lightness
  Toggle grid (2×8):  Faces, Edges, F0–F7, xyz≥0, Side
  Buttons/text:  save/load settings (JSON), save figure (PNG/PDF)

Dependencies:
  orientation_ops, laue_ops, grid_FZ, riesz_energy, batlow
"""

import colorsys
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from batlow import batlow_cmap
from src.grid_FZ import cu_kr_grid
from src.laue_ops import laue_dist_to_fz_boundary, ori_to_fz_laue
from src.orientation_ops import cu2ho, ho2qu, qu2rf, qu_std
from src.riesz_energy import riesz_energies_fused
from figures.common import (
    ControlWindow,
    PAPER_COL_W,
    PAPER_DPI,
    PAPER_FONT,
    add_panel_label,
    export_figure_with_tex,
)

# ── Config ────────────────────────────────────────────────────────────────────
LAUE_ID = 10  # id for T (tetrahedral Laue group "32")
LAUE_CARD = 12  # |K| for T
SEED = 42
DEVICE = torch.device("cpu")
DPI = PAPER_DPI
_HERE = Path(__file__).resolve().parent
DEFAULT_STEM = "so3t_rejection_vs_kr"
DEFAULT_SETTINGS_JSON = str(_HERE / "settings" / "so3t_rejection_vs_kr.json")
A_CU = (math.pi ** (2.0 / 3.0)) / 2.0

N_TARGET_FZ_OPTIONS: List[int] = [
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    20000,
    30000,
    40000,
]

# ── Octahedron FZ geometry ────────────────────────────────────────────────────
OCT_VERTS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=float,
)

OCT_FACES = [
    (0, 2, 4),
    (0, 4, 3),
    (0, 3, 5),
    (0, 5, 2),
    (1, 4, 2),
    (1, 3, 4),
    (1, 5, 3),
    (1, 2, 5),
]

OCT_EDGES = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
]

# Legacy face-colour presets (for loading old JSON files)
_FACE_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "Light gray": (0.92, 0.92, 0.92),
    "White": (1.0, 1.0, 1.0),
    "Off-white": (0.98, 0.98, 0.95),
    "Light blue": (0.88, 0.92, 0.98),
    "Light yellow": (0.99, 0.98, 0.88),
    "Pale pink": (0.98, 0.93, 0.93),
}


# ── Colour helpers ────────────────────────────────────────────────────────────
def hls_to_rgb(h: float, l: float, s: float = 1.0) -> Tuple[float, float, float]:
    return colorsys.hls_to_rgb(float(h), float(l), float(s))


def rgb_to_hl(r: float, g: float, b: float) -> Tuple[float, float]:
    h, l, _s = colorsys.rgb_to_hls(float(r), float(g), float(b))
    return h, l


# ── Settings ──────────────────────────────────────────────────────────────────
@dataclass
class Settings:
    n_target_index: int = 0
    view_elev: float = 30.0
    view_azim: float = -90.0
    zoom_r: float = 0.8
    zoom_center_z: float = 0.0
    cut_angle_deg: float = 90.0
    cut_plane_offset: float = 0.0
    keep_side_sign: float = 1.0
    font_size: float = PAPER_FONT["label"]
    title_size: float = PAPER_FONT["title"]
    subtitle_size: float = PAPER_FONT["subtitle"]
    panel_label_size: float = PAPER_FONT["panel_label"]
    panel_label_x: float = 0.02
    panel_label_y: float = 0.98
    subtitle_x: float = 0.52
    subtitle_y: float = 0.98
    cbar_font_size: float = PAPER_FONT["cbar"]
    cbar_width: float = 0.09
    point_size: float = 28.0
    boundary_thresh_deg: float = 10.0
    only_octant_xyz_pos: bool = False
    draw_faces: bool = True
    draw_face_edges: bool = True
    wire_thickness: float = 1.0
    face_alpha: float = 1.0
    face_color_hue: float = 0.0
    face_color_lightness: float = 0.92
    face_enabled: Tuple[bool, ...] = (
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    )
    p_lo: float = 1.0
    p_hi: float = 99.0
    panel_layout: str = "column"  # "column" (stacked) or "side_by_side" (like figure1.py)
    fig_width: float = PAPER_COL_W
    fig_height: float = 5.5
    # subplot margins
    margin_left: float = 0.02
    margin_right: float = 0.82
    margin_top: float = 0.96
    margin_bottom: float = 0.02
    gs_hspace: float = 0.12
    gs_wspace: float = 0.10
    cbar_ratio: float = 0.09
    title_pad: float = 2.0


# ── Grid helpers ──────────────────────────────────────────────────────────────
def semi_edge_for_target_fz(n_target_fz: int, laue_card: int) -> Tuple[int, int]:
    semi_kr = int(round((n_target_fz ** (1 / 3) - 1) / 2))
    semi_rej = int(round(((n_target_fz * laue_card) ** (1 / 3) - 1) / 2))
    print(f"semi_rej: {semi_rej}, semi_kr: {semi_kr} for n_target_fz: {n_target_fz}")
    return semi_rej, semi_kr


# ── 3-D plot helpers ──────────────────────────────────────────────────────────
def style_3d(ax):
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a.pane.set_visible(False)
        except Exception:
            pass
    ax.grid(False)


def set_equal_zoom(ax, r: float):
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))


def rotation_normal_z(deg: float) -> np.ndarray:
    th = math.radians(deg)
    return np.array([math.cos(th), math.sin(th), 0.0])


def cut_mask(rf, n, sign, offset=0.0):
    return (rf @ n - offset) * sign < 0.0


def octant_pos(rf):
    return (rf[:, 0] >= 0) & (rf[:, 1] >= 0) & (rf[:, 2] >= 0)


def _view_plane_normal(elev: float, azim: float) -> np.ndarray:
    elev_r = math.radians(elev)
    azim_r = math.radians(azim)
    return np.array(
        [
            math.cos(elev_r) * math.cos(azim_r),
            math.cos(elev_r) * math.sin(azim_r),
            math.sin(elev_r),
        ],
        dtype=np.float64,
    )


def _split_segments_by_view(
    segments: List[np.ndarray], elev: float, azim: float, eps: float = 1e-12
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    normal = _view_plane_normal(elev, azim)
    behind: List[np.ndarray] = []
    front: List[np.ndarray] = []

    for seg in segments:
        if seg.shape[0] < 2:
            continue
        midpoint = 0.5 * (seg[0] + seg[-1])
        side = float(np.dot(midpoint, normal))
        if side > eps:
            front.append(seg)
        else:
            behind.append(seg)

    return behind, front


# ── Dataset ───────────────────────────────────────────────────────────────────
@dataclass
class Dataset:
    n_target_fz: int
    rf_rej_in: np.ndarray
    nn_rej_in: np.ndarray
    bd_rej_in: np.ndarray
    rf_kr_in: np.ndarray
    nn_kr_in: np.ndarray
    bd_kr_in: np.ndarray


@torch.no_grad()
def build_dataset(n_target_fz: int, seed: int) -> Dataset:
    torch.manual_seed(seed)
    np.random.seed(seed)
    semi_rej, semi_kr = semi_edge_for_target_fz(n_target_fz, LAUE_CARD)

    lin = torch.linspace(-A_CU, A_CU, 2 * semi_rej + 1, device=DEVICE)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    cu = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
    qu_raw = qu_std(ho2qu(cu2ho(cu)))
    qu_fz = ori_to_fz_laue(qu_raw, LAUE_ID)
    in_fz = (qu_raw * qu_fz).sum(-1).abs() > (1.0 - 1e-4)
    qu_rej = qu_raw[in_fz]

    qu_kr = cu_kr_grid(semi_kr, LAUE_ID, DEVICE)
    _, _, _, nn_rej = riesz_energies_fused(qu_rej, LAUE_ID, return_nn=True)
    _, _, _, nn_kr = riesz_energies_fused(qu_kr, LAUE_ID, return_nn=True)
    bd_rej = laue_dist_to_fz_boundary(qu_rej, LAUE_ID)
    bd_kr = laue_dist_to_fz_boundary(qu_kr, LAUE_ID)

    return Dataset(
        n_target_fz=n_target_fz,
        rf_rej_in=qu2rf(qu_rej).cpu().numpy(),
        nn_rej_in=nn_rej.cpu().numpy(),
        bd_rej_in=bd_rej.cpu().numpy(),
        rf_kr_in=qu2rf(qu_kr).cpu().numpy(),
        nn_kr_in=nn_kr.cpu().numpy(),
        bd_kr_in=bd_kr.cpu().numpy(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive renderer
# ══════════════════════════════════════════════════════════════════════════════
class FigureInteractive:

    # ── construction ──────────────────────────────────────────────
    def __init__(
        self,
        datasets: List[Dataset],
        settings: Settings,
        *,
        create_controls: bool = True,
    ):
        self.datasets = datasets
        self.s = settings
        self._loading = False
        self._syncing = False
        self._locked_roll = 0.0
        self._locked_dist = None
        self._show_panel_labels = True

        # --- figure layout ---
        s = self.s
        self.fig = plt.figure(figsize=(s.fig_width, s.fig_height), dpi=110)
        if create_controls and hasattr(self.fig.canvas, "manager") and self.fig.canvas.manager:
            self.fig.canvas.manager.set_window_title("Figure 1")
        cbar_frac = max(0.05, float(s.cbar_ratio))
        if s.panel_layout == "side_by_side":
            self.gs = GridSpec(
                1,
                3,
                width_ratios=[1.0, 1.0, cbar_frac],
                wspace=s.gs_wspace,
                left=s.margin_left,
                right=s.margin_right,
                top=s.margin_top,
                bottom=s.margin_bottom,
            )
            self.axL = self.fig.add_subplot(self.gs[0, 0], projection="3d")
            self.axR = self.fig.add_subplot(self.gs[0, 1], projection="3d")
            self.cax = self.fig.add_subplot(self.gs[0, 2])
        else:
            self.gs = GridSpec(
                2,
                2,
                width_ratios=[1.0, cbar_frac],
                height_ratios=[1.0, 1.0],
                wspace=s.gs_wspace,
                hspace=s.gs_hspace,
                left=s.margin_left,
                right=s.margin_right,
                top=s.margin_top,
                bottom=s.margin_bottom,
            )
            self.axL = self.fig.add_subplot(self.gs[0, 0], projection="3d")
            self.axR = self.fig.add_subplot(self.gs[1, 0], projection="3d")
            self.cax = self.fig.add_subplot(self.gs[:, 1])
        self.cmap = batlow_cmap(reverse=True)
        self._exporting_tex = False

        for ax in (self.axL, self.axR):
            style_3d(ax)
            ax.computed_zorder = False
        self.axL.set_title("(a)  Rejection sampling", fontsize=11, pad=2)
        self.axR.set_title("(b)  KR rearrangement", fontsize=11, pad=2)

        # scatter / collection handles
        self.scL = self.scR = self.cb = None

        # Store original axes positions (before any cbar adjustment)
        self._orig_pos_L = self.axL.get_position()
        self._orig_pos_R = self.axR.get_position()
        self._orig_pos_cax = self.cax.get_position()

        # --- widgets ---
        self.widgets: dict = {}
        self.fig_ui = None
        if create_controls:
            self.fig_ui = plt.figure("Figure 1 — Controls", figsize=(11.0, 7.0), dpi=110)
            self.fig_ui.canvas.manager.set_window_title("Figure 1 Controls")
            self.fig_ui.set_facecolor("0.98")
            self._build_ui()
            self._bind_callbacks()
            self._bind_view_sync()
        else:
            self._cw = None

        # --- first paint ---
        self._apply_view()
        self.redraw(full_colorbar=True)

    # ── UI construction ───────────────────────────────────────────
    def _build_ui(self):
        """Build a two-column control panel: shared global controls + figure-specific controls."""
        s = self.s
        cw = ControlWindow(self.fig_ui, left=0.05, right=0.92, top=0.96, col_gap=0.04, col1_frac=0.40)
        self._cw = cw
        self.widgets = cw.widgets

        # Column 1: typography & layout (text entries — fire on Enter)
        cw.section(1, "Typography")
        cw.number_row(1, [("font_size", "Font", s.font_size), ("title_size", "Title", s.title_size)])
        cw.number_row(1, [("subtitle_size", "Sub", s.subtitle_size), ("panel_label_size", "Panel", s.panel_label_size)])
        cw.number_row(1, [("cbar_font_size", "Cbar F", s.cbar_font_size), ("title_pad", "T pad", s.title_pad)])

        cw.section(1, "Positions")
        cw.number_row(1, [("panel_label_x", "Lbl x", s.panel_label_x, "{:.3f}"), ("panel_label_y", "Lbl y", s.panel_label_y, "{:.3f}")])
        cw.number_row(1, [("subtitle_x", "Sub x", s.subtitle_x, "{:.3f}"), ("subtitle_y", "Sub y", s.subtitle_y, "{:.3f}")])

        cw.section(1, "Layout")
        cw.number_row(1, [("margin_left", "Left", s.margin_left, "{:.3f}"), ("margin_right", "Right", s.margin_right, "{:.3f}")])
        cw.number_row(1, [("margin_top", "Top", s.margin_top, "{:.3f}"), ("margin_bottom", "Bot", s.margin_bottom, "{:.3f}")])
        cw.number_row(1, [("gs_hspace", "Hspace", s.gs_hspace, "{:.3f}"), ("gs_wspace", "Wspace", s.gs_wspace, "{:.3f}")])
        cw.number_row(1, [("cbar_width", "Cbar W", s.cbar_width, "{:.3f}"), ("cbar_ratio", "Cbar R", s.cbar_ratio, "{:.3f}")])
        cw.number_row(1, [("fig_w", "Fig W", s.fig_width, "{:.2f}"), ("fig_h", "Fig H", s.fig_height, "{:.2f}")])

        cw.section(1, "Color")
        cw.number_row(1, [("p_lo", "% lo", s.p_lo, "{:.1f}"), ("p_hi", "% hi", s.p_hi, "{:.1f}")])

        cw.section(1, "File I/O")
        cw.textbox(1, "settings_path", "", DEFAULT_SETTINGS_JSON)
        cw.button_row(1, [("load", "Load"), ("save", "Save")])
        cw.textbox(1, "stem", "", DEFAULT_STEM)
        cw.button_row(1, [("save_png", "PNG"), ("save_pdf", "PDF")])

        # Column 2: figure-specific controls
        cw.section(2, "View")
        cw.slider(2, "view_elev", "Elevation", -90, 90, s.view_elev)
        cw.slider(2, "view_azim", "Azimuth", -180, 180, s.view_azim)
        cw.slider(2, "zoom_r", "Zoom", 0.15, 2.0, s.zoom_r)
        cw.slider(2, "zoom_cz", "Zoom center z", -1.5, 1.5, s.zoom_center_z)

        cw.section(2, "Data")
        cw.slider(
            2,
            "n_target_idx",
            "Dataset",
            0,
            len(self.datasets) - 1,
            float(s.n_target_index),
            step=1,
        )
        cw.slider(2, "cut_deg", "Cut angle", 0, 359, s.cut_angle_deg)
        cw.slider(2, "cut_off", "Cut offset", -1.5, 1.5, s.cut_plane_offset)
        cw.slider(2, "pt_size", "Point size", 1, 120, s.point_size)
        cw.slider(2, "bd_thresh", "Boundary deg", 0, 45, s.boundary_thresh_deg)

        cw.section(2, "FZ Style")
        cw.slider(2, "wire_thick", "Wire width", 0.25, 3.0, s.wire_thickness)
        cw.slider(2, "face_alpha", "Face alpha", 0, 1, s.face_alpha)

        cw.section(2, "Toggles")
        items = [
            ("draw_faces", "Faces", s.draw_faces),
            ("draw_face_edges", "Edges", s.draw_face_edges),
            ("only_octant_xyz_pos", "xyz>=0", s.only_octant_xyz_pos),
            ("keep_side_sign", "Side +", s.keep_side_sign > 0),
        ]
        cw.checkbox_grid(2, "tgrid", items, n_cols=2, row_h=0.040, label_size=7.0)

        cw.connect_scroll()

    # ── callbacks ─────────────────────────────────────────────────
    def _bind_callbacks(self):
        w = self.widgets
        w["n_target_idx"].on_changed(lambda _: self._on_change(full_cb=True))
        # Text entries — typography-only restyle (no data recompute)
        for k in (
            "font_size", "title_size", "subtitle_size", "panel_label_size",
            "panel_label_x", "panel_label_y", "subtitle_x", "subtitle_y",
            "cbar_width", "cbar_font_size", "title_pad",
        ):
            w[k].on_submit(lambda _, _k=k: self._on_restyle())
        # Text entries — layout (needs gridspec update, full redraw)
        for k in (
            "margin_left", "margin_right", "margin_top", "margin_bottom",
            "gs_hspace", "gs_wspace", "cbar_ratio", "fig_w", "fig_h",
        ):
            w[k].on_submit(lambda _, _k=k: self._on_layout_change())
        # Text entries — color percentiles (full redraw with colorbar)
        for k in ("p_lo", "p_hi"):
            w[k].on_submit(lambda _, _k=k: self._on_change(full_cb=True))
        # Sliders — visual-only (update artist properties in-place, no rebuild)
        for k in ("zoom_r", "pt_size", "wire_thick", "face_alpha"):
            w[k].on_changed(lambda _: self._on_visual_change())
        # Sliders — data-dependent (full redraw needed)
        for k in ("zoom_cz", "cut_deg", "cut_off"):
            w[k].on_changed(lambda _: self._on_change())
        w["bd_thresh"].on_changed(lambda _: self._on_change(full_cb=True))
        w["view_elev"].on_changed(lambda _: self._on_view_change())
        w["view_azim"].on_changed(lambda _: self._on_view_change())

        for chk in w["tgrid"]:
            chk.on_clicked(lambda _: self._on_change())

        w["load"].on_clicked(lambda _: self.load_settings(w["settings_path"].text))
        w["save"].on_clicked(lambda _: self.save_settings(w["settings_path"].text))
        w["save_png"].on_clicked(lambda _: self.save_figure(w["stem"].text, "png"))
        w["save_pdf"].on_clicked(lambda _: self.save_figure(w["stem"].text, "pdf"))

    def _on_change(self, full_cb: bool = False):
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self.redraw(full_colorbar=full_cb)

    def _on_restyle(self):
        """Lightweight: typography / cbar width only — no data recompute."""
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self._restyle()

    def _on_layout_change(self):
        """Update subplot margins / figure size — needs gridspec update + full redraw."""
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self._apply_layout()
        self.redraw(full_colorbar=True)

    def _apply_layout(self):
        """Push margin / gridspec / figsize settings into the figure."""
        s = self.s
        self.fig.set_size_inches(s.fig_width, s.fig_height)
        self.gs.update(
            left=s.margin_left, right=s.margin_right,
            top=s.margin_top, bottom=s.margin_bottom,
            hspace=s.gs_hspace, wspace=s.gs_wspace,
        )
        # Store new original positions (for cbar layout reference)
        self._orig_pos_L = self.axL.get_position()
        self._orig_pos_R = self.axR.get_position()
        self._orig_pos_cax = self.cax.get_position()

    def _on_view_change(self):
        if self._loading:
            return
        self._apply_view()

    def _on_visual_change(self):
        """Fast path: update artist properties in-place (no scatter rebuild)."""
        if self._loading:
            return
        self._sync_widgets_to_settings()
        self._apply_visual()

    def _apply_visual(self):
        """Update point size, zoom, wire thickness, face alpha without rebuilding."""
        s = self.s
        # zoom (axis limits only)
        for ax in (self.axL, self.axR):
            set_equal_zoom(ax, s.zoom_r)
        # point size
        ps = max(0.5, float(s.point_size))
        for sc in (self.scL, self.scR):
            if sc is not None:
                sc.set_sizes([ps])
        # wire thickness
        lw = max(0.5, float(s.wire_thickness))
        for ax in (self.axL, self.axR):
            for art in ax.collections:
                if isinstance(art, Line3DCollection):
                    art.set_linewidths([lw])
        # face alpha
        for ax in (self.axL, self.axR):
            for art in ax.collections:
                if isinstance(art, Poly3DCollection):
                    rgb = hls_to_rgb(s.face_color_hue, s.face_color_lightness)
                    art.set_facecolor([(*rgb, s.face_alpha)])
        self.fig.canvas.draw_idle()

    # ── view ──────────────────────────────────────────────────────
    def _apply_view(self):
        """Push elev/azim from sliders (or settings) into both 3-D axes."""
        if self._syncing:
            return
        if self.widgets:
            elev = float(self.widgets["view_elev"].val)
            azim = float(self.widgets["view_azim"].val)
        else:
            elev = float(self.s.view_elev)
            azim = float(self.s.view_azim)
        self._syncing = True
        try:
            for ax in (self.axL, self.axR):
                ax.view_init(elev=elev, azim=azim)
                if hasattr(ax, "roll"):
                    ax.roll = self._locked_roll
                if self._locked_dist is not None and hasattr(ax, "dist"):
                    ax.dist = self._locked_dist
            self.fig.canvas.draw_idle()
        finally:
            self._syncing = False

    def _bind_view_sync(self):
        """On mouse-release in a 3-D axis, copy its view to the other + update sliders."""

        def on_release(evt):
            if self._syncing or self._loading:
                return
            if evt.inaxes is self.axL:
                src, dst = self.axL, self.axR
            elif evt.inaxes is self.axR:
                src, dst = self.axR, self.axL
            else:
                return
            self._syncing = True
            try:
                dst.view_init(elev=float(src.elev), azim=float(src.azim))
                if hasattr(dst, "roll"):
                    dst.roll = self._locked_roll
                if self._locked_dist is not None and hasattr(dst, "dist"):
                    dst.dist = self._locked_dist
                # Feed back into sliders (suppressed: _syncing is True)
                self._loading = True
                self.widgets["view_elev"].set_val(float(src.elev))
                self.widgets["view_azim"].set_val(float(src.azim))
                self._loading = False
                self.fig.canvas.draw_idle()
            finally:
                self._syncing = False

        self.fig.canvas.mpl_connect("button_release_event", on_release)

    # ── settings ↔ widgets ────────────────────────────────────────
    def _sync_widgets_to_settings(self):
        """Read every widget value into self.s."""
        w, s = self.widgets, self.s
        cw = self._cw
        s.n_target_index = int(round(w["n_target_idx"].val))
        # Typography
        s.font_size = cw.get_val("font_size", s.font_size)
        s.title_size = cw.get_val("title_size", s.title_size)
        s.subtitle_size = cw.get_val("subtitle_size", s.subtitle_size)
        s.panel_label_size = cw.get_val("panel_label_size", s.panel_label_size)
        s.cbar_font_size = cw.get_val("cbar_font_size", s.cbar_font_size)
        s.title_pad = cw.get_val("title_pad", s.title_pad)
        # Positions
        s.panel_label_x = cw.get_val("panel_label_x", s.panel_label_x)
        s.panel_label_y = cw.get_val("panel_label_y", s.panel_label_y)
        s.subtitle_x = cw.get_val("subtitle_x", s.subtitle_x)
        s.subtitle_y = cw.get_val("subtitle_y", s.subtitle_y)
        # Layout
        s.margin_left = cw.get_val("margin_left", s.margin_left)
        s.margin_right = cw.get_val("margin_right", s.margin_right)
        s.margin_top = cw.get_val("margin_top", s.margin_top)
        s.margin_bottom = cw.get_val("margin_bottom", s.margin_bottom)
        s.gs_hspace = cw.get_val("gs_hspace", s.gs_hspace)
        s.gs_wspace = cw.get_val("gs_wspace", s.gs_wspace)
        s.cbar_width = cw.get_val("cbar_width", s.cbar_width)
        s.cbar_ratio = cw.get_val("cbar_ratio", s.cbar_ratio)
        s.fig_width = cw.get_val("fig_w", s.fig_width)
        s.fig_height = cw.get_val("fig_h", s.fig_height)
        # Color
        s.p_lo = cw.get_val("p_lo", s.p_lo)
        s.p_hi = cw.get_val("p_hi", s.p_hi)
        # View sliders
        s.view_elev = float(w["view_elev"].val)
        s.view_azim = float(w["view_azim"].val)
        s.zoom_r = float(w["zoom_r"].val)
        s.zoom_center_z = float(w["zoom_cz"].val)
        s.cut_angle_deg = float(w["cut_deg"].val)
        s.cut_plane_offset = float(w["cut_off"].val)
        s.point_size = float(w["pt_size"].val)
        s.wire_thickness = float(w["wire_thick"].val)
        s.boundary_thresh_deg = float(w["bd_thresh"].val)
        s.face_alpha = float(w["face_alpha"].val)
        s.face_color_hue = 0.0
        s.face_color_lightness = 0.92

        g = w["tgrid"]
        s.draw_faces = bool(g[0].get_status()[0])
        s.draw_face_edges = bool(g[1].get_status()[0])
        s.only_octant_xyz_pos = bool(g[2].get_status()[0])
        s.keep_side_sign = 1.0 if bool(g[3].get_status()[0]) else -1.0
        s.face_enabled = (True, True, True, True, True, True, True, True)

    def _push_settings_to_widgets(self):
        """Write self.s into every widget (must be called with _loading=True)."""
        w, s = self.widgets, self.s
        cw = self._cw
        w["n_target_idx"].set_val(float(int(s.n_target_index)))
        # Typography
        cw.set_val("font_size", s.font_size, "{:.1f}")
        cw.set_val("title_size", s.title_size, "{:.1f}")
        cw.set_val("subtitle_size", s.subtitle_size, "{:.1f}")
        cw.set_val("panel_label_size", s.panel_label_size, "{:.1f}")
        cw.set_val("cbar_font_size", s.cbar_font_size, "{:.1f}")
        cw.set_val("title_pad", s.title_pad, "{:.1f}")
        # Positions
        cw.set_val("panel_label_x", s.panel_label_x, "{:.3f}")
        cw.set_val("panel_label_y", s.panel_label_y, "{:.3f}")
        cw.set_val("subtitle_x", s.subtitle_x, "{:.3f}")
        cw.set_val("subtitle_y", s.subtitle_y, "{:.3f}")
        # Layout
        cw.set_val("margin_left", s.margin_left, "{:.3f}")
        cw.set_val("margin_right", s.margin_right, "{:.3f}")
        cw.set_val("margin_top", s.margin_top, "{:.3f}")
        cw.set_val("margin_bottom", s.margin_bottom, "{:.3f}")
        cw.set_val("gs_hspace", s.gs_hspace, "{:.3f}")
        cw.set_val("gs_wspace", s.gs_wspace, "{:.3f}")
        cw.set_val("cbar_width", s.cbar_width, "{:.3f}")
        cw.set_val("cbar_ratio", s.cbar_ratio, "{:.3f}")
        cw.set_val("fig_w", s.fig_width, "{:.2f}")
        cw.set_val("fig_h", s.fig_height, "{:.2f}")
        # Color
        cw.set_val("p_lo", s.p_lo, "{:.1f}")
        cw.set_val("p_hi", s.p_hi, "{:.1f}")
        # Sliders
        w["view_elev"].set_val(float(s.view_elev))
        w["view_azim"].set_val(float(s.view_azim))
        w["zoom_r"].set_val(float(s.zoom_r))
        w["zoom_cz"].set_val(float(s.zoom_center_z))
        w["cut_deg"].set_val(float(s.cut_angle_deg))
        w["cut_off"].set_val(float(s.cut_plane_offset))
        w["pt_size"].set_val(float(s.point_size))
        w["wire_thick"].set_val(float(s.wire_thickness))
        w["bd_thresh"].set_val(float(s.boundary_thresh_deg))
        w["face_alpha"].set_val(float(s.face_alpha))

        desired = [
            s.draw_faces,
            s.draw_face_edges,
            s.only_octant_xyz_pos,
            s.keep_side_sign > 0,
        ]
        for i, d in enumerate(desired):
            if bool(w["tgrid"][i].get_status()[0]) != bool(d):
                w["tgrid"][i].set_active(0)

    # ── settings IO ───────────────────────────────────────────────
    def _to_dict(self) -> dict:
        d = asdict(self.s)
        d["face_enabled"] = list(d["face_enabled"])
        return d

    def save_settings(self, path: str):
        self._sync_widgets_to_settings()
        fig_w, fig_h = self.fig.get_size_inches()
        self.s.fig_width = float(fig_w)
        self.s.fig_height = float(fig_h)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def load_settings(self, path: str):
        with open(path) as f:
            d = json.load(f)

        # Populate self.s from JSON
        for k, v in d.items():
            if hasattr(self.s, k):
                setattr(self.s, k, v)
        self.s.n_target_index = int(
            np.clip(self.s.n_target_index, 0, len(self.datasets) - 1)
        )
        if "face_enabled" in d:
            self.s.face_enabled = tuple(bool(x) for x in d["face_enabled"])
        # Legacy colour compat
        if "face_color" in d and isinstance(d["face_color"], str):
            rgb = _FACE_PRESETS.get(d["face_color"], (0.92, 0.92, 0.92))
            self.s.face_color_hue, self.s.face_color_lightness = rgb_to_hl(*rgb)
        elif all(k in d for k in ("face_color_r", "face_color_g", "face_color_b")):
            self.s.face_color_hue, self.s.face_color_lightness = rgb_to_hl(
                d["face_color_r"], d["face_color_g"], d["face_color_b"]
            )

        try:
            self._apply_layout()
        except Exception:
            pass

        # Push into widgets with callbacks suppressed
        self._loading = True
        self._push_settings_to_widgets()
        self._loading = False  # ← must be False BEFORE apply_view / redraw

        self._apply_view()
        self.redraw(full_colorbar=True)

    # ── save figure ───────────────────────────────────────────────
    def save_figure(self, stem: str, ext: str):
        self._sync_widgets_to_settings()
        stem = (stem or "").strip() or DEFAULT_STEM
        ext = (ext or "").strip().lower()
        out_path = f"{stem}.{ext}"
        with open(f"{stem}.json", "w") as f:
            json.dump(self._to_dict(), f, indent=2)
        self._exporting_tex = True
        try:
            export_figure_with_tex(
                self.fig,
                out_path,
                redraw_callback=lambda: self.redraw(full_colorbar=False),
                dpi=DPI,
            )
            print(f"Saved {out_path} (TeX export)")
        except Exception as exc:
            print(f"Export failed for {out_path}: {exc}")
            raise
        finally:
            self._exporting_tex = False
            self.redraw(full_colorbar=False)

    # ── main redraw ───────────────────────────────────────────────
    def redraw(self, full_colorbar: bool = False, show_panel_labels: bool | None = None):
        s = self.s
        if show_panel_labels is None:
            show_panel_labels = self._show_panel_labels
        ds = self.datasets[int(np.clip(s.n_target_index, 0, len(self.datasets) - 1))]
        fs = int(round(s.font_size))
        title_fs = int(round(s.title_size))
        sub_fs = int(round(s.subtitle_size))
        use_tex = bool(self._exporting_tex)
        title_left, title_right, subtitle_left, subtitle_right = self._panel_titles(
            ds, use_tex=use_tex, show_panel_labels=show_panel_labels,
        )
        cbar_label = "$\\textrm{NN chordal (FZ)}$" if use_tex else "NN chordal (FZ)"

        t_left = self.axL.set_title(title_left, fontsize=title_fs, pad=s.title_pad)
        t_right = self.axR.set_title(title_right, fontsize=title_fs, pad=s.title_pad)
        t_left.set_usetex(use_tex)
        t_right.set_usetex(use_tex)

        if show_panel_labels:
            add_panel_label(
                self.axL,
                "(a)",
                x=s.panel_label_x,
                y=s.panel_label_y,
                fontsize=s.panel_label_size,
                use_tex=use_tex,
            )
            add_panel_label(
                self.axR,
                "(b)",
                x=s.panel_label_x,
                y=s.panel_label_y,
                fontsize=s.panel_label_size,
                use_tex=use_tex,
            )
        if subtitle_left is not None and subtitle_right is not None:
            self._set_subtitle(self.axL, subtitle_left, s, sub_fs, use_tex)
            self._set_subtitle(self.axR, subtitle_right, s, sub_fs, use_tex)
        else:
            self._clear_subtitle(self.axL)
            self._clear_subtitle(self.axR)

        # transforms & masks
        zc = np.array([0.0, 0.0, s.zoom_center_z])
        rf_rej, rf_kr = ds.rf_rej_in - zc, ds.rf_kr_in - zc
        cn = rotation_normal_z(s.cut_angle_deg)
        m_rej = cut_mask(rf_rej, cn, s.keep_side_sign, s.cut_plane_offset)
        m_kr = cut_mask(rf_kr, cn, s.keep_side_sign, s.cut_plane_offset)
        if s.only_octant_xyz_pos:
            m_rej &= octant_pos(rf_rej)
            m_kr &= octant_pos(rf_kr)
        bd = math.radians(s.boundary_thresh_deg)
        if bd > 0:
            m_rej &= ds.bd_rej_in < bd
            m_kr &= ds.bd_kr_in < bd

        rf_rej_f, nn_rej_f = rf_rej[m_rej], ds.nn_rej_in[m_rej]
        rf_kr_f, nn_kr_f = rf_kr[m_kr], ds.nn_kr_in[m_kr]

        # depth sort
        for arr in ((rf_rej_f, nn_rej_f), (rf_kr_f, nn_kr_f)):
            if arr[0].shape[0]:
                o = np.argsort(arr[0][:, 2])
                arr[0][:], arr[1][:] = arr[0][o], arr[1][o]

        # shared colour norm
        all_nn = (
            np.concatenate([nn_rej_f, nn_kr_f])
            if nn_rej_f.size + nn_kr_f.size
            else np.array([0.0, 1.0])
        )
        vmin = float(np.percentile(all_nn, s.p_lo)) if all_nn.size else 0.0
        vmax = float(np.percentile(all_nn, s.p_hi)) if all_nn.size else 1.0
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        # clear old artists
        self._clear_collections()
        self._clear_scatters()
        for ax in (self.axL, self.axR):
            set_equal_zoom(ax, s.zoom_r)

        # FZ boundary: draw faces + back edges first, then points, then front edges.
        z_face = -2.0e6
        z_back = -1.0e6
        z_points = 0.0
        z_front = 1.0e6

        edge_segments = self._fz_edge_segments(zc)

        self._draw_fz_faces(self.axL, zc, zorder=z_face)
        self._draw_fz_faces(self.axR, zc, zorder=z_face)

        back_l, front_l = _split_segments_by_view(
            edge_segments, elev=float(self.axL.elev), azim=float(self.axL.azim)
        )
        back_r, front_r = _split_segments_by_view(
            edge_segments, elev=float(self.axR.elev), azim=float(self.axR.azim)
        )

        self._draw_fz_edges(self.axL, back_l, zorder=z_back)
        self._draw_fz_edges(self.axR, back_r, zorder=z_back)

        # scatter
        kw = dict(
            cmap=self.cmap,
            norm=norm,
            alpha=1.0,
            s=s.point_size,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            zorder=z_points,
        )
        if rf_rej_f.shape[0]:
            self.scL = self.axL.scatter(
                rf_rej_f[:, 0], rf_rej_f[:, 1], rf_rej_f[:, 2], c=nn_rej_f, **kw
            )
        if rf_kr_f.shape[0]:
            self.scR = self.axR.scatter(
                rf_kr_f[:, 0], rf_kr_f[:, 1], rf_kr_f[:, 2], c=nn_kr_f, **kw
            )

        self._draw_fz_edges(self.axL, front_l, zorder=z_front)
        self._draw_fz_edges(self.axR, front_r, zorder=z_front)

        # colorbar
        if full_colorbar or self.cb is None:
            self.cax.cla()
            mappable = self.scR or self.scL
            if mappable is None:
                mappable = ScalarMappable(norm=norm, cmap=self.cmap)
                mappable.set_array([])
            self.cb = self.fig.colorbar(mappable, cax=self.cax, orientation="vertical")

        # always refresh colorbar font (separate control from title font)
        if self.cb is not None:
            cfs = int(round(s.cbar_font_size))
            self.cb.set_label(
                cbar_label,
                fontsize=cfs,
            )
            self.cb.ax.yaxis.label.set_usetex(use_tex)
            self.cb.ax.tick_params(labelsize=max(6, cfs - 1))
            for tick_label in self.cb.ax.get_yticklabels(minor=False):
                tick_label.set_usetex(use_tex)
            for tick_label in self.cb.ax.get_yticklabels(minor=True):
                tick_label.set_usetex(use_tex)

        self.fig.canvas.draw_idle()

    def _apply_cbar_layout(self, s):
        """Reposition axes + cbar using *original* stored positions (no drift)."""
        x_right = s.margin_right + (1.0 - s.margin_right) * 0.85   # leave small right gutter
        cb_w = max(0.03, min(0.18, float(s.cbar_width)))
        cb_x0 = x_right - cb_w
        for ax, orig in ((self.axL, self._orig_pos_L), (self.axR, self._orig_pos_R)):
            ax.set_position([orig.x0, orig.y0, max(0.2, cb_x0 - 0.02 - orig.x0), orig.height])
        p0 = self._orig_pos_cax
        self.cax.set_position([cb_x0, p0.y0, cb_w, p0.height])

    @staticmethod
    def _clear_subtitle(ax) -> None:
        """Remove in-axes N_FZ overlay (used when counts move into the title)."""
        tag = "_subtitle_text"
        for child in list(ax.get_children()):
            if hasattr(child, "_sub_tag") and child._sub_tag == tag:
                child.remove()

    @staticmethod
    def _set_subtitle(ax, text, s, fontsize, use_tex):
        """Update-or-create a subtitle text on a 3D axis."""
        tag = "_subtitle_text"
        for child in ax.get_children():
            if hasattr(child, '_sub_tag') and child._sub_tag == tag:
                child.set_position((s.subtitle_x, s.subtitle_y))
                child.set_text(text)
                child.set_fontsize(fontsize)
                return
        fn = ax.text2D if hasattr(ax, "text2D") else ax.text
        t = fn(s.subtitle_x, s.subtitle_y, text,
               transform=ax.transAxes, ha="left", va="top", fontsize=fontsize)
        t._sub_tag = tag

    def _panel_titles(
        self,
        ds: Dataset,
        *,
        use_tex: bool,
        show_panel_labels: bool,
    ) -> tuple[str, str, str | None, str | None]:
        """Return (title_L, title_R, subtitle_L, subtitle_R); subtitles None if unused."""
        n_rej = int(ds.rf_rej_in.shape[0])
        n_kr = int(ds.rf_kr_in.shape[0])
        if self.s.panel_layout == "side_by_side":
            if use_tex:
                title_left = (
                    f"$\\textrm{{Rejection sampling}}\\ "
                    f"\\left(N_{{\\mathrm{{FZ}}}}={n_rej}\\right)$"
                )
                title_right = (
                    f"$\\textrm{{KR rearrangement}}\\ "
                    f"\\left(N_{{\\mathrm{{FZ}}}}={n_kr}\\right)$"
                )
            else:
                title_left = f"Rejection sampling (N_FZ={n_rej})"
                title_right = f"KR rearrangement (N_FZ={n_kr})"
            return title_left, title_right, None, None

        title_left = "$\\textrm{Rejection sampling}$" if use_tex else "Rejection sampling"
        title_right = "$\\textrm{KR rearrangement}$" if use_tex else "KR rearrangement"
        subtitle_left = (
            f"$N_{{\\mathrm{{FZ}}}}={n_rej}$" if use_tex else f"N_FZ={n_rej}"
        )
        subtitle_right = (
            f"$N_{{\\mathrm{{FZ}}}}={n_kr}$" if use_tex else f"N_FZ={n_kr}"
        )
        return title_left, title_right, subtitle_left, subtitle_right

    def _restyle(self):
        """Lightweight update: only typography / cbar width, no scatter rebuild."""
        s = self.s
        use_tex = bool(self._exporting_tex)
        title_fs = int(round(s.title_size))
        sub_fs = int(round(s.subtitle_size))
        ds = self.datasets[int(np.clip(s.n_target_index, 0, len(self.datasets) - 1))]
        title_left, title_right, subtitle_left, subtitle_right = self._panel_titles(
            ds, use_tex=use_tex, show_panel_labels=self._show_panel_labels,
        )

        self.axL.set_title(title_left, fontsize=title_fs, pad=s.title_pad)
        self.axR.set_title(title_right, fontsize=title_fs, pad=s.title_pad)

        if self._show_panel_labels:
            add_panel_label(self.axL, "(a)", x=s.panel_label_x, y=s.panel_label_y, fontsize=s.panel_label_size, use_tex=use_tex)
            add_panel_label(self.axR, "(b)", x=s.panel_label_x, y=s.panel_label_y, fontsize=s.panel_label_size, use_tex=use_tex)

        if subtitle_left is not None and subtitle_right is not None:
            self._set_subtitle(self.axL, subtitle_left, s, sub_fs, use_tex)
            self._set_subtitle(self.axR, subtitle_right, s, sub_fs, use_tex)
        else:
            self._clear_subtitle(self.axL)
            self._clear_subtitle(self.axR)

        self._apply_cbar_layout(s)

        if self.cb is not None:
            cfs = int(round(s.cbar_font_size))
            cbar_label = "$\\textrm{NN chordal (FZ)}$" if use_tex else "NN chordal (FZ)"
            self.cb.set_label(cbar_label, fontsize=cfs)
            self.cb.ax.yaxis.label.set_usetex(use_tex)
            self.cb.ax.tick_params(labelsize=max(6, cfs - 1))

        self.fig.canvas.draw_idle()

    # ── artist helpers ────────────────────────────────────────────
    def _clear_collections(self):
        for ax in (self.axL, self.axR):
            for art in [
                a
                for a in ax.collections
                if isinstance(a, (Poly3DCollection, Line3DCollection))
            ]:
                try:
                    art.remove()
                except Exception:
                    pass

    def _clear_scatters(self):
        for sc in (self.scL, self.scR):
            if sc is not None:
                try:
                    sc.remove()
                except Exception:
                    pass
        self.scL = self.scR = None

    def _fz_edge_segments(self, zoom_center: np.ndarray) -> List[np.ndarray]:
        s = self.s
        vt = OCT_VERTS - zoom_center
        edge_set = set()
        for fi, on in enumerate(s.face_enabled):
            if not on:
                continue
            f = OCT_FACES[fi]
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                edge_set.add(tuple(sorted((a, b))))

        return [
            np.stack([vt[i], vt[j]], axis=0)
            for i, j in OCT_EDGES
            if tuple(sorted((i, j))) in edge_set
        ]

    def _draw_fz_faces(self, ax, zoom_center: np.ndarray, zorder: float = 0.0):
        s = self.s
        if not s.draw_faces:
            return

        vt = OCT_VERTS - zoom_center
        tris = [vt[list(f)] for fi, f in enumerate(OCT_FACES) if s.face_enabled[fi]]
        if not tris:
            return

        rgb = hls_to_rgb(s.face_color_hue, s.face_color_lightness)
        ax.add_collection3d(
            Poly3DCollection(
                tris,
                facecolor=(*rgb, s.face_alpha),
                edgecolor="none",
                linewidths=1.0,
                alpha=1.0,
                zorder=zorder,
            )
        )

    def _draw_fz_edges(self, ax, segments: List[np.ndarray], zorder: float = 0.0):
        s = self.s
        if not s.draw_face_edges or not segments:
            return

        lines = [[seg[0], seg[1]] for seg in segments if seg.shape[0] >= 2]
        if not lines:
            return

        ax.add_collection3d(
            Line3DCollection(
                lines,
                colors=[(0.18, 0.18, 0.18, 1.0)],
                linewidths=max(1.0, float(s.wire_thickness)),
                alpha=1.0,
                zorder=zorder,
            )
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Figure 1 column layout (interactive or headless export)"
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

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    datasets = [
        build_dataset(n, seed=SEED + 1000 * i)
        for i, n in enumerate(N_TARGET_FZ_OPTIONS)
    ]
    s = Settings()
    s.n_target_index = int(np.clip(s.n_target_index, 0, len(datasets) - 1))
    app = FigureInteractive(datasets, s)

    settings_path = (args.settings or "").strip()
    if settings_path and os.path.exists(settings_path):
        try:
            app.load_settings(settings_path)
            print(f"Loaded settings from '{settings_path}'")
        except Exception as exc:
            print(f"Warning: could not load settings '{settings_path}': {exc}")

    if args.export:
        stem = (args.stem or "").strip() or DEFAULT_STEM
        app.save_figure(stem, "png")
        app.save_figure(stem, "pdf")
        plt.close("all")
        return

    plt.show()


if __name__ == "__main__":
    main()
