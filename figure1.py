#!/usr/bin/env python3
"""
figure1_interactive.py — Interactive Figure 1 (SO(3)/T): rejection vs KR

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
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from batlow import batlow_cmap
from grid_FZ import cu_kr_grid
from laue_ops import laue_dist_to_fz_boundary, ori_to_fz_laue
from orientation_ops import cu2ho, ho2qu, qu2rf, qu_std
from riesz_energy import riesz_energies_fused

# ── Config ────────────────────────────────────────────────────────────────────
LAUE_ID = 10  # id for T (tetrahedral Laue group "32")
LAUE_CARD = 12  # |K| for T
SEED = 42
DEVICE = torch.device("cpu")
DPI = 300
DEFAULT_STEM = "figure1"
DEFAULT_SETTINGS_JSON = "figure1_settings.json"
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
    font_size: float = 10.0
    cbar_font_size: float = 10.0
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
    fig_width: float = 12.8
    fig_height: float = 6.6


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
    def __init__(self, datasets: List[Dataset], settings: Settings):
        self.datasets = datasets
        self.s = settings
        self._loading = False
        self._syncing = False
        self._locked_roll = 0.0
        self._locked_dist = None

        # --- figure layout ---
        self.fig = plt.figure(figsize=(12.8, 6.6), dpi=110)
        self.fig.canvas.manager.set_window_title("Figure 1")
        gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.06)
        self.axL = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.axR = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.cax = self.fig.add_subplot(gs[0, 2])
        self.cmap = batlow_cmap(reverse=True)
        self._exporting_tex = False

        for ax in (self.axL, self.axR):
            style_3d(ax)
            ax.computed_zorder = False
        self.axL.set_title("(a)  Rejection sampling", fontsize=11, pad=2)
        self.axR.set_title("(b)  KR rearrangement", fontsize=11, pad=2)

        # scatter / collection handles
        self.scL = self.scR = self.cb = None

        # --- widgets ---
        self.widgets: dict = {}
        self.fig_ui = plt.figure("Figure 1 — Controls", figsize=(4.4, 8.8), dpi=110)
        self.fig_ui.canvas.manager.set_window_title("Figure 1 Controls")
        self.fig_ui.set_facecolor("0.98")
        self._build_ui()
        self._bind_callbacks()
        self._bind_view_sync()

        # --- first paint ---
        self._apply_view()
        self.redraw(full_colorbar=True)

    # ── UI construction ───────────────────────────────────────────
    def _build_ui(self):
        """Build a compact, grouped control panel (separate control window)."""
        L, W = 0.14, 0.78
        y = 0.96
        SH, DY, BH, GAP, HDR_H = 0.020, 0.026, 0.026, 0.014, 0.012

        def _section(title):
            nonlocal y
            y -= GAP
            ax = self.fig_ui.add_axes([L, y - HDR_H, W, HDR_H])
            ax.set_axis_off()
            ax.text(
                0.0,
                0.2,
                title.upper(),
                fontsize=7,
                fontweight="bold",
                color="0.4",
                transform=ax.transAxes,
                va="bottom",
            )
            y -= HDR_H + 0.003

        def _slider(name, label, lo, hi, val, step=None):
            nonlocal y
            ax = self.fig_ui.add_axes([L, y, W, SH])
            ax.set_facecolor("0.96")
            sl = Slider(ax, label, lo, hi, valinit=val, valstep=step)
            for t in (sl.label, sl.valtext):
                t.set_fontsize(8)
            self.widgets[name] = sl
            y -= DY
            return sl

        def _button_row(pairs):
            nonlocal y
            n = len(pairs)
            pad = 0.004
            bw = (W - pad * (n - 1)) / n
            for i, (name, label) in enumerate(pairs):
                bx = L + i * (bw + pad)
                ax = self.fig_ui.add_axes([bx, y, bw, BH])
                b = Button(ax, label, color="0.93", hovercolor="0.86")
                b.label.set_fontsize(8)
                self.widgets[name] = b
            y -= BH + 0.006

        def _textbox(name, label, initial):
            nonlocal y
            ax = self.fig_ui.add_axes([L, y, W, BH])
            tb = TextBox(ax, label, initial=initial)
            tb.label.set_fontsize(8)
            self.widgets[name] = tb
            y -= BH + 0.005

        s = self.s

        # ── VIEW ──────────────────────────
        _section("View")
        _slider("font_size", "Font", 6, 30, s.font_size)
        _slider("cbar_font_size", "Cbar", 6, 30, s.cbar_font_size)
        _slider("view_elev", "Elev", -90, 90, s.view_elev)
        _slider("view_azim", "Azim", -180, 180, s.view_azim)
        _slider("zoom_r", "Zoom", 0.15, 2, s.zoom_r)
        _slider("zoom_cz", "Z ctr", -1.5, 1.5, s.zoom_center_z)

        # ── DATA ──────────────────────────
        _section("Data")
        _slider(
            "n_target_idx",
            "N",
            0,
            len(self.datasets) - 1,
            float(s.n_target_index),
            step=1,
        )
        _slider("cut_deg", "Cut", 0, 359, s.cut_angle_deg)
        _slider("cut_off", "C off", -1.5, 1.5, s.cut_plane_offset)
        _slider("pt_size", "Size", 1, 120, s.point_size)
        _slider("bd_thresh", "Bdry", 0, 45, s.boundary_thresh_deg)

        # ── FZ STYLE ─────────────────────
        _section("FZ Style")
        _slider("wire_thick", "Wire", 0.25, 3.0, s.wire_thickness)
        _slider("face_alpha", "Alpha", 0, 1, s.face_alpha)

        # hue gradient strip
        bar_h = SH
        ax_bar = self.fig_ui.add_axes([L, y, W, bar_h])
        ax_bar.set_axis_off()
        grad = np.array([hls_to_rgb(i / 255, 0.5) for i in range(256)]).reshape(
            1, 256, 3
        )
        ax_bar.imshow(grad, aspect="auto", extent=[0, 1, 0, 1], origin="lower")
        y -= bar_h + 0.005

        _slider("face_hue", "Hue", 0, 1, s.face_color_hue)
        _slider("face_lit", "Light", 0, 1, s.face_color_lightness)

        # colour preview swatch
        pw, ph = W * 0.32, 0.014
        ax_p = self.fig_ui.add_axes([L, y, pw, ph])
        ax_p.set_axis_off()
        self._color_patch = Rectangle(
            (0, 0),
            1,
            1,
            facecolor=hls_to_rgb(s.face_color_hue, s.face_color_lightness),
            edgecolor="0.5",
            linewidth=0.6,
        )
        ax_p.add_patch(self._color_patch)
        ax_p.set_xlim(0, 1)
        ax_p.set_ylim(0, 1)
        y -= ph + 0.004

        # ── TOGGLES ──────────────────────
        _section("Toggles")
        COLS, ROWS = 4, 3
        gh = 0.045
        cw = W / COLS
        labels = [
            "Faces",
            "Edges",
            "xyz\u22650",
            "Side",
            "F0",
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
            "F6",
            "F7",
        ]
        states = [
            s.draw_faces,
            s.draw_face_edges,
            s.only_octant_xyz_pos,
            s.keep_side_sign > 0,
            *s.face_enabled,
        ]
        g0 = y - gh * ROWS
        self.widgets["tgrid"] = []
        for idx in range(12):
            r, c = idx // COLS, idx % COLS
            bx = L + c * cw
            by = g0 + (ROWS - 1 - r) * gh
            ax_c = self.fig_ui.add_axes([bx, by, cw - 0.002, gh - 0.003])
            chk = CheckButtons(ax_c, [labels[idx]], [states[idx]])
            for txt in chk.labels:
                txt.set_fontsize(6.5)
            self.widgets["tgrid"].append(chk)
        y = g0 - 0.006

        # ── FILE I/O ─────────────────────
        _section("File I/O")
        _textbox("settings_path", "", DEFAULT_SETTINGS_JSON)
        _button_row([("load", "Load"), ("save", "Save")])
        _textbox("stem", "", DEFAULT_STEM)
        _button_row([("save_png", "PNG"), ("save_pdf", "PDF")])

    # ── callbacks ─────────────────────────────────────────────────
    def _bind_callbacks(self):
        w = self.widgets
        w["n_target_idx"].on_changed(lambda _: self._on_change(full_cb=True))
        for k in (
            "font_size",
            "cbar_font_size",
            "zoom_r",
            "zoom_cz",
            "cut_deg",
            "cut_off",
            "pt_size",
            "wire_thick",
            "face_alpha",
        ):
            w[k].on_changed(lambda _: self._on_change())
        w["bd_thresh"].on_changed(lambda _: self._on_change(full_cb=True))
        w["view_elev"].on_changed(lambda _: self._on_view_change())
        w["view_azim"].on_changed(lambda _: self._on_view_change())

        def _on_hl(_):
            rgb = hls_to_rgb(float(w["face_hue"].val), float(w["face_lit"].val))
            self._color_patch.set_facecolor(rgb)
            self._on_change()

        w["face_hue"].on_changed(_on_hl)
        w["face_lit"].on_changed(_on_hl)

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

    def _on_view_change(self):
        if self._loading:
            return
        self._apply_view()

    # ── view ──────────────────────────────────────────────────────
    def _apply_view(self):
        """Push elev/azim from sliders into both 3-D axes."""
        if self._syncing:
            return
        elev = float(self.widgets["view_elev"].val)
        azim = float(self.widgets["view_azim"].val)
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
        s.n_target_index = int(round(w["n_target_idx"].val))
        s.font_size = float(w["font_size"].val)
        s.cbar_font_size = float(w["cbar_font_size"].val)
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
        s.face_color_hue = float(w["face_hue"].val)
        s.face_color_lightness = float(w["face_lit"].val)

        g = w["tgrid"]
        s.draw_faces = bool(g[0].get_status()[0])
        s.draw_face_edges = bool(g[1].get_status()[0])
        s.only_octant_xyz_pos = bool(g[2].get_status()[0])
        s.keep_side_sign = 1.0 if bool(g[3].get_status()[0]) else -1.0
        s.face_enabled = tuple(bool(g[i].get_status()[0]) for i in range(4, 12))

    def _push_settings_to_widgets(self):
        """Write self.s into every widget (must be called with _loading=True)."""
        w, s = self.widgets, self.s
        w["n_target_idx"].set_val(float(int(s.n_target_index)))
        w["font_size"].set_val(float(s.font_size))
        w["cbar_font_size"].set_val(float(s.cbar_font_size))
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
        w["face_hue"].set_val(float(s.face_color_hue))
        w["face_lit"].set_val(float(s.face_color_lightness))
        self._color_patch.set_facecolor(
            hls_to_rgb(s.face_color_hue, s.face_color_lightness)
        )

        desired = [
            s.draw_faces,
            s.draw_face_edges,
            s.only_octant_xyz_pos,
            s.keep_side_sign > 0,
            *s.face_enabled,
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
            fw = float(getattr(self.s, "fig_width", 12.8))
            fh = float(getattr(self.s, "fig_height", 6.6))
            self.fig.set_size_inches(fw, fh, forward=True)
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
            with plt.rc_context({"text.usetex": True}):
                self.redraw(full_colorbar=False)
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
            self.redraw(full_colorbar=False)

    # ── main redraw ───────────────────────────────────────────────
    def redraw(self, full_colorbar: bool = False):
        s = self.s
        ds = self.datasets[int(np.clip(s.n_target_index, 0, len(self.datasets) - 1))]
        fs = int(round(s.font_size))
        use_tex = bool(self._exporting_tex)
        if self._exporting_tex:
            title_left = "$\\mathbf{(a)}\\ \\textrm{Rejection sampling}\\ \\left(N_{\\mathrm{FZ}}=" + str(int(ds.rf_rej_in.shape[0])) + "\\right)$"
            title_right = "$\\mathbf{(b)}\\ \\textrm{KR rearrangement}\\ \\left(N_{\\mathrm{FZ}}=" + str(int(ds.rf_kr_in.shape[0])) + "\\right)$"
            cbar_label = "$\\textrm{NN chordal (FZ)}$"
        else:
            title_left = f"(a)  Rejection sampling   (N={int(ds.rf_rej_in.shape[0])})"
            title_right = f"(b)  KR rearrangement    (N={int(ds.rf_kr_in.shape[0])})"
            cbar_label = "NN chordal (FZ)"

        t_left = self.axL.set_title(
            title_left,
            fontsize=fs,
            pad=2,
        )
        t_right = self.axR.set_title(
            title_right,
            fontsize=fs,
            pad=2,
        )
        t_left.set_usetex(use_tex)
        t_right.set_usetex(use_tex)

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
                linewidths=0.9,
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
                linewidths=0.9 * max(0.25, float(s.wire_thickness)),
                alpha=1.0,
                zorder=zorder,
            )
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    datasets = [
        build_dataset(n, seed=SEED + 1000 * i)
        for i, n in enumerate(N_TARGET_FZ_OPTIONS)
    ]
    s = Settings()
    s.n_target_index = int(np.clip(s.n_target_index, 0, len(datasets) - 1))
    _app = FigureInteractive(datasets, s)
    plt.show()


if __name__ == "__main__":
    main()
