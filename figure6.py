#!/usr/bin/env python3
"""
figure6.py — Primitive-cubic KR vs FCC-KR for Laue O in 2x3 views.

Top row: primitive cubic KR input grid (group O)
Bottom row: FCC KR input grid (group O)

Columns:
  1) Cubochoric space (cube wireframe)
  2) Homochoric space (ball wireframe)
  3) O fundamental zone points

Coloring rule (matches Figure 1 intent):
  - Compute nearest-neighbor chordal distance ONLY after mapping to FZ (qu_fz).
  - Reuse that same FZ NN distance to color CU / HO / FZ points for each row.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider, TextBox

from batlow import batlow_cmap
from grid_FZ import kr_sample_laue
from orientation_ops import cu2ho, cu2qu, qu_norm, qu_std, qu2ho, rf2qu

LAUE_O = 11
CARD_O = 24
SETTINGS_FILE = "figure6_settings.json"
DEFAULT_STEM = "figure6"
SETTINGS_VERSION = 2

CU_MAX = 0.5 * math.pi ** (2.0 / 3.0)
HO_MAX = (3.0 * math.pi / 4.0) ** (1.0 / 3.0)

COLUMN_LABELS = (
    r"$\mathrm{Cubochoric}$",
    r"$\mathrm{Homochoric}$",
    r"$\mathrm{FZ}$",
)
PANEL_LETTERS = np.array([["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]])


@dataclass
class Match:
    h_pc: int
    h_fcc: int
    n_pc: int
    n_fcc: int
    n_s3_pc: int
    n_s3_fcc: int


@dataclass
class Settings:
    elev: float = 18.0
    azim: float = 35.0
    point_size: float = 7.0
    wire_thickness: float = 1.0
    text_size: float = 11.0
    zoom_cu: float = 0.70
    zoom_ho: float = 0.55
    zoom_fz: float = 1.35
    p_lo: float = 2.0
    p_hi: float = 98.0


def settings_from_raw(raw: dict, base: Settings | None = None) -> Settings:
    s = Settings() if base is None else base

    if "text_size" not in raw and "font_size" in raw:
        raw = dict(raw)
        raw["text_size"] = raw["font_size"]

    for k, v in raw.items():
        if hasattr(s, k):
            setattr(s, k, v)

    s.point_size = float(np.clip(s.point_size, 1.0, 25.0))
    s.wire_thickness = float(np.clip(s.wire_thickness, 0.25, 3.00))
    s.text_size = float(np.clip(s.text_size, 8.0, 20.0))
    s.zoom_cu = float(np.clip(s.zoom_cu, 0.20, 1.60))
    s.zoom_ho = float(np.clip(s.zoom_ho, 0.20, 1.60))
    s.zoom_fz = float(np.clip(s.zoom_fz, 0.30, 3.00))
    s.p_lo = float(np.clip(s.p_lo, 0.0, 30.0))
    s.p_hi = float(np.clip(s.p_hi, 70.0, 100.0))
    if s.p_lo >= s.p_hi:
        s.p_hi = min(100.0, s.p_lo + 0.5)

    return s


def settings_to_raw(s: Settings) -> dict:
    out = asdict(s)
    out["version"] = SETTINGS_VERSION
    return out


def cubochoric_primitive_grid(h: int, device: torch.device) -> torch.Tensor:
    u = torch.linspace(-CU_MAX, CU_MAX, 2 * h + 2, device=device, dtype=torch.float64)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def cubochoric_fcc_grid(
    h: int, device: torch.device, even_parity: bool = True
) -> torch.Tensor:
    u = torch.linspace(-CU_MAX, CU_MAX, 2 * h + 2, device=device, dtype=torch.float64)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")

    idx = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
    mask = ((i + j + k) % 2) == (
        0 if even_parity else 1
    )  # 0 retains identity rotation at (0,0,0)

    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


def n_base_pc(h: int) -> int:
    return (2 * h + 1) ** 3


def n_base_fcc(h: int) -> int:
    return (n_base_pc(h) + (1 if (h % 2 == 0) else -1)) // 2


def n_s3_from_n_base(n_base: int, card: int = CARD_O) -> int:
    return 2 * card * n_base


def find_best_match(h_min: int, h_max: int, n_s3_min: int, n_s3_max: int) -> Match:
    pcs = []
    fccs = []
    for h in range(h_min, h_max + 1):
        n_pc = n_base_pc(h)
        n_fcc = n_base_fcc(h)
        n_s3_pc = n_s3_from_n_base(n_pc)
        n_s3_fcc = n_s3_from_n_base(n_fcc)
        if n_s3_min <= n_s3_pc <= n_s3_max:
            pcs.append((h, n_pc, n_s3_pc))
        if n_s3_min <= n_s3_fcc <= n_s3_max:
            fccs.append((h, n_fcc, n_s3_fcc))

    if not pcs or not fccs:
        raise ValueError("No candidate h values found in the requested N_S^3 range.")

    best = None
    best_key = None

    for h_pc, n_pc, ns3_pc in pcs:
        for h_fcc, n_fcc, ns3_fcc in fccs:
            diff = abs(ns3_pc - ns3_fcc)
            rel_diff = diff / max(ns3_pc, ns3_fcc)
            size_bias = abs(h_pc - h_fcc)
            key = (rel_diff, diff, size_bias, ns3_pc + ns3_fcc)
            if best_key is None or key < best_key:
                best_key = key
                best = Match(
                    h_pc=h_pc,
                    h_fcc=h_fcc,
                    n_pc=n_pc,
                    n_fcc=n_fcc,
                    n_s3_pc=ns3_pc,
                    n_s3_fcc=ns3_fcc,
                )

    return best


@torch.no_grad()
def nn_chordal_dist(qu: torch.Tensor) -> np.ndarray:
    q = qu_norm(qu_std(qu.to(dtype=torch.float64)))
    d1 = torch.cdist(q, q)
    d2 = torch.cdist(q, -q)
    d = torch.minimum(d1, d2)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    return nn.detach().cpu().numpy()


def _cube_wire_segments() -> list[np.ndarray]:
    c = CU_MAX
    edges = [
        ((-c, -c, -c), (c, -c, -c)),
        ((-c, c, -c), (c, c, -c)),
        ((-c, -c, c), (c, -c, c)),
        ((-c, c, c), (c, c, c)),
        ((-c, -c, -c), (-c, c, -c)),
        ((c, -c, -c), (c, c, -c)),
        ((-c, -c, c), (-c, c, c)),
        ((c, -c, c), (c, c, c)),
        ((-c, -c, -c), (-c, -c, c)),
        ((c, -c, -c), (c, -c, c)),
        ((-c, c, -c), (-c, c, c)),
        ((c, c, -c), (c, c, c)),
    ]
    return [np.asarray([a, b], dtype=np.float64) for a, b in edges]


def _polyline_to_segments(
    polyline: np.ndarray, closed: bool = False
) -> list[np.ndarray]:
    if polyline.shape[0] < 2:
        return []

    segments = [polyline[i : i + 2] for i in range(polyline.shape[0] - 1)]
    if closed:
        segments.append(np.stack([polyline[-1], polyline[0]], axis=0))
    return segments


def _ball_wire_segments() -> list[np.ndarray]:
    u_open = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    v = np.linspace(0, np.pi, 36)
    segments = []

    # meridians
    for ui in u_open[::4]:
        x = HO_MAX * np.cos(ui) * np.sin(v)
        y = HO_MAX * np.sin(ui) * np.sin(v)
        z = HO_MAX * np.cos(v)
        meridian = np.column_stack((x, y, z))
        segments.extend(_polyline_to_segments(meridian, closed=False))

    # parallels (latitude rings) split into arc segments so front/back ordering is view-consistent.
    for vi in v[2:-2:4]:
        x = HO_MAX * np.cos(u_open) * np.sin(vi)
        y = HO_MAX * np.sin(u_open) * np.sin(vi)
        z = HO_MAX * np.full_like(u_open, np.cos(vi))
        ring = np.column_stack((x, y, z))
        segments.extend(_polyline_to_segments(ring, closed=True))
    return segments


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


def _split_wire_segments_by_view(
    segments: list[np.ndarray], elev: float, azim: float, eps: float = 1e-12
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    normal = _view_plane_normal(elev, azim)
    behind: list[np.ndarray] = []
    front: list[np.ndarray] = []

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


def _draw_wire_segments_with_zorder(
    ax,
    segments: list[np.ndarray],
    color: str = "0.10",
    lw: float = 1.0,
    zorder: float = 0.0,
):
    for seg in segments:
        if seg.shape[0] < 2:
            continue
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color=color,
            lw=lw,
            alpha=1.0,
            zorder=zorder,
        )


def _plot_points_depth(
    ax, xyz: np.ndarray, values: np.ndarray, cmap, norm, point_size: float
):
    _plot_points_depth_with_zorder(ax, xyz, values, cmap, norm, point_size, zorder=0.0)


def _plot_points_depth_with_zorder(
    ax,
    xyz: np.ndarray,
    values: np.ndarray,
    cmap,
    norm,
    point_size: float,
    zorder: float = 0.0,
):
    if xyz.size == 0:
        return
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=values,
        cmap=cmap,
        norm=norm,
        s=max(0.5, float(point_size)),
        marker="o",
        linewidths=0.0,
        edgecolors="none",
        depthshade=False,
        alpha=1.0,
        zorder=zorder,
    )


def _style_common(ax, lim: float, elev: float, azim: float):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)


def _rotate_about_axis(
    vec: np.ndarray, axis: np.ndarray, angle_rad: float
) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _truncated_cube_rf_vertices() -> np.ndarray:
    a = math.sqrt(2.0) - 1.0
    b = 3.0 - 2.0 * math.sqrt(2.0)
    base = np.array([a, a, b], dtype=np.float64)

    axis_111 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    tri = np.stack(
        [
            base,
            _rotate_about_axis(base, axis_111, 2.0 * math.pi / 3.0),
            _rotate_about_axis(base, axis_111, 4.0 * math.pi / 3.0),
        ],
        axis=0,
    )

    verts = []
    for sx, sy, sz in product([-1.0, 1.0], repeat=3):
        sign = np.array([sx, sy, sz], dtype=np.float64)
        verts.append(tri * sign[None, :])

    verts = np.concatenate(verts, axis=0)
    verts = np.unique(np.round(verts, decimals=12), axis=0)
    return verts


def _truncated_cube_rf_edges(vertices: np.ndarray) -> list[tuple[int, int]]:
    diff = vertices[:, None, :] - vertices[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[dist < 1e-12] = np.inf
    edge_len = float(np.min(dist))
    tol = max(1e-6, 1e-4 * edge_len)

    n = vertices.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dist[i, j] - edge_len) <= tol:
                edges.append((i, j))
    return edges


def _fz_wire_segments_ho(samples_per_edge: int = 64) -> list[np.ndarray]:
    verts = _truncated_cube_rf_vertices()
    edges = _truncated_cube_rf_edges(verts)

    t = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float64)[:, None]
    segs_ho = []
    for i, j in edges:
        rf_line = (1.0 - t) * verts[i][None, :] + t * verts[j][None, :]
        rf_t = torch.from_numpy(rf_line)
        qu = qu_norm(qu_std(rf2qu(rf_t)))
        ho = qu2ho(qu).detach().cpu().numpy()
        segs_ho.append(ho)
    return segs_ho


@torch.no_grad()
def build_row(cu: torch.Tensor, laue_id: int):
    qu_in = qu_norm(qu_std(cu2qu(cu)))
    ho_in = cu2ho(cu)

    qu_fz = qu_norm(qu_std(kr_sample_laue(qu_in, laue_id)))
    ho_fz = qu2ho(qu_fz)

    # Figure1-style: NN computed in FZ domain only.
    nn_fz = nn_chordal_dist(qu_fz)

    return {
        "cu": cu.detach().cpu().numpy(),
        "ho": ho_in.detach().cpu().numpy(),
        "ho_fz": ho_fz.detach().cpu().numpy(),
        "nn_fz": nn_fz,
    }


class Figure7Interactive:
    def __init__(
        self,
        row_top: dict,
        row_bot: dict,
        match: Match,
        settings: Settings,
        *,
        figsize: tuple[float, float] = (14.6, 8.8),
        dpi: int = 50,
        create_controls: bool = True,
        show_panel_labels: bool = True,
    ):
        self.row_top = row_top
        self.row_bot = row_bot
        self.match = match
        self.s = settings
        self._loading = False
        self._show_panel_labels = show_panel_labels

        self.cmap = batlow_cmap(reverse=True)

        self.fz_extent = float(
            max(
                np.max(np.abs(self.row_top["ho_fz"])),
                np.max(np.abs(self.row_bot["ho_fz"])),
            )
        )
        self.fz_extent = max(self.fz_extent, 1e-6)
        self.fz_wire_ho = _fz_wire_segments_ho(samples_per_edge=72)
        self.cu_wire_segments = _cube_wire_segments()
        self.ho_wire_segments = _ball_wire_segments()

        self.fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = self.fig.add_gridspec(
            2, 4, width_ratios=[1.0, 1.0, 1.0, 0.07], wspace=0.02
        )

        self.axes = np.empty((2, 3), dtype=object)
        for r in range(2):
            for c in range(3):
                self.axes[r, c] = self.fig.add_subplot(gs[r, c], projection="3d")
                self.axes[r, c].computed_zorder = False

        self.cax = self.fig.add_subplot(gs[:, 3])

        self._colorbar = None
        self.fig_ui = None
        if create_controls:
            self.fig_ui = plt.figure("Figure7 Controls", figsize=(6.8, 6.3), dpi=110)
            self.fig_ui.patch.set_facecolor("white")
            self._build_ui()

        self.redraw(full_cbar=True)

    def _build_ui(self):
        y = 0.93
        x = 0.06
        w = 0.90
        h = 0.024
        dy = 0.058

        def _add_slider(
            attr_ax: str,
            attr_sl: str,
            label: str,
            vmin: float,
            vmax: float,
            vinit: float,
            vstep: float,
        ):
            nonlocal y
            ax = self.fig_ui.add_axes([x, y, w, h])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=vstep)
            slider.label.set_fontsize(9.5)
            slider.label.set_ha("left")
            slider.label.set_va("bottom")
            slider.label.set_position((0.0, 1.15))
            slider.valtext.set_fontsize(9.0)
            slider.valtext.set_ha("right")
            slider.valtext.set_va("bottom")
            slider.valtext.set_position((1.0, 1.15))
            setattr(self, attr_ax, ax)
            setattr(self, attr_sl, slider)
            y -= dy

        _add_slider("ax_elev", "sl_elev", "Elev", -90.0, 90.0, self.s.elev, 1.0)
        _add_slider("ax_azim", "sl_azim", "Azim", -180.0, 180.0, self.s.azim, 1.0)
        _add_slider("ax_pt", "sl_pt", "Point size", 1.0, 50.0, self.s.point_size, 0.5)
        _add_slider(
            "ax_wire",
            "sl_wire",
            "Wire thickness",
            0.25,
            3.00,
            self.s.wire_thickness,
            0.05,
        )
        _add_slider("ax_text", "sl_text", "Text size", 8.0, 20.0, self.s.text_size, 0.5)
        _add_slider(
            "ax_zoom_cu", "sl_zoom_cu", "Zoom CU", 0.20, 1.60, self.s.zoom_cu, 0.01
        )
        _add_slider(
            "ax_zoom_ho", "sl_zoom_ho", "Zoom HO", 0.20, 1.60, self.s.zoom_ho, 0.01
        )
        _add_slider(
            "ax_zoom_fz", "sl_zoom_fz", "Zoom FZ", 0.30, 3.00, self.s.zoom_fz, 0.01
        )
        _add_slider("ax_plo", "sl_plo", "Pctl lo", 0.0, 30.0, self.s.p_lo, 0.5)
        _add_slider("ax_phi", "sl_phi", "Pctl hi", 70.0, 100.0, self.s.p_hi, 0.5)
        y -= 0.05

        def _add_textbox(attr_ax: str, attr_tb: str, label: str, initial: str):
            nonlocal y
            ax = self.fig_ui.add_axes([x, y, w, h])
            textbox = TextBox(ax, label, initial=initial)
            textbox.label.set_fontsize(9.5)
            textbox.label.set_ha("left")
            textbox.label.set_va("bottom")
            textbox.label.set_position((0.0, 1.15))
            if hasattr(textbox, "text_disp"):
                textbox.text_disp.set_fontsize(9.0)
            setattr(self, attr_ax, ax)
            setattr(self, attr_tb, textbox)
            y -= dy

        _add_textbox("ax_settings_path", "tb_settings_path", "Settings", SETTINGS_FILE)
        _add_textbox("ax_stem", "tb_stem", "Stem", DEFAULT_STEM)
        y -= 0.06

        bw = (w - 0.04) / 3.0
        self.ax_bsave = self.fig_ui.add_axes([x, y, bw, 0.05])
        self.ax_bload = self.fig_ui.add_axes([x + bw + 0.02, y, bw, 0.05])
        self.ax_bfig = self.fig_ui.add_axes([x + 2 * (bw + 0.02), y, bw, 0.05])

        self.bt_save = Button(self.ax_bsave, "Save cfg")
        self.bt_load = Button(self.ax_bload, "Load cfg")
        self.bt_fig = Button(self.ax_bfig, "Save fig")

        slider_map = {
            "elev": self.sl_elev,
            "azim": self.sl_azim,
            "point_size": self.sl_pt,
            "wire_thickness": self.sl_wire,
            "text_size": self.sl_text,
            "zoom_cu": self.sl_zoom_cu,
            "zoom_ho": self.sl_zoom_ho,
            "zoom_fz": self.sl_zoom_fz,
            "p_lo": self.sl_plo,
            "p_hi": self.sl_phi,
        }
        for slider_name, slider in slider_map.items():
            slider.on_changed(lambda _v, name=slider_name: self._on_slider(name))

        self.bt_save.on_clicked(self._save_settings)
        self.bt_load.on_clicked(self._load_settings)
        self.bt_fig.on_clicked(self._save_figure)

    def _sync_settings_from_widgets(self):
        lo = float(self.sl_plo.val)
        hi = float(self.sl_phi.val)
        if lo >= hi:
            hi = min(100.0, lo + 0.5)
            self._loading = True
            self.sl_phi.set_val(hi)
            self._loading = False

        self.s.elev = float(self.sl_elev.val)
        self.s.azim = float(self.sl_azim.val)
        self.s.point_size = float(self.sl_pt.val)
        self.s.wire_thickness = float(self.sl_wire.val)
        self.s.text_size = float(self.sl_text.val)
        self.s.zoom_cu = float(self.sl_zoom_cu.val)
        self.s.zoom_ho = float(self.sl_zoom_ho.val)
        self.s.zoom_fz = float(self.sl_zoom_fz.val)
        self.s.p_lo = lo
        self.s.p_hi = hi

    def _on_slider(self, slider_name: str):
        if self._loading:
            return
        self._sync_settings_from_widgets()
        self.redraw(full_cbar=slider_name in {"p_lo", "p_hi", "text_size"})

    def _draw_row(self, row_idx: int, row_data: dict, norm: Normalize):
        z_back = -1.0e6
        z_points = 0.0
        z_front = 1.0e6

        column_specs = (
            ("cu", CU_MAX * self.s.zoom_cu, self.cu_wire_segments, "0.22", 1.15),
            ("ho", HO_MAX * self.s.zoom_ho, self.ho_wire_segments, "0.24", 0.65),
            (
                "ho_fz",
                self.fz_extent * self.s.zoom_fz,
                self.fz_wire_ho,
                "0.10",
                1.0,
            ),
        )

        for col_idx, (key, lim, wire_segments, wire_color, wire_lw) in enumerate(
            column_specs
        ):
            ax = self.axes[row_idx, col_idx]
            lw_effective = wire_lw * self.s.wire_thickness

            wire_behind, wire_front = _split_wire_segments_by_view(
                wire_segments, elev=self.s.elev, azim=self.s.azim
            )
            _draw_wire_segments_with_zorder(
                ax, wire_behind, color=wire_color, lw=lw_effective, zorder=z_back
            )
            _plot_points_depth_with_zorder(
                ax,
                row_data[key],
                row_data["nn_fz"],
                self.cmap,
                norm,
                self.s.point_size,
                zorder=z_points,
            )
            _draw_wire_segments_with_zorder(
                ax, wire_front, color=wire_color, lw=lw_effective, zorder=z_front
            )

            _style_common(ax, lim, self.s.elev, self.s.azim)
            ax.set_axis_off()
            ax.set_title("")

    def _draw_annotations(self, use_tex: bool = False, show_panel_labels: bool | None = None):
        if show_panel_labels is None:
            show_panel_labels = self._show_panel_labels
        panel_letters = (
            np.array(
                [
                    ["$\\mathbf{(a)}$", "$\\mathbf{(b)}$", "$\\mathbf{(c)}$"],
                    ["$\\mathbf{(d)}$", "$\\mathbf{(e)}$", "$\\mathbf{(f)}$"],
                ],
                dtype=object,
            )
            if use_tex
            else PANEL_LETTERS
        )
        if use_tex:
            column_labels = (
                "$\\textrm{Cubochoric}$",
                "$\\textrm{Homochoric}$",
                "$\\textrm{FZ}$",
            )
        else:
            column_labels = COLUMN_LABELS

        if show_panel_labels:
            for r in range(2):
                for c in range(3):
                    self.axes[r, c].text2D(
                        0.02,
                        0.96,
                        panel_letters[r, c],
                        transform=self.axes[r, c].transAxes,
                        fontsize=self.s.text_size,
                        fontweight=("bold" if not use_tex else None),
                        ha="left",
                        va="top",
                    )

        for c, label in enumerate(column_labels):
            self.axes[0, c].text2D(
                0.50,
                1.03,
                label,
                transform=self.axes[0, c].transAxes,
                fontsize=self.s.text_size,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

        if use_tex:
            row_labels = (
                "$\\textrm{Cubic Primitive}\\;(N_{\\mathrm{FZ}}="
                + str(self.match.n_pc)
                + ")$",
                "$\\textrm{FCC}\\;(N_{\\mathrm{FZ}}=" + str(self.match.n_fcc) + ")$",
            )
        else:
            row_labels = (
                rf"$\mathrm{{Cubic\;Primitive}}\;(N_{{FZ}}={self.match.n_pc})$",
                rf"$\mathrm{{FCC}}\;(N_{{FZ}}={self.match.n_fcc})$",
            )
        for r, row_label in enumerate(row_labels):
            self.axes[r, 0].text2D(
                -0.17,
                0.50,
                row_label,
                transform=self.axes[r, 0].transAxes,
                fontsize=self.s.text_size,
                fontweight="bold",
                rotation=90,
                ha="center",
                va="center",
            )

    def _save_settings(self, _event):
        self._sync_settings_from_widgets()
        path = self.tb_settings_path.text.strip() or SETTINGS_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings_to_raw(self.s), f, indent=2)
        print(f"Saved settings: {path}")

    def _load_settings(self, _event):
        path = self.tb_settings_path.text.strip() or SETTINGS_FILE
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            print(f"Settings file not found: {path}")
            return

        self.s = settings_from_raw(raw, self.s)

        self._loading = True
        try:
            self.sl_elev.set_val(self.s.elev)
            self.sl_azim.set_val(self.s.azim)
            self.sl_pt.set_val(self.s.point_size)
            self.sl_text.set_val(self.s.text_size)
            self.sl_zoom_cu.set_val(self.s.zoom_cu)
            self.sl_zoom_ho.set_val(self.s.zoom_ho)
            self.sl_zoom_fz.set_val(self.s.zoom_fz)
            self.sl_plo.set_val(self.s.p_lo)
            self.sl_phi.set_val(self.s.p_hi)
        finally:
            self._loading = False

        self._sync_settings_from_widgets()
        self.redraw(full_cbar=True)
        print(f"Loaded settings: {path}")

    def _save_figure(self, _event):
        stem = self.tb_stem.text.strip() or DEFAULT_STEM
        with plt.rc_context({"text.usetex": True}):
            self.redraw(full_cbar=True, use_tex=True)
            self.fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
            self.fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        self.redraw(full_cbar=False, use_tex=False)
        print(f"Saved {stem}.png and {stem}.pdf (TeX export)")

    def _shared_norm(self) -> Normalize:
        nn = np.concatenate([self.row_top["nn_fz"], self.row_bot["nn_fz"]])
        lo = float(np.nanpercentile(nn, self.s.p_lo))
        hi = float(np.nanpercentile(nn, self.s.p_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.nanmin(nn)), float(np.nanmax(nn))
            if lo >= hi:
                hi = lo + 1e-6
        return Normalize(vmin=lo, vmax=hi)

    def redraw(self, full_cbar: bool = False, use_tex: bool = False):
        for ax in self.axes.ravel():
            ax.cla()

        norm = self._shared_norm()

        self._draw_row(0, self.row_top, norm)
        self._draw_row(1, self.row_bot, norm)

        # self.fig.suptitle(
        #     f"Figure 6: KR grid comparison for O (N_S^3: primitive={self.match.n_s3_pc:,}, fcc={self.match.n_s3_fcc:,})",
        #     fontsize=self.s.text_size + 1.5,
        #     y=0.995,
        # )

        self._draw_annotations(use_tex=use_tex)

        if full_cbar or self._colorbar is None:
            self.cax.cla()
            sm = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
            self._colorbar = self.fig.colorbar(sm, cax=self.cax)
            if use_tex:
                cbar_label = "$\\textrm{NN chordal (FZ)}$"
            else:
                cbar_label = r"$\mathrm{NN\;chordal\;(FZ)}$"
            self._colorbar.set_label(
                cbar_label, fontsize=max(8.0, self.s.text_size - 1.0)
            )
            self._colorbar.ax.tick_params(labelsize=max(7.0, self.s.text_size - 2.0))

        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(
        description="Figure 6: primitive vs FCC KR grid views for O"
    )
    ap.add_argument("--n-s3-min", type=int, default=1_000)
    ap.add_argument("--n-s3-max", type=int, default=200_000)
    ap.add_argument("--h-min", type=int, default=1)
    ap.add_argument("--h-max", type=int, default=30)
    ap.add_argument(
        "--save", action="store_true", help="Save figure7.png and figure7.pdf and exit"
    )
    ap.add_argument("--settings", type=str, default=SETTINGS_FILE)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match = find_best_match(args.h_min, args.h_max, args.n_s3_min, args.n_s3_max)
    rel_diff = abs(match.n_s3_pc - match.n_s3_fcc) / max(match.n_s3_pc, match.n_s3_fcc)
    print(
        f"Matched sizes: primitive h={match.h_pc} (N_S^3={match.n_s3_pc:,}) vs "
        f"fcc h={match.h_fcc} (N_S^3={match.n_s3_fcc:,}), "
        f"rel_diff={100.0 * rel_diff:.3f}% (abs_diff={abs(match.n_s3_pc - match.n_s3_fcc):,})"
    )

    cu_pc = cubochoric_primitive_grid(match.h_pc, dev)
    cu_fcc = cubochoric_fcc_grid(match.h_fcc, dev)

    row_top = build_row(cu_pc, LAUE_O)
    row_bot = build_row(cu_fcc, LAUE_O)

    settings = Settings()
    try:
        with open(args.settings, "r", encoding="utf-8") as f:
            raw = json.load(f)
        settings = settings_from_raw(raw, settings)
        print(f"Loaded settings from {args.settings}")
    except FileNotFoundError:
        pass

    ui = Figure7Interactive(row_top, row_bot, match, settings)
    ui.tb_settings_path.set_val(args.settings)

    if args.save:
        with plt.rc_context({"text.usetex": True}):
            ui.redraw(full_cbar=True, use_tex=True)
            ui.fig.savefig(f"{DEFAULT_STEM}.png", dpi=100, bbox_inches="tight")
            ui.fig.savefig(f"{DEFAULT_STEM}.pdf", bbox_inches="tight")
        ui.redraw(full_cbar=False, use_tex=False)
        print(f"Saved {DEFAULT_STEM}.png and {DEFAULT_STEM}.pdf (TeX export)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
