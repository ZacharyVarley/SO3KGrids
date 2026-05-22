#!/usr/bin/env python3
"""
stereo3d_end_to_end.py

3D-only stereo axes + rewritten Figure7 usage (UI + GIF + save).

Key fixes vs your broken rendition:
  1) No dist= passed to Axes3D.view_init (Matplotlib version-safe).
  2) Uses the reference shareview + quaternion-linked view_init override.
  3) Explicit stereo mode:
       - mode="parallel": left panel = left eye, right panel = right eye
       - mode="cross":    left panel = right eye, right panel = left eye
     This matches what your eyes actually do. If you cross-view, use mode="cross"
     OR set ipd negative (reference convention). This code supports both.

Dependencies: your project modules:
  grid_FZ.py, laue_ops.py, orientation_ops.py
"""

from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import _api
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d.axes3d import Axes3D, _Quaternion

from PIL import Image


# =========================
# Stereo 3D axes (reference-faithful)
# =========================


def _view_init_linked(
    self: Axes3D, elev=None, azim=None, roll=None, vertical_axis="z", share=False
):
    """
    Reference-style override for Axes3D.view_init:
      - updates shared 'view' siblings
      - preserves relative stereo offsets via quaternion correction
    """
    # IMPORTANT: do not hard-reset dist here; that was a subtle bug in many renditions.
    # Reference sets self._dist=10 unconditionally; that *fights* external zoom control.
    # Keep current zoom instead.
    if not hasattr(self, "_dist"):
        self._dist = 10

    if elev is None:
        elev = self.initial_elev
    if azim is None:
        azim = self.initial_azim
    if roll is None:
        roll = self.initial_roll

    vertical_axis = _api.check_getitem(dict(x=0, y=1, z=2), vertical_axis=vertical_axis)

    if share:
        axes = {sib for sib in self._shared_axes["view"].get_siblings(self)}
        axes.remove(self)
        axes = [self] + list(axes)
    else:
        axes = [self]

    # Apply the requested angles to self, then derive sibling angles by rotating
    # around z by the difference in stereo_offset
    for ax in axes:
        _elev, _azim, _roll = elev, azim, roll

        if (
            hasattr(ax, "stereo_offset")
            and hasattr(self, "stereo_offset")
            and ax is not self
        ):
            q = _Quaternion.from_cardan_angles(*np.deg2rad((_elev, _azim, _roll)))
            th = np.deg2rad(self.stereo_offset - ax.stereo_offset)
            k = np.array([0.0, 0.0, 1.0])
            dq = _Quaternion(np.cos(th), k * np.sin(th))
            q2 = dq * q
            _elev, _azim, _roll = np.rad2deg(q2.as_cardan_angles())

        ax.elev = _elev
        ax.azim = _azim
        ax.roll = _roll
        ax._vertical_axis = vertical_axis


def _set_zoom(ax: Axes3D, dist: float) -> None:
    # Matplotlib versions differ: some have ax.dist property, but _dist exists everywhere.
    if hasattr(ax, "dist"):
        try:
            ax.dist = float(dist)
            return
        except Exception:
            pass
    ax._dist = float(dist)


class AxesStereo3D:
    """
    3D-only stereo axes wrapper.

    Conventions:
      - center azim/elev are the cyclopean camera angles
      - per-eye is azim +/- offset

    mode:
      - "parallel": left subplot shows left-eye view
      - "cross":    left subplot shows right-eye view (for cross-eyed viewing)
    """

    def __init__(
        self,
        fig: Optional[Figure] = None,
        axs: Optional[Tuple[Axes3D, Axes3D]] = None,
        *,
        eye_balance: float = -1.0,
        d: float = 350.0,
        ipd: float = 65.0,
        mode: str = "cross",
    ):
        self.eye_balance = float(eye_balance)
        self.d = float(d)
        self.ipd = float(ipd)
        self.mode = str(mode).lower()
        if self.mode not in ("parallel", "cross"):
            raise ValueError("mode must be 'parallel' or 'cross'")

        if fig is None and axs is None:
            fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
            axL, axR = axs[0], axs[1]
        elif axs is None:
            axL = fig.add_subplot(121, projection="3d")
            axR = fig.add_subplot(122, projection="3d")
        else:
            fig = axs[0].figure
            axL, axR = axs

        self.fig = fig
        self.ax_left = axL
        self.ax_right = axR

        self.ax_left.sharex(self.ax_right)
        self.ax_left.sharey(self.ax_right)
        self.ax_left.sharez(self.ax_right)

        # Share view state and install reference-style linked view_init
        self.ax_left.shareview(self.ax_right)
        self.ax_left.view_init = _view_init_linked.__get__(self.ax_left, Axes3D)
        self.ax_right.view_init = _view_init_linked.__get__(self.ax_right, Axes3D)

    def calc_3d_offsets(self) -> Tuple[float, float]:
        # Reference formula
        ang = 90.0 - np.rad2deg(np.arctan(2.0 * self.d / max(abs(self.ipd), 1e-12)))
        offset = (ang / 2.0) * np.sign(self.ipd)
        offL = (self.eye_balance + 1.0) / 2.0 * offset
        offR = (1.0 - self.eye_balance) / 2.0 * offset
        return float(offL), float(offR)

    def set_view(
        self,
        *,
        elev: float,
        azim: float,
        roll: Optional[float] = None,
        zoom_dist: Optional[float] = None,
    ):
        # Set zoom separately (no dist= keyword)
        if zoom_dist is not None:
            _set_zoom(self.ax_left, zoom_dist)
            _set_zoom(self.ax_right, zoom_dist)

        offL, offR = self.calc_3d_offsets()

        # For "cross" mode, swap which eye goes to which subplot.
        # This matches: left eye looks at right subplot.
        if self.mode == "parallel":
            # left subplot = left eye; right subplot = right eye
            self.ax_left.stereo_offset = -offL
            self.ax_right.stereo_offset = +offR
            self.ax_left.view_init(elev=elev, azim=azim - offL, roll=roll, share=True)
            self.ax_right.view_init(elev=elev, azim=azim + offR, roll=roll, share=True)
        else:
            # left subplot = right eye; right subplot = left eye
            self.ax_left.stereo_offset = +offR
            self.ax_right.stereo_offset = -offL
            self.ax_left.view_init(elev=elev, azim=azim + offR, roll=roll, share=True)
            self.ax_right.view_init(elev=elev, azim=azim - offL, roll=roll, share=True)

    def __getattr__(self, name: str):
        # Pass-through that forces stereo view offsets before any plotting call
        def method(*args, **kwargs):
            ax_method = getattr(self.ax_left, name, None)
            if ax_method is None:
                raise AttributeError(name)

            # Compute offsets and apply *relative to current shared azim*
            offL, offR = self.calc_3d_offsets()

            # In shareview, ax_left.azim represents the shared base after interaction.
            base_elev = float(getattr(self.ax_left, "elev", 0.0))
            base_azim = float(getattr(self.ax_left, "azim", 0.0))
            base_roll = (
                float(getattr(self.ax_left, "roll", 0.0))
                if hasattr(self.ax_left, "roll")
                else None
            )

            # Re-apply stereo separation around the shared base
            if self.mode == "parallel":
                self.ax_left.stereo_offset = -offL
                self.ax_right.stereo_offset = +offR
                self.ax_left.view_init(
                    elev=base_elev, azim=base_azim - offL, roll=base_roll, share=True
                )
                self.ax_right.view_init(
                    elev=base_elev, azim=base_azim + offR, roll=base_roll, share=True
                )
            else:
                self.ax_left.stereo_offset = +offR
                self.ax_right.stereo_offset = -offL
                self.ax_left.view_init(
                    elev=base_elev, azim=base_azim + offR, roll=base_roll, share=True
                )
                self.ax_right.view_init(
                    elev=base_elev, azim=base_azim - offL, roll=base_roll, share=True
                )

            resL = getattr(self.ax_left, name)(*args, **kwargs)
            resR = getattr(self.ax_right, name)(*args, **kwargs)
            return (resL, resR)

        return method


# =========================
# Rewritten Figure7 usage
# =========================

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

LAUE_O = 11
CARD_O = 24
H_MAX = (3.0 * math.pi / 4.0) ** (1.0 / 3.0)
N_AUG_MIN = 100
N_AUG_MAX = 10_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SETTINGS_FILE = Path("figure7.json")
DEFAULT_STEM = "figure7"

# Your project modules
from grid_FZ import cu_kr_grid, kr_sample_laue
from laue_ops import laue_elements
from orientation_ops import cu2qu, qu_norm, qu_prod, qu_std, qu2ho


@dataclass
class Settings:
    point_size: float = 5.0
    lattice_mode: str = "fcc"  # "fcc" (default) or "primitive"
    grid_index: int = 0
    operator_enabled: List[bool] = None

    style_axis_off: bool = True
    style_ortho: bool = True
    style_panes_visible: bool = False
    style_grid: bool = False

    d: float = 350.0
    ipd: float = 65.0
    zoom_dist: float = 7.0

    stereo_mode: str = "cross"  # fixed: always cross

    gif_frames_per_360: int = 36
    gif_elevation: float = 20.0
    gif_duration_ms: int = 100
    gif_size_by_distance: bool = False

    def __post_init__(self):
        if self.operator_enabled is None:
            self.operator_enabled = [True] * CARD_O


@dataclass
class PrecomputedGrid:
    lattice_mode: str
    semi_edge: int
    h_lattice: int
    n_fz: int
    n_aug: int
    ho: np.ndarray
    op_idx: np.ndarray


_GIF_POOL_DATA: Dict[str, Any] = {}


def _init_gif_pool(payload: Dict[str, Any]):
    global _GIF_POOL_DATA
    _GIF_POOL_DATA = payload


def _render_gif_frame(task: Tuple[int, float]) -> Tuple[int, bytes]:
    idx, azC = task
    cfg = _GIF_POOL_DATA

    fig = Figure(figsize=cfg["figsize"], dpi=cfg["dpi"])
    _ = FigureCanvas(fig)
    axL = fig.add_subplot(121, projection="3d")
    axR = fig.add_subplot(122, projection="3d")
    ax_stereo = AxesStereo3D(
        fig=fig,
        axs=(axL, axR),
        eye_balance=-1.0,
        d=cfg["d"],
        ipd=cfg["ipd"],
        mode="cross",
    )
    ax_stereo.set_view(elev=cfg["elev"], azim=float(azC), zoom_dist=cfg["zoom_dist"])

    r = float(cfg["r"])
    for ax in (ax_stereo.ax_left, ax_stereo.ax_right):
        ax.cla()
        set_equal_zoom(ax, r)
        style_3d(
            ax,
            axis_off=cfg["style_axis_off"],
            ortho=cfg["style_ortho"],
            panes_visible=cfg["style_panes_visible"],
            grid=cfg["style_grid"],
        )

    ho = cfg["ho"]
    c = cfg["c"]
    if ho.shape[0] > 0:
        cmap, norm = cyclic_cmap_24()
        if cfg["gif_size_by_distance"]:
            elev_r = np.deg2rad(float(cfg["elev"]))
            ar = np.deg2rad(float(azC))
            cam = float(cfg["zoom_dist"]) * np.array(
                [
                    np.cos(elev_r) * np.cos(ar),
                    np.cos(elev_r) * np.sin(ar),
                    np.sin(elev_r),
                ]
            )
            dists = np.linalg.norm(ho - cam, axis=1)
            ref = max(float(np.median(dists)), 1e-6)
            sizes = float(cfg["point_size"]) * ref / dists
            sizes = np.clip(
                sizes, float(cfg["point_size"]) * 0.2, float(cfg["point_size"]) * 5.0
            )
        else:
            sizes = float(cfg["point_size"])

        ax_stereo.scatter(
            ho[:, 0],
            ho[:, 1],
            ho[:, 2],
            c=c,
            cmap=cmap,
            norm=norm,
            s=sizes,
            edgecolors="none",
            alpha=1.0,
        )

    _add_tracker_sphere(ax_stereo, r)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    return idx, buf.getvalue()


@torch.no_grad()
def _cubochoric_fcc_grid(h: int, device: torch.device) -> Tensor:
    cu_max = 0.5 * math.pi ** (2.0 / 3.0)
    u = torch.linspace(-cu_max, cu_max, 2 * h + 2, device=device, dtype=torch.float64)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")

    idx = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
    mask = ((i + j + k) % 2) == 0
    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


@torch.no_grad()
def _kr_fcc_grid(h: int) -> Tensor:
    cu = _cubochoric_fcc_grid(h, DEVICE)
    qu = qu_norm(qu_std(cu2qu(cu)))
    return qu_norm(qu_std(kr_sample_laue(qu, LAUE_O)))


@torch.no_grad()
def _augment_qu_fz_to_homochoric(
    qu_fz: Tensor, ops: Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    qu_fz = qu_norm(qu_std(qu_fz)).to(DEVICE)
    n_fz = qu_fz.shape[0]
    ho_list: List[Tensor] = []
    op_idx_list: List[Tensor] = []
    for i in range(ops.shape[0]):
        q_op = qu_norm(qu_prod(qu_fz, ops[i : i + 1].expand(n_fz, -1)))
        ho_list.append(qu2ho(q_op))
        op_idx_list.append(torch.full((n_fz,), i, dtype=torch.long, device=DEVICE))
    ho = torch.cat(ho_list, dim=0).cpu().numpy()
    op_idx = torch.cat(op_idx_list, dim=0).cpu().numpy()
    return ho, op_idx


def precompute_grids(lattice_mode: str) -> List[PrecomputedGrid]:
    ops = laue_elements(LAUE_O).to(device=DEVICE, dtype=torch.float64)
    grids: List[PrecomputedGrid] = []
    for h in range(2, 20):
        if lattice_mode == "fcc":
            n_fz = ((2 * h + 1) ** 3 + 1) // 2
        else:
            n_fz = (2 * h + 1) ** 3
        n_aug = CARD_O * n_fz
        if n_aug < N_AUG_MIN:
            continue
        if n_aug > N_AUG_MAX:
            break
        if lattice_mode == "fcc":
            qu_fz = _kr_fcc_grid(h)
        else:
            qu_fz = cu_kr_grid(h, LAUE_O, DEVICE)
        n_actual = qu_fz.shape[0]
        if n_actual != n_fz:
            n_fz = n_actual
            n_aug = CARD_O * n_fz
            if n_aug > N_AUG_MAX:
                continue
        ho, op_idx = _augment_qu_fz_to_homochoric(qu_fz, ops)
        grids.append(
            PrecomputedGrid(
                lattice_mode=lattice_mode,
                semi_edge=h,
                h_lattice=h,
                n_fz=n_fz,
                n_aug=n_aug,
                ho=ho,
                op_idx=op_idx,
            )
        )
    return grids


def cyclic_cmap_24() -> Tuple[ListedColormap, BoundaryNorm]:
    colors = [plt.cm.tab20(i / 19) for i in range(20)]
    colors += [plt.cm.tab20b((i - 20) / 3) for i in range(20, 24)]
    colors = [(*c[:3], 1.0) for c in colors]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(25) - 0.5, cmap.N)
    return cmap, norm


def sort_operators_by_w(ops: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    w = ops[:, 0].detach().cpu().numpy()
    idx = np.argsort(w)[::-1]
    return idx, w[idx]


def style_3d(
    ax: Axes3D, *, axis_off: bool, ortho: bool, panes_visible: bool, grid: bool
) -> None:
    ax.set_axis_off() if axis_off else ax.set_axis_on()
    ax.set_proj_type("ortho" if ortho else "persp")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a.pane.set_visible(panes_visible)
        except Exception:
            pass
    ax.grid(grid)


def set_equal_zoom(ax: Axes3D, r: float) -> None:
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))


def _add_tracker_sphere(ax_stereo: AxesStereo3D, r_plot: float) -> None:
    cx, cy, cz = 0.0, 0.0, -1.05 * r_plot
    rad = 0.1 * r_plot
    n_u, n_v = 24, 14
    u = np.linspace(0, 2 * np.pi, n_u + 1)
    v = np.linspace(0, np.pi, n_v + 1)
    U, V = np.meshgrid(u, v)
    X = cx + rad * np.sin(V) * np.cos(U)
    Y = cy + rad * np.sin(V) * np.sin(U)
    Z = cz + rad * np.cos(V)
    C = (np.sin(4 * U) * np.sin(4 * V) > 0).astype(float)
    cmap = ListedColormap(["#000000", "#FFFFFF"])
    facecolors = cmap(C)
    kwargs = dict(facecolors=facecolors, shade=False, rstride=1, cstride=1)
    ax_stereo.ax_left.plot_surface(X, Y, Z, **kwargs)
    ax_stereo.ax_right.plot_surface(X, Y, Z, **kwargs)


def plot_homochoric_ball(
    grid: PrecomputedGrid,
    ax_stereo: AxesStereo3D,
    *,
    point_size: float,
    operator_mask: Optional[np.ndarray],
    size_by_distance: bool,
    style_axis_off: bool,
    style_ortho: bool,
    style_panes_visible: bool,
    style_grid: bool,
):
    ho = grid.ho.copy()
    c = grid.op_idx.copy()

    if operator_mask is not None:
        m = operator_mask[grid.op_idx]
        ho = ho[m]
        c = c[m]

    r = 1.05 * H_MAX
    for ax in (ax_stereo.ax_left, ax_stereo.ax_right):
        ax.cla()
        set_equal_zoom(ax, r)
        style_3d(
            ax,
            axis_off=style_axis_off,
            ortho=style_ortho,
            panes_visible=style_panes_visible,
            grid=style_grid,
        )

    cmap, norm = cyclic_cmap_24()

    if ho.shape[0] > 0:
        if size_by_distance:
            # Cyclopean camera pos from the shared view (base angles)
            elev = float(getattr(ax_stereo.ax_left, "elev", 0.0))
            azim = float(getattr(ax_stereo.ax_left, "azim", 0.0))
            dist = float(getattr(ax_stereo.ax_left, "_dist", 7.0))
            er = np.deg2rad(elev)
            ar = np.deg2rad(azim)
            cam = dist * np.array(
                [np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)]
            )
            dists = np.linalg.norm(ho - cam, axis=1)
            ref = max(float(np.median(dists)), 1e-6)
            sizes = point_size * ref / dists
            sizes = np.clip(sizes, point_size * 0.2, point_size * 5.0)
        else:
            sizes = point_size

        ax_stereo.scatter(
            ho[:, 0],
            ho[:, 1],
            ho[:, 2],
            c=c,
            cmap=cmap,
            norm=norm,
            s=sizes,
            edgecolors="none",
            alpha=1.0,
        )

    _add_tracker_sphere(ax_stereo, r)


class Figure7App:
    def __init__(self, grids_by_lattice: Dict[str, List[PrecomputedGrid]]):
        self.grids_by_lattice = grids_by_lattice
        self.s = Settings()
        self.s.stereo_mode = "cross"
        self._loading = False

        ops = laue_elements(LAUE_O)
        self._op_sorted_indices, self._op_sorted_w = sort_operators_by_w(ops)

        self.ax_stereo = AxesStereo3D(
            eye_balance=-1.0, d=self.s.d, ipd=self.s.ipd, mode="cross"
        )
        self.ax_stereo.set_view(elev=0.0, azim=0.0, zoom_dist=self.s.zoom_dist)

        self.fig_ui = plt.figure("Figure 6 — Controls", figsize=(5.8, 7.4), dpi=110)
        self.fig_ui.patch.set_facecolor("white")

        self._widgets: Dict[str, Any] = {}
        self._operator_panels: List[Tuple[List[int], CheckButtons]] = []
        self._ax_info = None

        self._build_ui()
        self._bind_callbacks()
        self.redraw()

    def _build_ui(self):
        self._widgets.clear()
        self._operator_panels.clear()

        xL, wL = 0.06, 0.41
        xR, wR = 0.53, 0.41
        h = 0.022
        dy = 0.048

        def _heading(x0: float, y0: float, w0: float, text: str):
            ax = self.fig_ui.add_axes([x0, y0, w0, 0.012])
            ax.set_axis_off()
            ax.text(
                0.0, 0.0, text.upper(), fontsize=8, fontweight="bold", color="#4b5563"
            )

        def _add_slider(
            name: str,
            x0: float,
            y0: float,
            w0: float,
            label: str,
            lo: float,
            hi: float,
            val: float,
            step=None,
        ):
            ax = self.fig_ui.add_axes([x0, y0, w0, h])
            slider = Slider(ax, label, lo, hi, valinit=val, valstep=step)
            slider.label.set_fontsize(8.7)
            slider.label.set_ha("left")
            slider.label.set_va("bottom")
            slider.label.set_position((0.0, 1.10))
            slider.valtext.set_fontsize(8.2)
            slider.valtext.set_ha("right")
            slider.valtext.set_va("bottom")
            slider.valtext.set_position((1.0, 1.10))
            self._widgets[name] = slider
            return y0 - dy

        def _add_check(
            name: str, x0: float, y0: float, w0: float, label: str, checked: bool
        ):
            ax = self.fig_ui.add_axes([x0, y0, w0, h])
            cb = CheckButtons(ax, [label], [checked])
            for txt in cb.labels:
                txt.set_fontsize(8.4)
            self._widgets[name] = cb
            return y0 - dy * 0.82

        yL = 0.94
        yR = 0.94

        _heading(xL, yL, wL, "Grid")
        yL -= 0.040
        n_grids_default = len(self.grids_by_lattice.get("fcc", []))
        yL = _add_check("lattice_fcc", xL, yL, wL, "FCC KR lattice", True)
        yL = _add_slider(
            "grid_index", xL, yL, wL, "Grid", 0, max(0, n_grids_default - 1), 0, step=1
        )
        yL = _add_slider(
            "point_size", xL, yL, wL, "Point size", 1, 100, self.s.point_size
        )

        self._ax_info = self.fig_ui.add_axes([xL, yL - 0.006, wL, 0.050])
        self._ax_info.set_axis_off()
        yL -= 0.070

        _heading(xL, yL, wL, "Stereo (cross fixed)")
        yL -= 0.040
        yL = _add_slider("d", xL, yL, wL, "d (mm)", 200, 2000, self.s.d)
        yL = _add_slider("ipd", xL, yL, wL, "ipd (mm)", -80, 80, self.s.ipd)
        yL = _add_slider("zoom_dist", xL, yL, wL, "Zoom", 3, 25, self.s.zoom_dist)

        _heading(xL, yL, wL, "3D style")
        yL -= 0.042
        style_h = 0.028
        style_items = [
            ("style_axis_off", "Hide axis", self.s.style_axis_off),
            ("style_ortho", "Ortho", self.s.style_ortho),
            ("style_panes_visible", "Panes", self.s.style_panes_visible),
            ("style_grid", "Grid", self.s.style_grid),
        ]
        cw = wL / 2
        for i, (key, label, checked) in enumerate(style_items):
            r = i // 2
            c = i % 2
            ax_s = self.fig_ui.add_axes(
                [xL + c * cw, yL - r * (style_h + 0.006), cw - 0.006, style_h]
            )
            cb = CheckButtons(ax_s, [label], [checked])
            for txt in cb.labels:
                txt.set_fontsize(8.2)
            self._widgets[key] = cb
        yL -= 2 * (style_h + 0.006) + 0.012

        _heading(xR, yR, wR, "GIF")
        yR -= 0.040
        yR = _add_slider(
            "gif_frames_per_360",
            xR,
            yR,
            wR,
            "Frames / 360°",
            4,
            360,
            self.s.gif_frames_per_360,
            step=1,
        )
        yR = _add_slider(
            "gif_elevation", xR, yR, wR, "Elevation (°)", -90, 90, self.s.gif_elevation
        )
        yR = _add_slider(
            "gif_duration_ms",
            xR,
            yR,
            wR,
            "Duration (ms)",
            20,
            500,
            self.s.gif_duration_ms,
            step=10,
        )
        yR = _add_check(
            "gif_size_by_distance",
            xR,
            yR,
            wR,
            "Size ∝ 1/distance",
            self.s.gif_size_by_distance,
        )

        _heading(xR, yR, wR, "File")
        yR -= 0.040
        ax_set = self.fig_ui.add_axes([xR, yR, wR, h])
        tb_set = TextBox(ax_set, "Settings", initial=str(SETTINGS_FILE))
        tb_set.label.set_fontsize(8.7)
        tb_set.label.set_ha("left")
        tb_set.label.set_va("bottom")
        tb_set.label.set_position((0.0, 1.08))
        if hasattr(tb_set, "text_disp"):
            tb_set.text_disp.set_fontsize(8.2)
        self._widgets["settings_path"] = tb_set
        yR -= dy

        ax_stem = self.fig_ui.add_axes([xR, yR, wR, h])
        tb_stem = TextBox(ax_stem, "Stem", initial=DEFAULT_STEM)
        tb_stem.label.set_fontsize(8.7)
        tb_stem.label.set_ha("left")
        tb_stem.label.set_va("bottom")
        tb_stem.label.set_position((0.0, 1.08))
        if hasattr(tb_stem, "text_disp"):
            tb_stem.text_disp.set_fontsize(8.2)
        self._widgets["output_stem"] = tb_stem
        yR -= dy + 0.006

        bw = (wR - 0.02) / 2
        self._widgets["save_settings"] = Button(
            self.fig_ui.add_axes([xR, yR, bw, 0.042]), "Save cfg"
        )
        self._widgets["load_settings"] = Button(
            self.fig_ui.add_axes([xR + bw + 0.02, yR, bw, 0.042]), "Load cfg"
        )
        yR -= 0.052
        self._widgets["save_png"] = Button(
            self.fig_ui.add_axes([xR, yR, bw, 0.042]), "Save PNG"
        )
        self._widgets["save_pdf"] = Button(
            self.fig_ui.add_axes([xR + bw + 0.02, yR, bw, 0.042]), "Save PDF"
        )
        yR -= 0.052
        self._widgets["save_gif"] = Button(
            self.fig_ui.add_axes([xR, yR, wR, 0.042]), "Save GIF"
        )

        op_top = min(yL, yR) - 0.012
        _heading(0.06, op_top, 0.88, "Operators (sorted by w)")
        op_y = op_top - 0.215
        panel_h = 0.195

        left_idxs = list(self._op_sorted_indices[: CARD_O // 2])
        right_idxs = list(self._op_sorted_indices[CARD_O // 2 :])

        left_labels = [
            f"Op {idx:2d}  w={self._op_sorted_w[i]:.2f}"
            for i, idx in enumerate(left_idxs)
        ]
        left_states = [bool(self.s.operator_enabled[idx]) for idx in left_idxs]
        ax_ops_left = self.fig_ui.add_axes([0.06, op_y, 0.41, panel_h])
        cb_left = CheckButtons(ax_ops_left, left_labels, left_states)
        for txt in cb_left.labels:
            txt.set_fontsize(7.8)
        self._widgets["ops_left"] = cb_left
        self._operator_panels.append((left_idxs, cb_left))

        right_labels = [
            f"Op {idx:2d}  w={self._op_sorted_w[i + CARD_O // 2]:.2f}"
            for i, idx in enumerate(right_idxs)
        ]
        right_states = [bool(self.s.operator_enabled[idx]) for idx in right_idxs]
        ax_ops_right = self.fig_ui.add_axes([0.53, op_y, 0.41, panel_h])
        cb_right = CheckButtons(ax_ops_right, right_labels, right_states)
        for txt in cb_right.labels:
            txt.set_fontsize(7.8)
        self._widgets["ops_right"] = cb_right
        self._operator_panels.append((right_idxs, cb_right))

    def _sync_from_widgets(self):
        W = self._widgets
        self.s.point_size = float(W["point_size"].val)
        self.s.lattice_mode = "fcc" if W["lattice_fcc"].get_status()[0] else "primitive"
        self.s.grid_index = int(W["grid_index"].val)

        self.s.style_axis_off = W["style_axis_off"].get_status()[0]
        self.s.style_ortho = W["style_ortho"].get_status()[0]
        self.s.style_panes_visible = W["style_panes_visible"].get_status()[0]
        self.s.style_grid = W["style_grid"].get_status()[0]

        self.s.d = float(W["d"].val)
        self.s.ipd = float(W["ipd"].val)
        self.s.zoom_dist = float(W["zoom_dist"].val)
        self.s.stereo_mode = "cross"

        self.s.gif_frames_per_360 = int(W["gif_frames_per_360"].val)
        self.s.gif_elevation = float(W["gif_elevation"].val)
        self.s.gif_duration_ms = int(W["gif_duration_ms"].val)
        self.s.gif_size_by_distance = W["gif_size_by_distance"].get_status()[0]

        for indices, cb in self._operator_panels:
            states = cb.get_status()
            for i, orig_idx in enumerate(indices):
                self.s.operator_enabled[orig_idx] = bool(states[i])

    def _bind_callbacks(self):
        for name, w in self._widgets.items():
            if name in (
                "save_png",
                "save_pdf",
                "load_settings",
                "save_settings",
                "save_gif",
                "settings_path",
                "output_stem",
            ):
                continue
            if hasattr(w, "on_changed"):
                w.on_changed(lambda _: self.redraw())
            elif hasattr(w, "on_clicked"):
                w.on_clicked(lambda _: self.redraw())

        self._widgets["save_png"].on_clicked(self._save_png)
        self._widgets["save_pdf"].on_clicked(self._save_pdf)
        self._widgets["load_settings"].on_clicked(self._load_settings)
        self._widgets["save_settings"].on_clicked(self._save_settings)
        self._widgets["save_gif"].on_clicked(self._save_gif)

    def redraw(self):
        if self._loading:
            return
        self._sync_from_widgets()
        s = self.s

        grids = self.grids_by_lattice.get(s.lattice_mode, [])
        if not grids:
            fallback = "primitive" if s.lattice_mode == "fcc" else "fcc"
            grids = self.grids_by_lattice.get(fallback, [])
            s.lattice_mode = fallback
        if not grids:
            return

        idx = max(0, min(s.grid_index, len(grids) - 1))
        grid = grids[idx]

        # Update stereo parameters
        self.ax_stereo.d = s.d
        self.ax_stereo.ipd = s.ipd
        self.ax_stereo.mode = "cross"

        # Enforce a known base view every redraw (keeps UI deterministic).
        # Interaction still works between redraws; redraw resets.
        self.ax_stereo.set_view(elev=0.0, azim=0.0, zoom_dist=s.zoom_dist)

        operator_mask = np.array(s.operator_enabled, dtype=bool)

        plot_homochoric_ball(
            grid,
            self.ax_stereo,
            point_size=s.point_size,
            operator_mask=operator_mask,
            size_by_distance=s.gif_size_by_distance,
            style_axis_off=s.style_axis_off,
            style_ortho=s.style_ortho,
            style_panes_visible=s.style_panes_visible,
            style_grid=s.style_grid,
        )

        self._ax_info.clear()
        self._ax_info.set_axis_off()
        enabled_count = int(np.sum(operator_mask))
        self._ax_info.text(
            0,
            0.5,
            f"lattice={grid.lattice_mode}  h={grid.h_lattice}  N_FZ={grid.n_fz}  N_aug={grid.n_aug}  Ops enabled: {enabled_count}/{CARD_O}",
            fontsize=8,
            va="center",
            family="monospace",
        )

        self.ax_stereo.fig.canvas.draw_idle()
        self.fig_ui.canvas.draw_idle()

    def _save_png(self, _=None):
        stem = self._widgets["output_stem"].text.strip() or DEFAULT_STEM
        path = f"{stem}.png"
        self.ax_stereo.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

    def _save_pdf(self, _=None):
        stem = self._widgets["output_stem"].text.strip() or DEFAULT_STEM
        path = f"{stem}.pdf"
        self.ax_stereo.fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    def _save_gif(self, _=None):
        self._sync_from_widgets()
        s = self.s

        grids = self.grids_by_lattice.get(s.lattice_mode, [])
        if not grids:
            fallback = "primitive" if s.lattice_mode == "fcc" else "fcc"
            grids = self.grids_by_lattice.get(fallback, [])
        if not grids:
            return

        idx = max(0, min(s.grid_index, len(grids) - 1))
        grid = grids[idx]
        operator_mask = np.array(s.operator_enabled, dtype=bool)

        ho = grid.ho.copy()
        c = grid.op_idx.copy()
        if operator_mask is not None:
            m = operator_mask[grid.op_idx]
            ho = ho[m]
            c = c[m]

        r = 1.05 * H_MAX

        n_frames = max(2, int(s.gif_frames_per_360))
        step = 360.0 / n_frames
        azimuths = [i * step for i in range(n_frames)]
        elev = float(s.gif_elevation)
        dist = float(s.zoom_dist)

        frames: List[Image.Image] = []
        stem = self._widgets["output_stem"].text.strip() or DEFAULT_STEM
        outpath = f"{stem}.gif"
        dpi = 100
        # Suggested MP4 command (chosen settings from figure7compress.sh):
        # ffmpeg -y -v warning -i figure7.gif -vf
        # "fps=50,null,scale=960:-2:flags=lanczos" -an -c:v libx264 -preset slow
        # -crf 28 -pix_fmt yuv420p -movflags +faststart
        # sweep_out/rot_f50_w960_crf28.mp4

        payload = {
            "ho": ho,
            "c": c,
            "r": float(r),
            "point_size": float(s.point_size),
            "gif_size_by_distance": bool(s.gif_size_by_distance),
            "style_axis_off": bool(s.style_axis_off),
            "style_ortho": bool(s.style_ortho),
            "style_panes_visible": bool(s.style_panes_visible),
            "style_grid": bool(s.style_grid),
            "d": float(s.d),
            "ipd": float(s.ipd),
            "zoom_dist": float(dist),
            "elev": float(elev),
            "dpi": int(dpi),
            "figsize": tuple(float(v) for v in self.ax_stereo.fig.get_size_inches()),
        }

        tasks = list(enumerate(azimuths))
        max_workers = max(1, min(os.cpu_count() or 1, len(tasks)))
        print(f"Rendering GIF frames with pool ({max_workers} workers)...")

        rendered: List[Tuple[int, bytes]] = []
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=max_workers,
                initializer=_init_gif_pool,
                initargs=(payload,),
            ) as pool:
                rendered = pool.map(_render_gif_frame, tasks)
        except Exception as e:
            print(f"Pool rendering failed ({e}); falling back to sequential rendering.")
            _init_gif_pool(payload)
            rendered = [_render_gif_frame(t) for t in tasks]

        rendered.sort(key=lambda t: t[0])
        for _, png_bytes in rendered:
            buf = io.BytesIO(png_bytes)
            frames.append(Image.open(buf).copy())

        if not frames:
            return

        duration_ms = max(20, int(s.gif_duration_ms))
        frames[0].save(
            outpath,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Saved {outpath} ({len(frames)} frames)")
        self.redraw()

    def _load_settings(self, _=None):
        settings_path = Path(
            self._widgets["settings_path"].text.strip() or str(SETTINGS_FILE)
        )
        if not settings_path.exists():
            print(f"{settings_path} not found.")
            return
        with open(settings_path) as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(self.s, k):
                if k == "operator_enabled":
                    v = list(v) if isinstance(v, list) else [True] * CARD_O
                    v = [bool(x) for x in v[:CARD_O]]
                    v = v + [True] * (CARD_O - len(v))
                    setattr(self.s, k, v)
                elif k == "lattice_mode":
                    setattr(self.s, k, "fcc" if v == "fcc" else "primitive")
                elif k == "stereo_mode":
                    setattr(self.s, k, "cross")
                else:
                    setattr(self.s, k, v)

        self._loading = True
        try:
            self._widgets["point_size"].set_val(self.s.point_size)
            self._widgets["grid_index"].set_val(self.s.grid_index)
            self._widgets["d"].set_val(self.s.d)
            self._widgets["ipd"].set_val(self.s.ipd)
            self._widgets["zoom_dist"].set_val(self.s.zoom_dist)
            self._widgets["gif_frames_per_360"].set_val(self.s.gif_frames_per_360)
            self._widgets["gif_elevation"].set_val(self.s.gif_elevation)
            self._widgets["gif_duration_ms"].set_val(self.s.gif_duration_ms)

            fcc_checked = bool(self._widgets["lattice_fcc"].get_status()[0])
            want_fcc = self.s.lattice_mode == "fcc"
            if fcc_checked != want_fcc:
                self._widgets["lattice_fcc"].set_active(0)

            gif_checked = bool(self._widgets["gif_size_by_distance"].get_status()[0])
            if gif_checked != bool(self.s.gif_size_by_distance):
                self._widgets["gif_size_by_distance"].set_active(0)

            for key, desired in (
                ("style_axis_off", self.s.style_axis_off),
                ("style_ortho", self.s.style_ortho),
                ("style_panes_visible", self.s.style_panes_visible),
                ("style_grid", self.s.style_grid),
            ):
                got = bool(self._widgets[key].get_status()[0])
                if got != bool(desired):
                    self._widgets[key].set_active(0)

            for indices, cb in self._operator_panels:
                states = cb.get_status()
                for i, orig_idx in enumerate(indices):
                    got = bool(states[i])
                    want = bool(self.s.operator_enabled[orig_idx])
                    if got != want:
                        cb.set_active(i)
        finally:
            self._loading = False

        print(f"Loaded {settings_path}")
        self.redraw()

    def _save_settings(self, _=None):
        self._sync_from_widgets()
        self.s.stereo_mode = "cross"
        settings_path = Path(
            self._widgets["settings_path"].text.strip() or str(SETTINGS_FILE)
        )
        with open(settings_path, "w") as f:
            json.dump(asdict(self.s), f, indent=2)
        print(f"Saved {settings_path}")


def main():
    torch.set_default_dtype(torch.float64)
    print(
        "Precomputing grids (O, FCC + primitive KR, 100–10000 points after augmentation)..."
    )
    grids_fcc = precompute_grids("fcc")
    grids_primitive = precompute_grids("primitive")
    if not grids_fcc and not grids_primitive:
        print("No grids in range. Adjust N_AUG_MIN/N_AUG_MAX.")
        return
    print("Found FCC:", ", ".join(f"N_aug={g.n_aug}" for g in grids_fcc) or "none")
    print(
        "Found primitive:",
        ", ".join(f"N_aug={g.n_aug}" for g in grids_primitive) or "none",
    )
    _ = Figure7App({"fcc": grids_fcc, "primitive": grids_primitive})
    plt.show()


if __name__ == "__main__":
    main()
