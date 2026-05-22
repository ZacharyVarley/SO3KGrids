#!/usr/bin/env python
"""Figure 1 — 3D rejection vs KR scatter in FZ (manual / non-interactive)."""
import colorsys, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from batlow import batlow_cmap
from grid_FZ import cu_kr_grid
from laue_ops import laue_dist_to_fz_boundary, ori_to_fz_laue
from orientation_ops import cu2ho, ho2qu, qu2rf, qu_std
from riesz_energy import riesz_energies_fused
from figure_ui_common import add_panel_label

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS — edit these directly                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Output
STEM = "figure1_col"
DPI = 600

# Grid
LAUE_ID = 10        # tetrahedral
LAUE_CARD = 12
SEED = 42
DEVICE = torch.device("cpu")
N_TARGET_FZ = 1000  # index 0 of the old options list

# Figure size (IUCr single column)
FIG_W = 3.25
FIG_H = 5.5

# Subplot margins / gridspec
MARGIN_LEFT = 0.02
MARGIN_RIGHT = 0.82
MARGIN_TOP = 0.96
MARGIN_BOTTOM = 0.02
GS_HSPACE = 0.12
GS_WSPACE = 0.10
CBAR_RATIO = 0.09

# 3D view
VIEW_ELEV = 30.0
VIEW_AZIM = -90.0
ZOOM_R = 0.8
ZOOM_CENTER_Z = 0.0

# Cut plane
CUT_ANGLE_DEG = 90.0
CUT_PLANE_OFFSET = 0.0
KEEP_SIDE_SIGN = 1.0
ONLY_OCTANT_XYZ_POS = False
BOUNDARY_THRESH_DEG = 10.0

# Scatter
POINT_SIZE = 28.0
P_LO = 1.0
P_HI = 99.0

# FZ wireframe
DRAW_FACES = True
DRAW_FACE_EDGES = True
WIRE_THICKNESS = 1.0
FACE_ALPHA = 1.0
FACE_COLOR_HUE = 0.0
FACE_COLOR_LIGHTNESS = 0.92
FACE_ENABLED = (True, False, False, True, True, False, False, True)

# Typography
TITLE_SIZE = 9.0
SUBTITLE_SIZE = 7.5
PANEL_LABEL_SIZE = 9.0
PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
SUBTITLE_X = 0.52
SUBTITLE_Y = 0.98
CBAR_FONT_SIZE = 7.0
TITLE_PAD = 2.0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GEOMETRY                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

A_CU = (math.pi ** (2.0 / 3.0)) / 2.0

OCT_VERTS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=float,
)
OCT_FACES = [
    (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
    (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
]
OCT_EDGES = [
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (3, 4), (3, 5),
]


def hls_to_rgb(h, l, s=1.0):
    return colorsys.hls_to_rgb(float(h), float(l), float(s))


def set_equal_zoom(ax, r):
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))


def rotation_normal_z(deg):
    th = math.radians(deg)
    return np.array([math.cos(th), math.sin(th), 0.0])


def cut_mask(rf, n, sign, offset=0.0):
    return (rf @ n - offset) * sign < 0.0


def octant_pos(rf):
    return (rf[:, 0] >= 0) & (rf[:, 1] >= 0) & (rf[:, 2] >= 0)


def _view_plane_normal(elev, azim):
    er, ar = math.radians(elev), math.radians(azim)
    return np.array([math.cos(er) * math.cos(ar), math.cos(er) * math.sin(ar), math.sin(er)])


def _split_segments_by_view(segments, elev, azim, eps=1e-12):
    normal = _view_plane_normal(elev, azim)
    behind, front = [], []
    for seg in segments:
        if seg.shape[0] < 2:
            continue
        mid = 0.5 * (seg[0] + seg[-1])
        (front if float(np.dot(mid, normal)) > eps else behind).append(seg)
    return behind, front


def style_3d(ax):
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a.pane.set_visible(False)
        except Exception:
            pass
    ax.grid(False)


def semi_edge_for_target_fz(n_target_fz, laue_card):
    semi_kr = int(round((n_target_fz ** (1 / 3) - 1) / 2))
    semi_rej = int(round(((n_target_fz * laue_card) ** (1 / 3) - 1) / 2))
    return semi_rej, semi_kr


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA BUILDING                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def build_dataset(n_target_fz, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    semi_rej, semi_kr = semi_edge_for_target_fz(n_target_fz, LAUE_CARD)
    print(f"semi_rej={semi_rej}, semi_kr={semi_kr} for n_target_fz={n_target_fz}")

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

    return dict(
        rf_rej=qu2rf(qu_rej).cpu().numpy(),
        nn_rej=nn_rej.cpu().numpy(),
        bd_rej=bd_rej.cpu().numpy(),
        rf_kr=qu2rf(qu_kr).cpu().numpy(),
        nn_kr=nn_kr.cpu().numpy(),
        bd_kr=bd_kr.cpu().numpy(),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def draw_fz_faces(ax, zoom_center, zorder=-2e6):
    if not DRAW_FACES:
        return
    vt = OCT_VERTS - zoom_center
    tris = [vt[list(f)] for fi, f in enumerate(OCT_FACES) if FACE_ENABLED[fi]]
    if not tris:
        return
    rgb = hls_to_rgb(FACE_COLOR_HUE, FACE_COLOR_LIGHTNESS)
    ax.add_collection3d(
        Poly3DCollection(tris, facecolor=(*rgb, FACE_ALPHA), edgecolor="none",
                         linewidths=1.0, alpha=1.0, zorder=zorder)
    )


def draw_fz_edges(ax, segments, zorder=0.0):
    if not DRAW_FACE_EDGES or not segments:
        return
    lines = [[seg[0], seg[1]] for seg in segments if seg.shape[0] >= 2]
    if not lines:
        return
    ax.add_collection3d(
        Line3DCollection(lines, colors=[(0.18, 0.18, 0.18, 1.0)],
                         linewidths=max(1.0, WIRE_THICKNESS), alpha=1.0, zorder=zorder)
    )


def fz_edge_segments(zoom_center):
    vt = OCT_VERTS - zoom_center
    edge_set = set()
    for fi, on in enumerate(FACE_ENABLED):
        if not on:
            continue
        f = OCT_FACES[fi]
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_set.add(tuple(sorted((a, b))))
    return [np.stack([vt[i], vt[j]], axis=0) for i, j in OCT_EDGES if tuple(sorted((i, j))) in edge_set]


def main():
    print("Building dataset...")
    ds = build_dataset(N_TARGET_FZ, SEED)

    cmap = batlow_cmap(reverse=True)
    zc = np.array([0.0, 0.0, ZOOM_CENTER_Z])
    rf_rej = ds["rf_rej"] - zc
    rf_kr = ds["rf_kr"] - zc

    # Masks
    cn = rotation_normal_z(CUT_ANGLE_DEG)
    m_rej = cut_mask(rf_rej, cn, KEEP_SIDE_SIGN, CUT_PLANE_OFFSET)
    m_kr = cut_mask(rf_kr, cn, KEEP_SIDE_SIGN, CUT_PLANE_OFFSET)
    if ONLY_OCTANT_XYZ_POS:
        m_rej &= octant_pos(rf_rej)
        m_kr &= octant_pos(rf_kr)
    bd = math.radians(BOUNDARY_THRESH_DEG)
    if bd > 0:
        m_rej &= ds["bd_rej"] < bd
        m_kr &= ds["bd_kr"] < bd

    rf_rej_f, nn_rej_f = rf_rej[m_rej], ds["nn_rej"][m_rej]
    rf_kr_f, nn_kr_f = rf_kr[m_kr], ds["nn_kr"][m_kr]

    # Depth sort
    for pts, vals in ((rf_rej_f, nn_rej_f), (rf_kr_f, nn_kr_f)):
        if pts.shape[0]:
            o = np.argsort(pts[:, 2])
            pts[:], vals[:] = pts[o], vals[o]

    # Shared colour norm
    all_nn = np.concatenate([nn_rej_f, nn_kr_f]) if nn_rej_f.size + nn_kr_f.size else np.array([0.0, 1.0])
    vmin = float(np.percentile(all_nn, P_LO)) if all_nn.size else 0.0
    vmax = float(np.percentile(all_nn, P_HI)) if all_nn.size else 1.0
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # Create figure
    with plt.rc_context({"text.usetex": True}):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = GridSpec(2, 2, width_ratios=[1.0, CBAR_RATIO], height_ratios=[1.0, 1.0],
                      wspace=GS_WSPACE, hspace=GS_HSPACE,
                      left=MARGIN_LEFT, right=MARGIN_RIGHT, top=MARGIN_TOP, bottom=MARGIN_BOTTOM)
        axL = fig.add_subplot(gs[0, 0], projection="3d")
        axR = fig.add_subplot(gs[1, 0], projection="3d")
        cax = fig.add_subplot(gs[:, 1])

        for ax in (axL, axR):
            style_3d(ax)
            ax.computed_zorder = False
            set_equal_zoom(ax, ZOOM_R)
            ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

        title_left = r"$\textrm{Rejection sampling}$"
        title_right = r"$\textrm{KR rearrangement}$"
        n_rej = int(rf_rej_f.shape[0] + np.sum(~m_rej))
        n_kr = int(rf_kr_f.shape[0] + np.sum(~m_kr))
        sub_left = rf"$N_{{\mathrm{{FZ}}}}={n_rej}$"
        sub_right = rf"$N_{{\mathrm{{FZ}}}}={n_kr}$"

        axL.set_title(title_left, fontsize=int(round(TITLE_SIZE)), pad=TITLE_PAD)
        axR.set_title(title_right, fontsize=int(round(TITLE_SIZE)), pad=TITLE_PAD)

        add_panel_label(axL, "(a)", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)
        add_panel_label(axR, "(b)", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)

        fn = axL.text2D if hasattr(axL, "text2D") else axL.text
        fn(SUBTITLE_X, SUBTITLE_Y, sub_left, transform=axL.transAxes,
           ha="left", va="top", fontsize=int(round(SUBTITLE_SIZE)))
        fn2 = axR.text2D if hasattr(axR, "text2D") else axR.text
        fn2(SUBTITLE_X, SUBTITLE_Y, sub_right, transform=axR.transAxes,
            ha="left", va="top", fontsize=int(round(SUBTITLE_SIZE)))

        z_face, z_back, z_points, z_front = -2e6, -1e6, 0.0, 1e6
        edge_segs = fz_edge_segments(zc)

        # Faces
        draw_fz_faces(axL, zc, zorder=z_face)
        draw_fz_faces(axR, zc, zorder=z_face)

        # Back edges
        back_l, front_l = _split_segments_by_view(edge_segs, VIEW_ELEV, VIEW_AZIM)
        back_r, front_r = _split_segments_by_view(edge_segs, VIEW_ELEV, VIEW_AZIM)
        draw_fz_edges(axL, back_l, zorder=z_back)
        draw_fz_edges(axR, back_r, zorder=z_back)

        # Scatter
        kw = dict(cmap=cmap, norm=norm, alpha=1.0, s=POINT_SIZE,
                  edgecolors="none", depthshade=False, rasterized=True, zorder=z_points)
        scL = scR = None
        if rf_rej_f.shape[0]:
            scL = axL.scatter(rf_rej_f[:, 0], rf_rej_f[:, 1], rf_rej_f[:, 2], c=nn_rej_f, **kw)
        if rf_kr_f.shape[0]:
            scR = axR.scatter(rf_kr_f[:, 0], rf_kr_f[:, 1], rf_kr_f[:, 2], c=nn_kr_f, **kw)

        # Front edges
        draw_fz_edges(axL, front_l, zorder=z_front)
        draw_fz_edges(axR, front_r, zorder=z_front)

        # Colorbar
        mappable = scR or scL
        if mappable is None:
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
        cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
        cfs = int(round(CBAR_FONT_SIZE))
        cb.set_label(r"$\textrm{NN chordal (FZ)}$", fontsize=cfs)
        cb.ax.tick_params(labelsize=max(6, cfs - 1))

        # Save
        for ext in ("png", "pdf"):
            path = f"{STEM}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"[export] {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
