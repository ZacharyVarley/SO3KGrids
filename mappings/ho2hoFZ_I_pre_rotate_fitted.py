#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Icosahedral (532) KR map in crystallographic conventional setting
=================================================================

- Pentagonal face normal aligned to +z
- A 2-fold axis aligned to +x
- Local alignments are performed against these canonical axes
- Hard-coded Chebyshev inverse-CDF for azimuth on the 1/120 sector
- Saves multiple Plotly HTML views (base, E3-colored, NN-colored, ops-colored)
"""

import math
import argparse
from typing import Tuple

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
import plotly.graph_objects as go

from orientation_ops import cu2ho, ho2qu, qu2ho, qu_prod
from laue_ops import laue_elements
from riesz_energy import riesz_energies_fused

# ------------------------- Numeric/geom constants -------------------------
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
PI = math.pi
EPS = 1e-12

# Homochoric ball radius
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)

SQRT5 = math.sqrt(5.0)
PHI = 0.5 * (1.0 + SQRT5)

# Wedge widths for azimuth folding
WEDGE_72 = 2.0 * PI / 5.0
WEDGE_36 = WEDGE_72 / 2.0  # π/5 = 36°

# KR constants for I
ALPHA = math.sqrt(5.0 - 2.0 * SQRT5)  # tan(18°)
P_CONST = math.sqrt(5.0 + 2.0 * SQRT5)  # cot(18°)
COS_BETA = 1.0 / math.sqrt(5.0)
SIN_BETA = 2.0 / math.sqrt(5.0)


# ------------------------- Fixed canonical rotation (axes-only) -------------------------
def Rz(phi: float, device=None, dtype=DTYPE) -> torch.Tensor:
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device
    )


def Rx(theta: float, device=None, dtype=DTYPE) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=dtype, device=device
    )


def icosahedron_axes_neutral(device=None, dtype=DTYPE) -> torch.Tensor:
    V = torch.tensor(
        [
            [-1.0, PHI, 0.0],
            [1.0, PHI, 0.0],
            [-1.0, -PHI, 0.0],
            [1.0, -PHI, 0.0],
            [0.0, -1.0, PHI],
            [0.0, 1.0, PHI],
            [0.0, -1.0, -PHI],
            [0.0, 1.0, -PHI],
            [PHI, 0.0, -1.0],
            [PHI, 0.0, 1.0],
            [-PHI, 0.0, -1.0],
            [-PHI, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    return V / V.norm(dim=-1, keepdim=True)


def build_neighbor_indices(A: torch.Tensor) -> torch.Tensor:
    dots = A @ A.t()
    dots = dots - torch.eye(12, dtype=A.dtype, device=A.device) * 10.0
    return torch.argmax(dots, dim=-1)


def canonical_rotation_axes_only(device=None, dtype=DTYPE) -> torch.Tensor:
    """
    R_can = Rz(phi) @ Rx(theta), where:
      - Rx(theta) with theta = atan(1/phi) sends a 5-fold to +z,
      - Rz(phi) spins so that a *true 2-fold* lies on +x.
    Uses only the 12 5-fold axes; no faces/vertices.
    """
    A0 = icosahedron_axes_neutral(device=device, dtype=dtype)  # (12,3)
    theta = math.atan(1.0 / PHI)
    R1 = Rx(theta, device=device, dtype=dtype)

    A1 = A0 @ R1.t()  # rotate axes
    top = torch.argmax(A1[:, 2])  # index of axis with max +z
    # deterministic neighbor (closest other 5-fold). The 2-fold direction
    # that bisects top→neighbor lives in the top face's azimuthal zero.
    nb = build_neighbor_indices(A1)[top]

    # project neighbor into xy and compute spin so nb goes to +x
    v = A1[nb]
    phi = -math.atan2(float(v[1]), float(v[0]))
    R2 = Rz(phi, device=device, dtype=dtype)
    return R2 @ R1


# ------------------------- Small linear algebra helpers -------------------------
def z_axis(device=None, dtype=DTYPE) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)


def rodrigues_rotate_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    ax = axis
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    K = torch.zeros(ax.shape[:-1] + (3, 3), dtype=ax.dtype, device=ax.device)
    K[..., 0, 1] = -ax[..., 2]
    K[..., 0, 2] = ax[..., 1]
    K[..., 1, 0] = ax[..., 2]
    K[..., 1, 2] = -ax[..., 0]
    K[..., 2, 0] = -ax[..., 1]
    K[..., 2, 1] = ax[..., 0]
    I = torch.eye(3, dtype=ax.dtype, device=ax.device).expand_as(K)
    ax_outer = ax[..., :, None] * ax[..., None, :]
    return I * c + s * K + (1.0 - c) * ax_outer


def align_to_z(a: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + EPS)
    z = z_axis(device=a.device, dtype=a.dtype).expand_as(a)
    dot = (a * z).sum(dim=-1)
    v = torch.cross(a, z, dim=-1)
    v_norm = v.norm(dim=-1, keepdim=True)
    axis = torch.where(
        v_norm > EPS,
        v / (v_norm + 0.0),
        torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device),
    )
    ang = torch.atan2(v_norm.squeeze(-1), dot)
    R = rodrigues_rotate_matrix(axis, ang)
    close_pos = dot > 1.0 - 1e-12
    close_neg = dot < -1.0 + 1e-12
    if close_pos.any():
        R[close_pos] = torch.eye(3, dtype=a.dtype, device=a.device)
    if close_neg.any():
        R[close_neg] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=a.dtype,
            device=a.device,
        )
    return R


def spin_about_z(gamma: torch.Tensor, device=None, dtype=DTYPE) -> torch.Tensor:
    c = torch.cos(gamma)
    s = torch.sin(gamma)
    R = torch.zeros(gamma.shape + (3, 3), dtype=dtype, device=device)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1.0
    return R


# ------------------------- Tiny sector geometry & KR pieces -------------------------
def theta_max(psi: torch.Tensor) -> torch.Tensor:
    num = 1.0 - COS_BETA
    den = SIN_BETA * torch.clamp(torch.cos(psi), min=EPS)
    return torch.atan(num / den)


def R3_of_theta(theta: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(torch.cos(theta), -1.0 + 1e-15, 1.0 - 1e-15)
    term1 = torch.atan(1.0 / (P_CONST * c))
    term2 = (P_CONST * c) / (1.0 + (P_CONST * c) ** 2)
    return 1.5 * (term1 - term2)


def F_cap(theta: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(torch.cos(theta), -1.0 + 1e-15, 1.0 - 1e-15)
    return 1.5 * (
        0.5 * PI * (1.0 - c) - math.atan(P_CONST) + c * torch.atan(P_CONST * c)
    )


# ------------------------- Hardcoded Chebyshev ψ'(u) fit -------------------------
CHEB_NUM = torch.tensor(
    [
        0.3304924856175057,
        0.31583416591403574,
        -0.016688974639899247,
        -0.0016721216544865944,
        0.0003619870280989233,
        -3.4011730639492566e-06,
        -6.302515376491188e-06,
        6.421149861038183e-07,
        6.949651306457666e-08,
        -2.0250514424156133e-08,
        4.204414792689178e-10,
        4.1271900706295387e-10,
        -4.990132935908946e-11,
        -4.6492593956442474e-12,
        1.6306230055684584e-12,
        -5.1885682094719584e-14,
        -3.3872343958599216e-14,
        4.631902188019415e-15,
    ],
    dtype=DTYPE,
)

PSI_MAX = torch.tensor(math.pi / 5.0, dtype=DTYPE)


@torch.no_grad()
def chebyshev_clenshaw(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if coeffs.numel() == 1:
        return torch.full_like(t, coeffs[0])
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for c_k in coeffs[1:].flip(0):
        b0 = 2.0 * t * b1 - b2 + c_k
        b2, b1 = b1, b0
    return t * b1 - b2 + coeffs[0]


@torch.no_grad()
def psi_prime_from_u(u: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(u, 0.0, 1.0)
    t = 2.0 * u - 1.0
    psi = chebyshev_clenshaw(CHEB_NUM.to(device=u.device, dtype=u.dtype), t)
    return torch.clamp(psi, 0.0, float(PSI_MAX))


# ------------------------- Azimuth folding/unfolding -------------------------
def fold_azimuth(psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    psi = torch.remainder(psi, 2.0 * PI)
    k = torch.floor(psi / WEDGE_72).to(torch.int64)
    psi72 = psi - (k.to(psi.dtype) * WEDGE_72)
    mirror = psi72 > WEDGE_36
    psi_delta = torch.where(mirror, WEDGE_72 - psi72, psi72)
    return psi_delta, mirror, k


def unfold_azimuth(
    psi_prime: torch.Tensor, mirror: torch.Tensor, sector_idx: torch.Tensor
) -> torch.Tensor:
    psi72 = torch.where(mirror, WEDGE_72 - psi_prime, psi_prime)
    psi_full = psi72 + sector_idx.to(psi72.dtype) * WEDGE_72
    return torch.remainder(psi_full, 2.0 * PI)


# ------------------------- Main mapping: ho2ho_I (canonical frame) -------------------------
def ho2ho_I(h: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    device, dtype = h.device, h.dtype

    # unit directions
    rho = h.norm(dim=-1)
    nonzero = rho > eps
    uhat = torch.zeros_like(h)
    uhat[nonzero] = h[nonzero] / rho[nonzero].unsqueeze(-1)

    # canonical 5-fold axes (pre-rotated so top 5-fold at +z and a 2-fold at +x)
    R_can = canonical_rotation_axes_only(device=device, dtype=dtype)
    A_neu = icosahedron_axes_neutral(device=device, dtype=dtype)
    A = A_neu @ R_can.t()  # (12,3)

    # pick face & a deterministic neighbor
    dots = torch.einsum("nd,md->nm", uhat, A)
    idx_face = dots.argmax(dim=-1)
    a = A[idx_face]
    nb_of = build_neighbor_indices(A)
    b = A[nb_of[idx_face]]

    # local: a → +z, then spin so b lies on +x
    R_al = align_to_z(a)
    h_loc = torch.einsum("nij,nj->ni", R_al, h)
    b_al = torch.einsum("nij,nj->ni", R_al, b)
    gamma = -torch.atan2(b_al[:, 1], b_al[:, 0])
    R_sp = spin_about_z(gamma, device=device, dtype=dtype)
    h_loc = torch.einsum("nij,nj->ni", R_sp, h_loc)

    # spherical coords
    rho_loc = h_loc.norm(dim=-1)
    z = h_loc[:, 2]
    r_safe = torch.clamp(rho_loc, min=eps)
    cos_th = torch.clamp(z / r_safe, -1.0, 1.0)
    theta = torch.acos(cos_th)
    psi = torch.atan2(h_loc[:, 1], h_loc[:, 0])
    psi = torch.remainder(psi, 2.0 * PI)

    # fold to π/5 wedge → u
    psi_delta, mirror, sector_idx = fold_azimuth(psi)
    u = psi_delta / WEDGE_36

    # inverse-CDF fit
    psi_prime = psi_prime_from_u(u)

    # polar solve with ceiling
    th_ceiling = theta_max(psi_prime)
    y_src = (1.0 - torch.cos(theta)) / (1.0 - torch.cos(th_ceiling))
    y_src = torch.clamp(y_src, 0.0, 1.0)

    def polar_solve(y, th_max):
        lo = torch.zeros_like(y)
        hi = th_max.clone()
        denom = torch.clamp(F_cap(th_max), min=EPS)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            val = F_cap(mid) / denom - y
            hi = torch.where(val > 0, mid, hi)
            lo = torch.where(val <= 0, mid, lo)
            if (hi - lo).max() < 1e-14:
                break
        return 0.5 * (lo + hi)

    theta_prime = polar_solve(y_src, th_ceiling)

    # radial KR scale; slight retract
    R3 = R3_of_theta(theta_prime)
    R_cap = torch.clamp(R3, min=0.0) ** (1.0 / 3.0)
    rho_prime = (1.0 - 1e-6) * rho_loc * (R_cap / H_MAX)

    # rebuild, unfold, and back-transform
    psi_full = unfold_azimuth(psi_prime, mirror, sector_idx)
    sin_t = torch.sin(theta_prime)
    h_loc_prime = torch.stack(
        [
            rho_prime * sin_t * torch.cos(psi_full),
            rho_prime * sin_t * torch.sin(psi_full),
            rho_prime * torch.cos(theta_prime),
        ],
        dim=-1,
    )

    h_tmp = torch.einsum("nij,nj->ni", R_sp.transpose(-1, -2), h_loc_prime)
    h_out = torch.einsum("nij,nj->ni", R_al.transpose(-1, -2), h_tmp)
    h_out[~nonzero] = 0.0
    return h_out


# ------------------------- Stretched cubochoric grid (demo input) -------------------------
def so3_cubochoric_grid_stretch(
    cu_h: int,
    cu_z: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    cu = torch.linspace(
        -0.5 * torch.pi ** (2.0 / 3.0),
        0.5 * torch.pi ** (2.0 / 3.0),
        2 * cu_h + 2,
        dtype=dtype,
        device=device,
    )
    cu = cu[:-1]
    cu = cu + 0.5 * (cu[1] - cu[0])

    cu_z_lin = torch.linspace(
        -0.5 * torch.pi ** (2.0 / 3.0),
        0.5 * torch.pi ** (2.0 / 3.0),
        2 * cu_z + 2,
        dtype=dtype,
        device=device,
    )
    cu_z_lin = cu_z_lin[:-1]
    cu_z_lin = cu_z_lin + 0.5 * (cu_z_lin[1] - cu_z_lin[0])

    grid = torch.stack(torch.meshgrid(cu, cu, cu_z_lin, indexing="ij"), dim=-1).reshape(
        -1, 3
    )
    return grid


# ---------- discrete palette for up to 60 ops ----------
def palette60():
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    pal = []
    for k in range(6):
        f = 1.0 - 0.08 * k
        for c in base:
            r = int(int(c[1:3], 16) * f)
            g = int(int(c[3:5], 16) * f)
            b = int(int(c[5:7], 16) * f)
            pal.append(f"#{r:02x}{g:02x}{b:02x}")
    return pal[:60]


# ------------------------- CLI: saves multiple HTMLs -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="I(532) — cu2ho → KR(FZ, canonical) with ψ'(u) fit; saves base/E3/NN/ops plots"
    )
    ap.add_argument(
        "--h", type=int, default=11, help="half-resolution in x/y (2h cells per axis)"
    )
    ap.add_argument("--z", type=int, default=11, help="half-resolution in z (2z cells)")
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    ap.add_argument("--out_base", type=str, default="I532_base.html")
    ap.add_argument("--out_e3", type=str, default="I532_E3.html")
    ap.add_argument("--out_nn", type=str, default="I532_NN.html")
    ap.add_argument("--out_ops", type=str, default="I532_ops.html")
    ap.add_argument(
        "--plot_ops",
        action="store_true",
        help="also plot symmetry copies (color by op index)",
    )
    ap.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="max points per op trace (0 = no downsample)",
    )
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    torch.set_grad_enabled(False)

    # 1) cu → ho (source), slight retract from shell
    cu_grid = so3_cubochoric_grid_stretch(args.h, args.z, device=device, dtype=DTYPE)
    ho_src = cu2ho(cu_grid.to(dtype=DTYPE, device=device)) * 0.99999

    # 2) Map once to canonical RFZ
    ho_map = ho2ho_I(ho_src)

    # ----------------- Plot A: base cloud -----------------
    lim = float(H_MAX)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_src[:, 0].detach().cpu().numpy(),
            y=ho_src[:, 1].detach().cpu().numpy(),
            z=ho_src[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="Original (cu→ho)",
            marker=dict(size=2, opacity=0.35),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].detach().cpu().numpy(),
            y=ho_map[:, 1].detach().cpu().numpy(),
            z=ho_map[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="RFZ (canonical)",
            marker=dict(size=2, opacity=0.9),
        )
    )
    fig.update_layout(
        title="I(532) KR — canonical RFZ (top 5-fold at +z, 2-fold along +x)",
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="x"),
            yaxis=dict(range=[-lim, lim], title="y"),
            zaxis=dict(range=[-lim, lim], title="z"),
            aspectmode="cube",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=36, b=0),
        template="plotly_white",
    )
    fig.write_html(args.out_base, include_plotlyjs="cdn")
    print(f"[ok] wrote {args.out_base} with {ho_map.shape[0]} points")

    # 3) Quaternionize RFZ, energies & NN contributions
    q_fz = ho2qu(ho_map.to(dtype=DTYPE, device=device))
    q_fz = q_fz / torch.clamp(q_fz.norm(dim=-1, keepdim=True), min=1e-15)
    ops = laue_elements(12).to(dtype=DTYPE, device=device)  # (G,4)

    E1, E2, E3, _, _, S3_i, NN_i = riesz_energies_fused(
        q_fz, ops, return_contrib=True, return_nn=True
    )

    # ----------------- Plot B: color by E3 contribution -----------------
    s3 = S3_i.detach()
    s3 = s3 / torch.clamp(torch.median(s3), min=1e-15)
    s3_np = s3.cpu().numpy().astype(np.float32)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_src[:, 0].detach().cpu().numpy(),
            y=ho_src[:, 1].detach().cpu().numpy(),
            z=ho_src[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="Original (cu→ho)",
            marker=dict(size=2, opacity=0.25),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].detach().cpu().numpy(),
            y=ho_map[:, 1].detach().cpu().numpy(),
            z=ho_map[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="RFZ colored by E3 contrib",
            marker=dict(
                size=3, color=s3_np, colorscale="Turbo", showscale=True, opacity=1.0
            ),
        )
    )
    fig.update_layout(
        title=f"I(532) RFZ — colored by Riesz E3 contribution (E3 total = {float(E3):.6e})",
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="x"),
            yaxis=dict(range=[-lim, lim], title="y"),
            zaxis=dict(range=[-lim, lim], title="z"),
            aspectmode="cube",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=36, b=0),
        template="plotly_white",
    )
    fig.write_html(args.out_e3, include_plotlyjs="cdn")
    print(f"[ok] wrote {args.out_e3}  (E3 total = {float(E3):.6e})")

    # ----------------- Plot C: color by NN_i -----------------
    NN = NN_i.detach()
    NN = NN / torch.clamp(torch.median(NN), min=1e-15)
    NN_np = NN.cpu().numpy().astype(np.float32)

    print(
        f" NN stats —  min: {float(NN_i.min())},  median: {float(torch.median(NN_i))},  max: {float(NN_i.max())}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_src[:, 0].detach().cpu().numpy(),
            y=ho_src[:, 1].detach().cpu().numpy(),
            z=ho_src[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="Original (cu→ho)",
            marker=dict(size=2, opacity=0.25),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].detach().cpu().numpy(),
            y=ho_map[:, 1].detach().cpu().numpy(),
            z=ho_map[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="RFZ colored by NN",
            marker=dict(
                size=3, color=NN_np, colorscale="Turbo", showscale=True, opacity=1.0
            ),
        )
    )
    fig.update_layout(
        title=f"I(532) RFZ — colored by nearest-neighbor index (median = {float(torch.median(NN_i)):.6e})",
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="x"),
            yaxis=dict(range=[-lim, lim], title="y"),
            zaxis=dict(range=[-lim, lim], title="z"),
            aspectmode="cube",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=36, b=0),
        template="plotly_white",
    )
    fig.write_html(args.out_nn, include_plotlyjs="cdn")
    print(f"[ok] wrote {args.out_nn}")

    # ----------------- Plot D: symmetry copies (optional) -----------------
    # if args.plot_ops:
    G = ops.shape[0]
    colors = palette60()
    max_plot = args.downsample if args.downsample and args.downsample > 0 else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].detach().cpu().numpy(),
            y=ho_map[:, 1].detach().cpu().numpy(),
            z=ho_map[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="RFZ base (op 0)",
            marker=dict(size=2, color="#000000", opacity=0.35),
        )
    )

    for gi in range(G):
        g = ops[gi].unsqueeze(0).expand_as(q_fz)
        q_prime = qu_prod(g, q_fz)
        q_prime = q_prime / torch.clamp(q_prime.norm(dim=-1, keepdim=True), min=1e-15)
        ho_prime = qu2ho(q_prime)

        if max_plot is not None and ho_prime.shape[0] > max_plot:
            idx = torch.randint(
                0, ho_prime.shape[0], (max_plot,), device=ho_prime.device
            )
            xyz = ho_prime[idx].detach().cpu().numpy()
        else:
            xyz = ho_prime.detach().cpu().numpy()

        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=f"op {gi:02d}",
                marker=dict(size=2, color=colors[gi % len(colors)], opacity=1.0),
            )
        )

    fig.update_layout(
        title="I(532) — RFZ copies under all Laue ops (colored by op index)",
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="x"),
            yaxis=dict(range=[-lim, lim], title="y"),
            zaxis=dict(range=[-lim, lim], title="z"),
            aspectmode="cube",
        ),
        legend=dict(font=dict(size=10), x=0.02, y=0.98, itemsizing="constant"),
        margin=dict(l=0, r=0, t=36, b=0),
        template="plotly_white",
    )
    fig.write_html(args.out_ops, include_plotlyjs="cdn")
    print(f"[ok] wrote {args.out_ops} with {G} colored symmetry copies")


if __name__ == "__main__":
    main()
