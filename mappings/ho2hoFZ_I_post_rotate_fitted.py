#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Icosahedral (532) KR map with HARD-CODED Padé/Poly azimuth inverse CDF
======================================================================

This is a drop-in hardcoded version of your original I(532) script that uses
the *fitted* azimuthal inverse CDF ψ'(u) in Chebyshev T_k(t) basis (t=2u−1),
with denominator fixed to 1.0 (i.e., a pure polynomial in the Chebyshev basis).

Changes vs your original:
- Replaced the numeric C(ψ) inversion (Gauss–Legendre + bisection) with a
  direct Chebyshev-Clenshaw evaluation of ψ'(u) using your best-fit coefficients.
- Removed C_of_psi and invert_C; all other logic (folding/unfolding azimuth,
  θ ceiling, constant-Jacobian polar solve, post-rotation to canonical frame)
  remains identical.

Fit details (hardcoded below):
- degree_num = 17
- degree_den = 0 (Q(t) = 1)
- ψ'_max = π/5
- sup_error ≈ 1.22e−15, rms_error ≈ 3.44e−16
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
from orientation_ops import cu2ho, ho2qu, qu_prod, qu2ho
import plotly.graph_objects as go

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
COS_BETA = 1.0 / math.sqrt(5.0)  # adjacent 5-fold axis angle
SIN_BETA = 2.0 / math.sqrt(5.0)


# ------------------------- Icosahedron axes (neutral frame) -------------------------
def icosahedron_axes(device=None, dtype=DTYPE) -> torch.Tensor:
    """12 unit 5-fold axes from the standard icosahedron vertex set (neutral orientation)."""
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
    V = V / V.norm(dim=-1, keepdim=True)
    return V


def build_neighbor_indices(A: torch.Tensor) -> torch.Tensor:
    """Deterministically choose one neighbor per axis."""
    dots = A @ A.t()
    dots = dots - torch.eye(12, dtype=A.dtype, device=A.device) * 10.0
    return torch.argmax(dots, dim=-1)


# ------------------------- Canonical post-rotation (derived dynamically) -------------------------
_I_FACES = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
]


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-15:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _intersect_three_planes(n1, n2, n3, d):
    A = np.vstack([n1, n2, n3])
    b = np.array([d, d, d], dtype=float)
    return np.linalg.solve(A, b)


def compute_canonical_rotation_from_neutral_axes() -> np.ndarray:
    """
    Build the dodecahedron dual from the neutral icosahedron, then:
      1) rotate the face with max +z to +Z,
      2) spin so a vertex of that top face lies at y=0, x>0.
    Returns R (3x3) such that X_canonical = X_neutral @ R^T.
    """
    V = np.array(
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
        dtype=float,
    )
    N = V / np.linalg.norm(V, axis=1, keepdims=True)  # (12,3)

    rt5 = math.sqrt(5.0)
    r_face = math.sqrt(41.0 - 18.0 * rt5)  # matches user's script
    DodeV = np.array(
        [_intersect_three_planes(N[a], -N[b], N[c], r_face) for (a, b, c) in _I_FACES]
    )

    incident = [[] for _ in range(12)]
    for fi, (a, b, c) in enumerate(_I_FACES):
        incident[a].append(fi)
        incident[b].append(fi)
        incident[c].append(fi)

    def order_around_normal(normal, verts_idx):
        normal = normal / np.linalg.norm(normal)
        helper = (
            np.array([1.0, 0.0, 0.0])
            if abs(normal[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = helper - np.dot(helper, normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        pts = DodeV[verts_idx]
        ctr = pts.mean(axis=0)
        ang = []
        for idx in verts_idx:
            rel = DodeV[idx] - ctr
            x = np.dot(rel, u)
            y = np.dot(rel, v)
            ang.append((math.atan2(y, x), idx))
        ang.sort()
        return [idx for _, idx in ang]

    faces = [order_around_normal(N[i], incident[i]) for i in range(12)]

    top_idx = int(np.argmax(N[:, 2]))
    R1 = _rotation_from_a_to_b(N[top_idx], np.array([0.0, 0.0, 1.0]))
    N1 = (R1 @ N.T).T
    D1 = (R1 @ DodeV.T).T

    top_verts_idx = faces[top_idx]
    top_pts = D1[top_verts_idx]
    v_pick = top_pts[np.argmax(top_pts[:, 0])]
    theta = math.atan2(v_pick[1], v_pick[0])
    cT, sT = math.cos(-theta), math.sin(-theta)
    Rz = np.array([[cT, -sT, 0.0], [sT, cT, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ R1  # neutral → canonical
    return R


# ------------------------- Small linear algebra helpers (torch) -------------------------
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


# (kept for polar solve) — A_of_psi is implicit via θ_max and F_cap

# ------------------------- Hardcoded Chebyshev ψ'(u) fit -------------------------
# Chebyshev coefficients a[k] for P(t) = Σ a[k] T_k(t), t = 2u - 1, degree 17; denominator Q(t)=1
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
    """
    Clenshaw evaluation for Chebyshev T basis; coeffs on the last dim or 1-D.
    Works with arbitrary leading shapes via broadcasting on t.
    """
    # Expect coeffs flat (degree+1,)
    if coeffs.numel() == 1:
        return torch.full_like(t, coeffs[0])
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    # iterate from highest to T_1
    for c_k in coeffs[1:].flip(0):
        b0 = 2.0 * t * b1 - b2 + c_k
        b2, b1 = b1, b0
    # final combine with T_0
    return t * b1 - b2 + coeffs[0]


@torch.no_grad()
def psi_prime_from_u(u: torch.Tensor) -> torch.Tensor:
    """
    Evaluate ψ'(u) from hardcoded Chebyshev fit on t = 2u - 1.
    """
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


# ------------------------- Main mapping: ho2ho_I (neutral frame) -------------------------
def ho2ho_I_neutral(h: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    device, dtype = h.device, h.dtype
    rho = h.norm(dim=-1)
    nonzero = rho > eps
    uhat = torch.zeros_like(h)
    uhat[nonzero] = h[nonzero] / rho[nonzero].unsqueeze(-1)

    A = icosahedron_axes(device=device, dtype=dtype)  # (12,3)
    dots = torch.einsum("nd,md->nm", uhat, A)  # (N,12)
    idx_face = dots.argmax(dim=-1)  # (N,)
    a = A[idx_face]  # (N,3)

    nb_of = build_neighbor_indices(A)  # (12,)
    b = A[nb_of[idx_face]]  # (N,3)

    R_al = align_to_z(a)
    h_loc = torch.einsum("nij,nj->ni", R_al, h)
    b_al = torch.einsum("nij,nj->ni", R_al, b)
    gamma = -torch.atan2(b_al[:, 1], b_al[:, 0])
    R_sp = spin_about_z(gamma, device=device, dtype=dtype)
    h_loc = torch.einsum("nij,nj->ni", R_sp, h_loc)

    rho_loc = h_loc.norm(dim=-1)
    z = h_loc[:, 2]
    r_safe = torch.clamp(rho_loc, min=eps)
    cos_th = torch.clamp(z / r_safe, -1.0, 1.0)
    theta = torch.acos(cos_th)
    psi = torch.atan2(h_loc[:, 1], h_loc[:, 0])
    psi = torch.remainder(psi, 2.0 * PI)

    # Fold to canonical 36° wedge and compute u
    psi_delta, mirror, sector_idx = fold_azimuth(psi)
    u = psi_delta / WEDGE_36

    # HARD-CODED FIT: ψ' = ψ'(u)
    psi_prime = psi_prime_from_u(u)

    # Ceiling θ_max(ψ') and constant-Jacobian polar solve
    th_ceiling = theta_max(psi_prime)
    y_src = (1.0 - torch.cos(theta)) / (1.0 - torch.cos(th_ceiling))
    y_src = torch.clamp(y_src, 0.0, 1.0)

    def polar_solve(y, th_max):
        lo = torch.zeros_like(y)
        hi = th_max.clone()
        denom = torch.clamp(F_cap(th_max), min=EPS)
        for bisect_iter in range(100):
            mid = 0.5 * (lo + hi)
            val = F_cap(mid) / denom - y
            hi = torch.where(val > 0, mid, hi)
            lo = torch.where(val <= 0, mid, lo)
            if (hi - lo).max() < 1e-14:
                # print(f"Bisection converged in {bisect_iter + 1} iterations")
                break
        return 0.5 * (lo + hi)

    theta_prime = polar_solve(y_src, th_ceiling)

    R3 = R3_of_theta(theta_prime)
    R_cap = torch.clamp(R3, min=0.0) ** (1.0 / 3.0)
    # slight retract to avoid boundary self-intersection at the FZ shell
    rho_prime = (1.0 - 1e-6) * rho_loc * (R_cap / H_MAX)

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


def ho2ho_I(h: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    device, dtype = h.device, h.dtype
    ho_neutral = ho2ho_I_neutral(h, eps=eps)
    R_np = compute_canonical_rotation_from_neutral_axes()  # (3,3) numpy
    Rg = torch.tensor(R_np, dtype=dtype, device=device)  # torch, on device
    ho = (ho_neutral @ Rg.t()).to(dtype=dtype, device=device)
    return ho


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
