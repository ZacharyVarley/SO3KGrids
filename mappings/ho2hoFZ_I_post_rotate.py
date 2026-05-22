#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → Icosahedral RFZ (532) via constant-Jacobian KR, then post-rotate
into the canonical crystallography frame (top 5-fold at +z; ψ=0 along +x).

- Assumes `orientation_ops.cu2ho` is importable.
- Derives the canonical post-rotation *dynamically* from the neutral icosahedron,
  matching your dodecahedron construction (no guessing).
- Saves a Plotly HTML (no fig.show()).
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
from orientation_ops import cu2ho, ho2qu
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
WEDGE_36 = WEDGE_72 / 2.0

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
    # 12 unit icosahedron vertices (face normals of dodecahedron)
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

    # Dodecahedron vertices by intersecting 3 planes: n·x = d (we only need a fixed d>0)
    rt5 = math.sqrt(5.0)
    r_face = math.sqrt(41.0 - 18.0 * rt5)  # matches user's script
    DodeV = np.array(
        [_intersect_three_planes(N[a], N[b], N[c], r_face) for (a, b, c) in _I_FACES]
    )

    # incident faces per icosahedron vertex (5 each)
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

    # Step 1: rotate the face with max +z to +Z
    top_idx = int(np.argmax(N[:, 2]))
    R1 = _rotation_from_a_to_b(N[top_idx], np.array([0.0, 0.0, 1.0]))
    N1 = (R1 @ N.T).T
    D1 = (R1 @ DodeV.T).T

    # Step 2: spin about +Z so one vertex of that top pentagon is on y=0 with x>0
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


def A_of_psi(psi: torch.Tensor) -> torch.Tensor:
    return F_cap(theta_max(psi))


_GL16_X = torch.tensor(
    [
        -0.9894009349916499,
        -0.9445750230732326,
        -0.8656312023878317,
        -0.7554044083550030,
        -0.6178762444026437,
        -0.4580167776572274,
        -0.2816035507792589,
        -0.09501250983763744,
        0.09501250983763744,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026437,
        0.7554044083550030,
        0.8656312023878317,
        0.9445750230732326,
        0.9894009349916499,
    ],
    dtype=DTYPE,
)
_GL16_W = torch.tensor(
    [
        0.02715245941175409,
        0.06225352393864789,
        0.09515851168249278,
        0.12462897125553387,
        0.14959598881657673,
        0.16915651939500253,
        0.18260341504492358,
        0.1894506104550685,
        0.1894506104550685,
        0.18260341504492358,
        0.16915651939500253,
        0.14959598881657673,
        0.12462897125553387,
        0.09515851168249278,
        0.06225352393864789,
        0.02715245941175409,
    ],
    dtype=DTYPE,
)


def C_of_psi(psi: torch.Tensor) -> torch.Tensor:
    x = _GL16_X.to(device=psi.device)
    w = _GL16_W.to(device=psi.device)
    half = 0.5 * psi[..., None]
    phi = half * (x + 1.0)
    Aj = A_of_psi(phi)
    return (Aj * w).sum(dim=-1) * half.squeeze(-1)


def invert_C(
    u: torch.Tensor, C_max: float, tol: float = 1e-14, maxit: int = 60
) -> torch.Tensor:
    u = torch.clamp(u, 0.0, 1.0)
    lo = torch.zeros_like(u)
    hi = torch.full_like(u, WEDGE_36)
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        fm = C_of_psi(mid) / C_max - u
        hi = torch.where(fm > 0, mid, hi)
        lo = torch.where(fm <= 0, mid, lo)
        if (hi - lo).max() < tol:
            break
        # print(f" Max C error: {((hi - lo).max().item()):.6e}")
    return 0.5 * (lo + hi)


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

    psi_delta, mirror, sector_idx = fold_azimuth(psi)

    C_max = C_of_psi(torch.tensor([WEDGE_36], dtype=dtype, device=device))[0].item()
    u = psi_delta / WEDGE_36
    psi_prime = invert_C(u, C_max=C_max, tol=1e-12, maxit=60)

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
            # print(f" Max P error: {((hi - lo).max().item()):.6e}")
        return 0.5 * (lo + hi)

    theta_prime = polar_solve(y_src, th_ceiling)

    R3 = R3_of_theta(theta_prime)
    R_cap = torch.clamp(R3, min=0.0) ** (1.0 / 3.0)
    # rho_prime = rho_loc * (R_cap / H_MAX)
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


# # ------------------------- CLI demo (saves HTML) -------------------------
# def main():
#     ap = argparse.ArgumentParser(description="I(532) — cu2ho → KR(FZ) → post-rotate to canonical pose (saves Plotly HTML)")
#     ap.add_argument("--h", type=int, default=21, help="half-resolution in x/y (2h cells per axis)")
#     ap.add_argument("--z", type=int, default=21, help="half-resolution in z (2z cells)")
#     ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
#     ap.add_argument("--out", type=str, default="ho2ho_I_postrot_demo.html", help="output HTML filename")
#     args = ap.parse_args()

#     device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
#     torch.set_grad_enabled(False)

#     # 1) Build stretched cubochoric grid → homochoric source points
#     cu_grid = so3_cubochoric_grid_stretch(args.h, args.z, device=device, dtype=DTYPE)
#     ho_src = cu2ho(cu_grid.to(dtype=DTYPE, device=device))

#     # retract the homochoric vector from the boundary of the FZ
#     ho_src = ho_src * 0.99999

#     # 2) Map to RFZ (neutral frame)
#     ho_map_neutral = ho2ho_I_neutral(ho_src)

#     # 3) Compute the canonical post-rotation dynamically
#     R_np = compute_canonical_rotation_from_neutral_axes()        # (3,3) numpy
#     Rg = torch.tensor(R_np, dtype=DTYPE, device=device)          # torch, on user's device

#     # 4) Post-rotate into canonical pose
#     ho_map = ho_map_neutral @ Rg.t()

#     # # 5) Save Plotly HTML
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter3d(
#     #     x=ho_src[:, 0].detach().cpu().numpy(),
#     #     y=ho_src[:, 1].detach().cpu().numpy(),
#     #     z=ho_src[:, 2].detach().cpu().numpy(),
#     #     mode='markers',
#     #     name='Original (cu→ho)',
#     #     marker=dict(size=2)
#     # ))
#     # fig.add_trace(go.Scatter3d(
#     #     x=ho_map[:, 0].detach().cpu().numpy(),
#     #     y=ho_map[:, 1].detach().cpu().numpy(),
#     #     z=ho_map[:, 2].detach().cpu().numpy(),
#     #     mode='markers',
#     #     name='Mapped (I KR → RFZ) + post-rotate to canonical',
#     #     marker=dict(size=2)
#     # ))
#     # lim = float(H_MAX)
#     # fig.update_layout(
#     #     title='I (532) homochoric KR map — post-rotated to canonical RFZ (top 5-fold at +z, ψ=0 along +x)',
#     #     scene=dict(
#     #         xaxis=dict(range=[-lim, lim], title='x'),
#     #         yaxis=dict(range=[-lim, lim], title='y'),
#     #         zaxis=dict(range=[-lim, lim], title='z'),
#     #         aspectmode='cube'
#     #     ),
#     #     legend=dict(x=0.02, y=0.98),
#     #     margin=dict(l=0, r=0, t=36, b=0),
#     #     template='plotly_white'
#     # )
#     # fig.write_html(args.out, include_plotlyjs="cdn")
#     # print(f"[ok] wrote {args.out} with {ho_src.shape[0]} points")

#     # -------- Color by E3 (Riesz s=3) using ho2qu + laue_elements --------

#     # 1) Homochoric → quaternion (unit) using your implementation
#     q_fz = ho2qu(ho_map.to(dtype=DTYPE, device=device))        # (N,4)

#     # normalize just in case
#     q_fz = q_fz / torch.clamp(q_fz.norm(dim=-1, keepdim=True), min=1e-15)

#     # 2) Get Laue ops for Icosahedral (532); identity is first by contract
#     ops = laue_elements(12).to(dtype=DTYPE, device=device)     # (G,4) unit quats

#     # 3) Riesz energies — get per-point E3 contributions
#     E1, E2, E3, _, _, S3_i, NN_i = riesz_energies_fused(q_fz, ops, return_contrib=True, return_nn=True)

#     # # 4) Min–max normalize S3_i → [0,1] for coloring
#     # s3 = S3_i.detach()
#     # s3 = s3 - s3.min()
#     # s3 = s3 / torch.clamp(s3.max(), min=1e-15)
#     # s3_np = s3.cpu().numpy()

#     # print out the min and median of the NN_i
#     print(f" Min NN: {NN_i.min().item()}")
#     print(f" Median NN: {np.median(NN_i.cpu().numpy())}")
#     print(f" Max NN: {NN_i.max().item()}")

#     # ratio to the median
#     s3_np = S3_i.detach().cpu().numpy()
#     s3_np = s3_np / np.median(s3_np)
#     s3_np = s3_np.astype(np.float32)


#     # # 5) Save Plotly HTML — color by E3 contribution, opacity 1.0
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter3d(
#     #     x=ho_src[:, 0].detach().cpu().numpy(),
#     #     y=ho_src[:, 1].detach().cpu().numpy(),
#     #     z=ho_src[:, 2].detach().cpu().numpy(),
#     #     mode='markers',
#     #     name='Original (cu→ho)',
#     #     marker=dict(size=3, opacity=1.0)
#     # ))
#     # fig.add_trace(go.Scatter3d(
#     #     x=ho_map[:, 0].detach().cpu().numpy(),
#     #     y=ho_map[:, 1].detach().cpu().numpy(),
#     #     z=ho_map[:, 2].detach().cpu().numpy(),
#     #     mode='markers',
#     #     name='Mapped (colored by E3 contrib)',
#     #     marker=dict(size=4, color=s3_np, colorscale='Turbo', showscale=True, opacity=1.0)
#     # ))
#     # lim = float(H_MAX)
#     # fig.update_layout(
#     #     title=f'I (532) homochoric KR map — RFZ colored by Riesz E3 contribution (E3 total = {E3:.6e})',
#     #     scene=dict(
#     #         xaxis=dict(range=[-lim, lim], title='x'),
#     #         yaxis=dict(range=[-lim, lim], title='y'),
#     #         zaxis=dict(range=[-lim, lim], title='z'),
#     #         aspectmode='cube'
#     #     ),
#     #     legend=dict(x=0.02, y=0.98),
#     #     margin=dict(l=0, r=0, t=36, b=0),
#     #     template='plotly_white'
#     # )
#     # fig.write_html(args.out, include_plotlyjs="cdn")
#     # print(f"[ok] wrote {args.out} with {ho_src.shape[0]} points; E3 = {E3:.6e}")

#     # plot by median(NN_i) over NN_i
#     NN_i_np = NN_i.detach().cpu().numpy()
#     NN_i_np = NN_i_np / np.median(NN_i_np)
#     NN_i_np = NN_i_np.astype(np.float32)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter3d(
#         x=ho_src[:, 0].detach().cpu().numpy(),
#         y=ho_src[:, 1].detach().cpu().numpy(),
#         z=ho_src[:, 2].detach().cpu().numpy(),
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=ho_map[:, 0].detach().cpu().numpy(),
#         y=ho_map[:, 1].detach().cpu().numpy(),
#         z=ho_map[:, 2].detach().cpu().numpy(),
#         mode='markers',
#         name='Mapped (colored by NN)',
#         marker=dict(size=4, color=NN_i_np, colorscale='Turbo', showscale=True, opacity=1.0)
#     ))
#     fig.write_html(args.out, include_plotlyjs="cdn")
#     print(f"[ok] wrote {args.out} with {ho_src.shape[0]} points; NN = {np.median(NN_i.cpu().numpy()):.6e}")


# if __name__ == "__main__":
#     main()


# ---------- discrete palette for up to 60 ops ----------
def palette60():
    # 10 pleasant hues × 6 cycles (60 colors total)
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
        # slightly darken per cycle
        f = 1.0 - 0.08 * k
        for c in base:
            r = int(int(c[1:3], 16) * f)
            g = int(int(c[3:5], 16) * f)
            b = int(int(c[5:7], 16) * f)
            pal.append(f"#{r:02x}{g:02x}{b:02x}")
    return pal[:60]


# ------------------------- CLI demo (saves HTML) -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="I(532) — cu2ho → KR(FZ) → apply all Laue ops, remap to RFZ, color by op index"
    )
    ap.add_argument(
        "--h", type=int, default=7, help="half-resolution in x/y (2h cells per axis)"
    )
    ap.add_argument("--z", type=int, default=7, help="half-resolution in z (2z cells)")
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    ap.add_argument(
        "--out",
        type=str,
        default="figures/ho2ho_I_ops_colored.html",
        help="output HTML filename",
    )
    ap.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="plot at most this many points per op (0 = no downsample)",
    )
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    torch.set_grad_enabled(False)

    # 1) Build stretched cubochoric grid → homochoric (source)
    cu_grid = so3_cubochoric_grid_stretch(args.h, args.z, device=device, dtype=DTYPE)
    ho_src = cu2ho(cu_grid.to(dtype=DTYPE, device=device))

    # 2) Map ONCE to canonical RFZ (this is your base FZ set)
    ho_fz = ho2ho_I(ho_src)  # uses your ho2ho_I defined above

    # 3) Convert to unit quaternions
    q_fz = ho2qu(ho_fz.to(dtype=DTYPE, device=device))
    q_fz = q_fz / torch.clamp(q_fz.norm(dim=-1, keepdim=True), min=1e-15)

    # 4) Load Laue ops for Icosahedral (532). Identity should be ops[0] by contract.
    ops = laue_elements(12).to(dtype=DTYPE, device=device)  # (G,4) unit quats
    G = ops.shape[0]
    print(f"[info] Loaded {G} Laue operators")

    # 5) Prepare Plotly fig and base cloud (optional)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_fz[:, 0].detach().cpu().numpy(),
            y=ho_fz[:, 1].detach().cpu().numpy(),
            z=ho_fz[:, 2].detach().cpu().numpy(),
            mode="markers",
            name="RFZ base (op 0)",
            marker=dict(size=2, color="#000000", opacity=0.35),
        )
    )

    # 6) For each operator: q' = g ⊗ q_fz (left action), then back to ho and re-fold to RFZ
    from orientation_ops import qu_prod, qu2ho  # ensure import is here

    colors = palette60()
    max_plot = args.downsample if args.downsample and args.downsample > 0 else None

    for gi in range(G):
        g = ops[gi].unsqueeze(0).expand_as(q_fz)  # (N,4)
        # Depending on your convention, left action is typically correct: g ⊗ q
        q_prime = qu_prod(g, q_fz)  # (N,4)
        q_prime = q_prime / torch.clamp(q_prime.norm(dim=-1, keepdim=True), min=1e-15)

        # Back to homochoric, then map to canonical RFZ again to visualize where this op lands
        ho_prime = qu2ho(q_prime)  # (N,3), anywhere in ball
        # ho_prime = ho2ho_I(ho_prime)           # (N,3), folded to canonical RFZ

        # Optional downsample for large N
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

    lim = float(H_MAX)
    fig.update_layout(
        title="I (532) — RFZ copies under all Laue ops (colored by op index)",
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
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(
        f"[ok] wrote {args.out} with {ho_fz.shape[0]} base points and {G} colored symmetry copies"
    )


if __name__ == "__main__":
    main()
