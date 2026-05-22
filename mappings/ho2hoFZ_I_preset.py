#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → Icosahedral RFZ (532) via constant-Jacobian KR, canonical orientation.
- Canonical pose baked in: top pentagon normal at +z; ψ=0 along +x for the top face.
- Uses per-face in-plane reference tangents tied to the canonical frame.
- Assumes `orientation_ops.cu2ho` is importable.
- Saves a Plotly HTML (no fig.show()).

Run:
  python ho2ho_I_canonical.py --h 10 --z 8 --device cuda --out ho2ho_I_demo.html
"""
import math
import argparse
from typing import Tuple

import torch
import numpy as np


# ------------------------- Numeric/geom constants -------------------------
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
PI = math.pi
EPS = 1e-12

# Homochoric ball radius
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)

# Golden-ratio radicals (exact forms used to define axes)
SQRT5 = math.sqrt(5.0)
PHI = 0.5 * (1.0 + SQRT5)

a = 1.0 / math.sqrt(5.0)  # 1/√5
b = PHI / math.sqrt(5.0)  # φ/√5
c = (PHI - 1.0) / math.sqrt(5.0)  # (φ-1)/√5 = 1/(φ√5)
d = math.sqrt((5.0 - SQRT5) / 10.0)  # √((5-√5)/10)
e = math.sqrt((5.0 + SQRT5) / 10.0)  # √((5+√5)/10)
f = 2.0 / math.sqrt(5.0)  # 2/√5

# Canonical 12 five-fold axes (unit), arranged with +Z first, -Z last.
AXES_I_5FOLD_CANONICAL = [
    [0.0, 0.0, 1.0],
    [-b, +d, -a],
    [+c, +e, -a],
    [+f, 0.0, -a],
    [+c, -e, -a],
    [-b, -d, -a],
    [+b, -d, +a],
    [-c, -e, +a],
    [-f, 0.0, +a],
    [-c, +e, +a],
    [+b, +d, +a],
    [0.0, 0.0, -1.0],
]


# Per-face azimuth-zero (tangent) choice:
#   - For top/bottom faces, use +x.
#   - For all other faces, take eastward direction in equatorial plane:
#       T_i = normalize( e_z × A_i ) = (-y, x, 0)/√(x^2+y^2)
def canonical_face_tangents(axes: torch.Tensor) -> torch.Tensor:
    T = torch.zeros_like(axes)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=axes.dtype, device=axes.device)
    # top and bottom:
    T[0] = torch.tensor([1.0, 0.0, 0.0], dtype=axes.dtype, device=axes.device)
    T[-1] = torch.tensor([1.0, 0.0, 0.0], dtype=axes.dtype, device=axes.device)
    # others:
    mid = axes[1:-1]
    x, y = mid[:, 0], mid[:, 1]
    r = torch.sqrt(torch.clamp(x * x + y * y, min=1e-18))
    east = torch.stack([-y / r, x / r, torch.zeros_like(r)], dim=-1)
    T[1:-1] = east
    return T


# Wedge widths for azimuth folding
WEDGE_72 = 2.0 * PI / 5.0
WEDGE_36 = WEDGE_72 / 2.0

# KR constants for I (and dihedral/tetra/octa forms)
ALPHA = math.sqrt(5.0 - 2.0 * SQRT5)  # tan(18°)
P_CONST = math.sqrt(5.0 + 2.0 * SQRT5)  # cot(18°)
COS_BETA = 1.0 / math.sqrt(5.0)  # adjacent 5-fold axis angle
SIN_BETA = 2.0 / math.sqrt(5.0)


# ------------------------- Small linear algebra helpers -------------------------
def z_axis(device=None, dtype=DTYPE) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)


def rodrigues_rotate_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    axis: (...,3) unit; angle: (...,)
    returns: (...,3,3)
    """
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
    """
    Build rotation R that maps vector a -> +z, batched: a (N,3) unit.
    Returns R: (N,3,3)
    """
    a = a / (a.norm(dim=-1, keepdim=True) + EPS)
    z = z_axis(device=a.device, dtype=a.dtype).expand_as(a)
    dot = (a * z).sum(dim=-1)  # (N,)
    v = torch.cross(a, z, dim=-1)  # (N,3)
    v_norm = v.norm(dim=-1, keepdim=True)  # (N,1)
    axis = torch.where(
        v_norm > EPS,
        v / (v_norm + 0.0),
        torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device),
    )
    ang = torch.atan2(v_norm.squeeze(-1), dot)  # (N,)
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
    """Rotation about +z by angle gamma (N,) → (N,3,3)."""
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
    """Single active cap on the tiny wedge: tan θ_max = (1 - cosβ)/(sinβ cosψ)."""
    num = 1.0 - COS_BETA
    den = SIN_BETA * torch.clamp(torch.cos(psi), min=EPS)
    return torch.atan(num / den)


def R3_of_theta(theta: torch.Tensor) -> torch.Tensor:
    """Equal-volume radius^3 for the cap with c = P_CONST * cosθ."""
    c = torch.clamp(torch.cos(theta), -1.0 + 1e-15, 1.0 - 1e-15)
    term1 = torch.atan(1.0 / (P_CONST * c))
    term2 = (P_CONST * c) / (1.0 + (P_CONST * c) ** 2)
    return 1.5 * (term1 - term2)  # 3/2[...]


def F_cap(theta: torch.Tensor) -> torch.Tensor:
    """Primitive ∫ R^3 sinθ dθ on the cap (closed form)."""
    c = torch.clamp(torch.cos(theta), -1.0 + 1e-15, 1.0 - 1e-15)
    return 1.5 * (
        0.5 * PI * (1.0 - c) - math.atan(P_CONST) + c * torch.atan(P_CONST * c)
    )


def A_of_psi(psi: torch.Tensor) -> torch.Tensor:
    """Column area on the tiny wedge at azimuth ψ."""
    return F_cap(theta_max(psi))


# 16-point Gauss–Legendre nodes/weights on [-1,1] (hard-coded; no numpy)
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
    """
    C(ψ) = ∫_0^ψ A(φ) dφ via 16-point GL on [0,ψ] per element.
    psi: (N,)
    return: (N,)
    """
    x = _GL16_X.to(device=psi.device)
    w = _GL16_W.to(device=psi.device)
    half = 0.5 * psi[..., None]  # (N,1)
    phi = half * (x + 1.0)  # (N,16)
    Aj = A_of_psi(phi)  # (N,16)
    return (Aj * w).sum(dim=-1) * half.squeeze(-1)


def invert_C(
    u: torch.Tensor, C_max: float, tol: float = 1e-12, maxit: int = 50
) -> torch.Tensor:
    """Solve C(ψ)/C_max = u on [0,36°] by bisection."""
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
    return 0.5 * (lo + hi)


# ------------------------- Azimuth folding/unfolding -------------------------
def fold_azimuth(psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fold ψ ∈ [0,2π) to tiny wedge:
      k = floor(ψ / 72°) ∈ {0..4}; ψ72 = ψ - k*72°; mirror into 36° if needed.
    Returns (ψ_Δ, mirror_bit, k).
    """
    psi = torch.remainder(psi, 2.0 * PI)
    k = torch.floor(psi / WEDGE_72).to(torch.int64)
    psi72 = psi - (k.to(psi.dtype) * WEDGE_72)
    mirror = psi72 > WEDGE_36
    psi_delta = torch.where(mirror, WEDGE_72 - psi72, psi72)
    return psi_delta, mirror, k


def unfold_azimuth(
    psi_prime: torch.Tensor, mirror: torch.Tensor, sector_idx: torch.Tensor
) -> torch.Tensor:
    """Undo mirror, then add back k*72° to restore ψ ∈ [0,2π)."""
    psi72 = torch.where(mirror, WEDGE_72 - psi_prime, psi_prime)
    psi_full = psi72 + sector_idx.to(psi72.dtype) * WEDGE_72
    return torch.remainder(psi_full, 2.0 * PI)


# ------------------------- Main mapping: ho2ho_I -------------------------
def ho2ho_I(h: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """
    Homochoric → Icosahedral RFZ (homochoric), constant-Jacobian KR on the canonical frame.
    h: (N,3) tensor.
    return: (N,3) mapped points in canonical RFZ coordinates.
    """
    device, dtype = h.device, h.dtype

    # Radii and unit directions
    rho = h.norm(dim=-1)  # (N,)
    nonzero = rho > eps
    uhat = torch.zeros_like(h)
    uhat[nonzero] = h[nonzero] / rho[nonzero].unsqueeze(-1)

    # Canonical axes and per-face tangents (both on device/dtype)
    A = torch.tensor(AXES_I_5FOLD_CANONICAL, dtype=dtype, device=device)  # (12,3)
    T = canonical_face_tangents(A)  # (12,3)

    # Select nearest 5-fold face
    dots = torch.einsum("nd,md->nm", uhat, A)  # (N,12)
    idx_face = dots.argmax(dim=-1)  # (N,)
    a = A[idx_face]  # (N,3) face normal (canonical)
    b = T[idx_face]  # (N,3) in-plane reference (canonical)

    # Align to local frame: a → +z; then spin so b sits at local azimuth 0
    R_al = align_to_z(a)  # (N,3,3)
    h_loc = torch.einsum("nij,nj->ni", R_al, h)  # (N,3)
    b_al = torch.einsum("nij,nj->ni", R_al, b)  # (N,3)
    gamma = -torch.atan2(b_al[:, 1], b_al[:, 0])  # (N,)
    R_sp = spin_about_z(gamma, device=device, dtype=dtype)
    h_loc = torch.einsum("nij,nj->ni", R_sp, h_loc)

    # Local spherical (ρ_loc, θ, ψ)
    rho_loc = h_loc.norm(dim=-1)  # (N,)
    z = h_loc[:, 2]
    r_safe = torch.clamp(rho_loc, min=eps)
    cos_th = torch.clamp(z / r_safe, -1.0, 1.0)
    theta = torch.acos(cos_th)
    psi = torch.atan2(h_loc[:, 1], h_loc[:, 0])
    psi = torch.remainder(psi, 2.0 * PI)

    # Fold to tiny wedge
    psi_delta, mirror, sector_idx = fold_azimuth(psi)

    # KR (a): azimuth rearrangement by CDF inversion
    C_max = C_of_psi(torch.tensor([WEDGE_36], dtype=dtype, device=device))[0].item()
    u = psi_delta / WEDGE_36
    psi_prime = invert_C(u, C_max=C_max, tol=1e-12, maxit=60)

    # KR (b): polar rearrangement along the column (use target ceiling for normalization)
    th_ceiling = theta_max(psi_prime)
    y_src = (1.0 - torch.cos(theta)) / (1.0 - torch.cos(th_ceiling))
    y_src = torch.clamp(y_src, 0.0, 1.0)

    def polar_solve(y, th_max):
        lo = torch.zeros_like(y)
        hi = th_max.clone()
        denom = torch.clamp(F_cap(th_max), min=eps)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            val = F_cap(mid) / denom - y
            hi = torch.where(val > 0, mid, hi)
            lo = torch.where(val <= 0, mid, lo)
            if (hi - lo).max() < 1e-12:
                break
        return 0.5 * (lo + hi)

    theta_prime = polar_solve(y_src, th_ceiling)

    # KR (c): radial (uniform in ρ^3)
    R3 = R3_of_theta(theta_prime)
    R_cap = torch.clamp(R3, min=0.0) ** (1.0 / 3.0)
    rho_prime = rho_loc * (R_cap / H_MAX)

    # Local Cartesian with fully restored azimuth
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

    # Back to global (canonical!) frame: inverse rotations
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
    # x/y (uniform cells except last open interval)
    cu = torch.linspace(
        -0.5 * torch.pi ** (2.0 / 3.0),
        0.5 * torch.pi ** (2.0 / 3.0),
        2 * cu_h + 2,
        dtype=dtype,
        device=device,
    )
    cu = cu[:-1]
    cu = cu + 0.5 * (cu[1] - cu[0])
    # z (fewer slices)
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
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from orientation_ops import cu2ho, ho2qu, qu2ho, qu_prod
    from laue_ops import laue_elements
    from riesz_energy import riesz_energies_fused
    import plotly.graph_objects as go

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
    ap.add_argument("--out_base", type=str, default="figures/I532_base.html")
    ap.add_argument("--out_e3", type=str, default="figures/I532_E3.html")
    ap.add_argument("--out_nn", type=str, default="figures/I532_NN.html")
    ap.add_argument("--out_ops", type=str, default="figures/I532_ops.html")
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
