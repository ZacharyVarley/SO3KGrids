#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → Icosahedral RFZ (532) via constant-Jacobian KR, canonical orientation.
- Canonical pose baked in: top pentagon normal at +z; ψ=0 along +x for the top face.
- Uses per-face in-plane reference tangents tied to the canonical frame.
- Assumes `orientation_ops.cu2ho` is importable.

Run:
    python ho2ho_I_canonical.py --h 10 --z 8 --device cuda
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

def main():
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from src.orientation_ops import cu2ho, ho2qu
    from src.laue_ops import laue_elements
    from src.riesz_energy import riesz_energies_fused

    ap = argparse.ArgumentParser(
        description="I(532) — cu2ho → KR(FZ, canonical) with ψ'(u) fit; prints mapping and energy summary"
    )
    ap.add_argument(
        "--h", type=int, default=11, help="half-resolution in x/y (2h cells per axis)"
    )
    ap.add_argument("--z", type=int, default=11, help="half-resolution in z (2z cells)")
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    torch.set_grad_enabled(False)

    cu_grid = so3_cubochoric_grid_stretch(args.h, args.z, device=device, dtype=DTYPE)
    ho_src = cu2ho(cu_grid.to(dtype=DTYPE, device=device)) * 0.99999
    ho_map = ho2ho_I(ho_src)

    q_fz = ho2qu(ho_map.to(dtype=DTYPE, device=device))
    q_fz = q_fz / torch.clamp(q_fz.norm(dim=-1, keepdim=True), min=1e-15)
    ops = laue_elements(12).to(dtype=DTYPE, device=device)
    E1, E2, E3, _, _, _, NN_i = riesz_energies_fused(
        q_fz, ops, return_contrib=True, return_nn=True
    )

    print(f"[ok] mapped {ho_map.shape[0]} points into the canonical RFZ")
    print(f"[info] E1={float(E1):.6e}, E2={float(E2):.6e}, E3={float(E3):.6e}")
    print(
        f"[info] NN stats — min: {float(NN_i.min()):.6e}, "
        f"median: {float(torch.median(NN_i)):.6e}, max: {float(NN_i.max()):.6e}"
    )
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


# ------------------------- CLI summary -------------------------
def main():
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from src.orientation_ops import cu2ho, ho2qu
    from src.laue_ops import laue_elements
    from src.riesz_energy import riesz_energies_fused

    ap = argparse.ArgumentParser(
        description="I(532) — cu2ho → KR(FZ, canonical) with ψ'(u) fit; prints mapping and energy summary"
    )
    ap.add_argument(
        "--h", type=int, default=11, help="half-resolution in x/y (2h cells per axis)"
    )
    ap.add_argument("--z", type=int, default=11, help="half-resolution in z (2z cells)")
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    torch.set_grad_enabled(False)

    cu_grid = so3_cubochoric_grid_stretch(args.h, args.z, device=device, dtype=DTYPE)
    ho_src = cu2ho(cu_grid.to(dtype=DTYPE, device=device)) * 0.99999
    ho_map = ho2ho_I(ho_src)

    q_fz = ho2qu(ho_map.to(dtype=DTYPE, device=device))
    q_fz = q_fz / torch.clamp(q_fz.norm(dim=-1, keepdim=True), min=1e-15)
    ops = laue_elements(12).to(dtype=DTYPE, device=device)
    E1, E2, E3, _, _, _, NN_i = riesz_energies_fused(
        q_fz, ops, return_contrib=True, return_nn=True
    )

    print(f"[ok] mapped {ho_map.shape[0]} points into the canonical RFZ")
    print(f"[info] E1={float(E1):.6e}, E2={float(E2):.6e}, E3={float(E3):.6e}")
    print(
        f"[info] NN stats — min: {float(NN_i.min()):.6e}, "
        f"median: {float(torch.median(NN_i)):.6e}, max: {float(NN_i.max()):.6e}"
    )


if __name__ == "__main__":
    main()
