#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Icosahedral (532) KR map — minimal & clean
------------------------------------------
- 12 five-fold axes hardcoded (unit x,y,z only)
- Face = nearest 5-fold (max dot)
- Azimuth-0 rule (your corrected one):
    * a == +Z  → az0 = +X
    * a == -Z  → az0 = -X
    * a.z > 0  → az0 = +Z
    * a.z < 0  → az0 = -Z
- Local spherical from dot/cross only (no rotation matrices)
- Chebyshev ψ'(u) inverse-CDF on 1/120 sector
- Saves: base.html, E3.html, NN.html, ops.html
"""

import math
import argparse
import sys
from pathlib import Path
from typing import Tuple

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

# ---------- numerics ----------
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
PI = math.pi
EPS = 1e-12

# Homochoric ball radius
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)

# Wedge widths for azimuth folding
WEDGE_72 = 2.0 * PI / 5.0
WEDGE_36 = WEDGE_72 / 2.0  # π/5 = 36°

# KR constants
P_CONST = math.sqrt(5.0 + 2.0 * math.sqrt(5.0))  # cot(18°)
COS_BETA = 1.0 / math.sqrt(5.0)
SIN_BETA = 2.0 / math.sqrt(5.0)

# ---------- 12 five-fold axes (unit, pre-normalized; consistent with your 36° ops) ----------
AXES_5F = torch.tensor(
    [
        [-0.723606797749979, -0.5257311121191336, 0.4472135954999579],
        [-0.723606797749979, 0.5257311121191336, 0.4472135954999579],
        [0.0, 0.0, 1.0],
        [0.276393202250021, -0.8506508083520400, 0.4472135954999579],
        [0.276393202250021, 0.8506508083520400, 0.4472135954999579],
        [0.894427190999916, 0.0, 0.4472135954999579],
        [0.723606797749979, -0.5257311121191336, -0.4472135954999579],
        [-0.276393202250021, 0.8506508083520400, -0.4472135954999579],
        [0.0, 0.0, -1.0],
        [-0.276393202250021, -0.8506508083520400, -0.4472135954999579],
        [-0.894427190999916, 0.0, -0.4472135954999579],
        [0.723606797749979, 0.5257311121191336, -0.4472135954999579],
    ],
    dtype=DTYPE,
)


# ---------- tiny helpers (pure vectors, no matrices) ----------
def project_to_tangent(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    dot = (v * a).sum(dim=-1, keepdim=True)
    vt = v - dot * a
    n = vt.norm(dim=-1, keepdim=True)
    return torch.where(n > 1e-15, vt / n, vt)


# ---------- KR geometry pieces ----------
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


# ---------- hard-coded Chebyshev ψ'(u) fit ----------
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


# ---------- azimuth folding/unfolding ----------
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


# ---------- main mapping (face basis only) ----------
def ho2ho_I(h: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    device, dtype = h.device, h.dtype

    rho = h.norm(dim=-1)
    nonzero = rho > eps
    uhat = torch.zeros_like(h)
    uhat[nonzero] = h[nonzero] / rho[nonzero].unsqueeze(-1)

    A = AXES_5F.to(device=device, dtype=dtype)  # (12,3)

    # nearest face axis
    dots = torch.einsum("nd,md->nm", uhat, A)  # (N,12)
    idx = dots.argmax(dim=-1)  # (N,)
    a = A[idx]  # (N,3)

    # azimuth-0 direction (broadcasted constants)
    X = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand_as(a)
    Z = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand_as(a)

    az = torch.empty_like(a)
    az_posZ = (1.0 - a[:, 2]) < 1e-12  # a ≈ +Z
    az_negZ = (1.0 + a[:, 2]) < 1e-12  # a ≈ -Z

    az[az_posZ] = X[az_posZ]  # +Z face → +X
    az[az_negZ] = -X[az_negZ]  # -Z face → -X

    mid = ~(az_posZ | az_negZ)
    if mid.any():
        sign = torch.sign(a[mid, 2]).unsqueeze(-1)  # +1 if z>0 else -1
        az[mid] = sign * Z[mid]

    # tangent basis on face: e0 = proj(az, ⟂ a), e1 = a × e0
    e0 = project_to_tangent(az, a)
    e1 = torch.cross(a, e0, dim=-1)

    # local spherical: theta = arccos(uhat·a), psi = atan2(uhat·e1, uhat·e0)
    ua = (uhat * a).sum(dim=-1).clamp(-1.0, 1.0)
    theta = torch.acos(ua)
    ue0 = (uhat * e0).sum(dim=-1)
    ue1 = (uhat * e1).sum(dim=-1)
    psi = torch.atan2(ue1, ue0)
    psi = torch.remainder(psi, 2.0 * PI)

    # KR azimuth
    psi_delta, mirror, sector_idx = fold_azimuth(psi)
    u = psi_delta / WEDGE_36
    psi_prime = psi_prime_from_u(u)

    # KR polar
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
    rho_prime = (1.0 - 1e-12) * rho * (R_cap / H_MAX)

    # rebuild homochoric in global xyz using (e0,e1,a) local frame
    psi_full = unfold_azimuth(psi_prime, mirror, sector_idx)
    sin_t = torch.sin(theta_prime)
    xloc = rho_prime * sin_t * torch.cos(psi_full)
    yloc = rho_prime * sin_t * torch.sin(psi_full)
    zloc = rho_prime * torch.cos(theta_prime)
    h_out = xloc.unsqueeze(-1) * e0 + yloc.unsqueeze(-1) * e1 + zloc.unsqueeze(-1) * a
    h_out[~nonzero] = 0.0
    return h_out


# ---------- stretched cubochoric grid ----------
def so3_cubochoric_grid_stretch(
    cu_h: int, cu_z: int, device: torch.device, dtype: torch.dtype = torch.float64
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
    cz = torch.linspace(
        -0.5 * torch.pi ** (2.0 / 3.0),
        0.5 * torch.pi ** (2.0 / 3.0),
        2 * cu_z + 2,
        dtype=dtype,
        device=device,
    )
    cz = cz[:-1]
    cz = cz + 0.5 * (cz[1] - cz[0])
    return torch.stack(torch.meshgrid(cu, cu, cz, indexing="ij"), dim=-1).reshape(-1, 3)


# ---------- palette ----------
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


# ---------- CLI ----------
def main():
    import plotly.graph_objects as go
    from orientation_ops import cu2ho, ho2qu, qu2ho, qu_prod
    from laue_ops import laue_elements
    from riesz_energy import riesz_energies_fused

    ap = argparse.ArgumentParser(
        description="I(532) — cu2ho → KR(FZ, canonical) with ψ'(u); saves base/E3/NN/ops plots"
    )
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--z", type=int, default=5)
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    ap.add_argument("--out_base", type=str, default="I532_base.html")
    ap.add_argument("--out_e3", type=str, default="I532_E3.html")
    ap.add_argument("--out_nn", type=str, default="I532_NN.html")
    ap.add_argument("--out_ops", type=str, default="I532_ops.html")
    ap.add_argument("--downsample", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    torch.set_grad_enabled(False)

    cu_grid = so3_cubochoric_grid_stretch(args.h, args.z, device=device, dtype=DTYPE)
    ho_src = cu2ho(cu_grid.to(dtype=DTYPE, device=device))
    ho_map = ho2ho_I(ho_src)

    lim = float(H_MAX)
    # Base
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_src[:, 0].cpu(),
            y=ho_src[:, 1].cpu(),
            z=ho_src[:, 2].cpu(),
            mode="markers",
            name="Original (cu→ho)",
            marker=dict(size=2, opacity=0.35),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].cpu(),
            y=ho_map[:, 1].cpu(),
            z=ho_map[:, 2].cpu(),
            mode="markers",
            name="RFZ (canonical)",
            marker=dict(size=2, opacity=0.9),
        )
    )
    fig.update_layout(
        title="I(532) KR — canonical RFZ (minimal)",
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
    print(f"[ok] wrote {args.out_base} ({ho_map.shape[0]} pts)")

    # Energies & NN
    q_fz = ho2qu(ho_map.to(dtype=DTYPE, device=device))
    q_fz = q_fz / torch.clamp(q_fz.norm(dim=-1, keepdim=True), min=1e-15)
    ops = laue_elements(12).to(dtype=DTYPE, device=device)
    E1, E2, E3, _, _, S3_i, NN_i = riesz_energies_fused(
        q_fz, ops, return_contrib=True, return_nn=True
    )

    # E3 color
    s3 = (
        (S3_i / torch.clamp(torch.median(S3_i), min=1e-15))
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_src[:, 0].cpu(),
            y=ho_src[:, 1].cpu(),
            z=ho_src[:, 2].cpu(),
            mode="markers",
            name="Original (cu→ho)",
            marker=dict(size=2, opacity=0.25),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].cpu(),
            y=ho_map[:, 1].cpu(),
            z=ho_map[:, 2].cpu(),
            mode="markers",
            name="RFZ colored by E3",
            marker=dict(
                size=3, color=s3, colorscale="Turbo", showscale=True, opacity=1.0
            ),
        )
    )
    fig.update_layout(
        title=f"I(532) RFZ — Riesz E3 contrib (total={float(E3):.6e})",
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
    print(f"[ok] wrote {args.out_e3}")

    # NN color
    NN = (
        (NN_i / torch.clamp(torch.median(NN_i), min=1e-15))
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    print(
        f"NN stats — min: {float(NN_i.min())}, median: {float(torch.median(NN_i))}, max: {float(NN_i.max())}"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_src[:, 0].cpu(),
            y=ho_src[:, 1].cpu(),
            z=ho_src[:, 2].cpu(),
            mode="markers",
            name="Original (cu→ho)",
            marker=dict(size=2, opacity=0.25),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].cpu(),
            y=ho_map[:, 1].cpu(),
            z=ho_map[:, 2].cpu(),
            mode="markers",
            name="RFZ colored by NN",
            marker=dict(
                size=3, color=NN, colorscale="Turbo", showscale=True, opacity=1.0
            ),
        )
    )
    fig.update_layout(
        title=f"I(532) RFZ — NN index (median={float(torch.median(NN_i)):.6e})",
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

    # Ops copies (optional)
    G = ops.shape[0]
    colors = palette60()
    max_plot = args.downsample if args.downsample and args.downsample > 0 else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ho_map[:, 0].cpu(),
            y=ho_map[:, 1].cpu(),
            z=ho_map[:, 2].cpu(),
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

        pts = ho_prime.detach()
        if max_plot is not None and pts.shape[0] > max_plot:
            idx = torch.randint(0, pts.shape[0], (max_plot,), device=pts.device)
            pts = pts[idx]
        xyz = pts.cpu().numpy()

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
    print(f"[ok] wrote {args.out_ops} ({G} ops)")


if __name__ == "__main__":
    main()
