#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → T (tetrahedral, cubic-low) FZ mapper using hardcoded Padé φ'(u)

Pipeline
--------
1) Fold homochoric vector to canonical sector (first octant, x >= y)
2) Spherical: (r, θ, φ)
3) Azimuth:  φ' = φ_inv_T(u) with u = φ / (π/4)  (Padé inverse CDF in Chebyshev basis)
4) Polar:    solve y = (G/A)(θ'; φ') with safeguarded Newton, where
             y = 1 - cos θ,   A(φ') = ∫_0^{π/2} ρ^3(c_T(θ, φ')) sinθ dθ
             G(θ'; φ') = ∫_0^{θ'} ρ^3(c_T(θ, φ')) sinθ dθ
5) Radial:   ρ' = ρ(c_T(θ', φ')) * (r / H_MAX), with ρ = (ρ^3)^(1/3)

Notes
-----
- Uses float64 throughout; runs on CUDA if available.
- φ domain for T: [0, π/4].
- Broadcast-safe Gauss–Legendre integration (16-pt) for A and G.
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn.functional as F

# ------------------- Config & constants -------------------
PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# Homochoric ball radius
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)

# Domain
PHI_MAX_T = 0.25 * PI

# Epsilons / guards
EPS_C = 1e-15
EPS_SLOPE = 1e-15

# ------------------- Hardcoded Padé φ'(u) for T -------------------
# Numerator P(t) Chebyshev coeffs a[k], k=0..12
_A = torch.tensor([
    0.36781788207117677,
    0.3947242675359163,
    0.025552232221354085,
    -0.0019925177097101683,
    -0.0006826585228387496,
    -3.461937624041875e-05,
    1.0711964436465566e-05,
    8.266471280731193e-07,
    4.637247599777716e-07,
    6.556999583536072e-07,
    -1.4219967302931709e-08,
    4.576707508498134e-09,
    1.3484067046460952e-10,
], dtype=DTYPE, device=DEVICE)

# Denominator Q(t) = 1 + Σ_{k=1}^{10} b[k] T_k(t)  (b[0] fixed at 1.0)
_B = torch.tensor([
    1.0,
    2.9141418502709427e-06,
    -1.5269094632305715e-06,
    6.416353453852009e-07,
    1.6615935995742397e-06,
    -1.5174845503286455e-06,
    -1.8299029656639242e-06,
    -6.052492204511186e-06,
    4.967030741589562e-06,
    -5.04089587435592e-07,
    6.408342665323226e-08,
], dtype=DTYPE, device=DEVICE)

# ------------------- Gauss–Legendre (16-pt) -------------------
_GL16_X = torch.tensor([
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878317, -0.7554044083550031,
    -0.6178762444026437, -0.4580167776572274, -0.2816035507792589, -0.09501250983763744,
     0.09501250983763744,  0.2816035507792589,  0.4580167776572274,  0.6178762444026437,
     0.7554044083550031,  0.8656312023878317,  0.9445750230732326,  0.9894009349916499
], dtype=DTYPE, device=DEVICE)
_GL16_W = torch.tensor([
    0.027152459411754095, 0.06225352393864789, 0.09515851168249278, 0.12462897125553387,
    0.14959598881657673, 0.16915651939500254, 0.1826034150449236,  0.1894506104550685,
    0.1894506104550685,  0.1826034150449236,  0.16915651939500254, 0.14959598881657673,
    0.12462897125553387, 0.09515851168249278, 0.06225352393864789, 0.027152459411754095
], dtype=DTYPE, device=DEVICE)

@torch.no_grad()
def gl16_integrate(func, a, b) -> torch.Tensor:
    """16-pt Gauss–Legendre integrate func over [a,b]. Broadcast-safe."""
    a_t = torch.as_tensor(a, dtype=DTYPE, device=DEVICE)
    b_t = torch.as_tensor(b, dtype=DTYPE, device=DEVICE)
    a_b, b_b = torch.broadcast_tensors(a_t, b_t)

    half = 0.5 * (b_b - a_b)
    mid  = 0.5 * (b_b + a_b)

    xa = _GL16_X.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0) + mid.unsqueeze(0)
    wa = _GL16_W.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0)

    val = func(xa)  # (16, ...)
    wa_b = wa.reshape(wa.shape + (1,) * (val.ndim - wa.ndim))
    return (val * wa_b).sum(dim=0)


# ------------------- Spherical/cartesian helpers -------------------
def sph_from_cart(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = torch.linalg.norm(v, dim=-1)
    ct = torch.where(r > 0, (z / r).clamp(-1.0, 1.0), torch.ones_like(z))
    theta = torch.acos(ct)
    phi = torch.atan2(y, x)
    return r, theta, phi

def cart_from_sph(r: torch.Tensor, th: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    st, ct = torch.sin(th), torch.cos(th)
    cp, sp = torch.cos(ph), torch.sin(ph)
    return torch.stack([r * st * cp, r * st * sp, r * ct], dim=-1)

# ------------------- Homochoric radial law -------------------
@torch.no_grad()
def rho3_from_c(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp_min(EPS_C)
    # ρ^3
    return 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))

@torch.no_grad()
def R_from_c(c: torch.Tensor) -> torch.Tensor:
    return rho3_from_c(c).clamp_min(0.0).pow(1.0 / 3.0)

# ------------------- T support function and marginals -------------------
@torch.no_grad()
def c_T(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.cos(theta) + torch.sin(theta) * (torch.cos(phi) + torch.sin(phi))

@torch.no_grad()
def A_phi_T(phi: torch.Tensor) -> torch.Tensor:
    """A(φ) = ∫_0^{π/2} ρ^3(c_T(θ, φ)) sinθ dθ, vectorized over φ."""
    def fth(th):
        # th: (16,) -> (16, 1) ; phi: (N,) -> (1, N) to get (16, N)
        th_e  = th.unsqueeze(-1)
        phi_e = phi.unsqueeze(0)
        return rho3_from_c(c_T(th_e, phi_e)) * torch.sin(th_e)
    return gl16_integrate(fth, 0.0, 0.5 * PI).clamp_min(EPS_SLOPE)

@torch.no_grad()
def G_theta_phi_T(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """G(θ; φ) = ∫_0^{θ} ρ^3(c_T(τ, φ)) sinτ dτ ; θ, φ broadcast."""
    def fth(th):
        # th: (16, ...) ; phi: (...) ; make them broadcastable
        return rho3_from_c(c_T(th, phi)) * torch.sin(th)
    return gl16_integrate(fth, 0.0, theta)

# ------------------- Chebyshev / Padé evaluation -------------------
@torch.no_grad()
def chebyshev_clenshaw(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute Σ c_k T_k(t) via Clenshaw."""
    if coeffs.numel() == 1:
        return torch.full_like(t, coeffs[0])
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for c_k in coeffs[1:].flip(0):
        b0 = 2.0 * t * b1 - b2 + c_k
        b2, b1 = b1, b0
    return t * b1 - b2 + coeffs[0]

@torch.no_grad()
def phi_inv_T(u: torch.Tensor) -> torch.Tensor:
    """Hardcoded Padé inverse CDF for T on u ∈ [0,1] → φ' ∈ [0, π/4]."""
    u = u.clamp(0.0, 1.0)
    t = 2.0 * u - 1.0
    # P(t)
    P = chebyshev_clenshaw(_A, t)
    # Q(t) = 1 + Σ b[k] T_k, with b[0] fixed 1.0 (we only evaluate the Σ_{k>=1})
    if _B.numel() == 1:
        Q = torch.ones_like(P)
    else:
        d = _B.clone()
        d[0] = 0.0
        Q = 1.0 + chebyshev_clenshaw(d, t)
    phi = (P / Q).clamp(0.0, PHI_MAX_T)
    return phi

# ------------------- Main mapping: ho2ho_T -------------------
@torch.no_grad()
def ho2ho_T(ho: torch.Tensor, newton_iters: int = 8) -> torch.Tensor:
    """
    Map homochoric points to the T FZ using the fitted Padé azimuthal marginal.

    Args:
        ho: (..., 3) homochoric vectors
        newton_iters: polar Newton iterations (with damping)

    Returns:
        mapped: (..., 3) mapped homochoric vectors in the T FZ
    """
    v = ho.to(device=DEVICE, dtype=DTYPE).clone()

    # Fold to first octant and enforce x >= y
    sgn = torch.sign(v)
    sgn[sgn == 0] = 1.0
    v = v.abs()

    mask_xy = v[..., 1] > v[..., 0]
    if mask_xy.any():
        v_sw = v.clone()
        v_sw[..., 0], v_sw[..., 1] = v[..., 1].clone(), v[..., 0].clone()
        v = torch.where(mask_xy.unsqueeze(-1), v_sw, v)

    # Spherical
    r, th, ph = sph_from_cart(v)

    # Azimuth reparameterization via Padé inverse CDF
    # Use a simple monotone parameter u = φ / (π/4)
    u = (ph / PHI_MAX_T).clamp(0.0, 1.0)
    ph_p = phi_inv_T(u)

    # Polar via safeguarded Newton on y = (G/A)(θ'; φ')
    y = 1.0 - torch.cos(th)
    A = A_phi_T(ph_p)
    # Initial guess: preserve θ as θ'
    th_p = th.clone().clamp(0.0, 0.5 * PI)

    for _ in range(newton_iters):
        G = G_theta_phi_T(th_p, ph_p)
        Gn = G / A
        # slope at θ' : f(θ') / A
        f_theta = (rho3_from_c(c_T(th_p, ph_p)) * torch.sin(th_p)).clamp_min(EPS_SLOPE)
        fn = f_theta / A
        step = (Gn - y) / fn
        th_new = (th_p - step).clamp(0.0, 0.5 * PI)
        # damping for very large steps
        big = step.abs() > 0.25
        th_new = torch.where(big, 0.75 * th_p + 0.25 * th_new, th_new)
        # early break if converged
        if torch.max((th_new - th_p).abs()).item() < 1e-12:
            th_p = th_new
            break
        th_p = th_new

    # Radial
    rho = R_from_c(c_T(th_p, ph_p))
    r_p = rho * (r / H_MAX)
    mapped = cart_from_sph(r_p, th_p, ph_p)

    # Undo y/x swap and signs
    if mask_xy.any():
        mapped_sw = mapped.clone()
        mapped_sw[..., 0], mapped_sw[..., 1] = mapped[..., 1].clone(), mapped[..., 0].clone()
        mapped = torch.where(mask_xy.unsqueeze(-1), mapped_sw, mapped)
    mapped = mapped * sgn
    return mapped

# ------------------- Optional quick self-test -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Some random homochoric points in the ball
    N = 8
    ho = torch.randn(N, 3, device=DEVICE, dtype=DTYPE)
    # shrink to inside the homochoric ball
    norms = torch.linalg.norm(ho, dim=-1, keepdim=True).clamp_min(1e-12)
    ho = ho / norms * (0.9 * H_MAX)

    out = ho2ho_T(ho)
    print("in  shape:", tuple(ho.shape), "out shape:", tuple(out.shape))
    print("sample in[0]: ", ho[0].tolist())
    print("sample out[0]:", out[0].tolist())
