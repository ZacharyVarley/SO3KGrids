#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → O (octahedral) FZ mapper using hardcoded Padé φ'(u) (n=15, m=13)

Pipeline (ordered-simplex sector, first octant, x ≤ y ≤ z):
1) Fold homochoric vector to ordered simplex and record permutation/signs
2) Spherical: (r, θ, φ) on source sector
3) Azimuth (outer KR):
   - Compute source u_src = C_src(φ) / C_src(φ_hi), φ ∈ [π/4, π/2]
   - Map φ' = φ_inv_O(u_src) using Padé inverse CDF (Chebyshev in t = 2u-1)
4) Polar (inner KR):
   - y_src = (1 - cos θ) / (1 - cos θ_max_src(φ))
   - Solve y_src = G_tar(θ'; φ') / A_tar(φ') with safeguarded Newton (GL16 integrals)
5) Radial:
   - c(θ',φ') = max(c_max, c_sum), ρ' = (ρ^3)^(1/3), r' = ρ' * (r / H_MAX)
6) Undo permutation and restore original signs
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

# Azimuth domain for O (ordered simplex): [π/4, π/2]
PHI_LO_O = 0.25 * PI
PHI_HI_O = 0.50 * PI

# Guards
EPS_C = 1e-15
EPS_SLOPE = 1e-15

# Octahedral "max" face constant κ = √2 - 1
_KAPPA = math.sqrt(2.0) - 1.0

# ------------------- Hardcoded Padé φ'(u) for O (n=15, m=13) -------------------
# Numerator P(t) Chebyshev coeffs a[k], k=0..15
_A = torch.tensor([
    1.1822546284090303e+00,
    6.2667883943387037e-01,
    6.0044992184703982e-01,
    3.8819631496309054e-01,
    -4.4145326792933237e-01,
    -3.0362113191584633e-01,
    1.8863124699476139e-01,
    7.6897095197285464e-02,
    9.8597202925534602e-02,
    2.7635698327477154e-01,
    -2.0057747872002395e-02,
    6.1319952952553401e-02,
    -1.3776491479530030e-03,
    8.2840323424582154e-04,
    2.1685353523857393e-04,
    1.4705470527970237e-05,
], dtype=DTYPE, device=DEVICE)

# Denominator Q(t) = 1 + Σ_{k=1}^{13} b[k] T_k(t)  (b[0] fixed = 1)
_B = torch.tensor([
    1.0000000000000000e+00,
    1.2427500771143379e-01,
    4.2410008089771739e-01,
    3.3639100888253853e-01,
    -4.0743628876474214e-01,
    -2.3247960338991300e-01,
    2.0383650417779056e-01,
    2.5279219590521870e-02,
    3.7732606982845002e-02,
    2.4437987986578913e-01,
    -7.0431650497779760e-02,
    6.4426522575716655e-02,
    -1.1610440319712067e-02,
    2.0125442894998288e-03,
], dtype=DTYPE, device=DEVICE)

# ------------------- Gauss–Legendre (16-pt) -------------------
_GL16_X = torch.tensor([
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878317,
    -0.7554044083550031,
    -0.6178762444026437,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.09501250983763744,
     0.09501250983763744,
     0.2816035507792589,
     0.4580167776572274,
     0.6178762444026437,
     0.7554044083550031,
     0.8656312023878317,
     0.9445750230732326,
     0.9894009349916499,
], device=DEVICE)
_GL16_W = torch.tensor([
    0.027152459411754095,
    0.06225352393864789,
    0.09515851168249278,
    0.12462897125553387,
    0.14959598881657673,
    0.16915651939500254,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.16915651939500254,
    0.14959598881657673,
    0.12462897125553387,
    0.09515851168249278,
    0.06225352393864789,
    0.027152459411754095,
], device=DEVICE)

@torch.no_grad()
def gl16_integrate(func, a, b) -> torch.Tensor:
    """16-pt Gauss–Legendre integrate `func` over [a, b]. Broadcast-friendly."""
    a_t = torch.as_tensor(a, device=DEVICE)
    b_t = torch.as_tensor(b, device=DEVICE)
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
    return 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))

@torch.no_grad()
def R_from_c(c: torch.Tensor) -> torch.Tensor:
    return rho3_from_c(c).clamp_min(0.0).pow(1.0 / 3.0)

# ------------------- Octahedral geometry -------------------
@torch.no_grad()
def theta_max_O(phi: torch.Tensor) -> torch.Tensor:
    """θ_max(φ) = atan(1 / sin φ)."""
    s = torch.sin(phi).clamp_min(EPS_C)
    return torch.atan(1.0 / s)

@torch.no_grad()
def theta_switch_O(phi: torch.Tensor) -> torch.Tensor:
    """tan θ_s = √2 / (cos φ + sin φ)."""
    denom = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS_C)
    return torch.atan(torch.sqrt(torch.tensor(2.0, dtype=DTYPE, device=DEVICE)) / denom)

@torch.no_grad()
def c_O_components(theta: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(c_max, c_sum)"""
    c_max = (1.0 / _KAPPA) * torch.cos(theta)
    c_sum = torch.cos(theta) + torch.sin(theta) * (torch.cos(phi) + torch.sin(phi))
    return c_max, c_sum

# ------------------- Column area A(φ) (target) -------------------
@torch.no_grad()
def A_phi_O(phi: torch.Tensor) -> torch.Tensor:
    """A(φ) piecewise over θ ∈ [0, θ_max(φ)], switching at θ_s(φ)."""
    ths = theta_switch_O(phi)   # (N,)
    thh = theta_max_O(phi)      # (N,)

    def f_pc(th):
        cmax = (1.0 / _KAPPA) * torch.cos(th)
        return rho3_from_c(cmax) * torch.sin(th)

    A = torch.empty_like(phi)
    only_pc = ths >= thh
    if only_pc.any():
        A[only_pc] = gl16_integrate(f_pc, 0.0, thh[only_pc])

    if (~only_pc).any():
        mask = ~only_pc
        ths_loc = ths[mask]
        thh_loc = thh[mask]
        phi_loc = phi[mask]

        A1 = gl16_integrate(f_pc, 0.0, ths_loc)

        trig_phi = torch.cos(phi_loc) + torch.sin(phi_loc)

        def f_sum_loc(th):
            cs = torch.cos(th) + torch.sin(th) * trig_phi  # (16,M)
            return rho3_from_c(cs) * torch.sin(th)

        A2 = gl16_integrate(f_sum_loc, ths_loc, thh_loc)
        A[mask] = A1 + A2

    return A.clamp_min(EPS_SLOPE)

# ------------------- Source azimuthal CDF on ordered simplex -------------------
@torch.no_grad()
def C_src_phi_O(phi: torch.Tensor, phi_lo: float, phi_hi: float) -> torch.Tensor:
    """C_src(φ) = ∫_{φ_lo}^{φ} A_src(ψ) dψ,  A_src(ψ) = 1 - cos θ_max(ψ)."""
    def A_src(psi: torch.Tensor) -> torch.Tensor:
        thh = theta_max_O(psi)
        return 1.0 - torch.cos(thh)
    return gl16_integrate(A_src, torch.as_tensor(phi_lo, dtype=DTYPE, device=DEVICE), phi)

# ------------------- Chebyshev / Padé evaluation -------------------
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
def chebyshev_clenshaw_with_deriv(coeffs: torch.Tensor, t: torch.Tensor):
    # Returns (T(t), dT/dt) for Σ c_k T_k(t)
    if coeffs.numel() == 1:
        return torch.full_like(t, coeffs[0]), torch.zeros_like(t)
    # Standard Clenshaw for value
    b1 = torch.zeros_like(t); b2 = torch.zeros_like(t)
    for c_k in coeffs[1:].flip(0):
        b0 = 2.0 * t * b1 - b2 + c_k
        b2, b1 = b1, b0
    val = t * b1 - b2 + coeffs[0]
    # Derivative: d/dt T_k(t) = k U_{k-1}(t). Implement via a second Clenshaw on U, or
    # use recurrence for derivative directly:
    # Here’s a compact U-based evaluator:
    def eval_U(coeffsU):  # coeffsU[k] multiplies U_k
        u1 = torch.zeros_like(t); u2 = torch.zeros_like(t)  # Clenshaw for U
        for c_k in coeffsU[::-1]:
            u0 = 2.0 * t * u1 - u2 + c_k
            u2, u1 = u1, u0
        return u1  # evaluates Σ c_k U_k
    # Build U coefficients for the derivative: T_0' = 0, T_k' = k U_{k-1}
    if coeffs.numel() == 2:
        deriv = torch.zeros_like(t)
    else:
        deg = coeffs.numel() - 1
        Ucoeffs = torch.zeros(deg, device=t.device, dtype=t.dtype)  # up to U_{deg-1}
        # accumulate U_{k-1} coeffs with weights k*c_k
        # We'll evaluate Σ_{k>=1} k*c_k * U_{k-1}(t)
        Ucoeffs = torch.stack([ (k) * coeffs[k] for k in range(1, deg+1) ])
        # shift so index matches U_{k-1}:
        # eval_U expects coeffs over U_0..U_{deg-1}
        deriv = eval_U(torch.cat([Ucoeffs, torch.zeros(0, device=t.device, dtype=t.dtype)]))
    return val, deriv

@torch.no_grad()
def dphi_du_from_pade(u: torch.Tensor) -> torch.Tensor:
    # φ(u) = P(t)/Q(t),  t=2u-1  ⇒ dφ/du = 2*(P'Q - P Q')/Q^2
    u = u.clamp(0.0, 1.0)
    t = 2.0 * u - 1.0
    # Numerator P
    P, dPdt = chebyshev_clenshaw_with_deriv(_A, t)
    # Denominator Q = 1 + Σ b[k] T_k(t) with b[0]=1 fixed, we stored _B = [1, b1, ...]
    d = _B.clone(); d[0] = 0.0
    Q, dQdt = chebyshev_clenshaw_with_deriv(d, t)
    Q = 1.0 + Q
    dφ_dt = (dPdt * Q - P * dQdt) / (Q * Q)
    dφ_du = 2.0 * dφ_dt
    # guard positivity and poles
    dφ_du = dφ_du.clamp_min(1e-12)
    return dφ_du


@torch.no_grad()
def phi_inv_O(u: torch.Tensor) -> torch.Tensor:
    """Padé inverse CDF for O on u ∈ [0,1] → φ' ∈ [π/4, π/2]."""
    u = u.clamp(0.0, 1.0)
    t = 2.0 * u - 1.0
    P = chebyshev_clenshaw(_A, t)
    d = _B.clone()
    d[0] = 0.0
    Q = 1.0 + chebyshev_clenshaw(d, t)
    phi = (P / Q).clamp(PHI_LO_O, PHI_HI_O)
    return phi

# ------------------- Main mapping: ho2ho_O -------------------
@torch.no_grad()
def ho2ho_O(ho: torch.Tensor, newton_iters: int = 8) -> torch.Tensor:
    """
    Map homochoric points to the O FZ using the fitted Padé azimuthal marginal.

    Args:
        ho: (..., 3) homochoric vectors
        newton_iters: polar Newton iterations (with damping)

    Returns:
        mapped: (..., 3) mapped homochoric vectors in the O FZ
    """
    v0 = ho.to(device=DEVICE, dtype=DTYPE).clone()

    # Fold to first octant and enforce x ≤ y ≤ z (ordered simplex)
    sgn = torch.sign(v0); sgn[sgn == 0] = 1.0
    v = v0.abs()

    vals, idxs = torch.sort(v, dim=-1)  # ascending → ordered simplex
    v_sorted = vals

    # Build inverse permutation to undo later
    inv_idx = torch.empty_like(idxs)
    base = torch.arange(3, device=DEVICE).view(*([1] * (idxs.ndim - 1)), 3)
    inv_idx.scatter_(-1, idxs, base.expand_as(idxs))

    # Spherical on source sector
    r, th, ph = sph_from_cart(v_sorted)

    # Outer KR: source CDF on [π/4, π/2] then Padé inverse to φ'
    C_hi = C_src_phi_O(torch.as_tensor(PHI_HI_O, dtype=DTYPE, device=DEVICE), PHI_LO_O, PHI_HI_O)
    u_src = (C_src_phi_O(ph, PHI_LO_O, PHI_HI_O) / C_hi).clamp(0.0, 1.0)
    ph_p = phi_inv_O(u_src)

    # Inner KR: y_src normalized by source ceiling
    th_hi_src = theta_max_O(ph)
    denom = (1.0 - torch.cos(th_hi_src)).clamp_min(1e-12)
    y_src = ((1.0 - torch.cos(th)) / denom).clamp(0.0, 1.0)

    # Target column area A_tar(φ')
    A_tar = A_phi_O(ph_p)

    # Initial guess honoring target ceiling θ_max(φ')
    th_hi_tar = theta_max_O(ph_p)
    th_p = torch.acos((1.0 - y_src * (1.0 - torch.cos(th_hi_tar))).clamp(-1.0, 1.0))

    # Safeguarded Newton on y_src = G_tar(θ'; φ') / A_tar(φ')
    for _ in range(newton_iters):
        th_p = torch.max(torch.zeros_like(th_p), torch.min(th_p, th_hi_tar))

        ths = theta_switch_O(ph_p)
        only_pc = ths >= th_hi_tar

        G = torch.empty_like(th_p)

        # Case A: entirely polar-cap
        if only_pc.any():
            G[only_pc] = gl16_integrate(
                lambda t: rho3_from_c((1.0 / _KAPPA) * torch.cos(t)) * torch.sin(t),
                0.0, th_p[only_pc]
            )

        # Case B: column that switches
        if (~only_pc).any():
            mask = ~only_pc
            thp_loc = th_p[mask]
            ths_loc = ths[mask]
            phi_loc = ph_p[mask]
            trig_phi = torch.cos(phi_loc) + torch.sin(phi_loc)

            pc_branch = thp_loc <= ths_loc
            sum_branch = ~pc_branch
            G_loc = torch.empty_like(thp_loc)

            if pc_branch.any():
                G_loc[pc_branch] = gl16_integrate(
                    lambda t: rho3_from_c((1.0 / _KAPPA) * torch.cos(t)) * torch.sin(t),
                    0.0, thp_loc[pc_branch]
                )

            if sum_branch.any():
                thp2 = thp_loc[sum_branch]
                ths2 = ths_loc[sum_branch]

                def f_pc_local(t):
                    return rho3_from_c((1.0 / _KAPPA) * torch.cos(t)) * torch.sin(t)

                def f_sum_local(t):
                    cs = torch.cos(t) + torch.sin(t) * trig_phi[sum_branch]  # (16,M)
                    return rho3_from_c(cs) * torch.sin(t)

                G1 = gl16_integrate(f_pc_local, 0.0, ths2)
                G2 = gl16_integrate(f_sum_local, ths2, thp2)
                G_loc[sum_branch] = G1 + G2

            G[mask] = G_loc

        Gn = (G / A_tar).clamp(0.0, 1.0)

        # Slope f_tar/A_tar at θ'
        cmax, csum = c_O_components(th_p, ph_p)
        c_act = torch.maximum(cmax, csum).clamp_min(EPS_C)
        fn = (rho3_from_c(c_act) * torch.sin(th_p) / A_tar).clamp_min(EPS_SLOPE)

        step = (Gn - y_src) / fn
        th_new = th_p - step
        big = step.abs() > 0.25
        th_new = torch.where(big, 0.75 * th_p + 0.25 * th_new, th_new)
        th_new = torch.max(torch.zeros_like(th_new), torch.min(th_new, th_hi_tar))

        if torch.max((th_new - th_p).abs()).item() < 1e-12:
            th_p = th_new
            break
        th_p = th_new

    # Radial scaling
    cmax, csum = c_O_components(th_p, ph_p)
    rho = R_from_c(torch.maximum(cmax, csum))
    r_p = rho * (r / H_MAX)
    mapped_sorted = cart_from_sph(r_p, th_p, ph_p)

    # Undo permutation and signs
    mapped = torch.gather(mapped_sorted, dim=-1, index=inv_idx)
    mapped = mapped * sgn
    return mapped

# ------------------- Quick self-test -------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 8
    ho = torch.randn(N, 3, device=DEVICE, dtype=DTYPE)
    norms = torch.linalg.norm(ho, dim=-1, keepdim=True).clamp_min(1e-12)
    ho = ho / norms * (0.9 * H_MAX)

    out = ho2ho_O(ho)
    print("in  shape:", tuple(ho.shape), "out shape:", tuple(out.shape))
    print("sample in[0]: ", ho[0].tolist())
    print("sample out[0]:", out[0].tolist())
