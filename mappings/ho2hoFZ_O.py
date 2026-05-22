#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → Octahedral (O) FZ mapper (piecewise Padé inverse CDF)

Compute A(φ') from the fitted inverse CDF using the correct constant C_tot = π^2 / (24*48).

Why this is correct:
    u = C(φ)/C_tot,  φ' = Φ^{-1}(u)  ⇒  du/dφ = A(φ)/C_tot  ⇒  A(φ) = C_tot / (dφ/du).
For O on the ordered–simplex sector (abs + sort, x ≤ y ≤ z):
    C_tot = Vol(FZ on sector) = (π^2/|O|) * (1/48) = π^2 / (24*48) = π^2 / 384.

This script:
  * Uses piecewise Padé fits (P/Q in Chebyshev T_k(t_piece)) to get φ'(u) and dφ/du in one pass.
  * Sets A_tar(φ') = C_tot / (dφ/du).
  * Leaves the θ′ KR step and radial law unchanged.
  * Has an optional validation path to compare against direct column integrals.
"""

from __future__ import annotations
import math
from typing import Tuple
import torch

# ---- Core config ----
torch.set_default_dtype(torch.float64)
PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)
PHI_LO_O = 0.25 * PI
PHI_HI_O = 0.50 * PI
EPS_C = 1e-15
EPS_SLOPE = 1e-15

# |O| = 24; ordered–simplex sector uses 1/48 of directions
# C_tot = π^2 / 384
C_TOT_O_SECTOR = (PI * PI) / 384.0

_KAPPA = math.sqrt(2.0) - 1.0

# ---- Piecewise Padé coefficients for φ'(u) ----
# Breakpoint u*
U_BREAK = 0.5667905347333297

# Left piece: u in [0, U_BREAK], t = 2*(u-u0)/(u1-u0) - 1
_A_L = torch.tensor(
    [
        0.972852840588069,
        0.19114256303868526,
        -0.07200597429184963,
        0.0019036882372386453,
        -6.90762808742289e-05,
        -3.541269544600796e-05,
        1.2802239724117446e-06,
        -4.353221441400241e-08,
        -5.969310864624526e-09,
    ],
    dtype=DTYPE,
    device=DEVICE,
)

_B_L = torch.tensor(
    [
        1.0,
        0.003306117640028675,
        -0.08447285894067914,
        0.010338177449608622,
        -0.0007327163426372906,
        -4.420643674300553e-06,
        3.672842561317652e-06,
        -3.147154560490602e-07,
        1.105314525177379e-08,
    ],
    dtype=DTYPE,
    device=DEVICE,
)

# Right piece: u in [U_BREAK, 1], t = 2*(u-u0)/(u1-u0) - 1
_A_R = torch.tensor(
    [
        1.3707109341917165,
        0.1947432679225177,
        0.007269980754321227,
        0.0007151827313080849,
        -5.174273157437798e-05,
        -5.3091738401265805e-06,
        -1.659001355271721e-07,
        2.8004315443311014e-09,
        3.284693250980296e-10,
    ],
    dtype=DTYPE,
    device=DEVICE,
)

_B_R = torch.tensor(
    [
        1.0,
        -0.0017419236794533773,
        0.002750778936592661,
        0.0007074471694800329,
        -6.934266776154665e-05,
        -7.729908914691581e-07,
    ],
    dtype=DTYPE,
    device=DEVICE,
)

# ---- Gauss–Legendre 16-pt (for optional validation) ----
_GL16_X = torch.tensor(
    [
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
    ],
    device=DEVICE,
)
_GL16_W = torch.tensor(
    [
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
    ],
    device=DEVICE,
)


@torch.no_grad()
def gl16_integrate(func, a, b) -> torch.Tensor:
    a_t = torch.as_tensor(a, device=DEVICE, dtype=DTYPE)
    b_t = torch.as_tensor(b, device=DEVICE, dtype=DTYPE)
    a_b, b_b = torch.broadcast_tensors(a_t, b_t)
    half = 0.5 * (b_b - a_b)
    mid = 0.5 * (b_b + a_b)
    xa = _GL16_X.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0) + mid.unsqueeze(0)
    wa = _GL16_W.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0)
    val = func(xa)
    wa_b = wa.reshape(wa.shape + (1,) * (val.ndim - wa.ndim))
    return (val * wa_b).sum(dim=0)


# ---- Geometry helpers ----
@torch.no_grad()
def sph_from_cart(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = torch.linalg.norm(v, dim=-1)
    ct = torch.where(r > 0, (z / r).clamp(-1.0, 1.0), torch.ones_like(z))
    theta = torch.acos(ct)
    phi = torch.atan2(y, x)
    return r, theta, phi


@torch.no_grad()
def cart_from_sph(r: torch.Tensor, th: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    st, ct = torch.sin(th), torch.cos(th)
    cp, sp = torch.cos(ph), torch.sin(ph)
    return torch.stack([r * st * cp, r * st * sp, r * ct], dim=-1)


@torch.no_grad()
def rho3_from_c(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp_min(EPS_C)
    return 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))


@torch.no_grad()
def R_from_c(c: torch.Tensor) -> torch.Tensor:
    return rho3_from_c(c).clamp_min(0.0).pow(1.0 / 3.0)


@torch.no_grad()
def theta_max_O(phi: torch.Tensor) -> torch.Tensor:
    s = torch.sin(phi).clamp_min(EPS_C)
    return torch.atan(1.0 / s)


@torch.no_grad()
def theta_switch_O(phi: torch.Tensor) -> torch.Tensor:
    denom = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS_C)
    return torch.atan(torch.sqrt(torch.tensor(2.0, dtype=DTYPE, device=DEVICE)) / denom)


@torch.no_grad()
def c_O_components(
    theta: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_max = (1.0 / _KAPPA) * torch.cos(theta)
    c_sum = torch.cos(theta) + torch.sin(theta) * (torch.cos(phi) + torch.sin(phi))
    return c_max, c_sum


# ---- Source azimuthal CDF on ordered–simplex (unchanged) ----
@torch.no_grad()
def C_src_phi_O(phi: torch.Tensor, phi_lo: float, phi_hi: float) -> torch.Tensor:
    def A_src(psi: torch.Tensor) -> torch.Tensor:
        thh = theta_max_O(psi)
        return 1.0 - torch.cos(thh)

    return gl16_integrate(
        A_src, torch.as_tensor(phi_lo, dtype=DTYPE, device=DEVICE), phi
    )


# ---- Chebyshev T-series: value and derivative ----
@torch.no_grad()
def cheb_T_value(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    n = coeffs.numel() - 1
    if n < 0:
        return torch.zeros_like(t)
    if n == 0:
        return torch.full_like(t, coeffs[0])
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for k in range(n, 0, -1):
        b0 = 2.0 * t * b1 - b2 + coeffs[k]
        b2, b1 = b1, b0
    return t * b1 - b2 + coeffs[0]


@torch.no_grad()
def cheb_T_value_deriv(
    coeffs: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    f = cheb_T_value(coeffs, t)
    n = coeffs.numel() - 1
    if n <= 0:
        return f, torch.zeros_like(t)
    c = torch.stack([(j + 1) * coeffs[j + 1] for j in range(n)])  # (n,)
    U0 = torch.ones_like(t)
    if n == 1:
        return f, c[0] * U0
    U1 = 2.0 * t
    d = c[0] * U0 + c[1] * U1
    Ukm2, Ukm1 = U0, U1
    for j in range(2, n):
        Uk = 2.0 * t * Ukm1 - Ukm2
        d = d + c[j] * Uk
        Ukm2, Ukm1 = Ukm1, Uk
    return f, d


# ---- Piecewise inverse CDF φ'(u) and derivative dφ/du ----
@torch.no_grad()
def u_to_t_piece(u: torch.Tensor, u0: float, u1: float) -> torch.Tensor:
    return 2.0 * (u - u0) / (u1 - u0) - 1.0


@torch.no_grad()
def phi_and_dphi_du_from_pade_piece(
    u: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u = u.clamp(0.0, 1.0)

    uL0, uL1 = 0.0, U_BREAK
    uR0, uR1 = U_BREAK, 1.0

    use_left = u <= U_BREAK

    phi = torch.empty_like(u)
    dphi_du = torch.empty_like(u)

    if use_left.any():
        uu = u[use_left]
        t = u_to_t_piece(uu, uL0, uL1)
        P, dPdt = cheb_T_value_deriv(_A_L, t)
        d = _B_L.clone()
        d[0] = 0.0
        Q0, dQdt = cheb_T_value_deriv(d, t)
        Q = 1.0 + Q0
        phi_u = P / Q
        dphi_dt = (dPdt * Q - P * dQdt) / (Q * Q + 0.0)
        dt_du = 2.0 / (uL1 - uL0)
        dphi_du_u = (dt_du * dphi_dt).clamp_min(1e-12)
        phi[use_left] = phi_u
        dphi_du[use_left] = dphi_du_u

    if (~use_left).any():
        uu = u[~use_left]
        t = u_to_t_piece(uu, uR0, uR1)
        P, dPdt = cheb_T_value_deriv(_A_R, t)
        d = _B_R.clone()
        d[0] = 0.0
        Q0, dQdt = cheb_T_value_deriv(d, t)
        Q = 1.0 + Q0
        phi_u = P / Q
        dphi_dt = (dPdt * Q - P * dQdt) / (Q * Q + 0.0)
        dt_du = 2.0 / (uR1 - uR0)
        dphi_du_u = (dt_du * dphi_dt).clamp_min(1e-12)
        phi[~use_left] = phi_u
        dphi_du[~use_left] = dphi_du_u

    phi = phi.clamp(PHI_LO_O, PHI_HI_O)
    return phi, dphi_du


# ---- Optional: direct column area at ceiling (for validation) ----
@torch.no_grad()
def column_area_O(phi: torch.Tensor) -> torch.Tensor:
    ths = theta_switch_O(phi)
    thh = theta_max_O(phi)

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
            cs = torch.cos(th) + torch.sin(th) * trig_phi
            return rho3_from_c(cs) * torch.sin(th)

        A2 = gl16_integrate(f_sum_loc, ths_loc, thh_loc)
        A[mask] = A1 + A2

    return A.clamp_min(EPS_SLOPE)


# ---- Main mapper (uses A from inverse CDF with correct C_tot) ----
@torch.no_grad()
def ho2ho_O(
    ho: torch.Tensor, newton_iters: int = 8, validate_A: bool = False
) -> torch.Tensor:
    v0 = ho.to(device=DEVICE, dtype=DTYPE).clone()
    sgn = torch.sign(v0)
    sgn[sgn == 0] = 1.0
    v = v0.abs()
    vals, idxs = torch.sort(v, dim=-1)  # ordered simplex: x ≤ y ≤ z
    v_sorted = vals

    inv_idx = torch.empty_like(idxs)
    base = torch.arange(3, device=DEVICE).view(*([1] * (idxs.ndim - 1)), 3)
    inv_idx.scatter_(-1, idxs, base.expand_as(idxs))

    r, th, ph = sph_from_cart(v_sorted)

    # Outer KR: source CDF → u ∈ [0,1], then piecewise Padé inverse to φ'
    C_hi = C_src_phi_O(
        torch.as_tensor(PHI_HI_O, dtype=DTYPE, device=DEVICE), PHI_LO_O, PHI_HI_O
    )
    u_src = (C_src_phi_O(ph, PHI_LO_O, PHI_HI_O) / C_hi).clamp(0.0, 1.0)

    ph_p, dphi_du = phi_and_dphi_du_from_pade_piece(u_src)

    # Column area from inverse CDF with normalization
    A_tar = (C_TOT_O_SECTOR / dphi_du).clamp_min(EPS_SLOPE)

    if validate_A:
        A_true = column_area_O(ph_p)
        rel = (A_tar - A_true).abs() / A_true.clamp_min(1e-15)
        max_i = rel.argmax()
        print("validate_A: max_rel =", rel[max_i].item())

    # Normalize source polar CDF by its ceiling
    th_hi_src = theta_max_O(ph)
    denom = (1.0 - torch.cos(th_hi_src)).clamp_min(1e-12)
    y_src = ((1.0 - torch.cos(th)) / denom).clamp(0.0, 1.0)

    # Initial θ′ honoring target ceiling
    th_hi_tar = theta_max_O(ph_p)
    th_p = torch.acos((1.0 - y_src * (1.0 - torch.cos(th_hi_tar))).clamp(-1.0, 1.0))

    # Newton: solve y_src = G(θ′;φ′)/A(φ′)
    for _ in range(newton_iters):
        th_p = torch.max(torch.zeros_like(th_p), torch.min(th_p, th_hi_tar))
        ths = theta_switch_O(ph_p)
        only_pc = ths >= th_hi_tar

        G = torch.empty_like(th_p)

        if only_pc.any():
            G[only_pc] = gl16_integrate(
                lambda t: rho3_from_c((1.0 / _KAPPA) * torch.cos(t)) * torch.sin(t),
                0.0,
                th_p[only_pc],
            )

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
                    0.0,
                    thp_loc[pc_branch],
                )

            if sum_branch.any():
                thp2 = thp_loc[sum_branch]
                ths2 = ths_loc[sum_branch]

                def f_pc_local(t):
                    return rho3_from_c((1.0 / _KAPPA) * torch.cos(t)) * torch.sin(t)

                def f_sum_local(t):
                    cs = torch.cos(t) + torch.sin(t) * trig_phi[sum_branch]
                    return rho3_from_c(cs) * torch.sin(t)

                G1 = gl16_integrate(f_pc_local, 0.0, ths2)
                G2 = gl16_integrate(f_sum_local, ths2, thp2)
                G_loc[sum_branch] = G1 + G2

            G[mask] = G_loc

        Gn = (G / A_tar).clamp(0.0, 1.0)

        # slope d/dθ′ [G(θ′;φ′)/A_tar] = (ρ^3(c_act) sinθ′)/A_tar
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

    # Undo permutation & signs
    mapped = torch.gather(mapped_sorted, dim=-1, index=inv_idx)
    mapped = mapped * sgn
    return mapped


# ---- Smoke test ----
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 32
    ho = torch.randn(N, 3, device=DEVICE, dtype=DTYPE)
    ho = (
        ho
        / torch.linalg.norm(ho, dim=-1, keepdim=True).clamp_min(1e-12)
        * (0.9 * H_MAX)
    )

    out = ho2ho_O(ho, validate_A=True)
    print("ok", out.shape)
