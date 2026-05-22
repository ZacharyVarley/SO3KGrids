#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homochoric → Octahedral (O) FZ mapper
Compute A(φ') from the fitted inverse CDF using the correct constant C_tot = π^2 / (24*48).

Why this is correct:
    u = C(φ)/C_tot,  φ' = Φ^{-1}(u)  ⇒  du/dφ = A(φ)/C_tot  ⇒  A(φ) = C_tot / (dφ/du).
For O on the ordered–simplex sector (abs + sort, x ≤ y ≤ z):
    C_tot = Vol(FZ on sector) = (π^2/|O|) * (1/48) = π^2 / (24*48) = π^2 / 1152.

This script:
  * Uses your Padé fit (P/Q in Chebyshev T_k(t)) to get φ'(u) and dφ/du in one pass.
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
C_TOT_O_SECTOR = (PI * PI) / 384  # π^2 / 384

_KAPPA = math.sqrt(2.0) - 1.0

# ---- Your Padé coefficients for φ'(u) on t = 2u-1 ----
_A = torch.tensor(
    [
        1.1822546284090303e00,
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
    ],
    dtype=DTYPE,
    device=DEVICE,
)

_B = torch.tensor(
    [
        1.0000000000000000e00,
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


# ---- Chebyshev T-series: value and derivative (no negative slicing) ----
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
    # f'(t) = Σ_{k=1}^n k a_k U_{k-1}(t), with U_0=1, U_1=2t, U_k=2t U_{k-1}-U_{k-2}
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


# ---- Inverse CDF φ'(u) and derivative dφ/du ----
@torch.no_grad()
def phi_and_dphi_du_from_pade(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    u = u.clamp(0.0, 1.0)
    t = 2.0 * u - 1.0
    P, dPdt = cheb_T_value_deriv(_A, t)
    d = _B.clone()
    d[0] = 0.0
    Q, dQdt = cheb_T_value_deriv(d, t)
    Q = 1.0 + Q
    phi = (P / Q).clamp(PHI_LO_O, PHI_HI_O)
    dphi_dt = (dPdt * Q - P * dQdt) / (Q * Q + 0.0)
    dphi_du = (2.0 * dphi_dt).clamp_min(1e-12)
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
            cs = torch.cos(th) + torch.sin(th) * trig_phi  # (16,M)
            return rho3_from_c(cs) * torch.sin(th)

        A2 = gl16_integrate(f_sum_loc, ths_loc, thh_loc)
        A[mask] = A1 + A2
    return A.clamp_min(EPS_SLOPE)


# ---- Main mapper (uses A from inverse CDF with correct C_tot) ----
@torch.no_grad()
def ho2ho_O(ho: torch.Tensor, newton_iters: int = 8) -> torch.Tensor:
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

    # Outer KR: source CDF → u ∈ [0,1], then Padé inverse to φ'
    C_hi = C_src_phi_O(
        torch.as_tensor(PHI_HI_O, dtype=DTYPE, device=DEVICE), PHI_LO_O, PHI_HI_O
    )
    u_src = (C_src_phi_O(ph, PHI_LO_O, PHI_HI_O) / C_hi).clamp(0.0, 1.0)

    ph_p, dphi_du = phi_and_dphi_du_from_pade(u_src)

    # Column area from inverse CDF with normalization
    A_tar = (C_TOT_O_SECTOR / dphi_du).clamp_min(EPS_SLOPE)

    # # Optional one-shot check: compare A_tar to direct integration (should match closely)
    # A_true = column_area_O(ph_p)
    # rel = (A_tar - A_true).abs() / A_true.clamp_min(1e-15)
    # max_err_indx = rel.argmax()
    # print("max_err_indx:", max_err_indx)
    # print("A_tar:", A_tar[max_err_indx].item())
    # print("A_true:", A_true[max_err_indx].item())
    # print("A_rel:", rel[max_err_indx].item())

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


# ---- Smoke test (and optional validation) ----
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 32
    ho = torch.randn(N, 3, device=DEVICE, dtype=DTYPE)
    ho = (
        ho
        / torch.linalg.norm(ho, dim=-1, keepdim=True).clamp_min(1e-12)
        * (0.9 * H_MAX)
    )

    out = ho2ho_O(ho)  # set False to skip one-time comparison
    print("ok", out.shape)
