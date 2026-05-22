#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padé Approximation for the O (Octahedral) Azimuthal Marginal φ'(u)
==================================================================

- Domain: φ ∈ [π/4, π/2] on the ordered-simplex sector (x ≤ y ≤ z) in the first octant.
- Column measure A(φ) switches from a polar-cap (c_max) piece to a sum-face (c_sum) piece
  at θ_s(φ); the ceiling θ_max(φ) is φ-dependent.

Pipeline
--------
1) Build A(φ) with 16-pt Gauss–Legendre in θ (piecewise per φ).
2) Integrate A(φ) over φ ∈ [φ_lo, φ_hi] (φ_lo=π/4, φ_hi=π/2) → C(φ); normalize to [0,1].
3) Invert numerically to get ground-truth φ'(u).
4) Fit Padé P(t)/Q(t), t = 2u-1, via Levenberg–Marquardt with soft pole guard.
5) Choose the lowest (n,m) hitting target sup error and save to JSON.

Defaults are tuned for publication-precision; reduce grids if you’re prototyping.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch

# ============================== CONFIG ==============================
PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# Publication defaults (feel free to relax while iterating)
TARGET_SUP_ERROR = 5e-8
N_PHI_GRID = 20_000_001  # φ-grid for ground-truth CDF (Chebyshev-like)
N_TRAIN = 8195  # Chebyshev-like u grid for fitting
N_EVAL = 20001  # Uniform u grid for evaluation

# O azimuth domain on the ordered simplex
PHI_LO_O = 0.25 * PI
PHI_HI_O = 0.50 * PI

# Guards
EPS_C = 1e-15
EPS_SLOPE = 1e-15

# Degree ladder (continuation)
DEGREE_LADDER = [
    (5, 5),
    (5, 4),
    (5, 3),
    (5, 2),
    (5, 1),
    (5, 0),
    (10, 10),
    (10, 9),
    (10, 8),
    (10, 7),
    (10, 6),
    (10, 5),
    (10, 4),
    (10, 3),
    (10, 2),
    (10, 1),
    (10, 0),
    (15, 15),
    (15, 14),
    (15, 13),
    (15, 12),
    (15, 11),
    (15, 10),
    (15, 9),
    (15, 8),
    (15, 7),
    (15, 6),
    (15, 5),
    (15, 4),
    (15, 3),
    (15, 2),
    (15, 1),
    (15, 0),
]

# ========================= GL8 QUADRATURE ===========================
# 8-pt Gauss–Legendre nodes/weights on [-1, 1]
# ---------------- Gauss–Legendre (16-pt) ----------------
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
    """16-pt Gauss–Legendre integrate `func` over [a, b]. Broadcast-friendly."""
    a_t = torch.as_tensor(a, device=DEVICE)
    b_t = torch.as_tensor(b, device=DEVICE)
    a_b, b_b = torch.broadcast_tensors(a_t, b_t)

    half = 0.5 * (b_b - a_b)
    mid = 0.5 * (b_b + a_b)

    xa = _GL16_X.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0) + mid.unsqueeze(0)
    wa = _GL16_W.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0)

    val = func(xa)  # (16, ...)
    wa_b = wa.reshape(wa.shape + (1,) * (val.ndim - wa.ndim))
    return (val * wa_b).sum(dim=0)


# =================== HOMOCHORIC RADIAL (ρ^3) =======================
@torch.no_grad()
def rho3_from_c(c: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(c, min=EPS_C)
    return 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))


# =================== OCTAHEDRAL GEOMETRY (O) =======================
# Octahedral "max" face constant κ = √2 - 1
_KAPPA = math.sqrt(2.0) - 1.0


@torch.no_grad()
def theta_max_O(phi: torch.Tensor) -> torch.Tensor:
    """Ceiling in ordered-simplex (x ≤ y ≤ z): θ_max(φ) = atan(1 / sin φ)."""
    s = torch.sin(phi).clamp_min(EPS_C)
    return torch.atan(1.0 / s)


@torch.no_grad()
def theta_switch_O(phi: torch.Tensor) -> torch.Tensor:
    """Switch where c_max = c_sum ⇒ tan θ_s = √2 / (cos φ + sin φ)."""
    denom = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS_C)
    return torch.atan(torch.sqrt(torch.tensor(2.0, device=DEVICE)) / denom)


@torch.no_grad()
def c_O_components(
    theta: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (c_max, c_sum) at (θ, φ)."""
    c_max = (1.0 / _KAPPA) * torch.cos(theta)
    c_sum = torch.cos(theta) + torch.sin(theta) * (torch.cos(phi) + torch.sin(phi))
    return c_max, c_sum


# =================== TARGET COLUMN AREA A(φ) =======================
@torch.no_grad()
def A_phi_O(phi: torch.Tensor) -> torch.Tensor:
    """
    Target column area on the ordered-simplex sector:
      A(φ) = ∫_0^{min(θ_s, θ_max)} ρ^3(c_max) sinθ dθ
           + ∫_{θ_s}^{θ_max}       ρ^3(c_sum) sinθ dθ,  if θ_s < θ_max
           (otherwise, entire column is polar-cap).
    """
    ths = theta_switch_O(phi)  # switch
    thh = theta_max_O(phi)  # ceiling

    def f_pc(th):
        cmax = (1.0 / _KAPPA) * torch.cos(th)
        return rho3_from_c(cmax) * torch.sin(th)

    A = torch.empty_like(phi)

    # Case 1: none of the φ reaches sum-face
    only_pc = ths >= thh
    if only_pc.any():
        A[only_pc] = gl16_integrate(f_pc, 0.0, thh[only_pc])

    # Case 2: two-branch column
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


# ========== GROUND-TRUTH φ'(u): BUILD + INVERT ======================
@torch.no_grad()
def build_inverse_cdf_O(n_phi: int = N_PHI_GRID) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build φ-grid on [φ_lo, φ_hi] and its CDF grid C(φ) via trapezoidal integration of A(φ).
    Returns (phi_grid, C_grid) with C normalized to [0,1].
    """
    # Chebyshev-like spacing on [φ_lo, φ_hi]
    t = torch.linspace(0, 1, n_phi, device=DEVICE)
    phi_grid = 0.5 * (1 - torch.cos(PI * t)) * (PHI_HI_O - PHI_LO_O) + PHI_LO_O

    A = A_phi_O(phi_grid)

    dphi = phi_grid[1:] - phi_grid[:-1]
    A_avg = 0.5 * (A[1:] + A[:-1])

    C = torch.empty_like(phi_grid)
    C[0] = 0.0
    C[1:] = (A_avg * dphi).cumsum(0)
    C /= C[-1].clamp_min(1e-300)
    return phi_grid, C


@torch.no_grad()
def invert_cdf_lookup(
    u: torch.Tensor, phi_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """Binary-search + linear interpolation inversion: u ∈ [0,1] → φ."""
    u = u.clamp(0.0, 1.0)
    idx = torch.searchsorted(C_grid, u, right=True) - 1
    idx = idx.clamp(0, C_grid.numel() - 2)
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    p0, p1 = phi_grid[idx], phi_grid[idx + 1]
    t = (u - c0) / (c1 - c0 + 1e-300)
    return (1 - t) * p0 + t * p1


# ===================== CHEBYSHEV / PADÉ ============================
@torch.no_grad()
def u_to_t(u: torch.Tensor) -> torch.Tensor:
    return 2.0 * u - 1.0


@torch.no_grad()
def chebyshev_vandermonde(t: torch.Tensor, degree: int) -> torch.Tensor:
    T0 = torch.ones_like(t)
    if degree == 0:
        return T0.unsqueeze(-1)
    T1 = t
    cols = [T0, T1]
    for _ in range(1, degree):
        cols.append(2.0 * t * cols[-1] - cols[-2])
    return torch.stack(cols[: degree + 1], dim=-1)


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
def evaluate_pade(
    u: torch.Tensor, num: torch.Tensor, den: torch.Tensor
) -> torch.Tensor:
    t = u_to_t(u)
    P = chebyshev_clenshaw(num, t)
    if den.numel() == 1:
        Q = torch.ones_like(P)
    else:
        d = den.clone()
        d[0] = 0.0
        Q = 1.0 + chebyshev_clenshaw(d, t)
    return P / Q


# ========== LM FIT WITH POLE GUARD (as in D_k / T) ==================
@torch.no_grad()
def fit_pade(
    u_train: torch.Tensor,
    phi_train: torch.Tensor,
    n_num: int,
    n_den: int,
    num_init: torch.Tensor | None = None,
    den_init: torch.Tensor | None = None,
    max_iterations: int = 120,
    lambda_init: float = 1e-2,
    lambda_max: float = 1e6,
    guard_points: int = 2048,
    q_floor: float = 1e-3,
    lambda_guard: float = 5e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_train = u_to_t(u_train.reshape(-1))
    y = phi_train.reshape(-1)

    Tn = chebyshev_vandermonde(t_train, n_num)  # (N, n_num+1)
    Td = chebyshev_vandermonde(t_train, n_den)[:, 1:]  # (N, n_den)

    t_guard = torch.cos(
        PI * (torch.arange(guard_points, device=DEVICE) + 0.5) / guard_points
    )
    Td_guard = chebyshev_vandermonde(t_guard, n_den)[:, 1:]

    a = torch.zeros(n_num + 1, device=DEVICE)
    b = torch.zeros(n_den, device=DEVICE)
    if num_init is not None:
        a[: min(a.numel(), num_init.numel())] = num_init[
            : min(a.numel(), num_init.numel())
        ]
    if den_init is not None:
        b[: min(b.numel(), den_init.numel())] = den_init[
            : min(b.numel(), den_init.numel())
        ]

    lam = lambda_init
    best_cost = float("inf")
    best = (a.clone(), b.clone())

    for _ in range(max_iterations):
        P = Tn @ a
        Q = 1.0 + Td @ b
        r = P / Q - y

        Qg = 1.0 + Td_guard @ b
        gv = torch.clamp(q_floor - Qg.abs(), min=0.0)
        rg = math.sqrt(lambda_guard) * gv

        cost = (r @ r) + (rg @ rg)
        if cost.item() < best_cost:
            best_cost = cost.item()
            best = (a.clone(), b.clone())

        Ja = Tn / Q.unsqueeze(-1)
        Jb = -(P / (Q * Q)).unsqueeze(-1) * Td

        active = gv > 0
        if active.any():
            sign = torch.sign(Qg[active]).unsqueeze(-1)
            Jg_b = -math.sqrt(lambda_guard) * sign * Td_guard[active]
            Jg = torch.cat(
                [torch.zeros((Jg_b.shape[0], n_num + 1), device=DEVICE), Jg_b], dim=1
            )
            J = torch.cat([torch.cat([Ja, Jb], dim=1), Jg], dim=0)
            rr = torch.cat([r, rg[active]], dim=0)
        else:
            J = torch.cat([Ja, Jb], dim=1)
            rr = r

        H = J.T @ J
        g = J.T @ rr
        H = H + lam * torch.eye(H.shape[0], device=DEVICE)
        try:
            delta = torch.linalg.solve(H, -g)
        except RuntimeError:
            delta = torch.linalg.lstsq(H, -g).solution

        a_new = a + delta[: n_num + 1]
        b_new = b + delta[n_num + 1 :]

        Pn = Tn @ a_new
        Qn = 1.0 + Td @ b_new
        rn = Pn / Qn - y

        Qgn = 1.0 + Td_guard @ b_new
        gvn = torch.clamp(q_floor - Qgn.abs(), min=0.0)
        rgn = math.sqrt(lambda_guard) * gvn
        cost_new = (rn @ rn) + (rgn @ rgn)

        if cost_new < cost:
            a, b = a_new, b_new
            lam = max(lam / 3.0, 1e-12)
        else:
            lam = min(lam * 3.0, lambda_max)

    a_best, b_best = best
    den = torch.empty(n_den + 1, device=DEVICE)
    den[0] = 1.0
    den[1:] = b_best
    return a_best, den


# ======================= DEGREE SEARCH ==============================
@dataclass
class PadeCoefficients:
    degree_num: int
    degree_den: int
    numerator: torch.Tensor
    denominator: torch.Tensor
    phi_max: float
    sup_error: float
    rms_error: float
    min_Q: float


@torch.no_grad()
def search_optimal_degree(
    phi_grid: torch.Tensor,
    C_grid: torch.Tensor,
    degree_ladder: List[Tuple[int, int]],
    target_error: float,
) -> PadeCoefficients:
    phi_max = float(phi_grid[-1].item())

    # Training: Chebyshev-like u; Evaluation: uniform u
    u_train = 0.5 * (1.0 - torch.cos(PI * torch.linspace(0, 1, N_TRAIN, device=DEVICE)))
    phi_train = invert_cdf_lookup(u_train, phi_grid, C_grid)

    u_eval = torch.linspace(0.0, 1.0, N_EVAL, device=DEVICE)
    phi_true = invert_cdf_lookup(u_eval, phi_grid, C_grid)

    prev_num = None
    prev_den = None
    best: PadeCoefficients | None = None

    print(f"\n{'─'*80}")
    print(f"  O group: φ'(u) approximation  |  φ'_max = {math.degrees(phi_max):.6f}°")
    print(f"{'─'*80}")
    print(
        f"  {'Degree':>12}  {'Total':>6}  {'Sup Error':>12}  {'RMS Error':>12}  {'Min |Q|':>10}"
    )
    print(f"  {'(n, m)':>12}  {'n+m':>6}  {'':>12}  {'':>12}  {'':>10}")
    print(f"{'─'*80}")

    for n_num, n_den in degree_ladder:
        if prev_num is not None:
            num_init = torch.zeros(n_num + 1, device=DEVICE)
            num_init[: min(num_init.numel(), prev_num.numel())] = prev_num[
                : min(num_init.numel(), prev_num.numel())
            ]
            den_init = torch.zeros(n_den, device=DEVICE)
            k_copy = min(prev_den.numel() - 1, n_den)
            if k_copy > 0:
                den_init[:k_copy] = prev_den[1 : 1 + k_copy]
        else:
            num_init = None
            den_init = None

        num, den = fit_pade(
            u_train, phi_train, n_num, n_den, num_init=num_init, den_init=den_init
        )

        phi_apx = evaluate_pade(u_eval, num, den).clamp(PHI_LO_O, phi_max)
        err = (phi_apx - phi_true).abs()
        sup_err = float(err.max().item())
        rms_err = float(err.square().mean().sqrt().item())

        # Denominator health
        t_check = torch.cos(PI * (torch.arange(4096, device=DEVICE) + 0.5) / 4096)
        d = den.clone()
        d[0] = 0.0
        Qvals = 1.0 + chebyshev_clenshaw(d, t_check)
        min_Q = float(Qvals.abs().min().item())

        status = "✓" if sup_err < target_error else " "
        total = n_num + n_den
        print(
            f"  {status} ({n_num:2d}, {n_den:2d})  {total:6d}  {sup_err:12.3e}  {rms_err:12.3e}  {min_Q:10.3e}"
        )

        if sup_err < target_error:
            if best is None or total < (best.degree_num + best.degree_den):
                best = PadeCoefficients(
                    degree_num=n_num,
                    degree_den=n_den,
                    numerator=num.detach().cpu(),
                    denominator=den.detach().cpu(),
                    phi_max=phi_max,
                    sup_error=sup_err,
                    rms_error=rms_err,
                    min_Q=min_Q,
                )

        prev_num, prev_den = num, den

    print(f"{'─'*80}\n")
    if best is None:
        raise ValueError(
            f"No degree pair achieved target error {target_error:.2e} for O."
        )
    return best


# ============================ MAIN =================================
@torch.no_grad()
def main():
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  PADÉ APPROXIMATION FOR O (OCTAHEDRAL) AZIMUTH φ'(u)")
    print("=" * 80)
    print(f"  Target Supremum Error: {TARGET_SUP_ERROR:.2e}")
    print(f"  Training Points:       {N_TRAIN:,}")
    print(f"  Evaluation Points:     {N_EVAL:,}")
    print(f"  Ground Truth φ-Grid:   {N_PHI_GRID:,}")
    print("=" * 80)

    # Ground-truth CDF on [φ_lo, φ_hi]
    phi_grid, C_grid = build_inverse_cdf_O(N_PHI_GRID)

    # Degree search
    coeffs = search_optimal_degree(phi_grid, C_grid, DEGREE_LADDER, TARGET_SUP_ERROR)

    # Summary
    print(f"  SELECTED for O:")
    print(
        f"    Degrees:     (n={coeffs.degree_num}, m={coeffs.degree_den})  [total = {coeffs.degree_num + coeffs.degree_den}]"
    )
    print(f"    Sup Error:   {coeffs.sup_error:.3e}")
    print(f"    RMS Error:   {coeffs.rms_error:.3e}")
    print(f"    Min |Q|:     {coeffs.min_Q:.3e}")
    print(f"    φ'_max:      {math.degrees(coeffs.phi_max):.6f}°\n")

    # Print final coefficients
    print("\n" + "=" * 80)
    print("  FINAL COEFFICIENTS (O)")
    print("=" * 80)
    print(f"\n{'─'*80}")
    print(
        f"  O  |  Degrees (n={coeffs.degree_num}, m={coeffs.degree_den})  |  Sup Error = {coeffs.sup_error:.3e}"
    )
    print(f"{'─'*80}")

    print(
        f"\n  Numerator P(t) — Chebyshev coefficients [T_0, T_1, ..., T_{coeffs.degree_num}]:"
    )
    for i, c in enumerate(coeffs.numerator.numpy()):
        print(f"    a[{i:2d}] = {c:+.16e}")

    print(f"\n  Denominator Q(t) = 1 + Σ b[k]·T_k(t) — Chebyshev coefficients:")
    den = coeffs.denominator.numpy()
    print(f"    b[0] = {den[0]:+.16e}  (fixed)")
    for i in range(1, len(den)):
        print(f"    b[{i:2d}] = {den[i]:+.16e}")

    # Save JSON
    out = {
        "O": {
            "degree_num": coeffs.degree_num,
            "degree_den": coeffs.degree_den,
            "phi_min": PHI_LO_O,
            "phi_max": coeffs.phi_max,
            "sup_error": coeffs.sup_error,
            "rms_error": coeffs.rms_error,
            "min_Q": coeffs.min_Q,
            "numerator": coeffs.numerator.numpy().tolist(),
            "denominator": coeffs.denominator.numpy().tolist(),
        }
    }
    out_file = str(Path(__file__).resolve().parent / "coeffs_azim_O.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 80)
    print(f"  Coefficients saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
