#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padé Approximation for the I (Icosahedral, 532) Azimuthal Marginal — φ'(u) on [0, π/5]
======================================================================================

This script fits a rational Padé approximation for ψ'(u), the inverse CDF of the
azimuthal angle ψ over the tiny spherical sector that forms 1/120 of the sphere,
corresponding to the canonical icosahedral RFZ wedge [0, π/5] (36°).

It mirrors the tetrahedral fitter:
- Cosine-clustered grid for ψ,
- Ground-truth marginal A(ψ) from the analytic KR cap integral,
- CDF via trapezoidal accumulate, normalized,
- Chebyshev-in-t Padé fit with Levenberg–Marquardt and a pole-guard on Q(t),
- Dense evaluation and reporting, plus JSON serialization of coefficients.

Broadcasting is fully torch-vectorized; default dtype is float64; CUDA used if present.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch

# ----------------------- Numeric defaults & device -----------------------
torch.set_default_dtype(torch.float64)
PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SUP_ERROR = 5e-15
N_PSI_GRID = 10_000_001  # ground-truth ψ-grid for CDF construction (cosine-clustered)
N_TRAIN = 8195  # training u-samples (cosine-clustered in t)
N_EVAL = 20001  # dense evaluation grid for reporting
PSI_MIN = 0.0
PSI_MAX = PI / 5.0  # 36° wedge for the canonical RFZ azimuth sector
EPS_SMALL = 1e-15

# Degree search ladder (high→low like your T script tail)
DEGREE_LADDER = [
    (12, 12),
    (12, 11),
    (12, 10),
    (12, 9),
    (12, 8),
    (12, 7),
    (12, 6),
    (12, 5),
    (12, 4),
    (12, 3),
    (12, 2),
    (12, 1),
    (12, 0),
    (13, 13),
    (13, 12),
    (13, 11),
    (13, 10),
    (13, 9),
    (13, 8),
    (13, 7),
    (13, 6),
    (13, 5),
    (13, 4),
    (13, 3),
    (13, 2),
    (13, 1),
    (13, 0),
    (14, 14),
    (14, 13),
    (14, 12),
    (14, 11),
    (14, 10),
    (14, 9),
    (14, 8),
    (14, 7),
    (14, 6),
    (14, 5),
    (14, 4),
    (14, 3),
    (14, 2),
    (14, 1),
    (14, 0),
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
    (16, 16),
    (16, 15),
    (16, 14),
    (16, 13),
    (16, 12),
    (16, 11),
    (16, 10),
    (16, 9),
    (16, 8),
    (16, 7),
    (16, 6),
    (16, 5),
    (16, 4),
    (16, 3),
    (16, 2),
    (16, 1),
    (16, 0),
    (17, 17),
    (17, 16),
    (17, 15),
    (17, 14),
    (17, 13),
    (17, 12),
    (17, 11),
    (17, 10),
    (17, 9),
    (17, 8),
    (17, 7),
    (17, 6),
    (17, 5),
    (17, 4),
    (17, 3),
    (17, 2),
    (17, 1),
    (17, 0),
]

# ----------------------- Icosahedral KR pieces (analytic) -----------------------
SQRT5 = math.sqrt(5.0)
PHI_G = 0.5 * (1.0 + SQRT5)
# KR constants (match your mapping code)
ALPHA = math.sqrt(5.0 - 2.0 * SQRT5)  # tan(18°)
P_CONST = math.sqrt(5.0 + 2.0 * SQRT5)  # cot(18°)
COS_BETA = 1.0 / math.sqrt(5.0)  # adjacent 5-fold axis angle
SIN_BETA = 2.0 / math.sqrt(5.0)


@torch.no_grad()
def theta_max(psi: torch.Tensor) -> torch.Tensor:
    """Ceiling polar angle as a function of azimuth ψ (radians)."""
    num = 1.0 - COS_BETA
    den = SIN_BETA * torch.clamp(torch.cos(psi), min=EPS_SMALL)
    return torch.atan(num / den)


@torch.no_grad()
def R3_of_theta(theta: torch.Tensor) -> torch.Tensor:
    """
    R^3(θ) radial primitive for constant-Jacobian KR (Icosahedral).
    c = cos θ (clamped), P = P_CONST.
    """
    c = torch.clamp(torch.cos(theta), -1.0 + 1e-15, 1.0 - 1e-15)
    P = P_CONST
    term1 = torch.atan(1.0 / (P * c))
    term2 = (P * c) / (1.0 + (P * c) ** 2)
    return 1.5 * (term1 - term2)


@torch.no_grad()
def F_cap(theta: torch.Tensor) -> torch.Tensor:
    """
    Polar cap integral (analytic) used as the θ-marginal primitive.
    """
    c = torch.clamp(torch.cos(theta), -1.0 + 1e-15, 1.0 - 1e-15)
    P = P_CONST
    return 1.5 * (0.5 * PI * (1.0 - c) - math.atan(P) + c * torch.atan(P * c))


@torch.no_grad()
def A_psi_I(psi: torch.Tensor) -> torch.Tensor:
    """
    A(ψ) = ∫_0^{θ_max(ψ)} ρ^3(...) sinθ dθ = F_cap(θ_max(ψ)).
    Vectorized over ψ.
    """
    return F_cap(theta_max(psi)).clamp_min(1e-30)


# ----------------------- Build ground-truth ψ'(u) -----------------------
@torch.no_grad()
def build_inverse_cdf_I(n_psi: int = N_PSI_GRID) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct (ψ_grid, C_grid) with cosine clustering on ψ, then trapezoidal CDF.
    """
    # cosine-clustered parameterization t ∈ [0,1] → ψ ∈ [0, PSI_MAX]
    t = torch.linspace(0, 1, n_psi, device=DEVICE)
    psi_grid = 0.5 * (1.0 - torch.cos(PI * t)) * (PSI_MAX - PSI_MIN) + PSI_MIN  # (N,)

    A = A_psi_I(psi_grid)  # (N,)
    dpsi = psi_grid[1:] - psi_grid[:-1]  # (N-1,)
    A_avg = 0.5 * (A[1:] + A[:-1])  # (N-1,)

    C = torch.empty_like(psi_grid)
    C[0] = 0.0
    C[1:] = (A_avg * dpsi).cumsum(0)
    C /= C[-1].clamp_min(1e-300)  # normalize to 1
    return psi_grid, C


@torch.no_grad()
def invert_cdf_lookup(
    u: torch.Tensor, psi_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """
    Cheap monotone linear interpolation of ψ given u ∈ [0,1].
    """
    u = u.clamp(0.0, 1.0)
    idx = torch.searchsorted(C_grid, u, right=True) - 1
    idx = idx.clamp(0, C_grid.numel() - 2)
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    p0, p1 = psi_grid[idx], psi_grid[idx + 1]
    t = (u - c0) / (c1 - c0 + 1e-300)
    return (1.0 - t) * p0 + t * p1


# ----------------------- Chebyshev basis & Padé evaluation -----------------------
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
    """
    ψ'(u) ≈ P(t)/Q(t), with t = 2u - 1, Q(t) = 1 + Σ b_k T_k(t), k≥1.
    """
    t = u_to_t(u)
    P = chebyshev_clenshaw(num, t)
    if den.numel() == 1:
        Q = torch.ones_like(P)
    else:
        d = den.clone()
        d[0] = 0.0
        Q = 1.0 + chebyshev_clenshaw(d, t)
    return P / Q


# ----------------------- LM fit with pole guard -----------------------
@torch.no_grad()
def fit_pade(
    u_train: torch.Tensor,
    psi_train: torch.Tensor,
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
):
    t_train = u_to_t(u_train.reshape(-1))
    y = psi_train.reshape(-1)

    Tn = chebyshev_vandermonde(t_train, n_num)  # (N, n_num+1)
    Td = (
        chebyshev_vandermonde(t_train, n_den)[:, 1:]
        if n_den > 0
        else torch.zeros((t_train.numel(), 0), device=DEVICE)
    )

    # Guard grid in t for Q(t) ≈ 1 + Td_guard @ b
    t_guard = torch.cos(
        PI * (torch.arange(guard_points, device=DEVICE) + 0.5) / guard_points
    )
    Td_guard = (
        chebyshev_vandermonde(t_guard, n_den)[:, 1:]
        if n_den > 0
        else torch.zeros((t_guard.numel(), 0), device=DEVICE)
    )

    a = torch.zeros(n_num + 1, device=DEVICE)
    b = torch.zeros(n_den, device=DEVICE)
    if num_init is not None:
        a[: min(a.numel(), num_init.numel())] = num_init[
            : min(a.numel(), num_init.numel())
        ]
    if den_init is not None and n_den > 0:
        b[: min(b.numel(), den_init.numel())] = den_init[
            : min(b.numel(), den_init.numel())
        ]

    lam = lambda_init
    best_cost = float("inf")
    best = (a.clone(), b.clone())

    for _ in range(max_iterations):
        P = Tn @ a
        # Q = 1.0 + (Td @ b if n_den > 0 else 0.0)
        if n_den > 0:
            Q = 1.0 + Td @ b
        else:
            Q = torch.ones_like(t_train)
        r = P / Q - y

        if n_den > 0:
            Qg = 1.0 + Td_guard @ b
            gv = torch.clamp(q_floor - Qg.abs(), min=0.0)
            rg = math.sqrt(lambda_guard) * gv
        else:
            Qg = torch.ones_like(t_guard)
            gv = torch.zeros_like(t_guard)
            rg = torch.zeros_like(t_guard)

        cost = (r @ r) + (rg @ rg)
        if cost.item() < best_cost:
            best_cost = cost.item()
            best = (a.clone(), b.clone())

        Ja = Tn / Q.unsqueeze(-1)
        if n_den > 0:
            Jb = -(P / (Q * Q)).unsqueeze(-1) * Td
            active = gv > 0
            if active.any():
                sign = torch.sign(Qg[active]).unsqueeze(-1)
                Jg_b = -math.sqrt(lambda_guard) * sign * Td_guard[active]
                Jg = torch.cat(
                    [torch.zeros((Jg_b.shape[0], n_num + 1), device=DEVICE), Jg_b],
                    dim=1,
                )
                J = torch.cat([torch.cat([Ja, Jb], dim=1), Jg], dim=0)
                rr = torch.cat([r, rg[active]], dim=0)
            else:
                J = torch.cat([Ja, Jb], dim=1)
                rr = r
        else:
            J = Ja
            rr = r

        H = J.T @ J
        g = J.T @ rr
        H = H + lam * torch.eye(H.shape[0], device=DEVICE)
        try:
            delta = torch.linalg.solve(H, -g)
        except RuntimeError:
            delta = torch.linalg.lstsq(H, -g).solution

        a_new = a + delta[: n_num + 1]
        b_new = b + delta[n_num + 1 :] if n_den > 0 else b

        Pn = Tn @ a_new
        Qn = 1.0 + (Td @ b_new if n_den > 0 else 0.0)
        rn = Pn / Qn - y

        if n_den > 0:
            Qgn = 1.0 + Td_guard @ b_new
            gvn = torch.clamp(q_floor - Qgn.abs(), min=0.0)
            rgn = math.sqrt(lambda_guard) * gvn
            cost_new = (rn @ rn) + (rgn @ rgn)
        else:
            cost_new = rn @ rn

        if cost_new < cost:
            a, b = a_new, b_new
            lam = max(lam / 3.0, 1e-12)
        else:
            lam = min(lam * 3.0, lambda_max)

    a_best, b_best = best
    den = torch.empty(n_den + 1, device=DEVICE)
    den[0] = 1.0
    if n_den > 0:
        den[1:] = b_best
    return a_best, den


# ----------------------- Packaging & degree search -----------------------
@dataclass
class PadeCoefficients:
    degree_num: int
    degree_den: int
    numerator: torch.Tensor
    denominator: torch.Tensor
    psi_max: float
    sup_error: float
    rms_error: float
    min_Q: float


@torch.no_grad()
def search_optimal_degree(
    psi_grid, C_grid, degree_ladder, target_error
) -> PadeCoefficients:
    psi_max = float(psi_grid[-1].item())
    # clustered training in u via cosine in t
    u_train = 0.5 * (1.0 - torch.cos(PI * torch.linspace(0, 1, N_TRAIN, device=DEVICE)))
    psi_train = invert_cdf_lookup(u_train, psi_grid, C_grid)

    # dense eval
    u_eval = torch.linspace(0.0, 1.0, N_EVAL, device=DEVICE)
    psi_true = invert_cdf_lookup(u_eval, psi_grid, C_grid)

    prev_num = None
    prev_den = None
    best = None

    print(f"\n{'─'*80}")
    print(f"  I (532): ψ'(u) approximation  |  ψ'_max = {math.degrees(psi_max):.6f}°")
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
            den_init = torch.zeros(max(n_den, 0), device=DEVICE)
            if prev_den is not None:
                k_copy = min(prev_den.numel() - 1, n_den)
                if k_copy > 0:
                    den_init[:k_copy] = prev_den[1 : 1 + k_copy]
        else:
            num_init = None
            den_init = None

        num, den = fit_pade(
            u_train, psi_train, n_num, n_den, num_init=num_init, den_init=den_init
        )

        psi_apx = evaluate_pade(u_eval, num, den).clamp(0.0, psi_max)
        err = (psi_apx - psi_true).abs()
        sup_err = float(err.max().item())
        rms_err = float(err.square().mean().sqrt().item())

        # Check Q(t) on a dense t-grid
        t_check = torch.cos(PI * (torch.arange(4096, device=DEVICE) + 0.5) / 4096)
        d = den.clone()
        d[0] = 0.0
        Qvals = (
            1.0 + chebyshev_clenshaw(d, t_check)
            if den.numel() > 1
            else torch.ones_like(t_check)
        )
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
                    psi_max=psi_max,
                    sup_error=sup_err,
                    rms_error=rms_err,
                    min_Q=min_Q,
                )

        prev_num, prev_den = num, den

    print(f"{'─'*80}\n")
    if best is None:
        raise ValueError(
            f"No degree pair achieved target error {target_error:.2e} for I (532)."
        )
    return best


# ----------------------- Main -----------------------
@torch.no_grad()
def main():
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  PADÉ APPROXIMATION FOR I (ICOSAHEDRAL, 532) AZIMUTH ψ'(u)")
    print("=" * 80)
    print(f"  Target Supremum Error: {TARGET_SUP_ERROR:.2e}")
    print(f"  Training Points:       {N_TRAIN:,}")
    print(f"  Evaluation Points:     {N_EVAL:,}")
    print(f"  Ground Truth ψ-Grid:   {N_PSI_GRID:,}")
    print("=" * 80)

    # Build ground truth inverse CDF via dense ψ-grid
    psi_grid, C_grid = build_inverse_cdf_I(N_PSI_GRID)
    coeffs = search_optimal_degree(psi_grid, C_grid, DEGREE_LADDER, TARGET_SUP_ERROR)

    print(f"  SELECTED for I (532):")
    print(
        f"    Degrees:     (n={coeffs.degree_num}, m={coeffs.degree_den})  [total = {coeffs.degree_num + coeffs.degree_den}]"
    )
    print(f"    Sup Error:   {coeffs.sup_error:.3e}")
    print(f"    RMS Error:   {coeffs.rms_error:.3e}")
    print(f"    Min |Q|:     {coeffs.min_Q:.3e}")
    print(f"    ψ'_max:      {math.degrees(coeffs.psi_max):.6f}°\n")

    print("\n" + "=" * 80)
    print("  FINAL COEFFICIENTS (I, 532)")
    print("=" * 80)
    print(f"\n{'─'*80}")
    print(
        f"  I  |  Degrees (n={coeffs.degree_num}, m={coeffs.degree_den})  |  Sup Error = {coeffs.sup_error:.3e}"
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

    out = {
        "I_532": {
            "degree_num": coeffs.degree_num,
            "degree_den": coeffs.degree_den,
            "psi_max": coeffs.psi_max,
            "sup_error": coeffs.sup_error,
            "rms_error": coeffs.rms_error,
            "min_Q": coeffs.min_Q,
            "numerator": coeffs.numerator.numpy().tolist(),
            "denominator": coeffs.denominator.numpy().tolist(),
        }
    }
    out_file = str(Path(__file__).resolve().parent / "coeffs_azim_I_532.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 80)
    print(f"  Coefficients saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
