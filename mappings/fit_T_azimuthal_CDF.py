#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padé Approximation for the T (Tetrahedral) Azimuthal Marginal — FIXED BROADCASTING
=================================================================================

This script fits a rational Padé approximation for φ'(u) for the tetrahedral
(T) group. It fixes the broadcasting issue that occurs during Gauss–Legendre
integration when φ is a long vector and the quadrature variable carries a
leading dimension.

Key fixes vs previous version:
- In `gl16_integrate`, weights are reshaped after calling `func(xa)` so they
  broadcast to the returned `val` shape.
- In `A_phi_T`, the integrand `fth` explicitly unsqueezes `th` and `phi` to
  broadcast as (quadrature, φ-grid, ...), i.e. (16, N, ...).
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch

torch.set_default_dtype(torch.float64)
PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SUP_ERROR = 5e-15
N_PHI_GRID = 10_000_001
N_TRAIN = 8195
N_EVAL = 20001
PHI_MIN_T = 0.0
PHI_MAX_T = 0.25 * PI
EPS_C = 1e-15

DEGREE_LADDER = [
    # (2, 2), (2, 1), (2, 0),
    # (3, 3), (3, 2), (3, 1), (3, 0),
    # (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
    # (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0),
    # (6, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0),
    # (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (7, 0),
    # (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0),
    # (9, 9), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (9, 1), (9, 0),
    # (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1), (10, 0),
    # (11, 11), (11, 10), (11, 9), (11, 8), (11, 7), (11, 6), (11, 5), (11, 4), (11, 3), (11, 2), (11, 1), (11, 0),
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

    xa = _GL16_X.view(16, *([1] * a_b.ndim)) * half.unsqueeze(0) + mid.unsqueeze(0)
    wa = _GL16_W.view(16, *([1] * a_b.ndim)) * half.unsqueeze(0)

    val = func(xa)  # shape (16, ...)
    wa_b = wa.view(16, *([1] * (val.ndim - 1)))
    return (val * wa_b).sum(dim=0)


# ---------------- T marginal pieces ----------------
@torch.no_grad()
def rho3_from_c(c: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(c, min=EPS_C)
    return 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))


@torch.no_grad()
def c_T(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.cos(theta) + torch.sin(theta) * (torch.cos(phi) + torch.sin(phi))


@torch.no_grad()
def A_phi_T(phi: torch.Tensor) -> torch.Tensor:
    """A(φ) = ∫_0^{π/2} ρ^3(c_T(θ, φ)) sinθ dθ, vectorized over φ.
    Broadcasting fix: shape becomes (16, Nφ) inside integrand and then reduces.
    """

    def fth(th):
        # th: (16,) -> (16, 1) so it can broadcast with phi: (1, Nφ)
        th_e = th.unsqueeze(-1)
        phi_e = phi.unsqueeze(0)
        return rho3_from_c(c_T(th_e, phi_e)) * torch.sin(th_e)

    return gl16_integrate(fth, 0.0, 0.5 * PI).clamp_min(1e-18)


# ---------------- Build ground-truth φ'(u) ----------------
@torch.no_grad()
def build_inverse_cdf_T(n_phi: int = N_PHI_GRID):
    t = torch.linspace(0, 1, n_phi, device=DEVICE)
    phi_grid = 0.5 * (1 - torch.cos(PI * t)) * (PHI_MAX_T - PHI_MIN_T) + PHI_MIN_T

    A = A_phi_T(phi_grid)
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
    u = u.clamp(0.0, 1.0)
    idx = torch.searchsorted(C_grid, u, right=True) - 1
    idx = idx.clamp(0, C_grid.numel() - 2)
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    p0, p1 = phi_grid[idx], phi_grid[idx + 1]
    t = (u - c0) / (c1 - c0 + 1e-300)
    return (1 - t) * p0 + t * p1


# ---------------- Chebyshev & Padé ----------------
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


# ---------------- LM fit with pole guard ----------------
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
):
    t_train = u_to_t(u_train.reshape(-1))
    y = phi_train.reshape(-1)

    Tn = chebyshev_vandermonde(t_train, n_num)
    Td = chebyshev_vandermonde(t_train, n_den)[:, 1:]

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
    phi_grid, C_grid, degree_ladder, target_error
) -> PadeCoefficients:
    phi_max = float(phi_grid[-1].item())
    u_train = 0.5 * (1.0 - torch.cos(PI * torch.linspace(0, 1, N_TRAIN, device=DEVICE)))
    phi_train = invert_cdf_lookup(u_train, phi_grid, C_grid)

    u_eval = torch.linspace(0.0, 1.0, N_EVAL, device=DEVICE)
    phi_true = invert_cdf_lookup(u_eval, phi_grid, C_grid)

    prev_num = None
    prev_den = None
    best = None

    print(f"\n{'─'*80}")
    print(f"  T group: φ'(u) approximation  |  φ'_max = {math.degrees(phi_max):.6f}°")
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

        phi_apx = evaluate_pade(u_eval, num, den).clamp(0.0, phi_max)
        err = (phi_apx - phi_true).abs()
        sup_err = float(err.max().item())
        rms_err = float(err.square().mean().sqrt().item())

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
            f"No degree pair achieved target error {target_error:.2e} for T."
        )
    return best


@torch.no_grad()
def main():
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  PADÉ APPROXIMATION FOR T (TETRAHEDRAL) AZIMUTH φ'(u)")
    print("=" * 80)
    print(f"  Target Supremum Error: {TARGET_SUP_ERROR:.2e}")
    print(f"  Training Points:       {N_TRAIN:,}")
    print(f"  Evaluation Points:     {N_EVAL:,}")
    print(f"  Ground Truth φ-Grid:   {N_PHI_GRID:,}")
    print("=" * 80)

    phi_grid, C_grid = build_inverse_cdf_T(N_PHI_GRID)
    coeffs = search_optimal_degree(phi_grid, C_grid, DEGREE_LADDER, TARGET_SUP_ERROR)

    print(f"  SELECTED for T:")
    print(
        f"    Degrees:     (n={coeffs.degree_num}, m={coeffs.degree_den})  [total = {coeffs.degree_num + coeffs.degree_den}]"
    )
    print(f"    Sup Error:   {coeffs.sup_error:.3e}")
    print(f"    RMS Error:   {coeffs.rms_error:.3e}")
    print(f"    Min |Q|:     {coeffs.min_Q:.3e}")
    print(f"    φ'_max:      {math.degrees(coeffs.phi_max):.6f}°\n")

    print("\n" + "=" * 80)
    print("  FINAL COEFFICIENTS (T)")
    print("=" * 80)
    print(f"\n{'─'*80}")
    print(
        f"  T  |  Degrees (n={coeffs.degree_num}, m={coeffs.degree_den})  |  Sup Error = {coeffs.sup_error:.3e}"
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
        "T": {
            "degree_num": coeffs.degree_num,
            "degree_den": coeffs.degree_den,
            "phi_max": coeffs.phi_max,
            "sup_error": coeffs.sup_error,
            "rms_error": coeffs.rms_error,
            "min_Q": coeffs.min_Q,
            "numerator": coeffs.numerator.numpy().tolist(),
            "denominator": coeffs.denominator.numpy().tolist(),
        }
    }
    out_file = str(Path(__file__).resolve().parent / "coeffs_azim_T.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 80)
    print(f"  Coefficients saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
