#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piecewise Padé Approximation for the O (Octahedral) Azimuthal Marginal φ(u)
==========================================================================

Split the inverse-CDF fit at the geometric regime boundary φ* = 3π/8.
Let u* = C(φ*). Fit two independent Padés:
  - Left:  u ∈ [0, u*]
  - Right: u ∈ [u*, 1]
Each piece uses its own affine map to t ∈ [-1, 1].

Pipeline
--------
1) Build A(φ) with 16-pt Gauss–Legendre in θ (piecewise per φ).
2) Integrate A(φ) over φ ∈ [π/4, π/2] → C(φ); normalize to [0,1].
3) Compute u* = C(3π/8).
4) Invert numerically to get ground-truth φ(u).
5) Fit Padé P(t)/Q(t) per piece via Levenberg–Marquardt with pole guard.
6) Choose the lowest (n,m) hitting target sup error per piece and save to JSON.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Tuple, List

import torch

# ============================== CONFIG ==============================
PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

TARGET_SUP_ERROR = 5e-14

N_PHI_GRID = 20_000_001  # φ-grid for ground-truth CDF (Chebyshev-like)
N_TRAIN = 8195  # Chebyshev-like u grid for fitting (per piece)
N_EVAL = 20001  # Uniform u grid for evaluation (global)

PHI_LO_O = 0.25 * PI
PHI_HI_O = 0.50 * PI

EPS_C = 1e-15
EPS_SLOPE = 1e-15

# DEGREE_LADDER = [
#     (5, 5),
# ]
# enumerate with list comprehension
DEGREE_LADDER = [(n, m) for n in range(5, 16) for m in range(5, 16)]

# ========================= GL16 QUADRATURE ==========================
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
_KAPPA = math.sqrt(2.0) - 1.0


@torch.no_grad()
def theta_max_O(phi: torch.Tensor) -> torch.Tensor:
    s = torch.sin(phi).clamp_min(EPS_C)
    return torch.atan(1.0 / s)


@torch.no_grad()
def theta_switch_O(phi: torch.Tensor) -> torch.Tensor:
    denom = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS_C)
    return torch.atan(torch.sqrt(torch.tensor(2.0, device=DEVICE)) / denom)


# =================== TARGET COLUMN AREA A(φ) =======================
@torch.no_grad()
def A_phi_O(phi: torch.Tensor) -> torch.Tensor:
    ths = theta_switch_O(phi)  # switch
    thh = theta_max_O(phi)  # ceiling

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


# ========== GROUND-TRUTH CDF: BUILD + LOOKUPS =======================
@torch.no_grad()
def build_inverse_cdf_O(n_phi: int = N_PHI_GRID) -> Tuple[torch.Tensor, torch.Tensor]:
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
    u = u.clamp(0.0, 1.0)
    idx = torch.searchsorted(C_grid, u, right=True) - 1
    idx = idx.clamp(0, C_grid.numel() - 2)
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    p0, p1 = phi_grid[idx], phi_grid[idx + 1]
    t = (u - c0) / (c1 - c0 + 1e-300)
    return (1 - t) * p0 + t * p1


@torch.no_grad()
def cdf_lookup_phi(
    phi: torch.Tensor, phi_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    phi = phi.clamp(phi_grid[0], phi_grid[-1])
    idx = torch.searchsorted(phi_grid, phi, right=True) - 1
    idx = idx.clamp(0, phi_grid.numel() - 2)
    p0, p1 = phi_grid[idx], phi_grid[idx + 1]
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    t = (phi - p0) / (p1 - p0 + 1e-300)
    return (1 - t) * c0 + t * c1


# ===================== CHEBYSHEV / PADÉ ============================
@torch.no_grad()
def u_to_t_piece(u: torch.Tensor, u0: float, u1: float) -> torch.Tensor:
    return 2.0 * (u - u0) / (u1 - u0) - 1.0


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
def evaluate_pade_piece(
    u: torch.Tensor, num: torch.Tensor, den: torch.Tensor, u0: float, u1: float
) -> torch.Tensor:
    t = u_to_t_piece(u, u0, u1)
    P = chebyshev_clenshaw(num, t)
    if den.numel() == 1:
        Q = torch.ones_like(P)
    else:
        d = den.clone()
        d[0] = 0.0
        Q = 1.0 + chebyshev_clenshaw(d, t)
    return P / Q


# ========== LM FIT WITH POLE GUARD (per piece) ======================
@torch.no_grad()
def fit_pade_piece(
    u_train: torch.Tensor,
    phi_train: torch.Tensor,
    u0: float,
    u1: float,
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
    t_train = u_to_t_piece(u_train.reshape(-1), u0, u1)
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


# ======================= DEGREE SEARCH (piece) ======================
@dataclass
class PadeCoefficientsPiece:
    degree_num: int
    degree_den: int
    numerator: torch.Tensor
    denominator: torch.Tensor
    u0: float
    u1: float
    sup_error: float
    rms_error: float
    min_Q: float


@torch.no_grad()
def search_piece(
    piece_name: str,
    u0: float,
    u1: float,
    phi_grid: torch.Tensor,
    C_grid: torch.Tensor,
    degree_ladder: List[Tuple[int, int]],
    target_error: float,
    u_break: float,
    phi_break: float,
) -> PadeCoefficientsPiece:
    # training grid in [u0,u1]
    s = torch.linspace(0.0, 1.0, N_TRAIN, device=DEVICE)
    u_train = 0.5 * (1.0 - torch.cos(PI * s)) * (u1 - u0) + u0

    # include exact breakpoint sample on both sides
    if piece_name == "left":
        u_train[-1] = u_break
    else:
        u_train[0] = u_break

    phi_train = invert_cdf_lookup(u_train, phi_grid, C_grid)
    if piece_name == "left":
        phi_train[-1] = phi_break
    else:
        phi_train[0] = phi_break

    # evaluation grid in [u0,u1]
    u_eval = torch.linspace(u0, u1, N_EVAL, device=DEVICE)
    phi_true = invert_cdf_lookup(u_eval, phi_grid, C_grid)

    prev_num = None
    prev_den = None
    best: PadeCoefficientsPiece | None = None

    print(f"\n{'─'*80}")
    print(f"  PIECE: {piece_name.upper()}   u ∈ [{u0:.16e}, {u1:.16e}]")
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

        num, den = fit_pade_piece(
            u_train=u_train,
            phi_train=phi_train,
            u0=u0,
            u1=u1,
            n_num=n_num,
            n_den=n_den,
            num_init=num_init,
            den_init=den_init,
        )

        phi_apx = evaluate_pade_piece(u_eval, num, den, u0=u0, u1=u1)
        err = (phi_apx - phi_true).abs()
        sup_err = float(err.max().item())
        rms_err = float(err.square().mean().sqrt().item())

        # Denominator health on t ∈ [-1,1]
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
                best = PadeCoefficientsPiece(
                    degree_num=n_num,
                    degree_den=n_den,
                    numerator=num.detach().cpu(),
                    denominator=den.detach().cpu(),
                    u0=float(u0),
                    u1=float(u1),
                    sup_error=sup_err,
                    rms_error=rms_err,
                    min_Q=min_Q,
                )

        prev_num, prev_den = num, den

    print(f"{'─'*80}\n")
    if best is None:
        raise ValueError(
            f"No degree pair achieved target error {target_error:.2e} for piece '{piece_name}'."
        )
    return best


# ============================ MAIN =================================
@torch.no_grad()
def main():
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  PIECEWISE PADÉ APPROXIMATION FOR O (OCTAHEDRAL) AZIMUTH φ(u)")
    print("=" * 80)
    print(f"  Target Supremum Error (per piece): {TARGET_SUP_ERROR:.2e}")
    print(f"  Training Points (per piece):       {N_TRAIN:,}")
    print(f"  Eval Points (per piece/global):    {N_EVAL:,} / {N_EVAL:,}")
    print(f"  Ground Truth φ-Grid:               {N_PHI_GRID:,}")
    print("=" * 80)

    phi_grid, C_grid = build_inverse_cdf_O(N_PHI_GRID)

    # geometric boundary in φ, and its u-breakpoint
    phi_star = torch.tensor(3.0 * PI / 8.0, device=DEVICE)  # 3π/8
    u_star = float(cdf_lookup_phi(phi_star, phi_grid, C_grid).item())

    # exact breakpoint values for training seam
    phi_break = float(phi_star.item())
    u_break = float(u_star)

    print(f"\n  Breakpoint:")
    print(f"    phi* = 3π/8 = {phi_break:.16e} rad = {math.degrees(phi_break):.12f}°")
    print(f"    u*   = C(phi*) = {u_break:.16e}\n")

    # Fit left and right pieces
    left = search_piece(
        piece_name="left",
        u0=0.0,
        u1=u_break,
        phi_grid=phi_grid,
        C_grid=C_grid,
        degree_ladder=DEGREE_LADDER,
        target_error=TARGET_SUP_ERROR,
        u_break=u_break,
        phi_break=phi_break,
    )

    right = search_piece(
        piece_name="right",
        u0=u_break,
        u1=1.0,
        phi_grid=phi_grid,
        C_grid=C_grid,
        degree_ladder=DEGREE_LADDER,
        target_error=TARGET_SUP_ERROR,
        u_break=u_break,
        phi_break=phi_break,
    )

    # Global validation (piecewise)
    u_eval = torch.linspace(0.0, 1.0, N_EVAL, device=DEVICE)
    phi_true = invert_cdf_lookup(u_eval, phi_grid, C_grid)

    maskL = u_eval <= u_break
    maskR = ~maskL
    phi_apx = torch.empty_like(phi_true)

    numL = left.numerator.to(DEVICE)
    denL = left.denominator.to(DEVICE)
    numR = right.numerator.to(DEVICE)
    denR = right.denominator.to(DEVICE)

    if maskL.any():
        phi_apx[maskL] = evaluate_pade_piece(
            u_eval[maskL], numL, denL, u0=0.0, u1=u_break
        )
    if maskR.any():
        phi_apx[maskR] = evaluate_pade_piece(
            u_eval[maskR], numR, denR, u0=u_break, u1=1.0
        )

    err = (phi_apx - phi_true).abs()
    sup_err = float(err.max().item())
    rms_err = float(err.square().mean().sqrt().item())

    print("\n" + "=" * 80)
    print("  SELECTED (PIECEWISE) SUMMARY")
    print("=" * 80)
    print(
        f"  LEFT : (n={left.degree_num}, m={left.degree_den}), sup={left.sup_error:.3e}, rms={left.rms_error:.3e}, min|Q|={left.min_Q:.3e}"
    )
    print(
        f"  RIGHT: (n={right.degree_num}, m={right.degree_den}), sup={right.sup_error:.3e}, rms={right.rms_error:.3e}, min|Q|={right.min_Q:.3e}"
    )
    print(f"  GLOBAL: sup={sup_err:.3e}, rms={rms_err:.3e}")
    print("=" * 80)

    # Save JSON
    os.makedirs("data", exist_ok=True)
    out = {
        "O": {
            "phi_min": PHI_LO_O,
            "phi_max": float(phi_grid[-1].item()),
            "phi_break": phi_break,
            "u_break": u_break,
            "left": {
                "u0": 0.0,
                "u1": u_break,
                "degree_num": left.degree_num,
                "degree_den": left.degree_den,
                "sup_error": left.sup_error,
                "rms_error": left.rms_error,
                "min_Q": left.min_Q,
                "numerator": left.numerator.numpy().tolist(),
                "denominator": left.denominator.numpy().tolist(),
            },
            "right": {
                "u0": u_break,
                "u1": 1.0,
                "degree_num": right.degree_num,
                "degree_den": right.degree_den,
                "sup_error": right.sup_error,
                "rms_error": right.rms_error,
                "min_Q": right.min_Q,
                "numerator": right.numerator.numpy().tolist(),
                "denominator": right.denominator.numpy().tolist(),
            },
            "global_sup_error": sup_err,
            "global_rms_error": rms_err,
        }
    }

    out_file = str(Path(__file__).resolve().parent / "coeffs_azim_O_piecewise.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Coefficients saved to: {out_file}\n")


if __name__ == "__main__":
    main()
