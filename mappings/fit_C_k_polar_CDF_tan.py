# -*- coding: utf-8 -*-
"""
Padé approximation for cyclic groups C_k (k=2,3,4,6) using tan-half-angle variables.

We fit   φ'(s) ≈ P(t)/Q(t),   where
  s   = tan(θ/2) ∈ [0,1]
  φ'  = tan(θ'/2) ∈ [0,1]
  t   = 2 s - 1  ∈ [-1,1]

Ground truth enforces equal-volume:
  C(θ') = G(θ') / G_tot  = u(θ) = 1 - cos θ,
and with s = tan(θ/2) we have the exact identity:
  u(s) = 2 s^2 / (1 + s^2).

We build C(θ') numerically (high-res θ' grid), invert it by monotone interpolation
to get θ'(u), then map to φ'(s) = tan(θ'(u(s))/2). LM fits Chebyshev Padé on t.

Author: (you)
"""

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# ────────────────────────────────────────────────────────────────────────────
# Global config
# ────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

PI = math.pi
EPS = 1e-18

# Resolution settings (safe + fast defaults; increase if you want tighter fits)
N_THETAP_GRID = 10000001  # θ' grid for CDF build (odd -> includes endpoints cleanly)
N_TRAIN = 8192  # training points for LM (Chebyshev-like)
N_EVAL = 32768  # evaluation points for model selection (uniform in s)
TARGET_SUP_ERROR_RAD = 2.5e-12  # target sup error in θ' (radians)

# Cyclic groups: τ = tan(π/(2k))
CYCLIC_GROUPS = {
    2: math.tan(PI / 4.0),  # 1.0
    3: math.tan(PI / 6.0),  # 1/√3
    4: math.tan(PI / 8.0),  # √2 - 1
    6: math.tan(PI / 12.0),  # 2 - √3
}

# Degree search ladder (continuation friendly)
DEGREE_LADDER = [
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (20, 20),
    (25, 25),
    (30, 30),
    (50, 50),
    (70, 70),
    (100, 100),
]

# ────────────────────────────────────────────────────────────────────────────
# Chebyshev utilities
# ────────────────────────────────────────────────────────────────────────────


def chebyshev_vandermonde(t: torch.Tensor, deg: int) -> torch.Tensor:
    """
    Return (N, deg+1) matrix with columns T_0..T_deg evaluated at t in [-1,1].
    """
    t = t.reshape(-1).to(DEVICE, dtype=DTYPE)
    N = t.numel()
    if deg < 0:
        return torch.zeros((N, 0), device=DEVICE, dtype=DTYPE)
    T = torch.empty((N, deg + 1), device=DEVICE, dtype=DTYPE)
    T[:, 0] = 1.0
    if deg >= 1:
        T[:, 1] = t
    for k in range(2, deg + 1):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]
    return T


def chebyshev_clenshaw(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Evaluate Σ coeffs[k] T_k(t) by Clenshaw.
    coeffs: (..., K+1)
    t: (N,)
    """
    t = t.to(DEVICE, dtype=DTYPE)
    c = coeffs.to(DEVICE, dtype=DTYPE)
    N = t.numel()
    K = c.shape[-1] - 1
    if K < 0:
        return torch.zeros_like(t)
    if K == 0:
        return torch.full((N,), c[0], device=DEVICE, dtype=DTYPE)

    b_kplus1 = torch.zeros((N,), device=DEVICE, dtype=DTYPE)
    b_kplus2 = torch.zeros((N,), device=DEVICE, dtype=DTYPE)
    two_t = 2.0 * t
    for k in range(K, 0, -1):
        b_k = c[k] + two_t * b_kplus1 - b_kplus2
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return c[0] + t * b_kplus1 - b_kplus2


# ────────────────────────────────────────────────────────────────────────────
# Cyclic FZ column-law (R^3 sin θ') and CDF build
# ────────────────────────────────────────────────────────────────────────────


def omega_max_cyclic(theta_p: torch.Tensor, tau: float) -> torch.Tensor:
    """ω_max(θ') = 2 arctan(τ sec θ')  with sec θ' = 1/cos θ'."""
    c = torch.clamp(torch.cos(theta_p), min=EPS)
    return 2.0 * torch.atan(torch.tensor(tau, dtype=DTYPE, device=DEVICE) / c)


def sin_omega(theta_p: torch.Tensor, tau: float) -> torch.Tensor:
    """sin ω_max for cyclic bound: 2 τ cosθ' / (1 + τ^2 cos^2 θ')."""
    c = torch.cos(theta_p)
    tau2 = tau * tau
    return (2.0 * tau * c) / (1.0 + tau2 * c * c)


def R_cyclic(theta_p: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Homochoric boundary radius R(θ'): R^3 = (3/4)(ω - sin ω).
    """
    om = omega_max_cyclic(theta_p, tau)
    s = sin_omega(theta_p, tau)
    R3 = 0.75 * (om - s)
    return torch.clamp(R3, min=0.0) ** (1.0 / 3.0)


def build_cyclic_inverse_cdf(
    k: int, n_theta: int = N_THETAP_GRID
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build high-res monotone CDF C(θ') on θ'∈[0, π/2] for C_k, then return (θ'_grid, C_grid).
    """
    assert k in CYCLIC_GROUPS
    tau = CYCLIC_GROUPS[k]

    # Cosine-spaced grid resolves both endpoints well
    j = torch.arange(n_theta, device=DEVICE, dtype=DTYPE)
    theta_p = 0.5 * PI * (0.5 - 0.5 * torch.cos(PI * j / (n_theta - 1)))  # [0, π/2]

    # Column density f(θ') = R(θ')^3 sin θ'
    R = R_cyclic(theta_p, tau)
    f = R**3 * torch.sin(
        theta_p
    )  # Note: R^3, so this is exactly 0.75(ω - sin ω) * sin θ'

    # Cumulative via trapezoid in θ' (monotone increasing)
    # Integrate f(θ') dθ'  → G(θ')
    dtheta = torch.diff(theta_p)
    avg = 0.5 * (f[:-1] + f[1:])
    G = torch.zeros_like(theta_p)
    G[1:] = torch.cumsum(avg * dtheta, dim=0)
    G_tot = G[-1].clone()
    C = (G / torch.clamp(G_tot, min=EPS)).clamp(0.0, 1.0)
    return theta_p, C


def invert_cdf_lookup(
    u: torch.Tensor, theta_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """
    Invert C_grid(θ') ≈ u for u∈[0,1] by monotone linear interpolation (torch-only).
    """
    u = u.to(DEVICE, dtype=DTYPE).clamp(0.0, 1.0)
    # searchsorted on CPU is faster; but we stay on DEVICE for simplicity
    idx = torch.searchsorted(C_grid, u, right=False)
    idx = torch.clamp(idx, 1, C_grid.numel() - 1)

    C0 = C_grid[idx - 1]
    C1 = C_grid[idx]
    T0 = theta_grid[idx - 1]
    T1 = theta_grid[idx]

    w = torch.where((C1 - C0) > EPS, (u - C0) / (C1 - C0), torch.zeros_like(u))
    return (1.0 - w) * T0 + w * T1


# ────────────────────────────────────────────────────────────────────────────
# Levenberg–Marquardt with pole guard (Padé φ'(s) on Chebyshev t=2s-1)
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def fit_pade_LM(
    s_train: torch.Tensor,
    phi_train: torch.Tensor,
    deg_num: int,
    deg_den: int,
    num_init: torch.Tensor = None,
    den_init: torch.Tensor = None,
    max_iterations: int = 120,
    lambda_init: float = 1e-2,
    lambda_max: float = 1e6,
    guard_points: int = 2048,
    q_floor: float = 1e-3,
    lambda_guard: float = 5e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit Padé on t=2s-1 for φ'(s) in [0,1]. Denominator is 1 + Σ b_k T_k(t), k≥1.
    """
    s_train = s_train.to(DEVICE, dtype=DTYPE).reshape(-1)
    phi_train = phi_train.to(DEVICE, dtype=DTYPE).reshape(-1)
    t_train = 2.0 * s_train - 1.0
    N = t_train.numel()

    # Vandermonde
    Tn = chebyshev_vandermonde(t_train, deg_num)  # (N, n+1)
    Td = chebyshev_vandermonde(t_train, deg_den)[:, 1:]  # (N, m) exclude T0

    # Guard on dense Chebyshev abscissae in t
    j = torch.arange(guard_points, device=DEVICE, dtype=DTYPE)
    t_guard = torch.cos(PI * (j + 0.5) / guard_points)
    Td_guard = chebyshev_vandermonde(t_guard, deg_den)[:, 1:]

    # Params
    a = torch.zeros(deg_num + 1, device=DEVICE, dtype=DTYPE)
    b = torch.zeros(deg_den, device=DEVICE, dtype=DTYPE)
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
        Q = 1.0 + (Td @ b)
        r = P / Q - phi_train

        # Guard: penalize |Q| < q_floor at guard points
        Qg = 1.0 + (Td_guard @ b)
        viol = torch.clamp(q_floor - Qg.abs(), min=0.0)
        r_guard = math.sqrt(lambda_guard) * viol

        cost = (r @ r) + (r_guard @ r_guard)
        if cost.item() < best_cost:
            best_cost = cost.item()
            best = (a.clone(), b.clone())

        # Jacobians
        Ja = Tn / Q.unsqueeze(-1)  # (N, n+1)
        Jb = -(P / (Q * Q)).unsqueeze(-1) * Td  # (N, m)

        J_data = torch.cat([Ja, Jb], dim=1)  # (N, n+1+m)

        active = viol > 0.0
        if active.any():
            signQ = torch.sign(Qg[active]).unsqueeze(-1)
            Jg_b = -math.sqrt(lambda_guard) * signQ * Td_guard[active]
            Jg = torch.cat(
                [
                    torch.zeros(
                        (Jg_b.shape[0], deg_num + 1), device=DEVICE, dtype=DTYPE
                    ),
                    Jg_b,
                ],
                dim=1,
            )
            J = torch.cat([J_data, Jg], dim=0)
            rr = torch.cat([r, r_guard[active]], dim=0)
        else:
            J = J_data
            rr = r

        JTJ = J.T @ J
        g = J.T @ rr
        H = JTJ.clone()
        idx = torch.arange(H.shape[0], device=DEVICE)
        H[idx, idx] += lam

        try:
            delta = torch.linalg.solve(H, -g)
        except RuntimeError:
            delta = torch.linalg.lstsq(H, -g).solution

        a_new = a + delta[: deg_num + 1]
        b_new = b + delta[deg_num + 1 :]

        # Trial
        Pn = Tn @ a_new
        Qn = 1.0 + (Td @ b_new)
        rn = Pn / Qn - phi_train
        Qgn = 1.0 + (Td_guard @ b_new)
        violn = torch.clamp(q_floor - Qgn.abs(), min=0.0)
        rg = math.sqrt(lambda_guard) * violn
        cost_new = (rn @ rn) + (rg @ rg)

        if cost_new < cost:
            a, b = a_new, b_new
            lam = max(lam / 3.0, 1e-12)
        else:
            lam = min(lam * 3.0, lambda_max)

    a_best, b_best = best
    den = torch.empty(deg_den + 1, device=DEVICE, dtype=DTYPE)
    den[0] = 1.0
    den[1:] = b_best
    return a_best, den


def evaluate_pade_phi_of_s(
    s: torch.Tensor, num: torch.Tensor, den: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate φ'(s) on s∈[0,1] with t=2s-1 Chebyshev series P/Q.
    """
    s = s.to(DEVICE, dtype=DTYPE).clamp(0.0, 1.0)
    t = 2.0 * s - 1.0
    Tn = chebyshev_vandermonde(t, num.numel() - 1)
    Td = chebyshev_vandermonde(t, den.numel() - 1)
    P = Tn @ num.to(DEVICE, dtype=DTYPE)
    # den[0] is constant term for Q; we already store the full vector
    Q = Td @ den.to(DEVICE, dtype=DTYPE)
    return (P / torch.clamp(Q, min=1e-14)).clamp(0.0, 1.0)


# ────────────────────────────────────────────────────────────────────────────
# Degree search & packaging
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class PadeCoeffs:
    k: int
    deg_num: int
    deg_den: int
    numerator: torch.Tensor  # Chebyshev coeffs for P(t)
    denominator: torch.Tensor  # Chebyshev coeffs for Q(t), full (includes T0)
    sup_error_theta: float  # sup error in θ' (radians) on eval set
    rms_error_theta: float  # rms error in θ' (radians) on eval set
    min_Q_guard: float  # min |Q| across dense guard grid


def search_optimal_degree_for_Ck(
    k: int,
    degree_ladder=DEGREE_LADDER,
    target_sup_error_rad: float = TARGET_SUP_ERROR_RAD,
) -> PadeCoeffs:
    """
    Build ground truth for C_k, then fit along degree ladder and return the
    smallest degree meeting the target sup error in θ'.
    """
    # 1) Ground truth CDF C(θ')
    theta_grid, C_grid = build_cyclic_inverse_cdf(k, n_theta=N_THETAP_GRID)
    theta_max = float(theta_grid[-1].item())  # = π/2

    # 2) Training and evaluation in s=tan(θ/2)
    # Chebyshev-like sampling in s to resolve endpoints
    s_train = 0.5 * (
        1.0 - torch.cos(PI * torch.linspace(0, 1, N_TRAIN, device=DEVICE, dtype=DTYPE))
    )
    s_eval = torch.linspace(0.0, 1.0, N_EVAL, device=DEVICE, dtype=DTYPE)

    # Exact u(s) = 1 - cos θ = 2 s^2/(1+s^2)
    def u_of_s(s: torch.Tensor) -> torch.Tensor:
        return (2.0 * s * s) / (1.0 + s * s)

    # θ'(u), then φ'(s) = tan(θ'/2)
    theta_p_train = invert_cdf_lookup(u_of_s(s_train), theta_grid, C_grid)
    phi_train = torch.tan(0.5 * theta_p_train).clamp(0.0, 1.0)

    theta_p_eval_true = invert_cdf_lookup(u_of_s(s_eval), theta_grid, C_grid)
    phi_eval_true = torch.tan(0.5 * theta_p_eval_true).clamp(0.0, 1.0)

    # Warm starts
    prev_num = None
    prev_den = None
    best: PadeCoeffs = None

    print(f"\n{'─'*80}")
    print(f"  C_{k}: fitting φ'(s) ∈ [0,1]  (θ'∈[0, {math.degrees(theta_max):.2f}°])")
    print(f"{'─'*80}")
    print(
        f"  {'(n,m)':>12}  {'deg':>5}  {'sup|Δθ\'| (rad)':>16}  {'rms|Δθ\'| (rad)':>16}  {'min|Q|':>10}"
    )

    for n, m in degree_ladder:
        # Continuation init
        if prev_num is not None:
            num_init = torch.zeros(n + 1, device=DEVICE, dtype=DTYPE)
            num_init[: min(prev_num.numel(), num_init.numel())] = prev_num[
                : min(prev_num.numel(), num_init.numel())
            ]
            den_init = torch.zeros(m, device=DEVICE, dtype=DTYPE)
            if prev_den.numel() > 1:
                kcopy = min(m, prev_den.numel() - 1)
                den_init[:kcopy] = prev_den[1 : 1 + kcopy]
        else:
            num_init = None
            den_init = None

        num, den = fit_pade_LM(
            s_train,
            phi_train,
            n,
            m,
            num_init=num_init,
            den_init=den_init,
            max_iterations=120,
            lambda_init=1e-2,
            lambda_max=1e6,
            guard_points=4096,
            q_floor=1e-3,
            lambda_guard=5e-3,
        )

        # Evaluate on uniform s grid
        phi_eval_approx = evaluate_pade_phi_of_s(s_eval, num, den)
        theta_eval_approx = 2.0 * torch.atan(phi_eval_approx)  # back to θ'
        err = (theta_eval_approx - theta_p_eval_true).abs()
        sup_err = float(err.max().item())
        rms_err = float(torch.sqrt(torch.mean(err * err)).item())

        # Denominator health (dense guard)
        j = torch.arange(8192, device=DEVICE, dtype=DTYPE)
        tguard = torch.cos(PI * (j + 0.5) / 8192)
        Tdg = chebyshev_vandermonde(tguard, den.numel() - 1)
        Qvals = Tdg @ den
        min_Q = float(torch.min(Qvals.abs()).item())

        print(
            f"  ({n:2d},{m:2d})  {n+m:5d}  {sup_err:16.3e}  {rms_err:16.3e}  {min_Q:10.3e}"
        )

        if sup_err <= target_sup_error_rad and (
            best is None or (n + m) < (best.deg_num + best.deg_den)
        ):
            best = PadeCoeffs(
                k=k,
                deg_num=n,
                deg_den=m,
                numerator=num.detach().cpu(),
                denominator=den.detach().cpu(),
                sup_error_theta=sup_err,
                rms_error_theta=rms_err,
                min_Q_guard=min_Q,
            )

        prev_num, prev_den = num, den

    if best is None:
        raise RuntimeError(
            f"No degree met target sup error {target_sup_error_rad:.2e} rad for C_{k}"
        )

    print(
        f"  → SELECTED  (n={best.deg_num}, m={best.deg_den}),  sup|Δθ'|={best.sup_error_theta:.3e} rad"
    )
    return best


# ────────────────────────────────────────────────────────────────────────────
# Main driver & JSON output
# ────────────────────────────────────────────────────────────────────────────


def main():
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  Padé fits for SO(3)/C_k inverse map using tan half-angle (φ' vs s)")
    print("=" * 80)
    print(f"  DEVICE: {DEVICE}")
    print(
        f"  N_THETAP_GRID: {N_THETAP_GRID:,}, N_TRAIN: {N_TRAIN:,}, N_EVAL: {N_EVAL:,}"
    )
    print(f"  Target sup error in θ': {TARGET_SUP_ERROR_RAD:.2e} rad")
    print("=" * 80)

    all_coeffs: Dict[int, PadeCoeffs] = {}

    for k in [2, 3, 4, 6]:
        coeffs = search_optimal_degree_for_Ck(k, DEGREE_LADDER, TARGET_SUP_ERROR_RAD)
        all_coeffs[k] = coeffs

    # Pretty print and save
    print("\n" + "=" * 80)
    print("  FINAL COEFFICIENTS (Chebyshev on t=2s-1)")
    print("=" * 80)

    for k, c in all_coeffs.items():
        print(f"\n{'─'*80}")
        print(
            f"  C_{k} | Degrees (n={c.deg_num}, m={c.deg_den}) | sup|Δθ'|={c.sup_error_theta:.3e} rad"
        )
        print(f"{'─'*80}")
        print("  Numerator P(t) coefficients [T_0..T_n]:")
        for i, ai in enumerate(c.numerator.numpy()):
            print(f"    a[{i:2d}] = {ai:+.16e}")
        print("\n  Denominator Q(t) coefficients [T_0..T_m] (full, includes T_0):")
        for i, bi in enumerate(c.denominator.numpy()):
            print(f"    b[{i:2d}] = {bi:+.16e}")
        print(f"\n  Health: min|Q| on dense guard = {c.min_Q_guard:.3e}")
        print(f"          RMS|Δθ'| = {c.rms_error_theta:.3e} rad")

    # Save JSON
    out = {
        str(k): {
            "degree_num": c.deg_num,
            "degree_den": c.deg_den,
            "sup_error_theta_rad": c.sup_error_theta,
            "rms_error_theta_rad": c.rms_error_theta,
            "min_Q_guard": c.min_Q_guard,
            # Store full denominator (including T0)
            "numerator": c.numerator.numpy().tolist(),
            "denominator": c.denominator.numpy().tolist(),
            # For convenience, note the variable: φ'(s) with t=2s-1
            "variable": {
                "input": "s = tan(theta/2) in [0,1]",
                "output": "phi' = tan(theta'/2) in [0,1]",
                "t_map": "t=2s-1",
            },
        }
        for k, c in all_coeffs.items()
    }
    out_file = str(Path(__file__).resolve().parent / "coeffs_polar_Ck_tanhalf.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 80)
    print(f"  Coefficients saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
