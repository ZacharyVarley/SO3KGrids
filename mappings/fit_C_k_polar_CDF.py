#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padé Approximation for Inverse CDF in SO(3)/C_k Sampling
========================================================

This script computes high-precision rational (Padé) approximations for the
inverse cumulative distribution function θ'(y) used in sampling quotient
spaces SO(3)/C_k, where C_k is the cyclic group of order k.

For C_k groups, the domain is radially symmetric, requiring only a polar
angle mapping. We use √y preconditioning: g(y) = θ'(y)/√y where y = 1 - cos(θ).

The approximations take the form P(t)/Q(t) where:
  - t = 2y - 1 (transformation from [0,1] to [-1,1])
  - P(t) is a Chebyshev polynomial of degree n
  - Q(t) = 1 + Σ_{i=1}^m b_i T_i(t) (denominator fixed at Q(0) = 1)

Author: Generated for SO(3)/C_k sampling publication
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import torch


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# Maximum radius in Rodrigues representation
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)

# Target precision for publication
TARGET_RMS_ERROR = 1e-13

# High-resolution grid for building the ground-truth inverse CDF
N_SIMPSON = 1_000_001  # Must be odd for Simpson's rule

# Training points for Padé fitting (Chebyshev-like distribution)
N_TRAIN = 4096

# Evaluation points for error measurement
N_EVAL = 200001

# Cyclic groups to process: (name, tau, laue_id)
CYCLIC_GROUPS = [
    ("C2", 1.0, 2),  # Monoclinic
    ("C3", 1.0 / math.sqrt(3.0), 6),  # Trigonal
    ("C4", math.sqrt(2.0) - 1.0, 4),  # Tetragonal
    ("C6", 2.0 - math.sqrt(3.0), 8),  # Hexagonal
]

# Tiny value for numerical stability
TINY = 1e-300


# ═══════════════════════════════════════════════════════════════════════════
#  C_k GEOMETRY PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════


def omega_max(theta: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Maximum rotation angle ω_max(θ) for the fundamental zone boundary.

    ω_max(θ; τ) = 2 arctan(τ sec θ)
    """
    cos_theta = torch.cos(theta)
    sec_theta = torch.where(cos_theta > 1e-16, 1.0 / cos_theta, torch.tensor(1e16))
    return 2.0 * torch.atan(tau * sec_theta)


def sin_omega(theta: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Sine of rotation angle: sin ω(θ; τ) = 2τ cos θ / (cos² θ + τ²)
    """
    cos_theta = torch.cos(theta)
    return (2.0 * tau * cos_theta) / (cos_theta * cos_theta + tau * tau)


def R_of_theta(theta: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Radius function R(θ; τ) for equal-volume mapping.

    R(θ; τ)³ = (3/4)[ω_max(θ; τ) - sin ω(θ; τ)]
    """
    omega = omega_max(theta, tau)
    sin_om = sin_omega(theta, tau)
    R_cubed = 0.75 * (omega - sin_om)
    return torch.pow(R_cubed, 1.0 / 3.0)


def integrand_G(theta: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Integrand for G(θ): R(θ; τ)³ sin(θ)
    """
    omega = omega_max(theta, tau)
    sin_om = sin_omega(theta, tau)
    R_cubed = 0.75 * (omega - sin_om)
    return R_cubed * torch.sin(theta)


# ═══════════════════════════════════════════════════════════════════════════
#  HIGH-RESOLUTION INVERSE CDF CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


def simpson_integrate(f: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Cumulative Simpson's rule integration.

    For f = [f_0, f_1, ..., f_n] with n even, computes:
    ∫_0^x f(u) du for each x in the grid.

    Args:
        f: Function values at equally-spaced points (must have odd length)
        dx: Grid spacing

    Returns:
        Cumulative integral values
    """
    n = f.shape[0]
    if n % 2 == 0:
        raise ValueError("Simpson's rule requires odd number of points")

    # Simpson coefficients: 1, 4, 2, 4, 2, ..., 4, 1
    coeffs = torch.ones_like(f)
    coeffs[1:-1:2] = 4.0  # Odd indices get 4
    coeffs[2:-1:2] = 2.0  # Even indices (except first/last) get 2

    # Cumulative integration
    integrand = f * coeffs * (dx / 3.0)
    cumsum = torch.zeros_like(f)
    cumsum[2::2] = (integrand[:-1:2] + integrand[1::2] + integrand[2::2]).cumsum(0)

    # Linear interpolation for odd-indexed points
    cumsum[1:-1:2] = 0.5 * (cumsum[0:-2:2] + cumsum[2::2])

    return cumsum


def build_inverse_cdf_Ck(
    tau: float, n_points: int = N_SIMPSON
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build high-resolution ground truth for the inverse CDF via Simpson integration.

    Computes G(θ) = ∫_0^θ R(u; τ)³ sin(u) du and normalizes to get CDF.

    Args:
        tau: Group parameter τ = tan(π/k)
        n_points: Number of grid points (must be odd)

    Returns:
        theta_grid: Grid of θ values in [0, π/2]
        C_grid: Corresponding CDF values in [0, 1]
        G_tot: Total integral G(π/2)
    """
    if n_points % 2 == 0:
        n_points += 1  # Ensure odd for Simpson

    # Uniform grid in θ
    theta_grid = torch.linspace(0, 0.5 * PI, n_points, device=DEVICE)
    dtheta = theta_grid[1] - theta_grid[0]

    # Compute integrand
    f = integrand_G(theta_grid, tau)

    # Cumulative integration using Simpson's rule
    G = simpson_integrate(f, float(dtheta))
    G_tot = float(G[-1].item())

    # Normalize to [0, 1]
    C = G / G[-1].clamp_min(TINY)

    return theta_grid, C, G_tot


def invert_cdf_lookup(
    y: torch.Tensor, theta_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """
    Invert the CDF to get θ'(y) via binary search + linear interpolation.

    Args:
        y: Values of 1 - cos(θ) in [0, 1]
        theta_grid: Pre-computed θ grid
        C_grid: Pre-computed CDF values

    Returns:
        θ' values corresponding to y (via equal-volume mapping)
    """
    # Convert y to CDF values: C = y * G_tot
    # But C_grid is already normalized to [0,1], so we need the total
    # Actually, the equal-volume condition is: 1 - cos(θ') = G(θ) / G_tot
    # So: C(θ) = G(θ) / G_tot, and we want θ' such that C(θ') = y

    u = y.clamp(0, 1)

    # Binary search to find bracketing interval
    idx = torch.searchsorted(C_grid, u, right=True) - 1
    idx = idx.clamp(0, C_grid.numel() - 2)

    # Linear interpolation within interval
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    t0, t1 = theta_grid[idx], theta_grid[idx + 1]

    t = (u - c0) / (c1 - c0 + TINY)
    return (1 - t) * t0 + t * t1


def compute_y_from_theta(
    theta: torch.Tensor, theta_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """
    Forward map: θ → y via C(θ) lookup.

    This is the inverse of invert_cdf_lookup, used for creating training data.
    """
    theta = theta.clamp(0, 0.5 * PI)

    # Find bracketing interval
    idx = torch.searchsorted(theta_grid, theta, right=True) - 1
    idx = idx.clamp(0, theta_grid.numel() - 2)

    # Linear interpolation
    t0, t1 = theta_grid[idx], theta_grid[idx + 1]
    c0, c1 = C_grid[idx], C_grid[idx + 1]

    t = (theta - t0) / (t1 - t0 + TINY)
    return (1 - t) * c0 + t * c1


# ═══════════════════════════════════════════════════════════════════════════
#  PRECONDITIONING CONSTANT
# ═══════════════════════════════════════════════════════════════════════════


def compute_c0(tau: float, G_tot: float) -> float:
    """
    Leading-order constant c₀ for g(y) = θ'(y)/√y as y → 0.

    From Taylor expansion: θ' ≈ c₀√y for small y
    where c₀ = √(2 G_tot / R(0)³)
    """
    # R(0; τ)³ = (3/4) * [2 arctan(τ) - 2τ/(1+τ²)]
    atan_tau = math.atan(tau)
    R0_cubed = 0.75 * (2.0 * atan_tau - 2.0 * tau / (1.0 + tau * tau))

    return math.sqrt(2.0 * G_tot / R0_cubed)


# ═══════════════════════════════════════════════════════════════════════════
#  CHEBYSHEV POLYNOMIAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def y_to_t(y: torch.Tensor) -> torch.Tensor:
    """Map y ∈ [0,1] to t ∈ [-1,1] for Chebyshev basis."""
    return 2.0 * y - 1.0


def chebyshev_vandermonde(t: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Construct Chebyshev Vandermonde matrix [T_0(t), T_1(t), ..., T_n(t)].

    Uses the recurrence relation: T_{k+1}(t) = 2t·T_k(t) - T_{k-1}(t)
    """
    T0 = torch.ones_like(t)
    if degree == 0:
        return T0.unsqueeze(-1)

    T1 = t
    columns = [T0, T1]

    for _ in range(1, degree):
        T_next = 2.0 * t * columns[-1] - columns[-2]
        columns.append(T_next)

    return torch.stack(columns[: degree + 1], dim=-1)


def chebyshev_clenshaw(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Evaluate Chebyshev polynomial using Clenshaw's algorithm for numerical stability.

    Given coefficients c = [c_0, c_1, ..., c_n], computes Σ c_k T_k(t).
    """
    if coeffs.numel() == 1:
        return torch.full_like(t, coeffs[0])

    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)

    # Backward recurrence
    for c_k in coeffs[1:].flip(0):
        b0 = 2.0 * t * b1 - b2 + c_k
        b2, b1 = b1, b0

    return t * b1 - b2 + coeffs[0]


def evaluate_pade_g(
    y: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate rational function g(y) = P(t)/Q(t) where t = 2y - 1.

    Args:
        y: Input values in [0, 1]
        numerator: Chebyshev coefficients for P(t)
        denominator: Chebyshev coefficients for Q(t) = 1 + Σ b_k T_k(t)

    Returns:
        g(y) = P(t)/Q(t)
    """
    t = y_to_t(y)

    # Evaluate numerator
    P = chebyshev_clenshaw(numerator, t)

    # Evaluate denominator (exclude constant term as it's fixed at 1)
    if denominator.numel() == 1:
        Q = torch.ones_like(P)
    else:
        den_shifted = denominator.clone()
        den_shifted[0] = 0.0
        Q = 1.0 + chebyshev_clenshaw(den_shifted, t)

    return P / Q


def theta_prime_from_pade(
    theta: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    c0: float = None,
) -> torch.Tensor:
    """
    Compute θ'(θ) using the Padé approximation for g(y).

    θ'(y) = √y · g(y) where y = 1 - cos(θ)

    Args:
        theta: Input angles in [0, π/2]
        numerator: Chebyshev coefficients for P(t)
        denominator: Chebyshev coefficients for Q(t)
        c0: Leading constant for small y (optional)

    Returns:
        θ' values
    """
    y = 1.0 - torch.cos(theta)
    sqrt_y = torch.sqrt(torch.maximum(y, torch.tensor(TINY, device=y.device)))

    g = evaluate_pade_g(y, numerator, denominator)

    # Use asymptotic form for very small y
    if c0 is not None:
        g = torch.where(y <= 1e-24, torch.tensor(c0, device=g.device), g)

    return sqrt_y * g


# ═══════════════════════════════════════════════════════════════════════════
#  LEVENBERG-MARQUARDT OPTIMIZER WITH POLE GUARD
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def fit_pade_approximation(
    y_train: torch.Tensor,
    g_train: torch.Tensor,
    degree_num: int,
    degree_den: int,
    num_init: torch.Tensor = None,
    den_init: torch.Tensor = None,
    max_iterations: int = 120,
    lambda_init: float = 1e-2,
    lambda_max: float = 1e6,
    guard_points: int = 4096,
    q_floor: float = 1e-7,
    lambda_guard: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a Padé approximation P(t)/Q(t) for g(y) using Levenberg-Marquardt.

    Includes a soft pole guard to prevent Q(t) from approaching zero on [-1,1].

    Args:
        y_train: Training inputs in [0, 1]
        g_train: Training targets (g = θ'/√y values)
        degree_num: Degree n of numerator P(t)
        degree_den: Degree m of denominator Q(t) - 1
        num_init: Initial numerator coefficients (warm start)
        den_init: Initial denominator coefficients (warm start)
        max_iterations: Maximum LM iterations
        lambda_init: Initial damping parameter
        lambda_max: Maximum damping parameter
        guard_points: Number of Chebyshev nodes for pole guard
        q_floor: Minimum allowed |Q(t)| on guard points
        lambda_guard: Weight for pole guard penalty

    Returns:
        numerator: Optimal Chebyshev coefficients for P(t)
        denominator: Optimal Chebyshev coefficients for Q(t), with den[0] = 1
    """
    # Prepare data
    y_train = y_train.to(DEVICE, dtype=torch.float64).reshape(-1)
    g_train = g_train.to(DEVICE, dtype=torch.float64).reshape(-1)
    t_train = 2.0 * y_train - 1.0
    N = y_train.numel()

    # Vandermonde matrices for training data
    T_num = chebyshev_vandermonde(t_train, degree_num)  # (N, n+1)
    T_den = chebyshev_vandermonde(t_train, degree_den)[:, 1:]  # (N, m), exclude T_0

    # Chebyshev nodes for pole guard penalty
    t_guard = torch.cos(
        PI * (torch.arange(guard_points, device=DEVICE) + 0.5) / guard_points
    )
    T_guard = chebyshev_vandermonde(t_guard, degree_den)[:, 1:]  # (G, m)

    # Initialize parameters
    a = torch.zeros(degree_num + 1, device=DEVICE)
    b = torch.zeros(degree_den, device=DEVICE)

    if num_init is not None:
        a[: min(a.numel(), num_init.numel())] = num_init[
            : min(a.numel(), num_init.numel())
        ]
    if den_init is not None:
        b[: min(b.numel(), den_init.numel())] = den_init[
            : min(b.numel(), den_init.numel())
        ]

    # LM state
    lam = lambda_init
    best_cost = float("inf")
    best_params = (a.clone(), b.clone())

    for iteration in range(max_iterations):
        # Forward pass
        P = T_num @ a  # (N,)
        Q = 1.0 + (T_den @ b)  # (N,)
        residual = P - g_train * Q  # (N,)  [Note: residual is P - g*Q, not P/Q - g]

        # Guard penalty: penalize |Q(t)| < q_floor on guard points
        Q_guard = 1.0 + (T_guard @ b)  # (G,)
        guard_violation = torch.clamp(q_floor - Q_guard.abs(), min=0.0)
        r_guard = math.sqrt(lambda_guard) * guard_violation

        # Total cost
        cost = (residual @ residual) + (r_guard @ r_guard)

        if cost.item() < best_cost:
            best_cost = cost.item()
            best_params = (a.clone(), b.clone())

        # Jacobian of residual w.r.t. numerator coefficients
        # ∂(P - g*Q)/∂a = T_num
        J_a = T_num  # (N, n+1)

        # Jacobian of residual w.r.t. denominator coefficients
        # ∂(P - g*Q)/∂b = -g * T_den
        J_b = -g_train.unsqueeze(-1) * T_den  # (N, m)

        # Stack Jacobians for data fitting
        J_data = torch.cat([J_a, J_b], dim=1)  # (N, n+1+m)

        # Guard Jacobian (active constraints only)
        active = guard_violation > 0.0
        if active.any():
            sign_Q = torch.sign(Q_guard[active]).unsqueeze(-1)
            J_guard_b = -math.sqrt(lambda_guard) * sign_Q * T_guard[active]
            J_guard = torch.cat(
                [
                    torch.zeros((J_guard_b.shape[0], degree_num + 1), device=DEVICE),
                    J_guard_b,
                ],
                dim=1,
            )

            J = torch.cat([J_data, J_guard], dim=0)
            r = torch.cat([residual, r_guard[active]], dim=0)
        else:
            J = J_data
            r = residual

        # Solve Levenberg-Marquardt system: (J^T J + λI) δ = -J^T r
        JTJ = J.T @ J
        g_vec = J.T @ r

        H = JTJ.clone()
        diag_idx = torch.arange(H.shape[0], device=DEVICE)
        H[diag_idx, diag_idx] += lam

        try:
            delta = torch.linalg.solve(H, -g_vec)
        except RuntimeError:
            delta = torch.linalg.lstsq(H, -g_vec).solution

        # Trial step
        a_new = a + delta[: degree_num + 1]
        b_new = b + delta[degree_num + 1 :]

        # Evaluate trial cost
        P_new = T_num @ a_new
        Q_new = 1.0 + (T_den @ b_new)
        residual_new = P_new - g_train * Q_new

        Q_guard_new = 1.0 + (T_guard @ b_new)
        guard_violation_new = torch.clamp(q_floor - Q_guard_new.abs(), min=0.0)
        r_guard_new = math.sqrt(lambda_guard) * guard_violation_new

        cost_new = (residual_new @ residual_new) + (r_guard_new @ r_guard_new)

        # Accept or reject step
        if cost_new < cost:
            a, b = a_new, b_new
            lam = max(lam / 3.0, 1e-12)
        else:
            lam = min(lam * 3.0, lambda_max)

    # Return best parameters found
    a_best, b_best = best_params

    # Format denominator with explicit constant term
    denominator = torch.empty(degree_den + 1, device=DEVICE)
    denominator[0] = 1.0
    denominator[1:] = b_best

    return a_best, denominator


# ═══════════════════════════════════════════════════════════════════════════
#  DEGREE SEARCH AND SELECTION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PadeCoefficients:
    """Container for Padé approximation coefficients and metadata."""

    group_name: str
    tau: float
    laue_id: int
    degree_num: int  # Numerator degree
    degree_den: int  # Denominator degree
    numerator: torch.Tensor  # Chebyshev coefficients for P(t)
    denominator: torch.Tensor  # Chebyshev coefficients for Q(t)
    c0: float  # Leading constant
    G_tot: float  # Total integral
    sup_error: float  # Supremum error on evaluation set
    rms_error: float  # RMS error on evaluation set
    min_Q: float  # Minimum |Q(t)| on Chebyshev nodes


def search_optimal_degree(
    group_name: str,
    tau: float,
    laue_id: int,
    theta_grid: torch.Tensor,
    C_grid: torch.Tensor,
    G_tot: float,
    degree_ladder: List[Tuple[int, int]],
    target_error: float = TARGET_RMS_ERROR,
) -> PadeCoefficients:
    """
    Search for the lowest total degree achieving target supremum error.

    Args:
        group_name: Group identifier (e.g., "C2", "C3")
        tau: Group parameter τ = tan(π/k)
        laue_id: Laue group identifier
        theta_grid: Ground truth θ grid
        C_grid: Ground truth CDF grid
        G_tot: Total integral value
        degree_ladder: List of (degree_num, degree_den) pairs to try
        target_error: Target supremum error threshold

    Returns:
        PadeCoefficients for the best approximation found
    """
    c0 = compute_c0(tau, G_tot)

    # Generate training data (Chebyshev-like distribution in theta)
    k_train = torch.arange(N_TRAIN, device=DEVICE)
    theta_train = 0.5 * PI * 0.5 * (1.0 - torch.cos(PI * k_train / (N_TRAIN - 1)))

    # Convert to y space
    y_train = compute_y_from_theta(theta_train, theta_grid, C_grid)

    # Get true θ' values
    theta_prime_train = (
        theta_train  # Since we start from theta, theta' = theta for training
    )

    # Actually, we need to go theta -> y -> theta' via the equal-volume mapping
    # Let me reconsider: we have theta in original space, we want theta' in FZ space
    # The mapping is: y = 1 - cos(theta), and theta' = invert_cdf_lookup(y, ...)
    y_train_from_cos = 1.0 - torch.cos(theta_train)
    theta_prime_train = invert_cdf_lookup(y_train, theta_grid, C_grid)

    # Compute g = θ'/√y with regularization near y=0
    sqrt_y = torch.sqrt(
        torch.maximum(y_train_from_cos, torch.tensor(TINY, device=DEVICE))
    )
    g_train = theta_prime_train / sqrt_y

    # Use asymptotic value for very small y
    g_train = torch.where(
        y_train_from_cos <= 1e-24, torch.tensor(c0, device=DEVICE), g_train
    )

    # Generate evaluation data (uniform in theta)
    theta_eval = torch.linspace(0.0, 0.5 * PI, N_EVAL, device=DEVICE)
    y_eval = compute_y_from_theta(theta_eval, theta_grid, C_grid)
    theta_prime_true = invert_cdf_lookup(y_eval, theta_grid, C_grid)

    # Warm-start parameters
    prev_num = None
    prev_den = None

    # Track best result meeting target
    best_result = None

    print(f"\n{'─' * 80}")
    print(f"  {group_name}: g(y) = θ'(y)/√y approximation  |  τ = {tau:.10f}")
    print(f"  G_tot = {G_tot:.16e}  |  c₀ = {c0:.16e}")
    print(f"{'─' * 80}")
    print(
        f"  {'Degree':>12}  {'Total':>6}  {'Sup Error':>12}  {'RMS Error':>12}  {'Min |Q|':>10}"
    )
    print(f"  {'(n, m)':>12}  {'n+m':>6}  {'':>12}  {'':>12}  {'':>10}")
    print(f"{'─' * 80}")

    for degree_num, degree_den in degree_ladder:
        total_degree = degree_num + degree_den

        # Initialize from previous degree (continuation)
        if prev_num is not None:
            num_init = torch.zeros(degree_num + 1, device=DEVICE)
            num_init[: min(num_init.numel(), prev_num.numel())] = prev_num[
                : min(num_init.numel(), prev_num.numel())
            ]

            den_init = torch.zeros(degree_den, device=DEVICE)
            k_copy = min(prev_den.numel() - 1, degree_den)
            if k_copy > 0:
                den_init[:k_copy] = prev_den[1 : 1 + k_copy]
        else:
            num_init = None
            den_init = None

        # Fit Padé approximation
        numerator, denominator = fit_pade_approximation(
            y_train_from_cos,
            g_train,
            degree_num,
            degree_den,
            num_init=num_init,
            den_init=den_init,
            max_iterations=120,
            lambda_init=1e-2,
            lambda_max=1e6,
            guard_points=4096,
            q_floor=1e-7,
            lambda_guard=1e-3,
        )

        # Evaluate error on θ' directly
        theta_prime_approx = theta_prime_from_pade(
            theta_eval, numerator, denominator, c0=c0
        )

        error = (theta_prime_approx - theta_prime_true).abs()
        sup_error = float(error.max().item())
        rms_error = float(error.square().mean().sqrt().item())

        # Check denominator health
        t_check = torch.cos(PI * (torch.arange(4096, device=DEVICE) + 0.5) / 4096)
        den_shifted = denominator.clone()
        den_shifted[0] = 0.0
        Q_vals = 1.0 + chebyshev_clenshaw(den_shifted, t_check)
        min_Q = float(Q_vals.abs().min().item())

        # Display result
        status = "✓" if rms_error < target_error else " "
        print(
            f"  {status} ({degree_num:2d}, {degree_den:2d})  "
            f"{total_degree:6d}  "
            f"{sup_error:12.3e}  "
            f"{rms_error:12.3e}  "
            f"{min_Q:10.3e}"
        )

        # Update best result if this meets target and has lower total degree
        if rms_error < target_error:
            if best_result is None or total_degree < (
                best_result.degree_num + best_result.degree_den
            ):
                best_result = PadeCoefficients(
                    group_name=group_name,
                    tau=tau,
                    laue_id=laue_id,
                    degree_num=degree_num,
                    degree_den=degree_den,
                    numerator=numerator.detach().cpu(),
                    denominator=denominator.detach().cpu(),
                    c0=c0,
                    G_tot=G_tot,
                    sup_error=sup_error,
                    rms_error=rms_error,
                    min_Q=min_Q,
                )

        # Update warm-start
        prev_num = numerator
        prev_den = denominator

    print(f"{'─' * 80}\n")

    if best_result is None:
        raise ValueError(
            f"No degree pair achieved target error {target_error:.2e} for {group_name}"
        )

    return best_result


# ═══════════════════════════════════════════════════════════════════════════
#  END-TO-END VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def sample_uniform_ho(n: int, seed: int = 42) -> torch.Tensor:
    """
    Sample n points uniformly from the Rodrigues ball of radius H_MAX.

    Returns:
        Array of shape (n, 3) with uniformly distributed points
    """
    torch.manual_seed(seed)

    # Uniform direction on sphere
    v = torch.randn(n, 3, device=DEVICE)
    v = v / torch.norm(v, dim=1, keepdim=True)

    # Uniform radius with correct volume scaling
    u = torch.rand(n, device=DEVICE)
    r = H_MAX * torch.pow(u, 1.0 / 3.0)

    return v * r.unsqueeze(-1)


def ho_to_hoCk_pade(
    h: torch.Tensor,
    tau: float,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    c0: float,
) -> torch.Tensor:
    """
    Map Rodrigues vectors to C_k fundamental zone using Padé approximation.

    Args:
        h: Input Rodrigues vectors, shape (..., 3)
        tau: Group parameter
        numerator: Padé numerator coefficients
        denominator: Padé denominator coefficients
        c0: Leading constant

    Returns:
        Mapped Rodrigues vectors in fundamental zone
    """
    h = h.to(DEVICE, dtype=torch.float64)
    out = torch.empty_like(h)

    x, y, z = h[..., 0], h[..., 1], h[..., 2]

    # Handle sign of z
    z_sign = torch.sign(z)
    z_sign = torch.where(z_sign == 0.0, torch.ones_like(z_sign), z_sign)
    z_abs = torch.abs(z)

    # Compute spherical coordinates
    rho = torch.norm(h, dim=-1)
    xy = torch.hypot(x, y)
    theta = torch.atan2(xy, z_abs)  # ∈ [0, π/2]

    # Map theta to theta' using Padé approximation
    theta_fz = theta_prime_from_pade(theta, numerator, denominator, c0=c0)

    # Compute radius scaling
    R = R_of_theta(theta_fz, tau)
    rho_prime = rho * (R / H_MAX)

    # Convert back to Cartesian
    azimuth = torch.atan2(y, x)
    sin_theta = torch.sin(theta_fz)
    cos_theta = torch.cos(theta_fz)

    out[..., 0] = rho_prime * sin_theta * torch.cos(azimuth)
    out[..., 1] = rho_prime * sin_theta * torch.sin(azimuth)
    out[..., 2] = rho_prime * cos_theta * z_sign

    return out


def ho_to_hoCk_exact(
    h: torch.Tensor, tau: float, theta_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """
    Map Rodrigues vectors to C_k fundamental zone using exact lookup.

    Args:
        h: Input Rodrigues vectors, shape (..., 3)
        tau: Group parameter
        theta_grid: Ground truth θ grid
        C_grid: Ground truth CDF grid

    Returns:
        Mapped Rodrigues vectors in fundamental zone
    """
    h = h.to(DEVICE, dtype=torch.float64)
    out = torch.empty_like(h)

    x, y, z = h[..., 0], h[..., 1], h[..., 2]

    # Handle sign of z
    z_sign = torch.sign(z)
    z_sign = torch.where(z_sign == 0.0, torch.ones_like(z_sign), z_sign)
    z_abs = torch.abs(z)

    # Compute spherical coordinates
    rho = torch.norm(h, dim=-1)
    xy = torch.hypot(x, y)
    theta = torch.atan2(xy, z_abs)  # ∈ [0, π/2]

    # Map theta to theta' using exact CDF inversion
    y_val = compute_y_from_theta(theta, theta_grid, C_grid)
    theta_fz = invert_cdf_lookup(y_val, theta_grid, C_grid)

    # Compute radius scaling
    R = R_of_theta(theta_fz, tau)
    rho_prime = rho * (R / H_MAX)

    # Convert back to Cartesian
    azimuth = torch.atan2(y, x)
    sin_theta = torch.sin(theta_fz)
    cos_theta = torch.cos(theta_fz)

    out[..., 0] = rho_prime * sin_theta * torch.cos(azimuth)
    out[..., 1] = rho_prime * sin_theta * torch.sin(azimuth)
    out[..., 2] = rho_prime * cos_theta * z_sign

    return out


def report_errors(exact: torch.Tensor, approx: torch.Tensor, metric: str = "θ'"):
    """Print detailed error statistics."""
    error = (approx - exact).abs()

    print(f"  {metric} error statistics:")
    print(f"    Maximum:     {error.max().item():.6e}")
    print(f"    Mean:        {error.mean().item():.6e}")
    print(f"    RMS:         {error.square().mean().sqrt().item():.6e}")
    print(f"    Median:      {error.median().item():.6e}")
    print(f"    99th %ile:   {torch.quantile(error, 0.99).item():.6e}")


def report_cartesian_errors(H_exact: torch.Tensor, H_approx: torch.Tensor):
    """Print Cartesian error statistics."""
    diff = H_approx - H_exact
    error = torch.norm(diff, dim=-1)

    print(f"  Cartesian error statistics:")
    print(f"    Maximum:     {error.max().item():.6e}")
    print(f"    Mean:        {error.mean().item():.6e}")
    print(f"    RMS:         {error.square().mean().sqrt().item():.6e}")
    print(f"    99th %ile:   {torch.quantile(error, 0.99).item():.6e}")

    # Relative error
    error_rel = error / H_MAX
    print(f"  Relative error (/ H_MAX):")
    print(f"    Maximum:     {error_rel.max().item():.6e}")
    print(f"    Mean:        {error_rel.mean().item():.6e}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Main driver for computing Padé approximations for all cyclic groups."""
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  PADÉ APPROXIMATION FOR SO(3)/C_k INVERSE CDF")
    print("=" * 80)
    print(f"  H_MAX = {H_MAX:.16f}")
    print(f"  Target RMS Error: {TARGET_RMS_ERROR:.2e}")
    print(f"  Simpson Integration Points: {N_SIMPSON:,}")
    print(f"  Training Points:            {N_TRAIN:,}")
    print(f"  Evaluation Points:          {N_EVAL:,}")
    print("=" * 80)

    # Degree ladder for continuation-based optimization
    DEGREE_LADDER = [
        (5, 5),
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
        (10, 8),
        (11, 9),
        (12, 10),
        (13, 11),
        (14, 12),
        (15, 13),
        (16, 14),
        (17, 15),
        (18, 16),
        (19, 17),
        (20, 18),
    ]

    all_coefficients: Dict[str, PadeCoefficients] = {}

    # Process each cyclic group
    for group_name, tau, laue_id in CYCLIC_GROUPS:
        print(f"\n{'═' * 80}")
        print(f"  Processing {group_name}  (τ = {tau:.16e})")
        print(f"{'═' * 80}")

        # Build high-resolution ground truth
        theta_grid, C_grid, G_tot = build_inverse_cdf_Ck(tau, n_points=N_SIMPSON)

        # Search for optimal degree
        coeffs = search_optimal_degree(
            group_name,
            tau,
            laue_id,
            theta_grid,
            C_grid,
            G_tot,
            DEGREE_LADDER,
            TARGET_RMS_ERROR,
        )
        all_coefficients[group_name] = coeffs

        # Print selected result
        print(f"  SELECTED for {group_name}:")
        print(
            f"    Degrees:     (n={coeffs.degree_num}, m={coeffs.degree_den})  "
            f"[total = {coeffs.degree_num + coeffs.degree_den}]"
        )
        print(f"    Sup Error:   {coeffs.sup_error:.6e}")
        print(f"    RMS Error:   {coeffs.rms_error:.6e}")
        print(f"    Min |Q|:     {coeffs.min_Q:.6e}")

        # Validation: θ' accuracy on random samples
        print(f"\n  Validation: θ' accuracy on random samples")
        torch.manual_seed(123)
        theta_val = 0.5 * PI * torch.rand(8000, device=DEVICE)
        y_val = compute_y_from_theta(theta_val, theta_grid, C_grid)
        theta_prime_exact = invert_cdf_lookup(y_val, theta_grid, C_grid)
        theta_prime_pade = theta_prime_from_pade(
            theta_val,
            coeffs.numerator.to(DEVICE),
            coeffs.denominator.to(DEVICE),
            c0=coeffs.c0,
        )
        report_errors(theta_prime_exact, theta_prime_pade, metric=f"θ' ({group_name})")

        # End-to-end Cartesian test
        print(f"\n  End-to-end Cartesian accuracy test")
        H = sample_uniform_ho(20000, seed=42)
        H_exact = ho_to_hoCk_exact(H, tau, theta_grid, C_grid)
        H_pade = ho_to_hoCk_pade(
            H,
            tau,
            coeffs.numerator.to(DEVICE),
            coeffs.denominator.to(DEVICE),
            coeffs.c0,
        )
        report_cartesian_errors(H_exact, H_pade)

        print()

    # ═══════════════════════════════════════════════════════════════════════
    #  OUTPUT RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  FINAL COEFFICIENTS")
    print("=" * 80)

    for group_name in [name for name, _, _ in CYCLIC_GROUPS]:
        coeffs = all_coefficients[group_name]
        print(f"\n{'─' * 80}")
        print(
            f"  {group_name}  |  Degrees (n={coeffs.degree_num}, m={coeffs.degree_den})  |  "
            f"Sup Error = {coeffs.sup_error:.6e}"
        )
        print(f"  τ = {coeffs.tau:.16e}  |  c₀ = {coeffs.c0:.16e}")
        print(f"{'─' * 80}")

        print(
            f"\n  Numerator P(t) — Chebyshev coefficients [T_0, T_1, ..., T_{coeffs.degree_num}]:"
        )
        num_array = coeffs.numerator.numpy()
        for i, coeff in enumerate(num_array):
            print(f"    a[{i:2d}] = {coeff:+.16e}")

        print(f"\n  Denominator Q(t) = 1 + Σ b[k]·T_k(t) — Chebyshev coefficients:")
        den_array = coeffs.denominator.numpy()
        print(f"    b[0] = {den_array[0]:+.16e}  (fixed)")
        for i in range(1, len(den_array)):
            print(f"    b[{i:2d}] = {den_array[i]:+.16e}")

        print()

    # ═══════════════════════════════════════════════════════════════════════
    #  SAVE TO JSON
    # ═══════════════════════════════════════════════════════════════════════

    output = {
        group_name: {
            "tau": coeffs.tau,
            "laue_id": coeffs.laue_id,
            "degree_num": coeffs.degree_num,
            "degree_den": coeffs.degree_den,
            "c0": coeffs.c0,
            "G_tot": coeffs.G_tot,
            "sup_error": coeffs.sup_error,
            "rms_error": coeffs.rms_error,
            "min_Q": coeffs.min_Q,
            "numerator": coeffs.numerator.numpy().tolist(),
            "denominator": coeffs.denominator.numpy().tolist(),
        }
        for group_name, coeffs in all_coefficients.items()
    }

    output_file = str(Path(__file__).resolve().parent / "coeffs_polar_Ck.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print(f"  Coefficients saved to: {output_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
