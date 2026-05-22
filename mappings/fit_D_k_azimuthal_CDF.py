#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Padé Approximation for Inverse CDF in SO(3)/K Sampling
=======================================================

This script computes high-precision rational (Padé) approximations for the
inverse cumulative distribution function φ'(u) used in sampling quotient
spaces SO(3)/D_k, where D_k is the dihedral group of order 2k.

The approximations take the form P(t)/Q(t) where:
  - t = 2u - 1 (transformation from [0,1] to [-1,1])
  - P(t) is a Chebyshev polynomial of degree n
  - Q(t) = 1 + Σ_{i=1}^m b_i T_i(t) (denominator fixed at Q(0) = 1)

Author: Generated for SO(3)/K sampling publication
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PI = math.pi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# Target precision for publication
TARGET_SUP_ERROR = 5e-15

# High-resolution grid for building the ground-truth inverse CDF
N_PHI_GRID = 10_000_001

# Training points for Padé fitting (Chebyshev-like distribution)
N_TRAIN = 8195  # 4097*2 + 1

# Evaluation points for error measurement
N_EVAL = 20001

# Dihedral groups to process
DIHEDRAL_GROUPS = [2, 3, 4, 6]


# ═══════════════════════════════════════════════════════════════════════════
#  D_k GEOMETRY PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════


def polar_coefficient_Dk(k: int) -> float:
    """Polar coefficient a = cot(π/2k) for dihedral group D_k."""
    return 1.0 / math.tan(PI / (2.0 * k))


def F_polar_cap(theta: torch.Tensor, a: float) -> torch.Tensor:
    """Integrated density contribution from polar cap region."""
    cos_theta = torch.cos(theta)
    term1 = 0.5 * PI * (1.0 - cos_theta)
    term2 = math.atan(a)
    term3 = cos_theta * torch.atan(a * cos_theta)
    return 1.5 * (term1 - term2 + term3)


def F_latitude(theta: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Integrated density contribution from latitude band region."""
    b_sin_theta = b * torch.sin(theta)
    term1 = 0.5 * PI - torch.atan(b_sin_theta)

    sqrt_1_plus_b2 = torch.sqrt(1.0 + b * b)
    numerator = sqrt_1_plus_b2 * torch.sin(theta)
    term2 = (b / sqrt_1_plus_b2) * torch.atan2(numerator, torch.cos(theta))

    return 1.5 * (0.5 * PI - torch.cos(theta) * term1 - term2)


def A_phi_Dk(phi: torch.Tensor, k: int) -> torch.Tensor:
    """
    Integrated area function A(φ') for the fundamental wedge of SO(3)/D_k.

    This represents the cumulative density as a function of the wedge angle φ'.
    """
    a = polar_coefficient_Dk(k)
    b = torch.cos(phi)

    # Transition angle between polar cap and latitude band
    theta_star = torch.atan2(torch.full_like(b, a), b)

    # Combine contributions from both regions
    latitude_contrib = F_latitude(theta_star, b)
    polar_cap_contrib = F_polar_cap(torch.full_like(phi, 0.5 * PI), a) - F_polar_cap(
        theta_star, a
    )

    return latitude_contrib + polar_cap_contrib


def phi_max_for_k(k: int) -> float:
    """Maximum wedge angle φ'_max = π/2k for dihedral group D_k."""
    return PI / (2.0 * k)


# ═══════════════════════════════════════════════════════════════════════════
#  HIGH-RESOLUTION INVERSE CDF CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


def build_inverse_cdf(
    k: int, n_phi: int = N_PHI_GRID
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build high-resolution ground truth for the inverse CDF C(φ') via numerical integration.

    Uses a Chebyshev-like (Lobatto) grid for improved resolution near boundaries.

    Args:
        k: Dihedral group order (D_k has order 2k)
        n_phi: Number of grid points for φ' discretization

    Returns:
        phi_grid: Grid of φ' values in [0, φ'_max]
        C_grid: Corresponding CDF values in [0, 1]
    """
    phi_max = phi_max_for_k(k)

    # Chebyshev-like grid: dense near endpoints
    t = torch.linspace(0, 1, n_phi, device=DEVICE)
    phi_grid = 0.5 * (1 - torch.cos(PI * t)) * phi_max

    # Compute area function A(φ')
    A = A_phi_Dk(phi_grid, k)

    # Integrate using trapezoidal rule
    dphi = phi_grid[1:] - phi_grid[:-1]
    A_avg = 0.5 * (A[1:] + A[:-1])

    C = torch.empty_like(phi_grid)
    C[0] = 0.0
    C[1:] = (A_avg * dphi).cumsum(0)

    # Normalize to [0, 1]
    C /= C[-1].clamp_min(1e-300)

    return phi_grid, C


def invert_cdf_lookup(
    u: torch.Tensor, phi_grid: torch.Tensor, C_grid: torch.Tensor
) -> torch.Tensor:
    """
    Invert the CDF to get φ'(u) via binary search + linear interpolation.

    Args:
        u: Uniform random values in [0, 1]
        phi_grid: Pre-computed φ' grid
        C_grid: Pre-computed CDF values

    Returns:
        φ' values corresponding to u
    """
    u = u.clamp(0, 1)

    # Binary search to find bracketing interval
    idx = torch.searchsorted(C_grid, u, right=True) - 1
    idx = idx.clamp(0, C_grid.numel() - 2)

    # Linear interpolation within interval
    c0, c1 = C_grid[idx], C_grid[idx + 1]
    p0, p1 = phi_grid[idx], phi_grid[idx + 1]

    t = (u - c0) / (c1 - c0 + 1e-300)
    return (1 - t) * p0 + t * p1


# ═══════════════════════════════════════════════════════════════════════════
#  CHEBYSHEV POLYNOMIAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def u_to_t(u: torch.Tensor) -> torch.Tensor:
    """Map u ∈ [0,1] to t ∈ [-1,1] for Chebyshev basis."""
    return 2.0 * u - 1.0


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


def evaluate_pade_approximation(
    u: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate rational function P(t)/Q(t) where t = 2u - 1.

    Args:
        u: Input values in [0, 1]
        numerator: Chebyshev coefficients for P(t)
        denominator: Chebyshev coefficients for Q(t) = 1 + Σ b_k T_k(t)

    Returns:
        P(t)/Q(t) evaluated at each u
    """
    t = u_to_t(u)

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


# ═══════════════════════════════════════════════════════════════════════════
#  LEVENBERG-MARQUARDT OPTIMIZER WITH POLE GUARD
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def fit_pade_approximation(
    u_train: torch.Tensor,
    phi_train: torch.Tensor,
    degree_num: int,
    degree_den: int,
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
    Fit a Padé approximation P(t)/Q(t) to training data using Levenberg-Marquardt.

    Includes a soft pole guard to prevent Q(t) from approaching zero on [-1,1].

    Args:
        u_train: Training inputs in [0, 1]
        phi_train: Training targets (ground truth φ' values)
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
    u_train = u_train.to(DEVICE, dtype=torch.float64).reshape(-1)
    phi_train = phi_train.to(DEVICE, dtype=torch.float64).reshape(-1)
    t_train = 2.0 * u_train - 1.0
    N = u_train.numel()

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
        residual = P / Q - phi_train  # (N,)

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
        J_a = T_num / Q.unsqueeze(-1)  # (N, n+1)

        # Jacobian of residual w.r.t. denominator coefficients
        J_b = -(P / (Q * Q)).unsqueeze(-1) * T_den  # (N, m)

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
        g = J.T @ r

        H = JTJ.clone()
        diag_idx = torch.arange(H.shape[0], device=DEVICE)
        H[diag_idx, diag_idx] += lam

        try:
            delta = torch.linalg.solve(H, -g)
        except RuntimeError:
            delta = torch.linalg.lstsq(H, -g).solution

        # Trial step
        a_new = a + delta[: degree_num + 1]
        b_new = b + delta[degree_num + 1 :]

        # Evaluate trial cost
        P_new = T_num @ a_new
        Q_new = 1.0 + (T_den @ b_new)
        residual_new = P_new / Q_new - phi_train

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

    k: int  # Dihedral group order
    degree_num: int  # Numerator degree
    degree_den: int  # Denominator degree
    numerator: torch.Tensor  # Chebyshev coefficients for P(t)
    denominator: torch.Tensor  # Chebyshev coefficients for Q(t)
    phi_max: float  # Maximum wedge angle
    sup_error: float  # Supremum error on evaluation set
    rms_error: float  # RMS error on evaluation set
    min_Q: float  # Minimum |Q(t)| on Chebyshev nodes


def search_optimal_degree(
    k: int,
    phi_grid: torch.Tensor,
    C_grid: torch.Tensor,
    degree_ladder: list,
    target_error: float = TARGET_SUP_ERROR,
) -> PadeCoefficients:
    """
    Search for the lowest total degree achieving target supremum error.

    Args:
        k: Dihedral group order
        phi_grid: Ground truth φ' grid
        C_grid: Ground truth CDF grid
        degree_ladder: List of (degree_num, degree_den) pairs to try
        target_error: Target supremum error threshold

    Returns:
        PadeCoefficients for the best approximation found
    """
    phi_max = float(phi_grid[-1].item())

    # Generate training data (Chebyshev-like distribution)
    u_train = 0.5 * (1.0 - torch.cos(PI * torch.linspace(0, 1, N_TRAIN, device=DEVICE)))
    phi_train = invert_cdf_lookup(u_train, phi_grid, C_grid)

    # Generate evaluation data (uniform distribution)
    u_eval = torch.linspace(0.0, 1.0, N_EVAL, device=DEVICE)
    phi_true = invert_cdf_lookup(u_eval, phi_grid, C_grid)

    # Warm-start parameters
    prev_num = None
    prev_den = None

    # Track best result meeting target
    best_result = None

    print(f"\n{'─' * 80}")
    print(f"  D_{k}: φ'(u) approximation  |  φ'_max = {math.degrees(phi_max):.4f}°")
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
            u_train,
            phi_train,
            degree_num,
            degree_den,
            num_init=num_init,
            den_init=den_init,
            max_iterations=120,
            lambda_init=1e-2,
            lambda_max=1e6,
            guard_points=2048,
            q_floor=1e-3,
            lambda_guard=5e-3,
        )

        # Evaluate error
        phi_approx = evaluate_pade_approximation(u_eval, numerator, denominator)
        phi_approx = phi_approx.clamp(0.0, phi_max)

        error = (phi_approx - phi_true).abs()
        sup_error = float(error.max().item())
        rms_error = float(error.square().mean().sqrt().item())

        # Check denominator health
        t_check = torch.cos(PI * (torch.arange(4096, device=DEVICE) + 0.5) / 4096)
        den_shifted = denominator.clone()
        den_shifted[0] = 0.0
        Q_vals = 1.0 + chebyshev_clenshaw(den_shifted, t_check)
        min_Q = float(Q_vals.abs().min().item())

        # Display result
        status = "✓" if sup_error < target_error else " "
        print(
            f"  {status} ({degree_num:2d}, {degree_den:2d})  "
            f"{total_degree:6d}  "
            f"{sup_error:12.3e}  "
            f"{rms_error:12.3e}  "
            f"{min_Q:10.3e}"
        )

        # Update best result if this meets target and has lower total degree
        if sup_error < target_error:
            if best_result is None or total_degree < (
                best_result.degree_num + best_result.degree_den
            ):
                best_result = PadeCoefficients(
                    k=k,
                    degree_num=degree_num,
                    degree_den=degree_den,
                    numerator=numerator.detach().cpu(),
                    denominator=denominator.detach().cpu(),
                    phi_max=phi_max,
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
            f"No degree pair achieved target error {target_error:.2e} for D_{k}"
        )

    return best_result


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Main driver for computing Padé approximations for all dihedral groups."""
    torch.manual_seed(0)

    print("\n" + "=" * 80)
    print("  PADÉ APPROXIMATION FOR SO(3)/D_k INVERSE CDF")
    print("=" * 80)
    print(f"  Target Supremum Error: {TARGET_SUP_ERROR:.2e}")
    print(f"  Training Points:       {N_TRAIN:,}")
    print(f"  Evaluation Points:     {N_EVAL:,}")
    print(f"  Ground Truth Grid:     {N_PHI_GRID:,}")
    print("=" * 80)

    # Degree ladder for continuation-based optimization
    DEGREE_LADDER = [
        # (2, 2), (2, 1), (2, 0),
        # (3, 3), (3, 2), (3, 1), (3, 0),
        # (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
        # (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0),
        # (6, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0),
        # (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (7, 0),
        # (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0),
        # (9, 9), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3), (9, 2), (9, 1), (9, 0),
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
        (11, 11),
        (11, 10),
        (11, 9),
        (11, 8),
        (11, 7),
        (11, 6),
        (11, 5),
        (11, 4),
        (11, 3),
        (11, 2),
        (11, 1),
        (11, 0),
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
    ]

    all_coefficients: Dict[int, PadeCoefficients] = {}

    # Process each dihedral group
    for k in DIHEDRAL_GROUPS:
        # Build high-resolution ground truth
        phi_grid, C_grid = build_inverse_cdf(k, n_phi=N_PHI_GRID)

        # Search for optimal degree
        coeffs = search_optimal_degree(
            k, phi_grid, C_grid, DEGREE_LADDER, TARGET_SUP_ERROR
        )
        all_coefficients[k] = coeffs

        # Print selected result
        print(f"  SELECTED for D_{k}:")
        print(
            f"    Degrees:     (n={coeffs.degree_num}, m={coeffs.degree_den})  "
            f"[total = {coeffs.degree_num + coeffs.degree_den}]"
        )
        print(f"    Sup Error:   {coeffs.sup_error:.3e}")
        print(f"    RMS Error:   {coeffs.rms_error:.3e}")
        print(f"    Min |Q|:     {coeffs.min_Q:.3e}")
        print(f"    φ'_max:      {math.degrees(coeffs.phi_max):.6f}°")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    #  OUTPUT RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  FINAL COEFFICIENTS")
    print("=" * 80)

    for k in DIHEDRAL_GROUPS:
        coeffs = all_coefficients[k]
        print(f"\n{'─' * 80}")
        print(
            f"  D_{k}  |  Degrees (n={coeffs.degree_num}, m={coeffs.degree_den})  |  "
            f"Sup Error = {coeffs.sup_error:.3e}"
        )
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
        str(k): {
            "degree_num": coeffs.degree_num,
            "degree_den": coeffs.degree_den,
            "phi_max": coeffs.phi_max,
            "sup_error": coeffs.sup_error,
            "rms_error": coeffs.rms_error,
            "min_Q": coeffs.min_Q,
            "numerator": coeffs.numerator.numpy().tolist(),
            "denominator": coeffs.denominator.numpy().tolist(),
        }
        for k, coeffs in all_coefficients.items()
    }

    output_file = str(Path(__file__).resolve().parent / "coeffs_azim_D.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 80)
    print(f"  Coefficients saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
