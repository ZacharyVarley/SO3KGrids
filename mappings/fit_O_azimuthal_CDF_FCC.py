#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit an asymptotic FCC-truncated-cube source azimuth model for the octahedral
ordered-simplex sector and save coefficients to coeffs_azim_O_TC_FCC.json.

Model
-----
This script fits only the source-side azimuthal forward CDF needed for the map

    truncated cube source  ->  homochoric ball sector  ->  O fundamental zone

The right azimuth piece is exact, and the breakpoint mass is exact. Only the
left mixed piece needs fitting.

Coordinates and geometry
------------------------
Ordered simplex sector in the positive octant:
    0 <= x <= y <= z
with spherical variables
    x = r sin(theta) cos(phi)
    y = r sin(theta) sin(phi)
    z = r cos(theta)

Azimuth domain:
    phi in [pi/4, pi/2]

For a truncated cube with cube face z = a and truncation plane x+y+z = tau,
write
    beta = (tau-a)/a,   tau = a (1+beta)

In the ordered simplex the radial ceiling is
    R(theta,phi) = min( a/cos(theta), tau/(cos(theta)+sin(theta)(cos(phi)+sin(phi))) )

The switch angle is exact:
    theta_* = atan( beta / (cos(phi)+sin(phi)) )

The azimuth break where theta_* = theta_max is exact:
    phi_* = atan( 1 / (beta-1) )

Only the left piece phi in [pi/4, phi_*] has mixed geometry.
The right piece phi in [phi_*, pi/2] is cube-face only and is exact.

Fitted quantity
---------------
Let w_L(t,beta) denote the *local normalized* left-piece forward CDF, where
    t in [-1,1]
    phi = 0.5*(t+1)*(phi_*(beta)-pi/4) + pi/4
and
    w_L(-1,beta)=0,
    w_L( 1,beta)=1.

We fit the endpoint-preserving residual form

    w_L(t,beta) = 0.5*(1+t) + (1-t^2) * sum_{k=0..K} sum_{j=0..J} c[k,j] * (beta-beta0)^k * T_j(t)

with beta0 = sqrt(2) and T_j the Chebyshev polynomials of the first kind.

The exact global source CDF is then reconstructed as

    u_break(beta) = M_left(beta) / C_tot(beta)

    u(phi;beta) =
        u_break(beta) * w_L(t,beta),                          phi <= phi_*(beta)
        u_break(beta) + (1-u_break(beta)) * w_R(phi;beta),   phi >= phi_*(beta)

where the right piece is exact:

    w_R(phi;beta) = 1 - cot(phi)/(beta-1)

and

    C_tot(beta)  = 0.5 - (2-beta)^3 / 12
    M_right(beta)= 0.5 * (beta-1)
    M_left(beta) = C_tot(beta) - M_right(beta)

FCC midpoint snapping
---------------------
For the FCC grid built from indices i,j,k in [-h,h] with parity constraint
(i+j+k) even, occupied {111} planes occur at even sum-indices. The truncation
plane is snapped to the nearest midpoint between adjacent occupied {111} planes,
so in units of the Cartesian step the truncation plane uses the nearest *odd*
index m to
    q = (1+sqrt(2))*(h+0.5)
Hence
    beta(h) = m/(h+0.5) - 1.

Output
------
Saves a JSON file named coeffs_azim_O_TC_FCC.json in the same directory as this
script.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# ============================== CONFIG ==============================
PI = math.pi
BETA0 = math.sqrt(2.0)
PHI_LO = 0.25 * PI
PHI_HI = 0.50 * PI

# The fit is asymptotic in beta -> sqrt(2), so avoid very coarse h.
H_MIN = 24
H_MAX = 512
TRAIN_PARITY = 0  # train on h % 2 == TRAIN_PARITY, validate on the opposite parity

# Dense 1D quadrature grid for the left-piece ground-truth CDF.
N_PHI_LEFT = 20001

# Chebyshev-like t-grid on (-1,1) for fitting and validation.
N_TRAIN_T = 1024
N_EVAL_T = 4096

# Degree search ladder for the left-piece residual model.
T_DEGREES = list(range(6, 19))
DELTA_DEGREES = list(range(0, 5))
RIDGE = 1.0e-28

# The fit is linear, so set the target to something realistic for an offline LS fit.
TARGET_SUP_ERROR = 5.0e-13


# ========================= BASIC GEOMETRY ==========================
def theta_max(phi: np.ndarray) -> np.ndarray:
    return np.arctan(1.0 / np.sin(phi))


def phi_break(beta: float) -> float:
    if beta <= 1.0:
        raise ValueError("beta must exceed 1 for the ordered-simplex truncated cube.")
    return math.atan(1.0 / (beta - 1.0))


def c_tot(beta: float) -> float:
    """Integral of A(phi) over the ordered-simplex sector.

    Note: this is 3 * (sector volume), because A(phi) = ∫ R^3 sin(theta) dtheta.
    """
    return 0.5 - (2.0 - beta) ** 3 / 12.0


def right_mass(beta: float) -> float:
    return 0.5 * (beta - 1.0)


def left_mass(beta: float) -> float:
    return c_tot(beta) - right_mass(beta)


def u_break(beta: float) -> float:
    return left_mass(beta) / c_tot(beta)


# ===================== FCC BETA MIDPOINT RULE ======================
def nearest_odd_integer(x: float) -> int:
    m = int(round(x))
    if m % 2 != 0:
        return m
    lo = m - 1
    hi = m + 1
    return lo if abs(x - lo) <= abs(x - hi) else hi


def beta_fcc_midpoint(h: int) -> float:
    if h < 1:
        raise ValueError("h must be >= 1.")
    q = (1.0 + math.sqrt(2.0)) * (h + 0.5)
    m = nearest_odd_integer(q)
    return m / (h + 0.5) - 1.0


# ================== EXACT SOURCE-SIDE LEFT PIECE ===================
def A_left(phi: np.ndarray, beta: float) -> np.ndarray:
    """Exact left-piece column density A(phi) on phi in [pi/4, phi_break(beta)].

    Scale a has been normalized to 1. The normalized forward CDF is independent
    of the cube scale.
    """
    tau = 1.0 + beta
    s = np.cos(phi) + np.sin(phi)
    t_sw = beta / s
    t_hi = 1.0 / np.sin(phi)

    st_hi = s * t_hi
    st_sw = s * t_sw

    # Antiderivative of t / (1 + s t)^3 dt:
    # F(t) = [ -1/(1+st) + 1/(2(1+st)^2) ] / s^2
    F_hi = (-1.0 / (1.0 + st_hi) + 0.5 / (1.0 + st_hi) ** 2) / (s * s)
    F_sw = (-1.0 / (1.0 + st_sw) + 0.5 / (1.0 + st_sw) ** 2) / (s * s)

    return 0.5 * t_sw * t_sw + tau**3 * (F_hi - F_sw)


def build_left_local_cdf_table(
    beta: float, n_phi: int = N_PHI_LEFT
) -> tuple[np.ndarray, np.ndarray]:
    """Return phi-grid and normalized local left-piece CDF on [pi/4, phi_break(beta)]."""
    phib = phi_break(beta)

    # Cosine-clustered grid for better endpoint resolution.
    s = np.linspace(0.0, 1.0, n_phi)
    phi_grid = 0.5 * (1.0 - np.cos(PI * s)) * (phib - PHI_LO) + PHI_LO

    A = A_left(phi_grid, beta)
    dphi = np.diff(phi_grid)
    C = np.empty_like(phi_grid)
    C[0] = 0.0
    C[1:] = np.cumsum(0.5 * (A[1:] + A[:-1]) * dphi)
    C /= C[-1]
    return phi_grid, C


def eval_left_local_cdf(
    beta: float, t: np.ndarray, phi_grid: np.ndarray, C_grid: np.ndarray
) -> np.ndarray:
    phib = phi_break(beta)
    phi = 0.5 * (t + 1.0) * (phib - PHI_LO) + PHI_LO
    return np.interp(phi, phi_grid, C_grid)


# ========================= RIGHT PIECE EXACT =======================
def right_local_cdf_exact(phi: np.ndarray, beta: float) -> np.ndarray:
    return 1.0 - 1.0 / ((beta - 1.0) * np.tan(phi))


def global_forward_cdf_exact(
    phi: np.ndarray,
    beta: float,
    left_phi_grid: np.ndarray,
    left_C_grid: np.ndarray,
) -> np.ndarray:
    phib = phi_break(beta)
    ub = u_break(beta)
    out = np.empty_like(phi)
    mask_left = phi <= phib
    if np.any(mask_left):
        t = 2.0 * (phi[mask_left] - PHI_LO) / (phib - PHI_LO) - 1.0
        out[mask_left] = ub * eval_left_local_cdf(beta, t, left_phi_grid, left_C_grid)
    if np.any(~mask_left):
        wR = right_local_cdf_exact(phi[~mask_left], beta)
        out[~mask_left] = ub + (1.0 - ub) * wR
    return out


# ======================== CHEBYSHEV HELPERS ========================
def chebyshev_T_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    T = np.empty((x.size, degree + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if degree >= 1:
        T[:, 1] = x
    for j in range(1, degree):
        T[:, j + 1] = 2.0 * x * T[:, j] - T[:, j - 1]
    return T


def chebyshev_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate sum_j coeffs[j] T_j(x) by Clenshaw."""
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be 1D")
    n = coeffs.size - 1
    if n == 0:
        return np.full_like(x, coeffs[0], dtype=np.float64)
    b1 = np.zeros_like(x, dtype=np.float64)
    b2 = np.zeros_like(x, dtype=np.float64)
    for k in range(n, 0, -1):
        b0 = 2.0 * x * b1 - b2 + coeffs[k]
        b2, b1 = b1, b0
    return x * b1 - b2 + coeffs[0]


# ====================== FIT MODEL AND EVALUATOR =====================
@dataclass
class LeftModel:
    t_degree: int
    delta_degree: int
    coeffs: np.ndarray  # shape (delta_degree+1, t_degree+1)
    beta_center: float = BETA0

    def eval_local(self, beta: float, t: np.ndarray) -> np.ndarray:
        delta = beta - self.beta_center
        z = np.zeros_like(t, dtype=np.float64)
        for k in range(self.delta_degree + 1):
            z += (delta**k) * chebyshev_eval(self.coeffs[k], t)
        w = 0.5 * (1.0 + t) + (1.0 - t * t) * z
        # exact endpoint repair against roundoff
        w = np.clip(w, 0.0, 1.0)
        return w

    def eval_global(self, beta: float, phi: np.ndarray) -> np.ndarray:
        phib = phi_break(beta)
        ub = u_break(beta)
        out = np.empty_like(phi, dtype=np.float64)
        mask_left = phi <= phib
        if np.any(mask_left):
            t = 2.0 * (phi[mask_left] - PHI_LO) / (phib - PHI_LO) - 1.0
            out[mask_left] = ub * self.eval_local(beta, t)
        if np.any(~mask_left):
            wR = right_local_cdf_exact(phi[~mask_left], beta)
            out[~mask_left] = ub + (1.0 - ub) * wR
        return out


def build_h_sets() -> tuple[list[int], list[int]]:
    hs = list(range(H_MIN, H_MAX + 1))
    train = [h for h in hs if (h % 2) == TRAIN_PARITY]
    valid = [h for h in hs if (h % 2) != TRAIN_PARITY]
    return train, valid


def fit_left_model(
    train_h: Iterable[int], t_degree: int, delta_degree: int, t_fit: np.ndarray
) -> LeftModel:
    Tt = chebyshev_T_matrix(t_fit, t_degree)
    basis_count = (delta_degree + 1) * (t_degree + 1)
    H = np.zeros((basis_count, basis_count), dtype=np.float64)
    g = np.zeros((basis_count,), dtype=np.float64)
    denom = 1.0 - t_fit * t_fit

    for h in train_h:
        beta = beta_fcc_midpoint(h)
        phi_grid, C_grid = build_left_local_cdf_table(beta)
        w = eval_left_local_cdf(beta, t_fit, phi_grid, C_grid)
        y = (w - 0.5 * (1.0 + t_fit)) / denom

        delta = beta - BETA0
        X = np.concatenate([Tt * (delta**k) for k in range(delta_degree + 1)], axis=1)
        H += X.T @ X
        g += X.T @ y

    H.flat[:: H.shape[0] + 1] += RIDGE
    coef = np.linalg.solve(H, g).reshape(delta_degree + 1, t_degree + 1)
    return LeftModel(t_degree=t_degree, delta_degree=delta_degree, coeffs=coef)


# ========================= VALIDATION HELPERS =======================
def validate_model(
    model: LeftModel, valid_h: Iterable[int], t_eval: np.ndarray
) -> dict[str, float]:
    sup_local = 0.0
    rms_local_acc = 0.0
    n_local = 0

    sup_global = 0.0
    rms_global_acc = 0.0
    n_global = 0

    min_local_slope = math.inf

    for h in valid_h:
        beta = beta_fcc_midpoint(h)

        # Left local piece
        phi_grid, C_grid = build_left_local_cdf_table(beta)
        w_true = eval_left_local_cdf(beta, t_eval, phi_grid, C_grid)
        w_fit = model.eval_local(beta, t_eval)
        err_local = np.abs(w_fit - w_true)
        sup_local = max(sup_local, float(err_local.max()))
        rms_local_acc += float(np.sum(err_local * err_local))
        n_local += err_local.size
        local_slope = np.min(np.diff(w_fit))
        min_local_slope = min(min_local_slope, float(local_slope))

        # Global CDF on a dense phi grid across both pieces
        phi_dense = np.linspace(PHI_LO, PHI_HI, t_eval.size)
        u_true = global_forward_cdf_exact(phi_dense, beta, phi_grid, C_grid)
        u_fit = model.eval_global(beta, phi_dense)
        err_global = np.abs(u_fit - u_true)
        sup_global = max(sup_global, float(err_global.max()))
        rms_global_acc += float(np.sum(err_global * err_global))
        n_global += err_global.size

    return {
        "sup_local_error": sup_local,
        "rms_local_error": math.sqrt(rms_local_acc / max(1, n_local)),
        "sup_global_error": sup_global,
        "rms_global_error": math.sqrt(rms_global_acc / max(1, n_global)),
        "min_forward_difference_on_eval_grid": min_local_slope,
    }


# ============================= SEARCH ==============================
def search_best_model() -> tuple[LeftModel, dict[str, float]]:
    train_h, valid_h = build_h_sets()

    t_fit = np.cos(PI * (np.arange(N_TRAIN_T, dtype=np.float64) + 0.5) / N_TRAIN_T)
    t_eval = np.cos(PI * (np.arange(N_EVAL_T, dtype=np.float64) + 0.5) / N_EVAL_T)

    best_model: LeftModel | None = None
    best_stats: dict[str, float] | None = None
    best_score = math.inf

    print("=" * 88)
    print("  ASYMPTOTIC FCC TRUNCATED-CUBE SOURCE AZIMUTH FIT FOR O")
    print("=" * 88)
    print(f"  h-range:                 [{H_MIN}, {H_MAX}]")
    print(f"  train parity:            h % 2 == {TRAIN_PARITY}")
    print(
        f"  train/valid counts:      {len(build_h_sets()[0])} / {len(build_h_sets()[1])}"
    )
    print(f"  left phi grid:           {N_PHI_LEFT:,}")
    print(f"  t-grid fit/eval:         {N_TRAIN_T:,} / {N_EVAL_T:,}")
    print(f"  target sup error:        {TARGET_SUP_ERROR:.2e}")
    print("=" * 88)
    print(
        f"{'(J,K)':>10}  {'#coef':>6}  {'sup_local':>12}  {'sup_global':>12}  {'rms_global':>12}  {'min Δw':>12}"
    )
    print("-" * 88)

    for J in T_DEGREES:
        for K in DELTA_DEGREES:
            model = fit_left_model(train_h, t_degree=J, delta_degree=K, t_fit=t_fit)
            stats = validate_model(model, valid_h, t_eval)
            ncoef = (J + 1) * (K + 1)
            print(
                f"{str((J, K)):>10}  {ncoef:6d}  {stats['sup_local_error']:12.3e}  "
                f"{stats['sup_global_error']:12.3e}  {stats['rms_global_error']:12.3e}  "
                f"{stats['min_forward_difference_on_eval_grid']:12.3e}"
            )

            score = stats["sup_global_error"]
            if score < best_score:
                best_score = score
                best_model = model
                best_stats = stats

            if stats["sup_global_error"] <= TARGET_SUP_ERROR:
                return model, stats

    if best_model is None or best_stats is None:
        raise RuntimeError("degree search failed to produce any model")
    return best_model, best_stats


# ============================== SAVE ===============================
def save_json(model: LeftModel, stats: dict[str, float], out_file: Path) -> None:
    train_h, valid_h = build_h_sets()
    train_betas = [beta_fcc_midpoint(h) for h in train_h]
    valid_betas = [beta_fcc_midpoint(h) for h in valid_h]

    data = {
        "O_TC_FCC": {
            "fit_type": "piecewise source forward CDF",
            "description": "Asymptotic FCC-truncated-cube source azimuth fit for truncated-cube -> ball-sector transport. Right piece and breakpoint mass are exact; only the left mixed piece is fitted.",
            "phi_min": PHI_LO,
            "phi_max": PHI_HI,
            "beta_center": BETA0,
            "beta_formula": "beta(h) = m/(h+0.5) - 1, where m is the nearest odd integer to (1+sqrt(2))*(h+0.5)",
            "phi_break_formula": "phi_break(beta) = atan(1/(beta-1))",
            "c_total_formula": "C_tot(beta) = 0.5 - (2-beta)^3 / 12",
            "u_break_formula": "u_break(beta) = (C_tot(beta) - 0.5*(beta-1)) / C_tot(beta)",
            "right_piece_formula": "w_R(phi,beta) = 1 - cot(phi)/(beta-1)",
            "left_piece": {
                "fit_basis": "w_L(t,beta) = 0.5*(1+t) + (1-t^2) * sum_{k=0..K} sum_{j=0..J} c[k,j] * (beta-beta0)^k * T_j(t)",
                "t_degree": model.t_degree,
                "delta_degree": model.delta_degree,
                "coefficients": model.coeffs.tolist(),
            },
            "training": {
                "h_min": H_MIN,
                "h_max": H_MAX,
                "train_parity": TRAIN_PARITY,
                "n_phi_left": N_PHI_LEFT,
                "n_train_t": N_TRAIN_T,
                "n_eval_t": N_EVAL_T,
                "train_h_count": len(train_h),
                "valid_h_count": len(valid_h),
                "train_beta_min": min(train_betas),
                "train_beta_max": max(train_betas),
                "valid_beta_min": min(valid_betas),
                "valid_beta_max": max(valid_betas),
            },
            "validation": stats,
        }
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================== MAIN ===============================
def main() -> None:
    model, stats = search_best_model()

    print("-" * 88)
    print(
        f"Selected model: J={model.t_degree}, K={model.delta_degree}, "
        f"sup_global={stats['sup_global_error']:.3e}, rms_global={stats['rms_global_error']:.3e}, "
        f"min Δw={stats['min_forward_difference_on_eval_grid']:.3e}"
    )
    print("-" * 88)

    out_file = Path(__file__).resolve().parent / "coeffs_azim_O_TC_FCC.json"
    save_json(model, stats, out_file)
    print(f"Saved coefficients to: {out_file}")


if __name__ == "__main__":
    main()
