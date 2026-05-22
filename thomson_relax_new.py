#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modification of Thomson relaxation on S^3 from:

Quey, Romain, et al. “Nearly Uniform Sampling of Crystal Orientations.” Journal
of Applied Crystallography, vol. 51, no. 4, Aug. 2018, pp. 1162–73. DOI.org
(Crossref), https://doi.org/10.1107/S1600576718009019.

Additional features:
  - Geodesic (Lie-group) update on S^3: q <- exp(Δ) ⊗ q
  - Per-point trust region / step clipping in angular units: ||Δ|| <= κ d_r
  - Optional diagonal curvature preconditioner from the same all-pairs pass:
        h_i ≈ Σ (1/d^3 + 1/(π-d)^3), then Δ_imag <- Δ_imag / (h_scaled + λ)
  - Stopping by high-percentile angular movement with patience:
        stop if step_q(quantile) < threshold for `stop_patience` consecutive
        iters (default 5)
  - Barzilai–Borwein step with selectable alpha strategy:
        bb_select="alternate" (original) or bb_select="energy" (evaluate trial
        energies on both and pick the smaller one)
"""


import math
import time
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from pykeops.torch import LazyTensor

from covering_radius import covering_radius, covering_radius_naive
from laue_ops import laue_elements, ori_to_fz_laue
from orientation_ops import qu_std, qu_norm
from grid_SO3 import so3_super_fibonacci
from grid_FZ import cu_rej_grid, kr_sample_laue
from covering_radius import covering_radius_star_deg


# ------------------------------
# Quaternion basics
# ------------------------------
def ensure_unit(q: Tensor, eps: float = 1e-12) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    # (w, x, y, z)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((w, x, y, z), dim=-1)


def is_identity_op(op: Tensor, tol: float = 1e-6) -> bool:
    vec_ok = op[..., 1:].abs().max().item() < tol
    w = float(op[..., 0].item())
    return vec_ok and (abs(abs(w) - 1.0) < tol)


def quat_exp_pure_imag(delta: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Exponential map from Lie algebra (pure imaginary quaternions) to S^3.
      exp([0,v]) = [cos(|v|), (v/|v|) sin(|v|)].
    """
    v = delta[:, 1:4]
    s = v.norm(dim=-1, keepdim=True)  # (N,1)
    w = torch.cos(s)

    sin_s = torch.sin(s)
    scale = torch.empty_like(s)
    small = s < 1e-4
    s2 = s * s
    # sin s / s ≈ 1 - s^2/6 + s^4/120
    scale[small] = 1.0 - s2[small] / 6.0 + (s2[small] * s2[small]) / 120.0
    scale[~small] = sin_s[~small] / s[~small].clamp_min(eps)

    xyz = v * scale
    out = torch.cat([w, xyz], dim=-1)
    return ensure_unit(out)


# ------------------------------
# S^3 geometry from the paper (§2.3)
# ------------------------------
def s3_cap_area(theta: float) -> float:
    # Eq. (18): S = π(θ - sin θ)
    return math.pi * (theta - math.sin(theta))


def solve_theta_from_area(S: float, iters: int = 60) -> float:
    lo, hi = 0.0, math.pi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Smid = s3_cap_area(mid)
        if Smid < S:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def average_orientation_radius_dr(N_star: int) -> float:
    # Eq. (17): S = π^2 / N*
    # Eq. (18): S = π(θ_r - sin θ_r), then d_r = θ_r / 2
    S = (math.pi**2) / float(N_star)
    theta_r = solve_theta_from_area(S)
    return 0.5 * theta_r


# ------------------------------
# KeOps helpers
# ------------------------------
def clamp_c_to_unit_interval(c: LazyTensor) -> LazyTensor:
    # KeOps-safe clamp to [-1, 1] using only relu:
    c = 1.0 - (1.0 - c).relu()
    c = (1.0 + c).relu() - 1.0
    return c


def offdiag_mask_full(N: int, device: torch.device) -> LazyTensor:
    idx = torch.arange(N, device=device, dtype=torch.float32)
    I = LazyTensor(idx[:, None, None])
    J = LazyTensor(idx[None, :, None])
    ondiag = 1.0 - ((I - J).abs() > 0)
    return 1.0 - ondiag


def offdiag_mask_subset(subset_idx: Tensor, N: int) -> LazyTensor:
    # subset_idx: (M,) int64
    device = subset_idx.device
    idx_i = subset_idx.to(torch.float32)
    idx_j = torch.arange(N, device=device, dtype=torch.float32)
    I = LazyTensor(idx_i[:, None, None])
    J = LazyTensor(idx_j[None, :, None])
    ondiag = 1.0 - ((I - J).abs() > 0)
    return 1.0 - ondiag


# ------------------------------
# Forces: paper Eq. (14) all-pairs + optional curvature proxy
# ------------------------------
@torch.no_grad()
def forces_eq14_allpairs(
    q: Tensor,
    ops: Tensor,
    *,
    eps: float = 1e-9,
    return_curvature: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """
    Returns:
      f : (N,4) pure imaginary quaternions (real part 0), paper Eq. (14)
      h : (N,1) optional curvature proxy h_i ≈ Σ (1/d^3 + 1/(π-d)^3)
    """
    device = q.device
    q = ensure_unit(q.to(torch.float32))
    ops = ensure_unit(ops.to(torch.float32))

    N = q.shape[0]
    Xi = LazyTensor(q[:, None, :])  # (N,1,4)
    off = offdiag_mask_full(N, device)  # (N,N,1)

    wi = Xi[:, :, 0]
    vix = Xi[:, :, 1]
    viy = Xi[:, :, 2]
    viz = Xi[:, :, 3]

    Fx = torch.zeros((N, 1), device=device, dtype=torch.float32)
    Fy = torch.zeros((N, 1), device=device, dtype=torch.float32)
    Fz = torch.zeros((N, 1), device=device, dtype=torch.float32)
    H = (
        torch.zeros((N, 1), device=device, dtype=torch.float32)
        if return_curvature
        else None
    )

    pi = math.pi

    for k in range(ops.shape[0]):
        y = ensure_unit(quat_mul(q, ops[k]))  # q_j^k
        Yj = LazyTensor(y[None, :, :])  # (1,N,4)

        wj = Yj[:, :, 0]
        vjx = Yj[:, :, 1]
        vjy = Yj[:, :, 2]
        vjz = Yj[:, :, 3]

        c = (Xi * Yj).sum(dim=2)
        c = clamp_c_to_unit_interval(c)
        d = c.acos()

        s = (1.0 - c * c).relu().sqrt() + eps

        d2 = d * d + eps
        dp = pi - d
        dp2 = dp * dp + eps

        fmag = (1.0 / d2) - (1.0 / dp2)

        # rel_v = -wj*vi + wi*vj + (vi x vj)
        cx = viy * vjz - viz * vjy
        cy = viz * vjx - vix * vjz
        cz = vix * vjy - viy * vjx
        relx = wi * vjx - wj * vix + cx
        rely = wi * vjy - wj * viy + cy
        relz = wi * vjz - wj * viz + cz

        rx = relx / s
        ry = rely / s
        rz = relz / s

        Fx = Fx + (off * (-fmag * rx)).sum(dim=1)
        Fy = Fy + (off * (-fmag * ry)).sum(dim=1)
        Fz = Fz + (off * (-fmag * rz)).sum(dim=1)

        if return_curvature:
            d3 = d2 * d + eps
            dp3 = dp2 * dp + eps
            hmag = (1.0 / d3) + (1.0 / dp3)
            H = H + (off * hmag).sum(dim=1)

    f = torch.cat(
        [torch.zeros((N, 1), device=device, dtype=torch.float32), Fx, Fy, Fz], dim=1
    )
    f[:, 0] = 0.0
    return f, H


# ------------------------------
# Energy for alpha selection: Φ(d) = 1/d + 1/(π-d)
# ------------------------------
@torch.no_grad()
def energy_phi_allpairs(
    q: Tensor,
    ops: Tensor,
    *,
    subset_idx: Tensor,
    eps: float = 1e-9,
) -> float:
    """
    Approximate energy on a subset of i's against all j's and all symmetry ops:
      E = Σ_{i in subset} Σ_{j≠i} Σ_k [ 1/d + 1/(π-d) ].
    """
    device = q.device
    q = ensure_unit(q.to(torch.float32))
    ops = ensure_unit(ops.to(torch.float32))

    N = q.shape[0]
    qs = q[subset_idx]  # (M,4)
    Xi = LazyTensor(qs[:, None, :])  # (M,1,4)
    off = offdiag_mask_subset(subset_idx, N)  # (M,N,1)

    pi = math.pi
    total = 0.0

    for k in range(ops.shape[0]):
        y = ensure_unit(quat_mul(q, ops[k]))  # (N,4)
        Yj = LazyTensor(y[None, :, :])  # (1,N,4)
        c = (Xi * Yj).sum(dim=2)
        c = clamp_c_to_unit_interval(c)
        d = c.acos()

        inv_d = 1.0 / (d + eps)
        inv_dp = 1.0 / ((pi - d) + eps)
        phi = inv_d + inv_dp

        Ei = (off * phi).sum(dim=1)  # (M,1)
        total = total + float(Ei.sum().item())

    return total


# ------------------------------
# Relaxation loop (all-pairs) with patience + optional energy-based BB selection
# ------------------------------
@torch.no_grad()
def relax_orientations_allpairs(
    q0: Tensor,
    ops: Tensor,
    *,
    max_iters: int = 200,
    # Paper step init
    beta: float = 0.8,
    gamma: float = 3.4,
    # BB step bounds
    alpha_min: float = 1e-12,
    alpha_max: float | None = None,
    # Trust-region clipping: ||Δ|| <= κ d_r
    step_clip_kappa: float = 0.5,
    # Preconditioning
    use_precond: bool = True,
    precond_lambda: float = 1e-2,
    precond_sample: int = 65536,
    # Stopping on high-percentile angular movement
    stop_quantile: float = 0.99,
    stop_angle_deg: float = 0.01,
    stop_sample: int = 65536,
    stop_patience: int = 5,
    # BB selection strategy
    bb_select: str = "alternate",  # "alternate" (paper) or "energy"
    bb_energy_sample: int = 1024,  # subset size for energy evaluation (only if bb_select="energy")
    # Optional fixed points
    fixed_mask: Tensor | None = None,
    # Optional projection hook
    project_fn=None,
    # Optional callback: (q: Tensor, iter_num: int) -> None. Called after each update.
    callback=None,
    # Numerics / logging
    eps: float = 1e-9,
    verbose: bool = True,
    log_every: int = 1,
):
    assert q0.ndim == 2 and q0.shape[1] == 4
    assert ops.ndim == 2 and ops.shape[1] == 4
    assert is_identity_op(
        ops[0].to(torch.float32)
    ), "ops[0] must be identity (±[1,0,0,0])."

    device = q0.device
    q = ensure_unit(q0.to(torch.float32))
    ops = ensure_unit(ops.to(torch.float32))

    N, n_c = q.shape[0], ops.shape[0]
    N_star = N * n_c
    d_r = average_orientation_radius_dr(N_star)

    alpha0 = (beta / gamma) * (d_r**3)  # paper Eq. (25)
    if alpha_max is None:
        alpha_max = max(1.0, 10.0 * d_r)

    if fixed_mask is not None:
        fixed_mask = fixed_mask.to(device).bool()

    pi = math.pi
    deg_per_rad = 180.0 / pi

    # Fixed subsamples for stability (quantiles / medians / energy selection)
    def fixed_subsample(n: int, m: int) -> Tensor:
        m = int(min(n, max(1, m)))
        return torch.randint(low=0, high=n, size=(m,), device=device)

    idx_precond = fixed_subsample(N, precond_sample)
    idx_stop = fixed_subsample(N, stop_sample)
    idx_energy = fixed_subsample(N, bb_energy_sample) if bb_select == "energy" else None

    # BB storage
    f_prev = None
    Delta_prev = None  # previous applied Δ (post-precond, post-clip), pure imaginary

    # Patience counter
    good_steps = 0

    # History
    hist = {
        "alpha": [],
        "alpha_mode": [],
        "force_rms": [],
        "rho_dimless": [],
        "step_q_deg": [],
        "step_max_deg": [],
        "good_steps": [],
        "iter_time_s": [],
    }

    for l in range(max_iters):
        t0 = time.time()

        # --- Forces + curvature proxy ---
        f, H = forces_eq14_allpairs(q, ops, eps=eps, return_curvature=use_precond)

        if fixed_mask is not None:
            f = f.clone()
            f[fixed_mask] = 0.0
            if H is not None:
                H = H.clone()
                H[fixed_mask] = 0.0

        f[:, 0] = 0.0
        fim = f[:, 1:]
        force_rms = torch.sqrt((fim * fim).sum(dim=1).mean()).item()
        rho_dimless = (float(d_r) ** 2 / float(gamma)) * force_rms

        # --- Build preconditioner denominator once per iter (independent of alpha) ---
        denom_precond = None
        if use_precond:
            assert H is not None
            Hc = H.clamp_min(0.0)
            H_med = torch.quantile(Hc[idx_precond, 0], 0.5).clamp_min(1e-12)
            H_scaled = Hc / H_med
            denom_precond = (H_scaled + float(precond_lambda)).clamp_min(1e-6)  # (N,1)

        # --- Compute BB candidates if l>=1 ---
        alpha_mode = "alpha0"
        if l == 0:
            alpha = float(alpha0)
        else:
            Delta_f = f - f_prev
            s = Delta_prev[:, 1:]  # (N,3)
            y = Delta_f[:, 1:]  # (N,3)

            sTy = (s * y).sum()
            yTy = (y * y).sum()
            sTs = (s * s).sum()

            # raw BB candidates
            if yTy.abs() > 1e-20 and torch.isfinite(yTy) and torch.isfinite(sTy):
                alpha1 = (sTy / yTy).abs()
            else:
                alpha1 = torch.tensor(alpha0, device=device)

            if sTy.abs() > 1e-20 and torch.isfinite(sTy) and torch.isfinite(sTs):
                alpha2 = (sTs / sTy).abs()
            else:
                alpha2 = torch.tensor(alpha0, device=device)

            # clamp
            alpha1 = alpha1.clamp_min(alpha_min).clamp_max(alpha_max)
            alpha2 = alpha2.clamp_min(alpha_min).clamp_max(alpha_max)

            if bb_select == "alternate":
                alpha = float((alpha1 if (l % 2) == 1 else alpha2).item())
                alpha_mode = "bb1" if (l % 2) == 1 else "bb2"
            elif bb_select == "energy":
                # Evaluate both candidates via trial energy on subset
                def make_trial_q(alpha_c: float) -> tuple[Tensor, Tensor]:
                    Delta_c = (alpha_c * f).clone()
                    Delta_c[:, 0] = 0.0
                    if denom_precond is not None:
                        Delta_c[:, 1:] = Delta_c[:, 1:] / denom_precond
                    # clip
                    s_norm = Delta_c[:, 1:].norm(dim=-1, keepdim=True)
                    s_max = float(step_clip_kappa) * float(d_r)
                    if s_max > 0.0:
                        scale = (s_max / s_norm.clamp_min(1e-12)).clamp_max(1.0)
                        Delta_c[:, 1:] = Delta_c[:, 1:] * scale
                    dq = quat_exp_pure_imag(Delta_c)
                    q_trial = ensure_unit(quat_mul(dq, q))
                    return q_trial, Delta_c

                a1 = float(alpha1.item())
                a2 = float(alpha2.item())

                q1, _ = make_trial_q(a1)
                q2, _ = make_trial_q(a2)

                E1 = energy_phi_allpairs(q1, ops, subset_idx=idx_energy, eps=eps)
                E2 = energy_phi_allpairs(q2, ops, subset_idx=idx_energy, eps=eps)

                if E1 <= E2:
                    alpha = a1
                    alpha_mode = "bb1"
                else:
                    alpha = a2
                    alpha_mode = "bb2"
            else:
                raise ValueError("bb_select must be 'alternate' or 'energy'.")

        alpha = max(alpha_min, min(alpha, alpha_max))

        # --- Construct applied Δ with chosen alpha ---
        Delta = (alpha * f).clone()
        Delta[:, 0] = 0.0

        if denom_precond is not None:
            Delta[:, 1:] = Delta[:, 1:] / denom_precond

        # trust-region / step clipping
        s_norm = Delta[:, 1:].norm(dim=-1, keepdim=True)
        s_max = float(step_clip_kappa) * float(d_r)
        if s_max > 0.0:
            scale = (s_max / s_norm.clamp_min(1e-12)).clamp_max(1.0)
            Delta[:, 1:] = Delta[:, 1:] * scale
        Delta[:, 0] = 0.0

        # --- Geodesic update q <- exp(Δ) ⊗ q ---
        dq = quat_exp_pure_imag(Delta)
        q = ensure_unit(quat_mul(dq, q))

        if project_fn is not None:
            q = ensure_unit(project_fn(q))

        if callback is not None:
            callback(q, l)

        # --- BB storage uses applied Δ ---
        f_prev = f
        Delta_prev = Delta

        # --- Movement metrics + patience stopping ---
        step_half = Delta[:, 1:].norm(dim=-1)
        step_deg = 2.0 * step_half * deg_per_rad

        step_q = float(torch.quantile(step_deg[idx_stop], float(stop_quantile)).item())
        step_max = float(step_deg.max().item())

        if step_q < float(stop_angle_deg):
            good_steps += 1
        else:
            good_steps = 0

        it_time = time.time() - t0

        hist["alpha"].append(alpha)
        hist["alpha_mode"].append(alpha_mode)
        hist["force_rms"].append(force_rms)
        hist["rho_dimless"].append(rho_dimless)
        hist["step_q_deg"].append(step_q)
        hist["step_max_deg"].append(step_max)
        hist["good_steps"].append(good_steps)
        hist["iter_time_s"].append(it_time)

        if verbose and (l % max(1, int(log_every)) == 0):
            print(
                f"l={l:5d}|"
                f"step_q({stop_quantile:.2f})={step_q: .3e}°|"
                f"f_rms={force_rms: .3e} | rho={rho_dimless: .3e}|"
                f"alpha={alpha: .3e}({alpha_mode})|good={good_steps}/{stop_patience}|"
                f"dt={it_time*1e3:.1f} ms"
            )

        if good_steps >= int(stop_patience):
            break

    return q, hist


# ------------------------------
# Minimal runnable example
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    laue_id = 10
    grid_half_edge = 40
    ops = laue_elements(laue_id).to(device=device, dtype=torch.float32)

    # # Initialization: cubochoric grid KR mapped to the FZ
    # q0 = cu_rej_grid(grid_half_edge, 1, device=device).to(torch.float32)
    # q0 = kr_sample_laue(q0, laue_id) # map all points to the FZ via KR mapping

    N = 10000
    q0 = so3_super_fibonacci(N, device=device).to(torch.float32)
    q0 = qu_norm(q0)
    q0 = qu_std(q0)

    # q0 = ensure_unit(torch.randn(N, 4, device=device, dtype=torch.float32))
    n_fz = q0.shape[0]
    cr_ideal = covering_radius_star_deg(n_fz * 2 * ops.shape[0])
    # cr_start = covering_radius_naive(q0, laue_id) * (180.0 / math.pi)
    cr_start_efficient = covering_radius(q0, laue_id) * (180.0 / math.pi)
    cr_start = cr_start_efficient

    # print lower bound of covering radius
    print(f"Number of quaternions: {n_fz}")
    print(f"Lower bound of covering radius: {cr_ideal:.6f}°")
    print(f"Covering radius start: {cr_start:.6f}°")

    q, hist = relax_orientations_allpairs(
        q0,
        ops,
        max_iters=100,
        beta=0.1,  # 0.8 in paper was for Random initializations
        gamma=3.4,  # 3.4 in paper was for Random initializations
        stop_quantile=0.99,
        stop_angle_deg=1e-4,
        stop_patience=10,
        use_precond=True,
        step_clip_kappa=0.5,
        verbose=True,
        log_every=10,
    )

    # some of the points may have drifted out of the FZ, so we send back
    q = ori_to_fz_laue(q, laue_id)
    # cr_final = covering_radius_naive(q, laue_id) * (180.0 / math.pi)
    cr_final_efficient = covering_radius(q, laue_id) * (180.0 / math.pi)
    cr_final = cr_final_efficient

    # print
    print(f"Number of quaternions: {q.shape[0]}")
    print(f"Lower bound of covering radius: {cr_ideal:.6f}°")
    print(f"Covering radius start naive: {cr_start:.6f}°")
    print(f"                  efficient: {cr_start_efficient:.6f}°")
    print(f"Covering radius final naive: {cr_final:.6f}°")
    print(f"                  efficient: {cr_final_efficient:.6f}°")
