#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riesz energy computation module. We fuse into a single reduction over the Laue
group operators which dramatically reduces the memory footprint and computation
time versus naively enumerating all symmetrically equivalent orientations and
their antipodes.

"""

from typing import Tuple, Union
import torch
from torch import Tensor
from pykeops.torch import LazyTensor
from src.laue_ops import laue_elements
from src.orientation_ops import qu_prod
import math


# ------------------------------
# Helpers
# ------------------------------
def _ensure_unit_quats(q: Tensor, eps: float = 1e-12) -> Tensor:
    n = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q / n


def _is_identity_op(op: Tensor, tol: float = 1e-6) -> bool:
    # Treat both +[1,0,0,0] and -[1,0,0,0] as identity rotation.
    return (op[..., 1:].abs().max() < tol) and (op[..., 0].abs() > 1.0 - tol)


def optimal_constants_S3(N_eff: int) -> Tuple[float, float, float]:
    """First order continuum asymptotics on S^3."""
    E1_opt = 8.0 * N_eff * N_eff / (3.0 * math.pi)
    E2_opt = float(N_eff * N_eff)
    E3_opt = (2.0 * N_eff * N_eff * math.log(max(N_eff, 2))) / (3.0 * math.pi)
    return float(E1_opt), float(E2_opt), float(E3_opt)


# ==========================================================
# 1) Naive (explicitly unfolded) evaluator
# ==========================================================
@torch.no_grad()
def riesz_energies_naive_unfolded(
    q_fz: Tensor,
    ops: Tensor,
    eps: float = 1e-9,
    return_contrib: bool = False,
) -> Union[
    Tuple[float, float, float], Tuple[float, float, float, Tensor, Tensor, Tensor]
]:
    """
    Explicit (group ⊗ antipodes) unfolding with a single KeOps reduction
    on the full matrix; energies are computed via row-sums with the diagonal
    removed. Per-input contributions are obtained by folding row-sums from
    the unfolded set back onto the original N inputs.

    Args
    ----
    q_fz : [N,4] unit quaternions (float32/64). Recommend float64 inputs.
    ops  : [G,4] unit quaternions (Laue ops). Identity must be present
           (either +[1,0,0,0] or -[1,0,0,0]).
    eps  : small positive epsilon for fp32 KeOps stability.
    return_contrib : if True, also return (S1_i, S2_i, S3_i) of length N.

    Returns
    -------
    E1, E2, E3  (and optionally S1_i, S2_i, S3_i)
    """
    assert q_fz.ndim == 2 and q_fz.shape[1] == 4, "q_fz must be [N,4]"
    assert ops.ndim == 2 and ops.shape[1] == 4, "ops must be [G,4]"

    device = q_fz.device

    # Normalize / cast
    q_fz64 = _ensure_unit_quats(q_fz.to(device=device, dtype=torch.float64))
    ops64 = _ensure_unit_quats(ops.to(device=device, dtype=torch.float64))

    # check the identity op is the first element
    assert _is_identity_op(ops64[0]), "First op must be identity"

    N = q_fz64.shape[0]
    G = ops64.shape[0]

    # Unfold by group
    q_ops = [qu_prod(q_fz64, ops64[g]) for g in range(G)]  # (G*N, 4) fp64
    Q_cat = torch.cat(q_ops, dim=0)

    # Add antipodes
    Q_full = torch.cat([Q_cat, -Q_cat], dim=0).contiguous()
    Q_full = _ensure_unit_quats(Q_full)

    # KeOps in fp32
    Q = Q_full.to(torch.float32)
    M = Q.shape[0]  # M = 2 * G * N

    Xi, Yj = LazyTensor(Q[:, None, :]), LazyTensor(Q[None, :, :])
    D2 = ((Xi - Yj) * (Xi - Yj)).sum(dim=2) + eps  # chordal^2 + eps

    inv_sqrtD2 = 1.0 / D2.sqrt()
    invD2 = 1.0 / D2
    invD3 = invD2 * inv_sqrtD2

    # Zero the diagonal: we want sum_{j != i} ...
    idx = torch.arange(M, device=Q.device, dtype=torch.float32)
    I, J = LazyTensor(idx[:, None, None]), LazyTensor(idx[None, :, None])
    ondiag = 1.0 - ((I - J).abs() > 0)  # 1 on diag, 0 off
    offdiag = 1.0 - ondiag

    # Row-sums on unfolded set
    S1_row = (offdiag * inv_sqrtD2).sum(dim=1)  # [M,1] LazyTensor → reduced
    S2_row = (offdiag * invD2).sum(dim=1)
    S3_row = (offdiag * invD3).sum(dim=1)

    # Energies are total row-sum across all rows
    E1 = float(S1_row.sum().item())
    E2 = float(S2_row.sum().item())
    E3 = float(S3_row.sum().item())

    if not return_contrib:
        return E1, E2, E3

    # Map unfolded rows back to original base indices and aggregate
    base_idx = torch.arange(M, device=device) % N
    S1_i = torch.zeros(N, device=device, dtype=torch.float32).index_add(
        0, base_idx, S1_row.view(-1)
    )
    S2_i = torch.zeros(N, device=device, dtype=torch.float32).index_add(
        0, base_idx, S2_row.view(-1)
    )
    S3_i = torch.zeros(N, device=device, dtype=torch.float32).index_add(
        0, base_idx, S3_row.view(-1)
    )

    # Cast to fp64 for downstream consistency
    return E1, E2, E3, S1_i, S2_i, S3_i



@torch.no_grad()
def riesz_energies_fused(
    q: Tensor,
    laue_id: int,
    eps: float = 1e-7,
    return_contrib: bool = False,
    return_nn: bool = False,
) -> Union[
    Tuple[float, float, float],
    Tuple[float, float, float, Tensor, Tensor, Tensor],
    Tuple[float, float, float, Tensor],
    Tuple[float, float, float, Tensor, Tensor, Tensor, Tensor],
]:
    """
    Single-reduction KeOps for Riesz energies (s=1,2,3) over Laue ops,
    and nearest-neighbor chordal distance on S^3/{±1} **considering Laue images**.

    For NN: for each op h, compute d_ij^(h) = sqrt(2*(1-|<q_i, q_j h>|)),
    mask true diagonal only for identity, take rowwise min over j,
    then keep the min across all h.

    Args
    ----
    q : [N,4] unit quaternions (float32/64).
    laue_id : laue group of quaternions to compute energies for
    eps  : small positive epsilon for fp32 KeOps stability.
    return_contrib : return per-point (S1_i, S2_i, S3_i) if True.
    return_nn      : return per-point nn_dist (length N) if True.

    Returns
    -------
    E1, E2, E3
    (+ optionally S1_i, S2_i, S3_i)
    (+ optionally nn_dist)
    """
    assert q.ndim == 2 and q.shape[1] == 4, "q must be [N,4]"
    ops = laue_elements(laue_id).to(q)

    device = q.device

    # Normalize / cast
    q64 = _ensure_unit_quats(q.to(device=device, dtype=torch.float64))
    ops64 = _ensure_unit_quats(ops.to(device=device, dtype=torch.float64))

    N = q64.shape[0]
    G = ops64.shape[0]
    factor = 2.0 * float(G)  # accounts for ± and op branches in energy sum

    q32 = q64.to(torch.float32)
    ops32 = ops64.to(torch.float32)

    Xi = LazyTensor(q32[:, None, :])  # (N,1,4)

    # Diagonal detector (1 on diag, 0 off)
    idx = torch.arange(N, device=device, dtype=torch.float32)
    I = LazyTensor(idx[:, None, None])
    J = LazyTensor(idx[None, :, None])
    ondiag = 1.0 - ((I - J).abs() > 0)

    # Energy constants
    k1 = 2.0 ** (-0.5)
    k2 = 0.5
    k3 = 2.0 ** (-1.5)

    S1 = None
    S2 = None
    S3 = None

    # Running best (squared) NN distance across ops
    nn_best_d2 = None
    BIG = 1e9

    for h_idx in range(G):
        op_h = ops32[h_idx]
        Yh = qu_prod(q32, op_h)  # (N,4)
        YH = LazyTensor(Yh[None, :, :])  # (1,N,4)

        # Inner products <q_i, q_j h>
        c = (Xi * YH).sum(dim=2)

        # ---------- Energies ----------
        A = (1.0 - c).relu() + eps
        B = (1.0 + c).relu() + eps

        sqrtA, sqrtB = A.sqrt(), B.sqrt()
        inv_sqrtA, inv_sqrtB = 1.0 / sqrtA, 1.0 / sqrtB
        invA, invB = 1.0 / A, 1.0 / B
        invA_sqrtA, invB_sqrtB = inv_sqrtA * invA, inv_sqrtB * invB

        m_plus = (1.0 - ondiag) if h_idx == 0 else 1.0

        t1 = k1 * (m_plus * inv_sqrtA + inv_sqrtB)  # s=1
        t2 = k2 * (m_plus * invA + invB)  # s=2
        t3 = k3 * (m_plus * invA_sqrtA + invB_sqrtB)  # s=3

        S1 = t1 if S1 is None else (S1 + t1)
        S2 = t2 if S2 is None else (S2 + t2)
        S3 = t3 if S3 is None else (S3 + t3)

        # ---------- Nearest-neighbor distance for this op ----------
        # chordal distance on S^3/{±1}: d^2 = 2 * (1 - |c|)
        if return_nn:
            d2_h = 2.0 * (1.0 - c.abs())
            if h_idx == 0:
                # mask ONLY true diagonal for identity
                d2_h = d2_h + BIG * ondiag
            # rowwise min over j
            nn_d2_h = d2_h.min(dim=1)  # (N,1)
            nn_best_d2 = (
                nn_d2_h if nn_best_d2 is None else torch.minimum(nn_best_d2, nn_d2_h)
            )

    # Row-sums for energies; apply global factor
    S1_row = factor * S1.sum(dim=1)  # (N,1)
    S2_row = factor * S2.sum(dim=1)
    S3_row = factor * S3.sum(dim=1)

    E1 = float(S1_row.sum().item())
    E2 = float(S2_row.sum().item())
    E3 = float(S3_row.sum().item())

    # Final NN distances
    if return_nn:
        nn_dist = nn_best_d2.sqrt().view(-1)  # (N,)
    else:
        nn_dist = None

    # ---------------- Returns ----------------
    if return_contrib and return_nn:
        S1_i = S1_row.view(-1)
        S2_i = S2_row.view(-1)
        S3_i = S3_row.view(-1)
        return E1, E2, E3, S1_i, S2_i, S3_i, nn_dist

    if return_contrib and not return_nn:
        S1_i = S1_row.view(-1)
        S2_i = S2_row.view(-1)
        S3_i = S3_row.view(-1)
        return E1, E2, E3, S1_i, S2_i, S3_i

    if (not return_contrib) and return_nn:
        return E1, E2, E3, nn_dist

    return E1, E2, E3


@torch.no_grad()
def laue_nn_chordal_fused(
    q: Tensor,
    laue_id: int,
    eps: float = 0.0,
) -> Tensor:
    """
    Fast NN-only KeOps reduction over Laue images (no Riesz energies).

    For each i:
      nn_i = min_{h in K, j != i if h=e} sqrt(2 * (1 - |<q_i, q_j h>|))

    Args
    ----
    q : [N,4] unit quaternions (float32/64).
    laue_id : Laue group id.
    eps : optional non-negative offset added to squared distances.

    Returns
    -------
    nn_dist : [N] chordal NN distances on S^3/{±1} considering Laue images.
    """
    assert q.ndim == 2 and q.shape[1] == 4, "q must be [N,4]"
    assert eps >= 0.0, "eps must be non-negative"

    ops = laue_elements(laue_id).to(q)
    device = q.device

    q32 = _ensure_unit_quats(q.to(device=device, dtype=torch.float32))
    ops32 = _ensure_unit_quats(ops.to(device=device, dtype=torch.float32))

    N = q32.shape[0]
    G = ops32.shape[0]

    Xi = LazyTensor(q32[:, None, :])  # (N,1,4)

    idx = torch.arange(N, device=device, dtype=torch.float32)
    I = LazyTensor(idx[:, None, None])
    J = LazyTensor(idx[None, :, None])
    ondiag = 1.0 - ((I - J).abs() > 0)

    nn_best_d2 = None
    BIG = 1e9

    for h_idx in range(G):
        Yh = qu_prod(q32, ops32[h_idx])
        YH = LazyTensor(Yh[None, :, :])  # (1,N,4)
        c = (Xi * YH).sum(dim=2)

        d2_h = 2.0 * (1.0 - c.abs()) + eps
        if h_idx == 0:
            d2_h = d2_h + BIG * ondiag

        nn_d2_h = d2_h.min(dim=1)  # (N,1)
        nn_best_d2 = nn_d2_h if nn_best_d2 is None else torch.minimum(nn_best_d2, nn_d2_h)

    return nn_best_d2.clamp_min(0.0).sqrt().view(-1)


@torch.no_grad()
def laue_cross_nn_chordal_fused(
    q_query: Tensor,
    q_ref: Tensor,
    laue_id: int,
    eps: float = 0.0,
) -> Tensor:
    """
    Fast cross-set NN-only KeOps reduction over Laue images (no energy terms).

    For each query i and ref j, over Laue op h:
      d2_ijh = 2 * (1 - |<q_query_i, q_ref_j h>|) + eps
    then nn_i = min_{h,j} sqrt(d2_ijh).

    Args
    ----
    q_query : [M,4] unit quaternions.
    q_ref   : [N,4] unit quaternions.
    laue_id : Laue group id.
    eps     : optional non-negative offset on squared distances.

    Returns
    -------
    nn_dist : [M] chordal NN distances on S^3/{±1} considering Laue images.
    """
    assert q_query.ndim == 2 and q_query.shape[1] == 4, "q_query must be [M,4]"
    assert q_ref.ndim == 2 and q_ref.shape[1] == 4, "q_ref must be [N,4]"
    assert eps >= 0.0, "eps must be non-negative"

    ops = laue_elements(laue_id).to(q_ref)
    device = q_ref.device

    q_query32 = _ensure_unit_quats(q_query.to(device=device, dtype=torch.float32))
    q_ref32 = _ensure_unit_quats(q_ref.to(device=device, dtype=torch.float32))
    ops32 = _ensure_unit_quats(ops.to(device=device, dtype=torch.float32))

    Xi = LazyTensor(q_query32[:, None, :])  # (M,1,4)
    G = ops32.shape[0]
    nn_best_d2 = None

    for h_idx in range(G):
        Yh = qu_prod(q_ref32, ops32[h_idx])
        YH = LazyTensor(Yh[None, :, :])  # (1,N,4)
        c = (Xi * YH).sum(dim=2)
        d2_h = 2.0 * (1.0 - c.abs()) + eps
        nn_d2_h = d2_h.min(dim=1)  # (M,1)
        nn_best_d2 = nn_d2_h if nn_best_d2 is None else torch.minimum(nn_best_d2, nn_d2_h)

    return nn_best_d2.clamp_min(0.0).sqrt().view(-1)
