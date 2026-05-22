#!/usr/bin/env python3
"""
Generate cached data for the maintained Figure 3 CDF overlays.

Generate cached data for consolidated Figure 3 plotting:
- witness->grid NN overlay data
- self-NN overlay data

This script keeps the datasets separate so existing caches can be reused.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.grid_FZ import cu_kr_grid, cu_rej_grid
from src.laue_ops import laue_elements
from src.laue_ops import ori_in_fz_laue
from src.orientation_ops import qu_prod, qu_std
from src.riesz_energy import laue_cross_nn_chordal_fused, riesz_energies_fused


_HERE = Path(__file__).resolve().parent
WITNESS_DATA_FILE = str(_HERE / "data" / "figure3_witness_nn.npz")
SELF_NN_DATA_FILE = str(_HERE / "data" / "figure3_self_nn.npz")
HIST_BINS_WITNESS = 10_000
HIST_BINS_SELF_NN = 10_000
MAX_WITNESS_FILTER_BATCH = 10_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SELF_NN_DEFAULT_N_EFF = 16_000_000
WITNESS_DEFAULT_N_EFF = 16_000_000

GROUPS = [
    {"name": "C2", "laue_id": 2, "card": 2},
    {"name": "C3", "laue_id": 6, "card": 3},
    {"name": "C4", "laue_id": 4, "card": 4},
    {"name": "C6", "laue_id": 8, "card": 6},
    {"name": "D2", "laue_id": 3, "card": 4},
    {"name": "D3", "laue_id": 7, "card": 6},
    {"name": "D4", "laue_id": 5, "card": 8},
    {"name": "D6", "laue_id": 9, "card": 12},
    {"name": "T", "laue_id": 10, "card": 12},
    {"name": "O", "laue_id": 11, "card": 24},
    {"name": "I", "laue_id": 12, "card": 60},
]
LOGQUAD_COEFFS: Dict[str, Tuple[float, float, float]] = {
    "C2": (+0.595700, +0.063365, -0.008238),
    "C3": (+0.505407, +0.068467, -0.008634),
    "C4": (+0.426731, +0.094131, -0.012107),
    "C6": (+0.401063, +0.092710, -0.012194),
    "D3": (+0.594580, +0.065238, -0.008549),
    "D4": (+0.489191, +0.026999, -0.000234),
    "D6": (+0.664528, -0.185023, +0.026121),
}


def rhat_star(name: str, cu_xy: int) -> float:
    coeffs = LOGQUAD_COEFFS.get(name)
    if coeffs is None:
        return 1.0
    b0, b1, b2 = coeffs
    x = math.log(max(int(cu_xy), 1))
    return b0 + b1 * x + b2 * x * x


def compute_semi_edges_kr(n_target: int, name: str) -> Tuple[int, int]:
    best_xy, best_z, best_n = 1, 1, 1
    for cu_xy in range(1, 500):
        cu_z = max(1, round(rhat_star(name, cu_xy) * cu_xy))
        n = (2 * cu_xy + 1) ** 2 * (2 * cu_z + 1)
        if abs(n - n_target) < abs(best_n - n_target):
            best_xy, best_z, best_n = cu_xy, cu_z, n
        if n > 2 * n_target:
            break
    return best_xy, best_z


def compute_semi_edge_rej(n_target: int, card: int) -> int:
    return max(1, int(round(((n_target * card) ** (1.0 / 3.0) - 1.0) / 2.0)))


def ensure_unit(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def chordal_to_disori_deg_torch(d: torch.Tensor) -> torch.Tensor:
    cos_half = torch.clamp(1.0 - 0.5 * d * d, -1.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(cos_half))


def rebin_histogram_torch(
    fine_counts: torch.Tensor,
    fine_edges: torch.Tensor,
    coarse_edges: torch.Tensor,
) -> torch.Tensor:
    mids = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    idx = torch.bucketize(mids, coarse_edges) - 1
    idx = idx.clamp(0, coarse_edges.numel() - 2)
    out = torch.zeros(coarse_edges.numel() - 1, device=fine_counts.device, dtype=torch.float64)
    out.scatter_add_(0, idx.long(), fine_counts.to(torch.float64))
    return out


def update_running_stats_torch(n: int, mean: float, m2: float, x: torch.Tensor) -> Tuple[int, float, float]:
    m = int(x.numel())
    if m == 0:
        return n, mean, m2

    mean_x = float(torch.mean(x).item())
    m2_x = float(torch.sum((x - mean_x) ** 2).item())
    if n == 0:
        return m, mean_x, m2_x

    n_new = n + m
    delta = mean_x - mean
    mean_new = mean + delta * (m / n_new)
    m2_new = m2 + m2_x + delta * delta * (n * m / n_new)
    return n_new, mean_new, m2_new


def so3_super_fibonacci_chunk(start: int, stop: int, n_total: int, device: torch.device) -> torch.Tensor:
    if stop <= start:
        return torch.empty((0, 4), device=device, dtype=torch.float64)

    phi = 2.0**0.5
    psi = 1.533751168755204288118041

    indices = torch.arange(start, stop, device=device, dtype=torch.float64)
    s = indices + 0.5
    t = s / float(n_total)
    d = 2.0 * torch.pi * s
    r = torch.sqrt(t)
    big_r = torch.sqrt(1.0 - t)
    alpha = d / phi
    beta = d / psi

    q = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            big_r * torch.sin(beta),
            big_r * torch.cos(beta),
        ],
        dim=1,
    )
    mask = q[..., 0] < 0
    q[mask] *= -1
    return q


def _build_laue_augmented_points(reference: torch.Tensor, laue_id: int):
    ref_cpu = reference.detach().to(device="cpu", dtype=torch.float64)
    ops = laue_elements(laue_id).to(ref_cpu)
    n_ref = ref_cpu.shape[0]

    aug_blocks = []
    base_idx_blocks = []
    id_pos_blocks = []

    base = torch.arange(n_ref, dtype=torch.int64)
    for op_idx in range(ops.shape[0]):
        ref_h = qu_prod(ref_cpu, ops[op_idx])
        aug_blocks.append(ref_h)
        aug_blocks.append(-ref_h)

        base_idx_blocks.append(base)
        base_idx_blocks.append(base)

        id_pos_blocks.append(torch.full((n_ref,), op_idx == 0, dtype=torch.bool))
        id_pos_blocks.append(torch.zeros(n_ref, dtype=torch.bool))

    aug_points = torch.cat(aug_blocks, dim=0).contiguous().numpy()
    aug_base_idx = torch.cat(base_idx_blocks, dim=0).contiguous().numpy()
    aug_id_pos = torch.cat(id_pos_blocks, dim=0).contiguous().numpy()
    return aug_points, aug_base_idx, aug_id_pos


def nn_chordal_cross_kdtree(query: torch.Tensor, reference: torch.Tensor, laue_id: int) -> torch.Tensor:
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for nn_method='kdtree'")

    q_np = query.detach().cpu().numpy()
    aug_points, _, _ = _build_laue_augmented_points(reference, laue_id)
    tree = cKDTree(aug_points)
    dists, _ = tree.query(q_np, k=1, workers=-1)
    d = np.asarray(dists, dtype=np.float64)
    return torch.from_numpy(d).to(query.device)


def nn_chordal_self_kdtree(points: torch.Tensor, laue_id: int) -> torch.Tensor:
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for nn_method='kdtree'")

    p_np = points.detach().cpu().numpy()
    n_pts = p_np.shape[0]
    aug_points, aug_base_idx, aug_id_pos = _build_laue_augmented_points(points, laue_id)
    tree = cKDTree(aug_points)

    max_k = aug_points.shape[0]
    k = min(8, max_k)
    unresolved = np.ones(n_pts, dtype=bool)
    nn = np.full(n_pts, np.inf, dtype=np.float64)

    while np.any(unresolved):
        q_idx = np.nonzero(unresolved)[0]
        dists, inds = tree.query(p_np[q_idx], k=k, workers=-1)
        dists = np.atleast_2d(dists)
        inds = np.atleast_2d(inds)

        for row, original_i in enumerate(q_idx):
            chosen = None
            for col in range(inds.shape[1]):
                cand = int(inds[row, col])
                if not (aug_base_idx[cand] == original_i and aug_id_pos[cand]):
                    chosen = float(dists[row, col])
                    break
            if chosen is not None:
                nn[original_i] = chosen
                unresolved[original_i] = False

        if np.any(unresolved):
            if k >= max_k:
                raise RuntimeError("KDTree self-NN failed to find valid non-diagonal candidate")
            k = min(max_k, k * 2)

    return torch.from_numpy(nn).to(points.device)


def generate_witness_data(n_eff_target: int, witness_factor: int, device: torch.device, nn_method: str) -> dict:
    t0 = time.time()
    n_groups = len(GROUPS)
    print(
        f"[witness] starting {n_groups} groups (n_eff={n_eff_target:,}, "
        f"witness_factor={witness_factor}, nn_method={nn_method})"
    )
    out = {
        "metadata": {
            "n_eff_target": int(n_eff_target),
            "witness_factor": int(witness_factor),
            "groups": [g["name"] for g in GROUPS],
        },
        "groups": {},
    }

    for idx, g in enumerate(GROUPS, start=1):
        tg = time.time()
        name, laue_id, card = g["name"], g["laue_id"], g["card"]
        n_fz_target = max(1, n_eff_target // (2 * card))
        n_wit_goal_fz = max(1, witness_factor * n_fz_target)

        cu_xy_kr, cu_z_kr = compute_semi_edges_kr(n_fz_target, name)
        q_kr = cu_kr_grid(cu_xy_kr, laue_id, device, z_semi_edge_length=cu_z_kr)
        n_kr = int(q_kr.shape[0])

        semi_rej = compute_semi_edge_rej(n_fz_target, card)
        q_rej = cu_rej_grid(semi_rej, laue_id, device)
        n_rej = int(q_rej.shape[0])

        while n_kr >= n_rej:
            semi_rej += 1
            q_rej = cu_rej_grid(semi_rej, laue_id, device)
            n_rej = int(q_rej.shape[0])

        n_fz_kr = int(q_kr.shape[0])
        n_fz_rej = int(q_rej.shape[0])

        n_so3 = int(math.ceil(n_wit_goal_fz * card * 1.01))
        n_witness = 0
        n_kr_stats, mean_kr, m2_kr = 0, 0.0, 0.0
        n_rej_stats, mean_rej, m2_rej = 0, 0.0, 0.0
        edges = torch.linspace(0.0, 1.01, HIST_BINS_WITNESS + 1, device=device, dtype=torch.float64)
        c_kr = torch.zeros(HIST_BINS_WITNESS, device=device, dtype=torch.float64)
        c_rej = torch.zeros(HIST_BINS_WITNESS, device=device, dtype=torch.float64)
        current_max_angle = 1.0

        for start in range(0, n_so3, MAX_WITNESS_FILTER_BATCH):
            stop = min(start + MAX_WITNESS_FILTER_BATCH, n_so3)
            q_chunk_so3 = so3_super_fibonacci_chunk(start, stop, n_so3, device)
            q_chunk_so3 = ensure_unit(qu_std(q_chunk_so3))
            mask_fz_chunk = ori_in_fz_laue(q_chunk_so3, laue_id)
            q_wit_chunk = q_chunk_so3[mask_fz_chunk]
            m_chunk = int(q_wit_chunk.shape[0])
            if m_chunk == 0:
                continue

            n_witness += m_chunk

            if nn_method == "kdtree":
                nn_kr_chunk = nn_chordal_cross_kdtree(q_wit_chunk, q_kr, laue_id)
            else:
                nn_kr_chunk = laue_cross_nn_chordal_fused(q_wit_chunk, q_kr, laue_id)
            ang_kr_chunk = chordal_to_disori_deg_torch(nn_kr_chunk)

            if nn_method == "kdtree":
                nn_rej_chunk = nn_chordal_cross_kdtree(q_wit_chunk, q_rej, laue_id)
            else:
                nn_rej_chunk = laue_cross_nn_chordal_fused(q_wit_chunk, q_rej, laue_id)
            ang_rej_chunk = chordal_to_disori_deg_torch(nn_rej_chunk)

            chunk_max_angle = max(float(torch.max(ang_kr_chunk).item()), float(torch.max(ang_rej_chunk).item()), 1.0)
            if chunk_max_angle > current_max_angle:
                new_edges = torch.linspace(0.0, 1.01 * chunk_max_angle, HIST_BINS_WITNESS + 1, device=device, dtype=torch.float64)
                c_kr = rebin_histogram_torch(c_kr, edges, new_edges)
                c_rej = rebin_histogram_torch(c_rej, edges, new_edges)
                edges = new_edges
                current_max_angle = chunk_max_angle

            idx_kr = torch.bucketize(ang_kr_chunk, edges) - 1
            idx_kr = idx_kr.clamp(0, HIST_BINS_WITNESS - 1)
            idx_rej = torch.bucketize(ang_rej_chunk, edges) - 1
            idx_rej = idx_rej.clamp(0, HIST_BINS_WITNESS - 1)

            c_kr += torch.bincount(idx_kr.long(), minlength=HIST_BINS_WITNESS).to(torch.float64)
            c_rej += torch.bincount(idx_rej.long(), minlength=HIST_BINS_WITNESS).to(torch.float64)

            n_kr_stats, mean_kr, m2_kr = update_running_stats_torch(n_kr_stats, mean_kr, m2_kr, ang_kr_chunk)
            n_rej_stats, mean_rej, m2_rej = update_running_stats_torch(n_rej_stats, mean_rej, m2_rej, ang_rej_chunk)

            if device.type == "cuda":
                del q_chunk_so3, mask_fz_chunk, q_wit_chunk, nn_kr_chunk, nn_rej_chunk
                torch.cuda.empty_cache()

        std_kr = math.sqrt(m2_kr / n_kr_stats) if n_kr_stats > 0 else float("nan")
        std_rej = math.sqrt(m2_rej / n_rej_stats) if n_rej_stats > 0 else float("nan")

        out["groups"][name] = {
            "laue_id": laue_id,
            "card": card,
            "cu_xy_kr": cu_xy_kr,
            "cu_z_kr": cu_z_kr,
            "semi_rej": semi_rej,
            "N_kr": n_kr,
            "N_rej": n_rej,
            "N_FZ_kr": n_fz_kr,
            "N_FZ_rej": n_fz_rej,
            "N_witness": n_witness,
            "mean_kr": float(mean_kr),
            "std_kr": float(std_kr),
            "mean_rej": float(mean_rej),
            "std_rej": float(std_rej),
            "bin_edges": edges.detach().cpu().numpy().astype(np.float32),
            "counts_kr": torch.round(c_kr).to(torch.int64).detach().cpu().numpy(),
            "counts_rej": torch.round(c_rej).to(torch.int64).detach().cpu().numpy(),
        }

        if device.type == "cuda":
            del q_kr, q_rej
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        avg = elapsed / idx
        eta = avg * (n_groups - idx)
        print(f"[witness] {idx}/{n_groups} {name} done in {time.time() - tg:.1f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m")

    return out


def generate_self_nn_data(n_eff_target: int, device: torch.device, nn_method: str) -> dict:
    t0 = time.time()
    n_groups = len(GROUPS)
    print(f"[self-nn] starting {n_groups} groups (n_eff={n_eff_target:,}, nn_method={nn_method})")
    results = {}
    for idx, g in enumerate(GROUPS, start=1):
        tg = time.time()
        name = g["name"]
        laue_id = g["laue_id"]
        card = g["card"]
        n_fz = max(1, n_eff_target // (2 * card))

        cu_xy_kr, cu_z_kr = compute_semi_edges_kr(n_fz, name)
        q_kr = cu_kr_grid(cu_xy_kr, laue_id, device, z_semi_edge_length=cu_z_kr)
        n_kr = int(q_kr.shape[0])

        semi_rej = compute_semi_edge_rej(n_fz, card)
        q_rej = cu_rej_grid(semi_rej, laue_id, device)
        n_rej = int(q_rej.shape[0])

        while n_kr < n_rej:
            cu_xy_kr += 1
            cu_z_kr = max(1, round(rhat_star(name, cu_xy_kr) * cu_xy_kr))
            q_kr = cu_kr_grid(cu_xy_kr, laue_id, device, z_semi_edge_length=cu_z_kr)
            n_kr = int(q_kr.shape[0])

        if nn_method == "kdtree":
            nn_kr = nn_chordal_self_kdtree(q_kr, laue_id)
        else:
            _, _, _, nn_kr = riesz_energies_fused(q_kr, laue_id, return_nn=True)
        ang_kr = chordal_to_disori_deg_torch(nn_kr)

        if nn_method == "kdtree":
            nn_rej = nn_chordal_self_kdtree(q_rej, laue_id)
        else:
            _, _, _, nn_rej = riesz_energies_fused(q_rej, laue_id, return_nn=True)
        ang_rej = chordal_to_disori_deg_torch(nn_rej)

        all_angles = torch.cat([ang_kr, ang_rej], dim=0)
        max_angle = float(torch.max(all_angles).item()) if all_angles.numel() > 0 else 10.0
        bin_edges = torch.linspace(0.0, max_angle * 1.01, HIST_BINS_SELF_NN + 1, device=device, dtype=torch.float64)
        idx_kr = torch.bucketize(ang_kr, bin_edges) - 1
        idx_kr = idx_kr.clamp(0, HIST_BINS_SELF_NN - 1)
        idx_rej = torch.bucketize(ang_rej, bin_edges) - 1
        idx_rej = idx_rej.clamp(0, HIST_BINS_SELF_NN - 1)
        counts_kr = torch.bincount(idx_kr.long(), minlength=HIST_BINS_SELF_NN)
        counts_rej = torch.bincount(idx_rej.long(), minlength=HIST_BINS_SELF_NN)

        results[name] = {
            "laue_id": laue_id,
            "card": card,
            "cu_xy_kr": cu_xy_kr,
            "cu_z_kr": cu_z_kr,
            "semi_rej": semi_rej,
            "N_kr": n_kr,
            "N_rej": n_rej,
            "bin_edges": bin_edges.detach().cpu().numpy().astype(np.float32),
            "counts_kr": counts_kr.to(torch.int64).detach().cpu().numpy(),
            "counts_rej": counts_rej.to(torch.int64).detach().cpu().numpy(),
            "mean_kr": float(torch.mean(ang_kr).item()),
            "std_kr": float(torch.std(ang_kr, unbiased=False).item()),
            "mean_rej": float(torch.mean(ang_rej).item()),
            "std_rej": float(torch.std(ang_rej, unbiased=False).item()),
        }

        if device.type == "cuda":
            del q_kr, q_rej, nn_kr, nn_rej
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        avg = elapsed / idx
        eta = avg * (n_groups - idx)
        print(f"[self-nn] {idx}/{n_groups} {name} done in {time.time() - tg:.1f}s | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m")

    return {
        "metadata": {
            "n_eff_target": n_eff_target,
            "groups": [g["name"] for g in GROUPS],
        },
        "groups": results,
    }


_WITNESS_SCALAR_KEYS = [
    "laue_id", "card", "cu_xy_kr", "cu_z_kr", "semi_rej", "N_kr", "N_rej",
    "N_FZ_kr", "N_FZ_rej", "N_witness", "mean_kr", "std_kr", "mean_rej", "std_rej"
]
_SELF_NN_SCALAR_KEYS = [
    "laue_id", "card", "cu_xy_kr", "cu_z_kr", "semi_rej", "N_kr", "N_rej",
    "mean_kr", "std_kr", "mean_rej", "std_rej"
]
_ARRAY_KEYS = ["bin_edges", "counts_kr", "counts_rej"]


def save_witness_data(data: dict, path: str) -> None:
    payload = {
        "_n_eff_target": np.array(int(data["metadata"]["n_eff_target"])),
        "_witness_factor": np.array(int(data["metadata"]["witness_factor"])),
        "_group_names": np.array(data["metadata"]["groups"], dtype=object),
    }
    for name, g in data["groups"].items():
        for k in _WITNESS_SCALAR_KEYS:
            payload[f"{name}_{k}"] = np.array(g[k])
        for k in _ARRAY_KEYS:
            payload[f"{name}_{k}"] = g[k]
    np.savez_compressed(path, **payload)


def save_self_nn_data(data: dict, path: str):
    payload = {
        "_n_eff_target": np.array(data["metadata"]["n_eff_target"]),
        "_group_names": np.array(data["metadata"]["groups"]),
    }
    for name, r in data["groups"].items():
        for k in _SELF_NN_SCALAR_KEYS:
            payload[f"{name}_{k}"] = np.array(r[k])
        for k in _ARRAY_KEYS:
            payload[f"{name}_{k}"] = r[k]
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 witness and self-NN NPZ caches"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate both outputs even if files already exist",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Fail if either expected NPZ cache file is missing",
    )
    parser.add_argument(
        "--n-eff-witness",
        type=int,
        default=WITNESS_DEFAULT_N_EFF,
        help="Target witness N_eff = 2*|K|*N_FZ per group",
    )
    parser.add_argument(
        "--witness-factor",
        type=int,
        default=8,
        help="Witness density multiplier for witness->grid generation",
    )
    parser.add_argument(
        "--n-eff-self-nn",
        type=int,
        default=SELF_NN_DEFAULT_N_EFF,
        help="Target self-NN N_eff = 2*|K|*N_FZ per group",
    )
    parser.add_argument(
        "--nn-method",
        choices=["kdtree", "keops"],
        default="kdtree",
        help="Exact NN backend: 'kdtree' (default) or 'keops'",
    )
    args = parser.parse_args()

    if args.nn_method == "kdtree" and cKDTree is None:
        print("Error: scipy is required for --nn-method kdtree")
        sys.exit(1)

    expected_files = [WITNESS_DATA_FILE, SELF_NN_DATA_FILE]
    if args.plot_only:
        missing = [path for path in expected_files if not os.path.exists(path)]
        if missing:
            print("Error: missing required cache file(s):")
            for path in missing:
                print(f"  - {path}")
            print("Run without --plot-only (or with --force) to generate them.")
            sys.exit(1)
        print("All required cache files are present.")
        return

    do_witness = args.force or not os.path.exists(WITNESS_DATA_FILE)
    do_self_nn = args.force or not os.path.exists(SELF_NN_DATA_FILE)

    if do_witness:
        witness_data = generate_witness_data(
            args.n_eff_witness,
            args.witness_factor,
            DEVICE,
            args.nn_method,
        )
        save_witness_data(witness_data, WITNESS_DATA_FILE)
    else:
        print(f"Using existing witness cache: {WITNESS_DATA_FILE}")

    if do_self_nn:
        self_nn_data = generate_self_nn_data(
            args.n_eff_self_nn,
            DEVICE,
            args.nn_method,
        )
        save_self_nn_data(self_nn_data, SELF_NN_DATA_FILE)
    else:
        print(f"Using existing self-NN cache: {SELF_NN_DATA_FILE}")


if __name__ == "__main__":
    main()
