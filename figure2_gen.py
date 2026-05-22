#!/usr/bin/env python3
"""
figure2_gen.py  —  Data generation for Figure 2: axial anisotropy in
cubochoric KR grids.

For each anisotropic Laue quotient and x/y semi-edge *cu_xy*, find the
integer z semi-edge *cu_z* that minimises the s=3 Riesz energy ratio:

    ratio(cu_z) = E₃(cu_z) / E₃_opt(N_eff)

N_eff definition
----------------
    N_eff = 2 · |K| · N_base
where:
    N_base = q.shape[0]   — grid points returned by cu_kr_grid
    |K|    = Laue group order
    2      = S³ double cover (both +q and −q map to the same rotation)
Together these match the ``factor = 2·G`` multiplied into every row-sum
inside ``riesz_energies_fused()``, making the fused energy exactly equal
to the naïve energy of the full unfolded set of 2·|K|·N_base points on S³.

Two-stage optimisation per (group, cu_xy)
-----------------------------------------
  1. Discrete unimodal bisection over admissible
       cu_z ∈ [⌈cu_z_lo_frac·cu_xy⌉ , ⌊cu_z_hi_frac·cu_xy⌋]
  2. Quadratic de-alias refinement around the discrete optimum:
       fit ratio vs r = cu_z/cu_xy → continuous minimiser r̂
       → ĉu_z = round(r̂·cu_xy)

Warm start:  each subsequent cu_xy seeds its bisection bracket from the
previous continuous optimum r̂_prev.

Data persistence
----------------
  • If the output file already exists → print summary and exit.
  • Otherwise → compute everything, then save.
  • Pass ``--force`` to recompute even if the file exists.
  • Pass ``--add-missing`` to only compute groups not yet in the existing file,
    then merge and save (no full recompute).
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from grid_FZ import cu_kr_grid
from riesz_energy import optimal_constants_S3, riesz_energies_fused

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA_FILE = "figure2_data.json"
VERSION = "figure2_gen v1.0"

# ── Groups to analyse ───────────────────────────────────────────────────────
# Excluded (equiaxial / isotropic in cubochoric grids):
#     C1 (laue_id=1)  — no symmetry, full cube
#     D2 (laue_id=3)  — orthorhombic: FZ is roughly isotropic
#     T  (laue_id=10) — cubic low
#     O  (laue_id=11) — cubic high
#     I  (laue_id=12) — icosahedral (532)
#
# C5 is NOT a crystallographic Laue group (5-fold rotational symmetry is
# incompatible with 3-D lattice periodicity) and does not appear in the
# Laue group tables.  It is therefore excluded.
GROUPS = [
    {"name": "C2", "laue_id": 2, "card": 2},
    {"name": "C3", "laue_id": 6, "card": 3},
    {"name": "C4", "laue_id": 4, "card": 4},
    {"name": "C6", "laue_id": 8, "card": 6},
    {"name": "D3", "laue_id": 7, "card": 6},
    {"name": "D4", "laue_id": 5, "card": 8},
    {"name": "D6", "laue_id": 9, "card": 12},
]

# ── Sweep parameters ────────────────────────────────────────────────────────
CU_XY_VALUES = (
    list(range(10, 26))  # every integer 10–25
    # + list(range(35, 51, 5))  # 35, 40, 45, 50
    # + [60, 70, 80, 90, 100]  # coarse tail
    + [30, 35, 40]  # coarse tail
)

CU_Z_LO_FRAC = 0.2  # cu_z ≥ ⌈0.2·cu_xy⌉
CU_Z_HI_FRAC = 0.8  # cu_z ≤ ⌊0.8·cu_xy⌋

REFINE_WINDOW = 2  # half-width for quadratic refinement

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return obj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Energy evaluation (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@torch.no_grad()
def compute_ratio(
    cu_xy: int,
    cu_z: int,
    laue_id: int,
    card: int,
    device: torch.device,
    cache: Dict[Tuple[int, int, int], Tuple[float, int, int]],
) -> Tuple[float, int, int]:
    """
    Compute  E₃(cu_z) / E₃_opt(N_eff).

    Parameters
    ----------
    cu_xy : int
        x/y semi-edge length passed to ``cu_kr_grid`` as ``semi_edge_length``.
    cu_z : int
        z semi-edge length passed as ``z_semi_edge_length``.
    laue_id : int
        Laue group identifier (1–12).
    card : int
        |K|, the order (cardinality) of the Laue group.
    device : torch.device
    cache : dict
        Maps ``(laue_id, cu_xy, cu_z)`` → ``(ratio, N_base, N_eff)``.

    Returns
    -------
    (ratio, N_base, N_eff)
    """
    key = (laue_id, cu_xy, cu_z)
    if key in cache:
        return cache[key]

    q = cu_kr_grid(cu_xy, laue_id, device, z_semi_edge_length=cu_z)
    N_base = int(q.shape[0])
    N_eff = 2 * card * N_base

    if N_base == 0:
        cache[key] = (float("inf"), 0, 0)
        return cache[key]

    _, _, E3 = riesz_energies_fused(q, laue_id)
    _, _, E3_opt = optimal_constants_S3(N_eff)

    ratio = E3 / E3_opt if E3_opt > 0 else float("inf")
    cache[key] = (ratio, N_base, N_eff)
    return cache[key]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 1 — discrete unimodal bisection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def discrete_bisection(
    cu_xy: int,
    laue_id: int,
    card: int,
    lo: int,
    hi: int,
    device: torch.device,
    cache: dict,
) -> Tuple[int, float]:
    """
    Find the integer *cu_z* in ``[lo, hi]`` that minimises ``ratio(cu_z)``,
    assuming the ratio is unimodal over this interval.

    Uses a simple bisection:  compare f(mid) vs f(mid+1) and discard the
    half that cannot contain the minimum.  The final 1–3 candidates are
    evaluated exhaustively.
    """
    while hi - lo > 2:
        mid = (lo + hi) // 2
        r_mid, _, _ = compute_ratio(cu_xy, mid, laue_id, card, device, cache)
        r_mid1, _, _ = compute_ratio(cu_xy, mid + 1, laue_id, card, device, cache)
        if r_mid < r_mid1:
            hi = mid  # minimum at or before mid
        else:
            lo = mid + 1  # minimum at or after mid+1

    # Exhaustively evaluate remaining candidates
    best_cu_z, best_r = lo, float("inf")
    for cz in range(lo, hi + 1):
        r, _, _ = compute_ratio(cu_xy, cz, laue_id, card, device, cache)
        if r < best_r:
            best_r, best_cu_z = r, cz
    return best_cu_z, best_r


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stage 2 — continuous quadratic de-alias refinement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def refine_continuous(
    cu_xy: int,
    cu_z_disc: int,
    laue_id: int,
    card: int,
    lo_abs: int,
    hi_abs: int,
    window: int,
    device: torch.device,
    cache: dict,
) -> dict:
    """
    Quadratic refinement around the discrete optimum *cu_z_disc*.

    1. Evaluate a window  cu_z ∈ {cu_z_disc − w, …, cu_z_disc + w}  (clamped).
    2. Fit  ratio ≈ a·r² + b·r + c   where r = cu_z / cu_xy.
    3. Continuous minimiser  r̂ = −b/(2a)  → ĉu_z = round(r̂·cu_xy).
    4. Best integer from candidates  {cu_z_disc, ĉu_z, window points}.

    Returns a dict of results and diagnostics.
    """
    w_lo = max(lo_abs, cu_z_disc - window)
    w_hi = min(hi_abs, cu_z_disc + window)
    cz_vals = list(range(w_lo, w_hi + 1))
    ratios = [
        compute_ratio(cu_xy, cz, laue_id, card, device, cache)[0] for cz in cz_vals
    ]

    r_vals = np.array([cz / cu_xy for cz in cz_vals])
    ratio_arr = np.array(ratios)

    curvature = float("nan")
    r_cont = cu_z_disc / cu_xy
    cu_z_fit = cu_z_disc

    if len(r_vals) >= 3:
        coeffs = np.polyfit(r_vals, ratio_arr, 2)
        a, b, _c = coeffs
        curvature = float(a)
        if a > 1e-15:  # concave ⇒ no minimum
            r_hat = -b / (2.0 * a)
            r_hat = float(np.clip(r_hat, lo_abs / cu_xy, hi_abs / cu_xy))
            r_cont = r_hat
            cu_z_fit = int(round(r_hat * cu_xy))
            cu_z_fit = max(lo_abs, min(hi_abs, cu_z_fit))

    # Candidate pool
    candidates = set(cz_vals) | {cu_z_disc, cu_z_fit}

    best_cu_z, best_ratio = cu_z_disc, float("inf")
    best_nb, best_ne = 0, 0
    for cz in sorted(candidates):
        if cz < lo_abs or cz > hi_abs:
            continue
        r, nb, ne = compute_ratio(cu_xy, cz, laue_id, card, device, cache)
        if r < best_ratio:
            best_ratio, best_cu_z = r, cz
            best_nb, best_ne = nb, ne

    return {
        "cu_z_best": best_cu_z,
        "E3_ratio_best": best_ratio,
        "N_base": best_nb,
        "N_eff": best_ne,
        "cu_z_discrete": cu_z_disc,
        "r_cont": r_cont,
        "curvature": curvature,
        "window": [w_lo, w_hi],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full sweep for one group
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def sweep_group(
    group: dict,
    cu_xy_values: Sequence[int],
    cu_z_lo_frac: float,
    cu_z_hi_frac: float,
    refine_window: int,
    device: torch.device,
) -> List[dict]:
    """
    Sweep *cu_xy* for one Laue group, using warm-start from the previous
    continuous optimum to seed each subsequent bisection bracket.
    """
    name = group["name"]
    laue_id = group["laue_id"]
    card = group["card"]

    cache: Dict[Tuple[int, int, int], Tuple[float, int, int]] = {}
    results: List[dict] = []
    prev_r_cont: Optional[float] = None

    for cu_xy in cu_xy_values:
        # Admissible cu_z range
        lo_abs = max(1, math.ceil(cu_z_lo_frac * cu_xy))
        hi_abs = max(lo_abs, math.floor(cu_z_hi_frac * cu_xy))
        if lo_abs > hi_abs:
            continue

        seed_cu_z: Optional[int] = None

        # ── Stage 1: bisection ──
        if prev_r_cont is not None:
            # Warm start: narrow bracket around seed
            seed_cu_z = int(round(prev_r_cont * cu_xy))
            seed_cu_z = max(lo_abs, min(hi_abs, seed_cu_z))

            narrow_lo = max(lo_abs, seed_cu_z - refine_window)
            narrow_hi = min(hi_abs, seed_cu_z + refine_window)

            cu_z_disc, _ = discrete_bisection(
                cu_xy, laue_id, card, narrow_lo, narrow_hi, device, cache
            )
            # Fall back to full range if optimum is at boundary of narrow bracket
            if (cu_z_disc == narrow_lo or cu_z_disc == narrow_hi) and (
                narrow_lo > lo_abs or narrow_hi < hi_abs
            ):
                cu_z_disc, _ = discrete_bisection(
                    cu_xy, laue_id, card, lo_abs, hi_abs, device, cache
                )
        else:
            # Cold start: search the full admissible range
            cu_z_disc, _ = discrete_bisection(
                cu_xy, laue_id, card, lo_abs, hi_abs, device, cache
            )

        # ── Stage 2: continuous refinement ──
        result = refine_continuous(
            cu_xy,
            cu_z_disc,
            laue_id,
            card,
            lo_abs,
            hi_abs,
            refine_window,
            device,
            cache,
        )
        result.update(
            {
                "group": name,
                "laue_id": laue_id,
                "cu_xy": cu_xy,
                "seed_cu_z": seed_cu_z,
            }
        )

        # Update warm-start for next cu_xy
        rc = result.get("r_cont")
        if rc is not None and isinstance(rc, (int, float)) and math.isfinite(rc):
            prev_r_cont = float(rc)
        else:
            prev_r_cont = result["cu_z_best"] / cu_xy

        results.append(result)

        print(
            f"  {name} cu_xy={cu_xy:3d}: "
            f"cu_z*={result['cu_z_best']:3d}  "
            f"r\u0302={prev_r_cont:.4f}  "
            f"ratio={result['E3_ratio_best']:.6f}  "
            f"N={result['N_base']:6d}  "
            f"N_eff={result['N_eff']:8d}"
        )

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    force = "--force" in sys.argv
    add_missing = "--add-missing" in sys.argv

    if os.path.exists(DATA_FILE) and not force and not add_missing:
        print(f"Data file '{DATA_FILE}' already exists (pass --force to recompute).")
        with open(DATA_FILE) as f:
            data = json.load(f)
        n_rec = len(data.get("results", []))
        n_grp = len(set(r["group"] for r in data.get("results", [])))
        print(f"  {n_rec} records for {n_grp} groups.")
        return

    group_order = {g["name"]: i for i, g in enumerate(GROUPS)}

    if add_missing and os.path.exists(DATA_FILE):
        # Load existing data, compute only missing groups, merge and save
        with open(DATA_FILE) as f:
            data = json.load(f)
        existing_results = data.get("results", [])
        existing_groups = set(r["group"] for r in existing_results)
        missing = [g for g in GROUPS if g["name"] not in existing_groups]
        if not missing:
            print(f"All groups already in '{DATA_FILE}' (nothing to add).")
            return
        print(f"Adding missing groups: {[g['name'] for g in missing]}")
        print(f"cu_xy  : {CU_XY_VALUES}")
        print(f"cu_z/cu_xy frac: [{CU_Z_LO_FRAC}, {CU_Z_HI_FRAC}]")
        print()

        t0 = time.time()
        new_results: List[dict] = []
        for group in missing:
            print(
                f"== {group['name']} "
                f"(laue_id={group['laue_id']}, |K|={group['card']}) =="
            )
            results = sweep_group(
                group,
                CU_XY_VALUES,
                CU_Z_LO_FRAC,
                CU_Z_HI_FRAC,
                REFINE_WINDOW,
                DEVICE,
            )
            new_results.extend(results)
            print()

        combined = existing_results + new_results
        combined.sort(key=lambda r: (group_order.get(r["group"], 99), r["cu_xy"]))
        elapsed = time.time() - t0

        meta = data.get("metadata", {})
        output = {
            "metadata": {
                **meta,
                "version": VERSION,
                "cu_xy_values": CU_XY_VALUES,
                "cu_z_lo_frac": CU_Z_LO_FRAC,
                "cu_z_hi_frac": CU_Z_HI_FRAC,
                "refine_window": REFINE_WINDOW,
                "groups": [g["name"] for g in GROUPS],
                "note_N_eff": (
                    "N_eff = 2 * |K| * N_base.  "
                    "Factor 2 = S^3 double cover (+/-q both represent the same "
                    "rotation); |K| = Laue group order.  This matches the "
                    "'factor = 2*G' inside riesz_energies_fused()."
                ),
                "note_C5": (
                    "C5 is not a crystallographic Laue group (5-fold rotational "
                    "symmetry violates the crystallographic restriction theorem) "
                    "and is therefore excluded."
                ),
            },
            "results": combined,
        }
        output = _sanitize_for_json(output)
        with open(DATA_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(
            f"Saved {len(combined)} records to '{DATA_FILE}' (+{len(new_results)} new, {elapsed:.1f} s)"
        )
        return

    print(f"Groups : {[g['name'] for g in GROUPS]}")
    print(f"cu_xy  : {CU_XY_VALUES}")
    print(f"cu_z/cu_xy frac: [{CU_Z_LO_FRAC}, {CU_Z_HI_FRAC}]")
    print(f"Window : +/-{REFINE_WINDOW}")
    print()

    t0 = time.time()
    all_results: List[dict] = []

    for group in GROUPS:
        print(
            f"== {group['name']} "
            f"(laue_id={group['laue_id']}, |K|={group['card']}) =="
        )
        results = sweep_group(
            group,
            CU_XY_VALUES,
            CU_Z_LO_FRAC,
            CU_Z_HI_FRAC,
            REFINE_WINDOW,
            DEVICE,
        )
        all_results.extend(results)
        print()

    elapsed = time.time() - t0

    output = {
        "metadata": {
            "version": VERSION,
            "cu_xy_values": CU_XY_VALUES,
            "cu_z_lo_frac": CU_Z_LO_FRAC,
            "cu_z_hi_frac": CU_Z_HI_FRAC,
            "refine_window": REFINE_WINDOW,
            "groups": [g["name"] for g in GROUPS],
            "note_N_eff": (
                "N_eff = 2 * |K| * N_base.  "
                "Factor 2 = S^3 double cover (+/-q both represent the same "
                "rotation); |K| = Laue group order.  This matches the "
                "'factor = 2*G' inside riesz_energies_fused()."
            ),
            "note_C5": (
                "C5 is not a crystallographic Laue group (5-fold rotational "
                "symmetry violates the crystallographic restriction theorem) "
                "and is therefore excluded."
            ),
        },
        "results": all_results,
    }

    # Sanitise before serialisation (NaN → null, numpy types → native)
    output = _sanitize_for_json(output)

    with open(DATA_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(all_results)} records to '{DATA_FILE}' ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
