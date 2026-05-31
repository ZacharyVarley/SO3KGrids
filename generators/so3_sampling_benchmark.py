#!/usr/bin/env python3
"""
SO(3) sampler benchmark: E3/E3* and covering-radius excess vs N_S3.

Methods (SPOT excluded):
  Hopf, HArDiSh, primitive cubochoric, super-Fibonacci (canonicalized), FCC-KR (O FZ).

Caches grids under figures/data/so3_sampling_benchmark/grids/ and per-method
metrics JSON under .../metrics/.  Optionally writes a consolidated NPZ for plotting.

Usage:
    python -m generators.so3_sampling_benchmark              # use cache; build NPZ
    python -m generators.so3_sampling_benchmark --pack-only
    python -m generators.so3_sampling_benchmark --refresh-metrics
    python -m generators.so3_sampling_benchmark --refresh-all
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
SO3_SAMPLING_CACHE_DIR = _ROOT / "figures" / "data" / "so3_sampling_benchmark"
SO3_SAMPLING_DATA = _ROOT / "figures" / "data" / "figure13_so3_sampling_methods.npz"
from src.covering_radius import covering_radius, covering_radius_star_deg
from src.riesz_energy import optimal_constants_S3, riesz_energies_fused
from src.so3_sampling import (
    grid_cubochoric,
    grid_fcc_kr,
    grid_fibonacci_all,
    grid_hardish,
    grid_hopf,
)
from src.so3_sampling.benchmark_cache import (
    cache_key,
    grid_cache_path,
    load_method_metrics,
    load_or_compute_grid,
    metric_payload,
    method_slug,
    save_metrics_cache,
)
from src.so3_sampling.grids import CARD_O, LAUE_ID_O, LAUE_ID_SO3

VERSION = "generators.so3_sampling_benchmark v1.0"
MIN_N_FZ = 30

DEFAULT_N_LIST = [
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
]
CUBOCHORIC_H_LIST = list(range(4, 40))
FCC_H_LIST = list(range(1, 18))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def n_eff_so3(n_fz: int) -> int:
    return 2 * n_fz


def n_eff_o(n_fz: int) -> int:
    return 2 * CARD_O * n_fz


def _to_numpy(q: torch.Tensor) -> np.ndarray:
    return q.detach().cpu().numpy().astype(np.float64)


def _grid_hopf_np(n: int) -> np.ndarray:
    return _to_numpy(grid_hopf(n, DEVICE))


def _grid_hardish_np(n: int) -> np.ndarray:
    return _to_numpy(grid_hardish(n, DEVICE))


def _grid_cubochoric_np(h: int) -> np.ndarray:
    return _to_numpy(grid_cubochoric(h, DEVICE))


def _grid_fibonacci_np(n: int) -> np.ndarray:
    return _to_numpy(grid_fibonacci_all(n, DEVICE))


def _grid_fcc_kr_np(h: int) -> np.ndarray:
    return _to_numpy(grid_fcc_kr(h, DEVICE))


METHOD_CONFIGS: list[tuple[str, Callable[[int], np.ndarray], Callable[[int], int], int, str, list[int]]] = [
    ("Hopf", _grid_hopf_np, n_eff_so3, LAUE_ID_SO3, "n", DEFAULT_N_LIST),
    ("HArDiSh", _grid_hardish_np, n_eff_so3, LAUE_ID_SO3, "n", DEFAULT_N_LIST),
    ("Cubochoric", _grid_cubochoric_np, n_eff_so3, LAUE_ID_SO3, "h", CUBOCHORIC_H_LIST),
    ("super-Fibonacci", _grid_fibonacci_np, n_eff_so3, LAUE_ID_SO3, "n", DEFAULT_N_LIST),
    ("FCC KR", _grid_fcc_kr_np, n_eff_o, LAUE_ID_O, "h", FCC_H_LIST),
]


def run_method(
    name: str,
    get_grid: Callable[[int], np.ndarray],
    get_n_eff: Callable[[int], int],
    laue_id: int,
    param_name: str,
    param_values: list[int],
    cache_root: Path,
    *,
    refresh_grid: bool = False,
    refresh_metrics: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    metrics_path, metrics_store = load_method_metrics(cache_root, name)

    for p in param_values:
        key = cache_key(param_name, p)
        grid_path = grid_cache_path(cache_root, name, param_name, p)
        payload = metrics_store["entries"].get(key)

        if payload is not None and not refresh_metrics:
            if refresh_grid or not grid_path.exists():
                try:
                    load_or_compute_grid(
                        name, get_grid, param_name, p, cache_root, refresh_grid
                    )
                except Exception as exc:
                    print(f"  {name} {param_name}={p}: {exc}")
                    continue
            if payload.get("skipped", False):
                print(
                    f"  {name} {param_name}={p}: cached skip ({payload.get('reason', 'n/a')})"
                )
                continue
            results.append(payload)
            print(f"  {name} {param_name}={p}: loaded cached metrics")
            continue

        try:
            q_np, grid_path = load_or_compute_grid(
                name, get_grid, param_name, p, cache_root, refresh_grid
            )
        except Exception as exc:
            print(f"  {name} {param_name}={p}: {exc}")
            continue

        n_fz = int(q_np.shape[0])
        n_eff = get_n_eff(n_fz)
        if n_fz < MIN_N_FZ:
            payload = metric_payload(
                name,
                param_name,
                p,
                laue_id,
                n_fz,
                n_eff,
                grid_path,
                cache_root,
                skipped=True,
                reason="n_fz_below_30",
            )
            metrics_store["entries"][key] = payload
            save_metrics_cache(metrics_path, metrics_store)
            print(f"  {name} {param_name}={p}: skipped (n_fz < {MIN_N_FZ})")
            continue

        q = torch.from_numpy(q_np).to(device=DEVICE, dtype=torch.float64)
        try:
            _, _, e3 = riesz_energies_fused(q, laue_id)
        except Exception as exc:
            print(f"  {name} E3 {param_name}={p}: {exc}")
            continue
        _, _, e3_opt = optimal_constants_S3(n_eff)
        e3_ratio = float(e3 / e3_opt) if e3_opt else float("nan")
        try:
            theta_rad = float(covering_radius(q, laue_id))
        except Exception as exc:
            print(f"  {name} CR {param_name}={p}: {exc}")
            theta_rad = float("nan")
        theta_deg = theta_rad * 180.0 / math.pi
        theta_star = float(covering_radius_star_deg(n_eff))
        cr_excess = (theta_deg / theta_star - 1.0) if theta_star > 0 else float("nan")

        payload = metric_payload(
            name,
            param_name,
            p,
            laue_id,
            n_fz,
            n_eff,
            grid_path,
            cache_root,
            e3_ratio=e3_ratio,
            theta_deg=theta_deg,
            theta_star=theta_star,
            cr_excess=cr_excess,
        )
        metrics_store["entries"][key] = payload
        save_metrics_cache(metrics_path, metrics_store)
        results.append(payload)
        print(f"  {name} {param_name}={p}: computed and cached")

    return results


def pack_npz(cache_root: Path, out_path: Path) -> None:
    """Consolidate per-method metrics JSON into one NPZ for figure plotting."""
    payload: dict[str, Any] = {"version": VERSION}
    for name, _, _, _, _, _ in METHOD_CONFIGS:
        slug = method_slug(name)
        path = cache_root / "metrics" / f"{slug}.json"
        if not path.exists():
            print(f"[pack] missing {path}")
            continue
        import json

        store = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            v
            for v in store.get("entries", {}).values()
            if not v.get("skipped", False)
        ]
        if not entries:
            continue
        entries.sort(key=lambda e: e["n_eff"])
        payload[f"{slug}_n_eff"] = np.array([e["n_eff"] for e in entries], dtype=np.int64)
        payload[f"{slug}_e3_ratio"] = np.array(
            [e["e3_ratio"] for e in entries], dtype=np.float64
        )
        payload[f"{slug}_cr_excess"] = np.array(
            [e["cr_excess"] for e in entries], dtype=np.float64
        )
        payload[f"{slug}_method"] = np.array(name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"[pack] Saved {out_path}")


def generate_all(
    cache_root: Path,
    *,
    refresh_grid: bool = False,
    refresh_metrics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {}
    for name, get_grid, get_n_eff, laue_id, param_name, params in METHOD_CONFIGS:
        print(f"Running {name}...")
        data[name] = run_method(
            name,
            get_grid,
            get_n_eff,
            laue_id,
            param_name,
            params,
            cache_root,
            refresh_grid=refresh_grid,
            refresh_metrics=refresh_metrics,
        )
        print(f"  {len(data[name])} points")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=SO3_SAMPLING_CACHE_DIR,
        help="Root for grids/ and metrics/ caches.",
    )
    parser.add_argument(
        "--npz-out",
        type=Path,
        default=SO3_SAMPLING_DATA,
        help="Consolidated NPZ path for figure plotting.",
    )
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help="Only rebuild NPZ from existing metrics JSON (no grid/metric recompute).",
    )
    parser.add_argument("--refresh-grid", action="store_true")
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--refresh-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = args.cache_dir.resolve()
    refresh_grid = args.refresh_grid or args.refresh_all
    refresh_metrics = args.refresh_metrics or args.refresh_all

    if args.pack_only:
        pack_npz(cache_root, args.npz_out.resolve())
        return 0

    generate_all(
        cache_root,
        refresh_grid=refresh_grid,
        refresh_metrics=refresh_metrics,
    )
    pack_npz(cache_root, args.npz_out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
