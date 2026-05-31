"""Grid and metric disk cache for the SO(3) sampler benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

METHOD_SLUGS = {
    "Hopf": "hopf",
    "HArDiSh": "hardish",
    "Cubochoric": "cubochoric",
    "super-Fibonacci": "super_fibonacci",
    "FCC KR": "fcc_kr",
}


def method_slug(name: str) -> str:
    if name in METHOD_SLUGS:
        return METHOD_SLUGS[name]
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "method"


def cache_key(param_name: str, param_value: int) -> str:
    return f"{param_name}_{param_value}"


def grid_cache_path(
    cache_root: Path, method_name: str, param_name: str, param_value: int
) -> Path:
    return (
        cache_root
        / "grids"
        / method_slug(method_name)
        / f"{cache_key(param_name, param_value)}.csv"
    )


def metrics_cache_path(cache_root: Path, method_name: str) -> Path:
    return cache_root / "metrics" / f"{method_slug(method_name)}.json"


def save_grid_cache(path: Path, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(q, dtype=np.float64), delimiter=",")


def load_grid_cache(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def save_metrics_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_metrics_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_metrics_store(method_name: str) -> dict:
    return {"method": method_name, "entries": {}}


def load_method_metrics(cache_root: Path, method_name: str) -> tuple[Path, dict]:
    path = metrics_cache_path(cache_root, method_name)
    if not path.exists():
        return path, empty_metrics_store(method_name)
    payload = load_metrics_cache(path)
    payload.setdefault("method", method_name)
    payload.setdefault("entries", {})
    return path, payload


def metric_payload(
    method_name: str,
    param_name: str,
    param_value: int,
    laue_id: int,
    n_fz: int,
    n_eff: int,
    grid_path: Path,
    cache_root: Path,
    *,
    e3_ratio: float | None = None,
    theta_deg: float | None = None,
    theta_star: float | None = None,
    cr_excess: float | None = None,
    skipped: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    rel = grid_path
    try:
        rel = grid_path.relative_to(cache_root)
    except ValueError:
        rel = grid_path
    payload: dict[str, Any] = {
        "method": method_name,
        "param_name": param_name,
        "param_value": int(param_value),
        "laue_id": int(laue_id),
        "n_fz": int(n_fz),
        "n_eff": int(n_eff),
        "grid_path": str(rel),
        "skipped": bool(skipped),
    }
    if reason is not None:
        payload["reason"] = reason
    if e3_ratio is not None:
        payload["e3_ratio"] = float(e3_ratio)
    if theta_deg is not None:
        payload["theta_deg"] = float(theta_deg)
    if theta_star is not None:
        payload["theta_star"] = float(theta_star)
    if cr_excess is not None:
        payload["cr_excess"] = float(cr_excess)
    return payload


def load_or_compute_grid(
    method_name: str,
    get_grid: Callable[[int], np.ndarray],
    param_name: str,
    param_value: int,
    cache_root: Path,
    refresh_grid: bool,
) -> tuple[np.ndarray, Path]:
    grid_path = grid_cache_path(cache_root, method_name, param_name, param_value)
    if grid_path.exists() and not refresh_grid:
        return load_grid_cache(grid_path), grid_path
    q = get_grid(param_value)
    save_grid_cache(grid_path, q)
    return q, grid_path
