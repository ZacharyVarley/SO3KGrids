#!/usr/bin/env python3
"""
Thomson relaxation: E3 ratio and covering radius vs iteration (O only).

Compares three initialization methods for Laue O during Thomson relaxation:
  1) KR from FCC cubochoric lattice
  2) KR from primitive-cubic cubochoric lattice
  3) Cubochoric rejection grid

Layout: 2 subplots
  1) E3 ratio (E3/E3*) during relaxation
  2) Covering radius (degrees) during relaxation

Usage:
        python -m generators.thomson_relaxation              # generate data (if needed) + interactive plot
        python -m generators.thomson_relaxation --force      # force-regenerate all data
        python -m generators.thomson_relaxation --plot-only  # plot from existing .npz
        python -m generators.thomson_relaxation --static     # non-interactive PNG/PDF export
        python -m generators.thomson_relaxation --quick      # 5 iters (for testing)
        python -m generators.thomson_relaxation --n-eff-min 50000 --n-eff-max 500000  # custom range
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from io import StringIO

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required.  pip install torch")

try:
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider
except ImportError:
    sys.exit("Matplotlib is required.  pip install matplotlib")

from src.grid_FZ import cu_kr_grid, cu_rej_grid, kr_sample_laue
from src.covering_radius import covering_radius, covering_radius_star_deg
from src.riesz_energy import riesz_energies_fused, optimal_constants_S3
from src.laue_ops import laue_elements, ori_to_fz_laue
from src.orientation_ops import cu2qu, qu_std
from repo_paths import FIGURE5_DATA, FIGURE5_GENERATOR_SETTINGS, FIGURE5_SUMMARY
from src.thomson_relax import relax_orientations_allpairs, ensure_unit

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

E3_YMIN = 0.925
E3_YMAX = 0.96
N_EFF_MIN = 400_000
N_EFF_MAX = 800_000
CR_EVERY_DEFAULT = 1
LOG_EVERY_DEFAULT = 10
MAX_ITERS = 100
DATA_FILE = str(FIGURE5_DATA)
SETTINGS_FILE = str(FIGURE5_GENERATOR_SETTINGS)
SUMMARY_FILE = str(FIGURE5_SUMMARY)
DPI = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "O"
LAUE_ID = 11
CARD = 24
CU_MAX = 0.5 * math.pi ** (2.0 / 3.0)

BASE_KEYS = ["O_rej", "O_KR_PC", "O_KR_FCC"]
PANEL_KEYS = ["E3", "CR"]
GROUP_CARD = {"O": CARD}


def _series_keys_from_data(data: dict) -> list[str]:
    """Sort keys by n_eff, then BASE_KEYS order."""
    keys = list(data.keys())

    def key_order(k):
        n_eff = int(data.get(k, {}).get("n_eff") or 0)
        order = BASE_KEYS.index(k) if k in BASE_KEYS else 99
        return (n_eff, order)

    return sorted(keys, key=key_order)


def rhat_star(name: str, cu_xy: int) -> float:
    if name in ("O", "T", "I"):
        return 1.0
    lx = math.log(max(cu_xy, 1))
    return 0.664528 - 0.185023 * lx + 0.026121 * lx * lx


def _kr_grid_n(cu_xy: int, cu_z: int) -> int:
    return (2 * cu_xy + 1) ** 2 * (2 * cu_z + 1)


def _fcc_grid_n(h: int) -> int:
    return ((2 * h + 1) ** 3 + 1) // 2


def compute_semi_edge_rej(n_target: int, card: int) -> int:
    return max(1, int(round(((n_target * card) ** (1 / 3) - 1) / 2)))


def _enumerate_pc_kr_sizes(
    n_eff_min: int, n_eff_max: int
) -> list[tuple[int, int, int]]:
    out = []
    for cu_xy in range(1, 500):
        cu_z = max(1, round(rhat_star(NAME, cu_xy) * cu_xy))
        n_pc = _kr_grid_n(cu_xy, cu_z)
        n_eff = 2 * CARD * n_pc
        if n_eff_min <= n_eff <= n_eff_max:
            out.append((cu_xy, cu_z, n_pc))
        if n_eff > n_eff_max:
            break
    return out


def _enumerate_fcc_kr_sizes(n_eff_min: int, n_eff_max: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, 500):
        n_fcc = _fcc_grid_n(h)
        n_eff = 2 * CARD * n_fcc
        if n_eff_min <= n_eff <= n_eff_max:
            out.append((h, n_fcc))
        if n_eff > n_eff_max:
            break
    return out


def _enumerate_rej_sizes(dev, n_eff_min: int, n_eff_max: int) -> list[tuple[int, int]]:
    out = []
    min_n_fz = max(1, n_eff_min // (2 * CARD))
    max_n_fz = n_eff_max // (2 * CARD)
    semi_lo = max(1, compute_semi_edge_rej(min_n_fz, CARD) - 2)
    semi_hi = compute_semi_edge_rej(max_n_fz, CARD) + 2

    for semi in range(semi_lo, semi_hi + 1):
        q_rej = cu_rej_grid(semi, LAUE_ID, dev)
        n_rej = int(q_rej.shape[0])
        n_eff = 2 * CARD * n_rej
        if n_eff_min <= n_eff <= n_eff_max:
            out.append((semi, n_rej))
    return out


def _closest_by_n(target_n: int, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    idx, n = min(candidates, key=lambda t: abs(t[1] - target_n))
    return idx, n


def _find_best_grid_sizes(
    dev, n_eff_min: int, n_eff_max: int
) -> tuple[int, int, int, int, int, int, int]:
    """
    Choose primitive KR in range, then closest FCC-KR and rejection sizes by N_fz.
    Returns: (cu_xy, cu_z, h_fcc, semi_rej, n_pc, n_fcc, n_rej)
    """
    pc_sizes = _enumerate_pc_kr_sizes(n_eff_min, n_eff_max)
    fcc_sizes = _enumerate_fcc_kr_sizes(n_eff_min, n_eff_max)
    rej_sizes = _enumerate_rej_sizes(dev, n_eff_min, n_eff_max)
    if not pc_sizes:
        raise ValueError("No primitive KR sizes in requested N_eff range")
    if not fcc_sizes:
        raise ValueError("No FCC KR sizes in requested N_eff range")
    if not rej_sizes:
        raise ValueError("No rejection sizes in requested N_eff range")

    best = None
    best_obj = None
    for cu_xy, cu_z, n_pc in pc_sizes:
        h_fcc, n_fcc = _closest_by_n(n_pc, fcc_sizes)
        semi_rej, n_rej = _closest_by_n(n_pc, rej_sizes)
        obj = (
            abs(n_pc - n_fcc) + abs(n_pc - n_rej) + abs(n_fcc - n_rej),
            abs(n_pc - n_fcc),
            abs(n_pc - n_rej),
            n_pc,
        )
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best = (cu_xy, cu_z, h_fcc, semi_rej, n_pc, n_fcc, n_rej)

    return best


@torch.no_grad()
def _cubochoric_fcc_grid(h: int, device: torch.device) -> torch.Tensor:
    """FCC lattice in cubochoric cube, matching figure7 construction."""
    u = torch.linspace(-CU_MAX, CU_MAX, 2 * h + 2, device=device, dtype=torch.float32)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")

    idx = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
    mask = ((i + j + k) % 2) == 0

    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


@torch.no_grad()
def _kr_fcc_grid(h: int, laue_id: int, dev) -> torch.Tensor:
    cu = _cubochoric_fcc_grid(h, dev)
    qu = ensure_unit(qu_std(cu2qu(cu)))
    q_fz = kr_sample_laue(qu, laue_id)
    return ensure_unit(qu_std(q_fz))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA GENERATION                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _metrics_at_iter(
    q: torch.Tensor,
    laue_id: int,
    card: int,
    quiet: bool = True,
    skip_cr: bool = False,
) -> dict:
    n_base = int(q.shape[0])
    n_eff = 2 * card * n_base
    E1, E2, E3 = riesz_energies_fused(q, laue_id)
    E1o, E2o, E3o = optimal_constants_S3(n_eff)
    cr_deg = float("nan")
    if not skip_cr:
        elc = min(20, max(5, n_base // 500))
        if quiet:
            with redirect_stdout(StringIO()):
                cr = covering_radius(
                    q,
                    laue_id,
                    edge_length_check=elc,
                    grid_eps_factor=2.0,
                    qhull_options="QJ",
                )
        else:
            cr = covering_radius(
                q,
                laue_id,
                edge_length_check=elc,
                grid_eps_factor=2.0,
                qhull_options="QJ",
            )
        cr_deg = float(cr) * 180.0 / math.pi
    return {"E3_ratio": E3 / E3o if E3o else float("nan"), "cr_deg": cr_deg}


def generate_data(
    dev,
    force: bool = False,
    max_iters: int = 100,
    cr_every: int = 10,
    n_eff_min: int | None = None,
    n_eff_max: int | None = None,
) -> dict:
    if os.path.exists(DATA_FILE) and not force:
        print(f"[data] Loading {DATA_FILE}")
        return _load_data()

    n_min = n_eff_min or N_EFF_MIN
    n_max = n_eff_max or N_EFF_MAX
    print(f"\nN_eff range: [{n_min:,}, {n_max:,}]")

    cu_xy, cu_z, h_fcc, semi_rej, n_pc, n_fcc, n_rej = _find_best_grid_sizes(
        dev, n_min, n_max
    )
    print("\n[O] Selected grid sizes:")
    print(f"  KR primitive: cu_xy={cu_xy}, cu_z={cu_z}")
    print(f"                N_fz={n_pc:,}, N_eff={2*CARD*n_pc:,}")
    print(f"  KR FCC:       h={h_fcc}")
    print(f"                N_fz={n_fcc:,}, N_eff={2*CARD*n_fcc:,}")
    print(f"  Rejection:    semi={semi_rej}")
    print(f"                N_fz={n_rej:,}, N_eff={2*CARD*n_rej:,}")

    print("\n[O] Regenerating grids...")
    q_pc = cu_kr_grid(cu_xy, LAUE_ID, dev, z_semi_edge_length=cu_z)
    q_fcc = _kr_fcc_grid(h_fcc, LAUE_ID, dev)
    q_rej = cu_rej_grid(semi_rej, LAUE_ID, dev)

    n_pc_actual = int(q_pc.shape[0])
    n_fcc_actual = int(q_fcc.shape[0])
    n_rej_actual = int(q_rej.shape[0])
    print(f"  KR primitive: N_fz={n_pc_actual:,}, N_eff={2*CARD*n_pc_actual:,}")
    print(f"  KR FCC:       N_fz={n_fcc_actual:,}, N_eff={2*CARD*n_fcc_actual:,}")
    print(f"  Rejection:    N_fz={n_rej_actual:,}, N_eff={2*CARD*n_rej_actual:,}")

    if cr_every > 1:
        print(f"  Covering radius every {cr_every} iterations")

    ops_O = laue_elements(LAUE_ID).to(device=dev, dtype=torch.float32)

    runs = [
        ("O_KR_FCC", q_fcc),
        ("O_KR_PC", q_pc),
        ("O_rej", q_rej),
    ]

    results = {}
    for key, q0 in runs:
        is_rand = key.endswith("_rand")
        laue_metrics = 1 if is_rand else LAUE_ID
        card_metrics = 1 if is_rand else CARD
        n_fz_actual = int(q0.shape[0])
        n_eff_actual = 2 * card_metrics * n_fz_actual
        print(f"  Running {key}: N_fz={n_fz_actual:,}, N_eff={n_eff_actual:,}")

        q0 = ensure_unit(q0.to(torch.float32))
        hist_e3 = []
        hist_cr = []
        last_cr = float("nan")

        ops_relax = laue_elements(1) if is_rand else ops_O
        project_fn = (
            (lambda q: ensure_unit(qu_std(q)))
            if is_rand
            else (lambda q: ori_to_fz_laue(q, LAUE_ID))
        )

        m0 = _metrics_at_iter(q0, laue_metrics, card_metrics, quiet=True, skip_cr=False)
        hist_e3.append(m0["E3_ratio"])
        last_cr = m0["cr_deg"]
        hist_cr.append(last_cr)

        def callback(q: torch.Tensor, iter_num: int):
            q_fz = project_fn(q)
            nonlocal last_cr
            skip_cr = cr_every > 1 and (iter_num + 1) % cr_every != 0
            m = _metrics_at_iter(
                q_fz, laue_metrics, card_metrics, quiet=True, skip_cr=skip_cr
            )
            hist_e3.append(m["E3_ratio"])
            if not skip_cr:
                last_cr = m["cr_deg"]
            hist_cr.append(last_cr)

        q_final, _ = relax_orientations_allpairs(
            q0,
            ops_relax,
            max_iters=max_iters,
            stop_quantile=0.99,
            stop_angle_deg=1e-10,
            stop_patience=1000,
            bb_select="alternate",
            use_precond=True,
            step_clip_kappa=0.5,
            project_fn=project_fn,
            callback=callback,
            verbose=True,
            log_every=LOG_EVERY_DEFAULT,
        )

        results[key] = {
            "E3_ratio": np.array(hist_e3),
            "cr_deg": np.array(hist_cr),
            "iters": np.arange(len(hist_e3)),
            "n_fz": n_fz_actual,
            "n_eff": n_eff_actual,
            "n_eff_run": n_eff_actual,
        }
        print(f"    {key}: {len(hist_e3)} points (init + {len(hist_e3)-1} iters)")

    save = {}
    for key, data in results.items():
        for k, v in data.items():
            if k in ("n_fz", "n_eff", "n_eff_run"):
                save[f"{key}_{k}"] = np.array(v)
            else:
                save[f"{key}_{k}"] = v
    np.savez(DATA_FILE, **save)
    print(f"\n[data] Saved → {DATA_FILE}")
    return results


def _load_data() -> dict:
    raw = dict(np.load(DATA_FILE, allow_pickle=True))
    keys = set()
    for k in raw:
        if k.endswith("_iters"):
            keys.add(k.replace("_iters", ""))
    results = {}
    for key in keys:
        if key.endswith("_rand"):
            continue

        def _scalar(v):
            return int(v.item()) if v is not None and hasattr(v, "item") else v

        n_fz = _scalar(raw.get(f"{key}_n_fz", None))
        n_eff = _scalar(raw.get(f"{key}_n_eff", None))
        n_eff_run = _scalar(raw.get(f"{key}_n_eff_run", None))
        results[key] = {
            "E3_ratio": raw.get(f"{key}_E3_ratio", np.array([])),
            "cr_deg": raw.get(f"{key}_cr_deg", np.array([])),
            "iters": raw.get(f"{key}_iters", np.array([])),
            "n_fz": n_fz,
            "n_eff": n_eff,
            "n_eff_run": n_eff_run,
        }
    return results


def _save_and_print_metric_summary(data: dict, path: str = SUMMARY_FILE) -> None:
    """Save and print initial/final E3 and CR metrics for the available approaches."""
    if not data:
        return

    keys = [k for k in BASE_KEYS if k in data]
    summary = {}

    print("\n--- Initial / Final values (iter 0 and last iter) ---")
    for key in keys:
        d = data[key]
        e3 = np.asarray(d.get("E3_ratio", np.array([])), dtype=float)
        cr = np.asarray(d.get("cr_deg", np.array([])), dtype=float)
        n_eff = d.get("n_eff")
        theta_star = (
            covering_radius_star_deg(n_eff) if n_eff is not None else float("nan")
        )

        e3_0 = float(e3[0]) if e3.size else float("nan")
        e3_f = float(e3[-1]) if e3.size else float("nan")
        cr_0 = float(cr[0]) if cr.size else float("nan")
        cr_f = float(cr[-1]) if cr.size else float("nan")
        crr_0 = (
            cr_0 / theta_star
            if np.isfinite(theta_star) and theta_star > 0
            else float("nan")
        )
        crr_f = (
            cr_f / theta_star
            if np.isfinite(theta_star) and theta_star > 0
            else float("nan")
        )

        summary[key] = {
            "n_fz": int(d.get("n_fz")) if d.get("n_fz") is not None else None,
            "n_eff": int(n_eff) if n_eff is not None else None,
            "iter_count": int(e3.size),
            "initial": {
                "E3_ratio": e3_0,
                "cr_deg": cr_0,
                "cr_ratio": crr_0,
            },
            "final": {
                "E3_ratio": e3_f,
                "cr_deg": cr_f,
                "cr_ratio": crr_f,
            },
        }

        print(
            f"  {key}: "
            f"E3_ratio {e3_0:.6f} -> {e3_f:.6f}, "
            f"cr_deg {cr_0:.4f} -> {cr_f:.4f}, "
            f"CR_ratio {crr_0:.6f} -> {crr_f:.6f}"
        )
    print("---\n")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] Saved → {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SETTINGS                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class Settings:
    label_size: float = 10.0
    tick_size: float = 8.5
    legend_size: float = 8.0
    line_lw: float = 1.5
    show_legend_b: bool = True
    plot_cr_ratio: bool = False
    show_grid: bool = True
    grid_alpha: float = 0.25
    xlog: dict = field(default_factory=lambda: {"E3": False, "CR": False})
    ylog: dict = field(default_factory=lambda: {"E3": False, "CR": False})
    show_theta_star: bool = True
    show: dict = field(default_factory=dict)


def _load_settings(path: str) -> Settings:
    s = Settings()
    if not os.path.exists(path):
        return s
    try:
        with open(path) as f:
            d = json.load(f)
        if "line_lw" not in d and "theta_star_lw" in d:
            d["line_lw"] = d["theta_star_lw"]
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        print(f"[settings] Loaded {path}")
    except Exception as e:
        print(f"[settings] Warning: {e}")
    return s


def _save_settings(s: Settings, path: str):
    with open(path, "w") as f:
        json.dump(asdict(s), f, indent=2)
    print(f"[settings] Saved → {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PALETTE = {
    "O_KR_FCC": "#000000",
    "O_KR_PC": "#757575",
    "O_rej": "#BBBBBB",
}
LINESTYLES = {"O_KR_FCC": "-", "O_KR_PC": "-", "O_rej": "-"}


def _format_series_label(base: str, d: dict, use_tex: bool = False) -> str:
    n_fz = d.get("n_fz")
    if n_fz is not None:
        n_fz = int(n_fz) if hasattr(n_fz, "item") else int(n_fz)

    if use_tex:
        if base == "O_KR_FCC":
            meth_tex = "\\textrm{KR\\ FCC}"
            pad = ""
        elif base == "O_KR_PC":
            meth_tex = "\\textrm{KR}"
            pad = "\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0"
        elif base == "O_rej":
            meth_tex = "\\textrm{REJ.}"
            pad = "\u00a0\u00a0\u00a0\u00a0"
        else:
            meth_tex = "\\textrm{RAND.}"
            pad = ""
    else:
        if base == "O_KR_FCC":
            meth_tex = r"\mathrm{KR\;FCC}"
            pad = ""
        elif base == "O_KR_PC":
            meth_tex = r"\mathrm{KR}"
            pad = "\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0"
        elif base == "O_rej":
            meth_tex = r"\mathrm{REJ.}"
            pad = "\u00a0\u00a0\u00a0\u00a0"
        else:
            meth_tex = r"\mathrm{RAND.}"
            pad = ""

    parts = []
    if n_fz is not None:
        n_str = f"{n_fz:,}".replace(",", r"{,}")
        if use_tex:
            parts.append("$N_{\\mathrm{FZ}}=" + n_str + "$")
        else:
            parts.append(rf"$N_{{\mathrm{{FZ}}}}={n_str}$")
    if parts:
        if use_tex:
            return "$O$ $" + meth_tex + "$" + pad + " (" + ", ".join(parts) + ")"
        return rf"$O$ ${meth_tex}${pad} ({', '.join(parts)})"
    if use_tex:
        return "$O$ $" + meth_tex + "$"
    return rf"$O$ ${meth_tex}$"


def _theta_star_per_group(data: dict, visible: list[str]) -> dict[str, int]:
    def _to_int(v):
        return (
            int(v.item())
            if v is not None and hasattr(v, "item")
            else (int(v) if v is not None else None)
        )

    group_n_eff: dict[str, int] = {}
    if not any(k.startswith("O_") for k in visible):
        return group_n_eff

    n_eff_vals = [_to_int(data.get(k, {}).get("n_eff")) for k in BASE_KEYS]
    n_eff = max(v or 0 for v in n_eff_vals)
    if n_eff > 0:
        group_n_eff["O"] = n_eff
    return group_n_eff


def _draw_panels(axes, data: dict, S: Settings, use_tex: bool = False):
    ax1, ax2 = axes[0], axes[1]
    series_keys = _series_keys_from_data(data)
    visible = [k for k in series_keys if S.show.get(k, True)]

    x_label = "$\\textrm{Iteration}$" if use_tex else r"$\mathrm{Iteration}$"
    panel_a = "$\\mathbf{(a)}$" if use_tex else "(a)"
    panel_b = "$\\mathbf{(b)}$" if use_tex else "(b)"
    e3_label = "$E_3\\,/\\,E_3^{\\,*}$" if use_tex else r"$E_3\,/\,E_3^{\,*}$"
    if S.plot_cr_ratio:
        cr_label = (
            "$(\\theta_{\\mathrm{S^3}}/\\theta_{\\mathrm{S^3}}^{*}) - 1$"
            if use_tex
            else r"$(\theta_{\mathrm{S^3}}/\theta_{\mathrm{S^3}}^{*}) - 1$"
        )
    else:
        cr_label = (
            "$\\theta_{\\mathrm{S^3}}(^\\circ)$"
            if use_tex
            else r"$\theta_{\mathrm{S^3}}(^\circ)$"
        )
    cr_ref_ratio = (
        "$\\theta_{\\mathrm{S^3}}/\\theta_{\\mathrm{S^3}}^* - 1 = 0$"
        if use_tex
        else r"$\theta_{\mathrm{S^3}}/\theta_{\mathrm{S^3}}^* - 1 = 0$"
    )
    cr_ref_star = (
        "$\\theta_{\\mathrm{S^3}}^*$" if use_tex else r"$\theta_{\mathrm{S^3}}^*$"
    )

    max_x = 1
    for key in visible:
        d = data[key]
        arr = d.get("E3_ratio", d.get("cr_deg", []))
        n = len(arr) if hasattr(arr, "__len__") else 0
        if n > 0:
            iters = d.get("iters", np.arange(n))
            iters = np.asarray(iters)
            if len(iters) > 0:
                max_x = max(max_x, float(np.max(iters)) + 1)

    ax1.cla()
    for key in visible:
        d = data[key]
        iters = d.get("iters", np.arange(len(d["E3_ratio"])))
        if len(iters) == 0:
            iters = np.arange(len(d["E3_ratio"]))
        x_vals = np.asarray(iters, dtype=float) + 1
        yy = np.asarray(d["E3_ratio"])
        ok = np.isfinite(yy)
        if S.ylog.get("E3", False):
            ok &= yy > 0
        if not ok.any():
            continue
        ax1.plot(
            x_vals[ok],
            yy[ok],
            color=PALETTE.get(key, "#666"),
            linestyle=LINESTYLES.get(key, "-"),
            label=_format_series_label(key, d, use_tex=use_tex),
            lw=S.line_lw,
        )
    ax1.set_xlabel(x_label, fontsize=S.label_size)
    ax1.set_ylabel(e3_label, fontsize=S.label_size)
    ax1.set_xscale("log" if S.xlog.get("E3", False) else "linear")
    ax1.set_yscale("log" if S.ylog.get("E3", False) else "linear")
    ax1.set_xlim(1, max_x)
    if E3_YMAX is not None and E3_YMAX > 0:
        ax1.set_ylim(E3_YMIN, E3_YMAX)
    ax1.legend(loc="upper right", fontsize=S.legend_size)
    ax1.tick_params(labelsize=S.tick_size)
    if S.show_grid:
        ax1.grid(True, alpha=S.grid_alpha)
    ax1.text(
        0.02,
        0.98,
        panel_a,
        transform=ax1.transAxes,
        fontsize=12,
        fontweight=("bold" if not use_tex else None),
        va="top",
        ha="left",
    )

    ax2.cla()
    for key in visible:
        d = data[key]
        iters = d.get("iters", np.arange(len(d["cr_deg"])))
        if len(iters) == 0:
            iters = np.arange(len(d["cr_deg"]))
        x_vals = np.asarray(iters, dtype=float) + 1
        yy = np.asarray(d["cr_deg"], dtype=float)
        if S.plot_cr_ratio:
            n_eff_i = d.get("n_eff")
            n_eff_i = int(n_eff_i.item()) if hasattr(n_eff_i, "item") else n_eff_i
            theta_star_i = (
                covering_radius_star_deg(int(n_eff_i))
                if n_eff_i is not None and int(n_eff_i) > 0
                else float("nan")
            )
            if np.isfinite(theta_star_i) and theta_star_i > 0:
                yy = yy / theta_star_i - 1.0
            else:
                yy = np.full_like(yy, np.nan, dtype=float)
        ok = np.isfinite(yy)
        if S.ylog.get("CR", False):
            ok &= yy > 0
        if not ok.any():
            continue
        ax2.plot(
            x_vals[ok],
            yy[ok],
            color=PALETTE.get(key, "#666"),
            linestyle=LINESTYLES.get(key, "-"),
            label=_format_series_label(key, d, use_tex=use_tex),
            lw=S.line_lw,
        )
    if S.show_theta_star and visible:
        if S.plot_cr_ratio:
            ax2.axhline(
                0.0,
                color="#000000",
                linestyle=":",
                lw=S.line_lw,
                alpha=0.8,
                label=cr_ref_ratio,
            )
        else:
            group_n_eff = _theta_star_per_group(data, visible)
            for _, n_eff in group_n_eff.items():
                theta_star = covering_radius_star_deg(n_eff)
                ax2.axhline(
                    theta_star,
                    color="#000000",
                    linestyle=":",
                    lw=S.line_lw,
                    alpha=0.8,
                    label=cr_ref_star,
                )
    ax2.set_xlabel(x_label, fontsize=S.label_size)
    ax2.set_ylabel(cr_label, fontsize=S.label_size)
    ax2.set_xscale("log" if S.xlog.get("CR", False) else "linear")
    ax2.set_yscale("log" if S.ylog.get("CR", False) else "linear")
    ax2.set_xlim(1, max_x)
    if S.show_legend_b:
        ax2.legend(loc="upper right", fontsize=S.legend_size)
    ax2.tick_params(labelsize=S.tick_size)
    if S.show_grid:
        ax2.grid(True, alpha=S.grid_alpha)
    ax2.text(
        0.02,
        0.98,
        panel_b,
        transform=ax2.transAxes,
        fontsize=12,
        fontweight=("bold" if not use_tex else None),
        va="top",
        ha="left",
    )


def plot_static(data: dict, s: Settings):
    with plt.rc_context({"text.usetex": True}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        _draw_panels(axes, data, s, use_tex=True)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            path = f"figure5.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"[export] {path} (TeX export)")
        plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  INTERACTIVE FIGURE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class InteractiveFigure:
    def __init__(self, data: dict, settings: Settings):
        self.data = data
        self.S = settings
        self._updating = False

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "mathtext.fontset": "cm",
                "axes.linewidth": 0.6,
                "xtick.direction": "in",
                "ytick.direction": "in",
            }
        )

        self.fig = plt.figure(figsize=(10, 4.5), num="Figure 5 — Thomson Relaxation")
        self.axes = self.fig.subplots(1, 2)

        self.ctrl_fig = plt.figure(figsize=(6.8, 5.6), num="Controls")
        self.ctrl_fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.01)
        self._widgets: dict = {}
        self._build_controls()
        self._connect_events()
        self.redraw()

    def _build_controls(self):
        S = self.S
        fig = self.ctrl_fig
        y_left = 0.965
        y_right = 0.965
        GAP = 0.008
        SH = 0.028
        SG = 0.042
        BH = 0.032
        CH = 0.032
        col_w = 0.43
        x_left = 0.05
        x_right = 0.52

        def _col_x(col):
            return x_left if col == 0 else x_right

        def _get_y(col):
            return y_left if col == 0 else y_right

        def _set_y(col, value):
            nonlocal y_left, y_right
            if col == 0:
                y_left = value
            else:
                y_right = value

        def _heading(col, text):
            y = _get_y(col) - GAP * 2
            ax = fig.add_axes([_col_x(col), y - 0.012, col_w, 0.014])
            ax.set_axis_off()
            ax.text(
                0,
                0,
                text.upper(),
                fontsize=7,
                fontweight="bold",
                color="#555",
                va="bottom",
            )
            _set_y(col, y - 0.020)

        def _slider(col, name, label, lo, hi, val):
            y = _get_y(col)
            ax = fig.add_axes([_col_x(col) + 0.11, y, col_w - 0.11, SH])
            sl = Slider(ax, label, lo, hi, valinit=val, color="#6366f1")
            sl.label.set_fontsize(6.5)
            sl.valtext.set_fontsize(6.5)
            self._widgets[name] = sl
            _set_y(col, y - SG)
            return sl

        def _checkbox_row(col, items_list, n_cols=2):
            y = _get_y(col)
            cw = col_w / n_cols
            out = []
            for idx, (wname, label, checked) in enumerate(items_list):
                c = idx % n_cols
                r = idx // n_cols
                ax_c = fig.add_axes(
                    [_col_x(col) + c * cw, y - (r + 1) * CH, cw - 0.004, CH - 0.004]
                )
                chk = CheckButtons(ax_c, [label], [checked])
                for txt in chk.labels:
                    txt.set_fontsize(6.0)
                out.append((wname, chk))
            n_rows = (len(items_list) + n_cols - 1) // n_cols
            _set_y(col, y - n_rows * CH - GAP)
            return out, n_rows

        def _btn(col, name, label, x_rel, w_rel):
            y = _get_y(col)
            ax = fig.add_axes([_col_x(col) + x_rel * col_w, y, w_rel * col_w, BH])
            b = Button(ax, label, color="#f3f4f6", hovercolor="#e5e7eb")
            b.label.set_fontsize(6.5)
            self._widgets[name] = b
            return b

        _heading(0, "Typography")
        _slider(0, "label_size", "Labels", 7, 14, S.label_size)
        _slider(0, "tick_size", "Ticks", 5, 12, S.tick_size)
        _slider(0, "legend_size", "Legend", 5, 12, S.legend_size)
        _slider(0, "line_lw", "Line lw", 0.5, 3.0, S.line_lw)

        _heading(0, "Axis Scales (log)")
        y = _get_y(0)
        fig.text(_col_x(0), y + 0.004, "X log:", fontsize=6.0, color="#666", va="top")
        _set_y(0, y - 0.010)
        x_items = [(f"xlog_{pk}", pk, S.xlog.get(pk, False)) for pk in PANEL_KEYS]
        xlog_checks, _ = _checkbox_row(0, x_items, n_cols=2)
        self._widgets["xlog_checks"] = xlog_checks

        y = _get_y(0)
        fig.text(_col_x(0), y + 0.004, "Y log:", fontsize=6.0, color="#666", va="top")
        _set_y(0, y - 0.010)
        y_items = [(f"ylog_{pk}", pk, S.ylog.get(pk, False)) for pk in PANEL_KEYS]
        ylog_checks, _ = _checkbox_row(0, y_items, n_cols=2)
        self._widgets["ylog_checks"] = ylog_checks

        _heading(1, "CR Plot")
        theta_items = [
            ("show_theta_star", r"$\theta^*$ line", S.show_theta_star),
            ("show_legend_b", "Legend (b)", S.show_legend_b),
            ("plot_cr_ratio", r"Plot $(\theta/\theta^*) - 1$", S.plot_cr_ratio),
        ]
        theta_checks, _ = _checkbox_row(1, theta_items, n_cols=1)
        self._widgets["theta_checks"] = theta_checks

        _heading(0, "Grid")
        grid_items = [("show_grid", "Show grid", S.show_grid)]
        grid_checks, _ = _checkbox_row(0, grid_items, n_cols=1)
        self._widgets["grid_chk"] = grid_checks
        _slider(0, "grid_alpha", "Alpha", 0.0, 0.6, S.grid_alpha)

        _heading(1, "Series Visibility")
        series_keys = _series_keys_from_data(self.data)
        ser_items = []
        for k in series_keys:
            d = self.data.get(k, {})
            lbl = _format_series_label(k, d)
            ser_items.append((k, lbl, S.show.get(k, True)))
        ser_checks, _ = _checkbox_row(1, ser_items, n_cols=1)
        for key, chk in ser_checks:
            for txt in chk.labels:
                txt.set_color(PALETTE.get(key, "#666"))
        self._widgets["series_checks"] = ser_checks

        _heading(1, "Export")
        _btn(1, "save_png", "Save PNG", 0.00, 0.48)
        _btn(1, "save_pdf", "Save PDF", 0.52, 0.48)
        _set_y(1, _get_y(1) - BH - GAP)
        _btn(1, "save_cfg", "Save Settings", 0.00, 0.48)
        _btn(1, "load_cfg", "Load Settings", 0.52, 0.48)

    def _connect_events(self):
        W = self._widgets
        for k in [
            "label_size",
            "tick_size",
            "legend_size",
            "line_lw",
            "grid_alpha",
        ]:
            W[k].on_changed(lambda _, _k=k: self._on_ui_change())
        for _, chk in W.get("grid_chk", []):
            chk.on_clicked(lambda _: self._on_ui_change())
        for _, chk in W.get("xlog_checks", []):
            chk.on_clicked(lambda _: self._on_ui_change())
        for _, chk in W.get("ylog_checks", []):
            chk.on_clicked(lambda _: self._on_ui_change())
        for _, chk in W.get("theta_checks", []):
            chk.on_clicked(lambda _: self._on_ui_change())
        for _, chk in W.get("series_checks", []):
            chk.on_clicked(lambda _: self._on_ui_change())
        W["save_png"].on_clicked(lambda _: self._export("png"))
        W["save_pdf"].on_clicked(lambda _: self._export("pdf"))
        W["save_cfg"].on_clicked(lambda _: _save_settings(self.S, SETTINGS_FILE))
        W["load_cfg"].on_clicked(lambda _: self._reload_settings())

    def _sync_from_widgets(self):
        W = self._widgets
        self.S.label_size = float(W["label_size"].val)
        self.S.tick_size = float(W["tick_size"].val)
        self.S.legend_size = float(W["legend_size"].val)
        self.S.line_lw = float(W["line_lw"].val)
        self.S.grid_alpha = float(W["grid_alpha"].val)
        for name, chk in W.get("grid_chk", []):
            if name == "show_grid":
                self.S.show_grid = bool(chk.get_status()[0])
        for name, chk in W.get("theta_checks", []):
            if name in ("show_theta_star", "show_legend_b", "plot_cr_ratio"):
                setattr(self.S, name, bool(chk.get_status()[0]))
        for name, chk in W.get("xlog_checks", []):
            pk = name.replace("xlog_", "")
            self.S.xlog[pk] = bool(chk.get_status()[0])
        for name, chk in W.get("ylog_checks", []):
            pk = name.replace("ylog_", "")
            self.S.ylog[pk] = bool(chk.get_status()[0])
        for key, chk in W.get("series_checks", []):
            self.S.show[key] = bool(chk.get_status()[0])

    def _push_settings_to_widgets(self):
        W = self._widgets
        for attr in [
            "label_size",
            "tick_size",
            "legend_size",
            "line_lw",
            "grid_alpha",
        ]:
            W[attr].set_val(getattr(self.S, attr))
        for name, chk in W.get("grid_chk", []):
            if bool(chk.get_status()[0]) != getattr(self.S, name, True):
                chk.set_active(0)
        for name, chk in W.get("theta_checks", []):
            if bool(chk.get_status()[0]) != getattr(self.S, name, True):
                chk.set_active(0)
        for name, chk in W.get("xlog_checks", []):
            pk = name.replace("xlog_", "")
            if bool(chk.get_status()[0]) != self.S.xlog.get(pk, False):
                chk.set_active(0)
        for name, chk in W.get("ylog_checks", []):
            pk = name.replace("ylog_", "")
            if bool(chk.get_status()[0]) != self.S.ylog.get(pk, False):
                chk.set_active(0)
        for key, chk in W.get("series_checks", []):
            if bool(chk.get_status()[0]) != self.S.show.get(key, True):
                chk.set_active(0)

    def _on_ui_change(self):
        if self._updating:
            return
        self._sync_from_widgets()
        self.redraw()

    def _reload_settings(self):
        self.S = _load_settings(SETTINGS_FILE)
        self._updating = True
        self._push_settings_to_widgets()
        self._updating = False
        self.redraw()

    def _export(self, fmt):
        self._sync_from_widgets()
        _save_settings(self.S, SETTINGS_FILE)
        path = f"figure5.{fmt}"
        with plt.rc_context({"text.usetex": True}):
            _draw_panels(self.axes, self.data, self.S, use_tex=True)
            self.fig.canvas.draw_idle()
            self.fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        self.redraw()
        print(f"[export] {path} (TeX export)")

    def redraw(self):
        _draw_panels(self.axes, self.data, self.S, use_tex=False)
        self.fig.canvas.draw_idle()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def main():
    parser = argparse.ArgumentParser(
        description="Figure 5: Thomson relaxation comparison (O, 3 initializations)"
    )
    parser.add_argument("--force", action="store_true", help="Force regenerate data")
    parser.add_argument(
        "--plot-only", action="store_true", help="Plot from existing data only"
    )
    parser.add_argument(
        "--static", action="store_true", help="Export PNG/PDF, no interactive"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test: 5 iters")
    parser.add_argument(
        "--n-eff-min",
        type=int,
        default=None,
        metavar="N",
        help=f"N_eff min (default: {N_EFF_MIN:,})",
    )
    parser.add_argument(
        "--n-eff-max",
        type=int,
        default=None,
        metavar="N",
        help=f"N_eff max (default: {N_EFF_MAX:,})",
    )
    args = parser.parse_args()

    if args.quick:
        max_iters = 5
        cr_every = 5
    else:
        max_iters = MAX_ITERS
        cr_every = CR_EVERY_DEFAULT

    if args.plot_only:
        if not os.path.exists(DATA_FILE):
            print(f"Data file {DATA_FILE} not found. Run without --plot-only first.")
            sys.exit(1)
        data = _load_data()
    else:
        data = generate_data(
            DEVICE,
            force=args.force,
            max_iters=max_iters,
            cr_every=cr_every,
            n_eff_min=args.n_eff_min,
            n_eff_max=args.n_eff_max,
        )

    settings = _load_settings(SETTINGS_FILE)
    _save_and_print_metric_summary(data)

    if args.static:
        plot_static(data, settings)
    else:
        app = InteractiveFigure(data, settings)
        plt.show()


if __name__ == "__main__":
    main()
