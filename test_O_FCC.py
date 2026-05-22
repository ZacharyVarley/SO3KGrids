#!/usr/bin/env python3
"""
test_O_FCC.py — Compare rejection, KR PC, KR FCC, and KR FCC TC SRC grids for Laue O.

For several grid sizes, computes:
  - E3 ratio (E3/E3*)
  - Covering radius excess (theta/theta* - 1)

Plots:
  1) N_S^3 vs E3 ratio
  2) N_S^3 vs theta/theta* - 1

Each method samples a different N_fz (and thus N_S^3 = 2*CARD*N_fz).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Matplotlib required. pip install matplotlib")

from grid_FZ import cu_kr_grid, cu_rej_grid, kr_sample_laue
from covering_radius import covering_radius, covering_radius_star_deg
from riesz_energy import riesz_energies_fused, optimal_constants_S3
from thomson_relax_new import ensure_unit
from orientation_ops import cu2qu, qu_std

# New KR FCC TC SRC mapping (FCC over truncated cube)
from mappings.ho2hoFZ_O_FCC import kr_grid_O_FCC

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

LAUE_ID = 11  # Cubic High O
CARD = 24
CU_MAX = 0.5 * math.pi ** (2.0 / 3.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grid size ranges (sweep parameters)
SEMI_REJ_RANGE = list(range(4, 18))  # rejection semi-edge
CU_XY_RANGE = list(range(10, 20, 2))  # KR PC cu_xy (cu_z = cu_xy for O)
H_FCC_RANGE = list(range(15, 25, 2))  # KR FCC and KR FCC TC SRC h


def rhat_star(name: str, cu_xy: int) -> float:
    if name in ("O", "T", "I"):
        return 1.0
    lx = math.log(max(cu_xy, 1))
    return 0.664528 - 0.185023 * lx + 0.026121 * lx * lx


def _kr_grid_n(cu_xy: int, cu_z: int) -> int:
    return (2 * cu_xy + 1) ** 2 * (2 * cu_z + 1)


def _fcc_grid_n(h: int) -> int:
    return ((2 * h + 1) ** 3 + 1) // 2


@torch.no_grad()
def _cubochoric_fcc_grid(h: int, device: torch.device) -> torch.Tensor:
    """FCC lattice in cubochoric cube."""
    u = torch.linspace(-CU_MAX, CU_MAX, 2 * h + 2, device=device, dtype=torch.float64)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")
    idx = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
    mask = ((i + j + k) % 2) == 0
    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


@torch.no_grad()
def _kr_fcc_grid(h: int, laue_id: int, dev: torch.device) -> torch.Tensor:
    cu = _cubochoric_fcc_grid(h, dev)
    qu = ensure_unit(qu_std(cu2qu(cu)))
    q_fz = kr_sample_laue(qu, laue_id)
    return ensure_unit(qu_std(q_fz))


def n_s3_from_n_fz(n_fz: int) -> int:
    return 2 * CARD * n_fz


def compute_metrics(q: torch.Tensor, laue_id: int, n_fz: int) -> dict:
    """Compute E3 ratio and covering radius excess."""
    n_eff = n_s3_from_n_fz(n_fz)
    E1, E2, E3 = riesz_energies_fused(q, laue_id)
    _, _, E3o = optimal_constants_S3(n_eff)
    e3_ratio = E3 / E3o if E3o else float("nan")

    elc = min(20, max(5, n_fz // 500))
    theta_rad = covering_radius(
        q,
        laue_id,
        edge_length_check=elc,
        grid_eps_factor=2.0,
        qhull_options="QJ",
    )
    theta_deg = float(theta_rad) * 180.0 / math.pi
    theta_star_deg = covering_radius_star_deg(n_eff)
    cr_excess = (
        (theta_deg / theta_star_deg - 1.0) if theta_star_deg > 0 else float("nan")
    )

    return {
        "n_fz": n_fz,
        "n_s3": n_eff,
        "e3_ratio": e3_ratio,
        "theta_deg": theta_deg,
        "theta_star_deg": theta_star_deg,
        "cr_excess": cr_excess,
    }


def collect_rejection(dev: torch.device) -> list[dict]:
    results = []
    for semi in SEMI_REJ_RANGE:
        q = cu_rej_grid(semi, LAUE_ID, dev)
        n_fz = int(q.shape[0])
        if n_fz < 50:
            continue
        m = compute_metrics(q, LAUE_ID, n_fz)
        m["param"] = semi
        results.append(m)
    return results


def collect_kr_pc(dev: torch.device) -> list[dict]:
    results = []
    for cu_xy in CU_XY_RANGE:
        cu_z = max(1, round(rhat_star("O", cu_xy) * cu_xy))
        q = cu_kr_grid(cu_xy, LAUE_ID, dev, z_semi_edge_length=cu_z)
        n_fz = int(q.shape[0])
        if n_fz < 50:
            continue
        m = compute_metrics(q, LAUE_ID, n_fz)
        m["param"] = (cu_xy, cu_z)
        results.append(m)
    return results


def collect_kr_fcc(dev: torch.device) -> list[dict]:
    results = []
    for h in H_FCC_RANGE:
        q = _kr_fcc_grid(h, LAUE_ID, dev)
        n_fz = int(q.shape[0])
        if n_fz < 50:
            continue
        m = compute_metrics(q, LAUE_ID, n_fz)
        m["param"] = h
        results.append(m)
    return results


def collect_kr_fcc_tc_src(dev: torch.device) -> list[dict]:
    """KR FCC TC SRC: FCC grid over truncated cube as src domain."""
    results = []
    for h in H_FCC_RANGE:
        q = kr_grid_O_FCC(h=h, device=dev, return_type="qu", check_fz=False)
        n_fz = int(q.shape[0])
        if n_fz < 50:
            continue
        m = compute_metrics(q, LAUE_ID, n_fz)
        m["param"] = h
        results.append(m)
    return results


def main():
    print("test_O_FCC: Comparing rejection, KR PC, KR FCC, KR FCC TC SRC")
    print(f"Device: {DEVICE}\n")

    data = {
        "rejection": collect_rejection(DEVICE),
        "KR PC": collect_kr_pc(DEVICE),
        "KR FCC": collect_kr_fcc(DEVICE),
        "KR FCC TC SRC": collect_kr_fcc_tc_src(DEVICE),
    }

    for name, lst in data.items():
        print(f"  {name}: {len(lst)} grid sizes")

    # Build plot data
    palette = {
        "rejection": "#BBBBBB",
        "KR PC": "#757575",
        "KR FCC": "#000000",
        "KR FCC TC SRC": "#E63946",
    }
    markers = {
        "rejection": "o",
        "KR PC": "s",
        "KR FCC": "^",
        "KR FCC TC SRC": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Subplot 1: N_S^3 vs E3 ratio
    ax1 = axes[0]
    for name, lst in data.items():
        if name == "rejection":
            continue
        if not lst:
            continue
        ns3 = np.array([m["n_s3"] for m in lst])
        e3 = np.array([m["e3_ratio"] for m in lst])
        ok = np.isfinite(e3) & (e3 > 0)
        if not ok.any():
            continue
        ax1.plot(
            ns3[ok],
            e3[ok],
            color=palette[name],
            marker=markers[name],
            markersize=5,
            linestyle="-",
            label=name,
            lw=1.5,
        )
    ax1.set_xlabel(r"$N_{\mathrm{S}^3}$", fontsize=11)
    ax1.set_ylabel(r"$E_3\,/\,E_3^{\,*}$", fontsize=11)
    ax1.set_xscale("log")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.02,
        0.98,
        "(a)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Subplot 2: N_S^3 vs theta/theta* - 1
    ax2 = axes[1]
    for name, lst in data.items():
        if name == "rejection":
            continue
        if not lst:
            continue
        ns3 = np.array([m["n_s3"] for m in lst])
        cr_ex = np.array([m["cr_excess"] for m in lst])
        ok = np.isfinite(cr_ex)
        if not ok.any():
            continue
        ax2.plot(
            ns3[ok],
            cr_ex[ok],
            color=palette[name],
            marker=markers[name],
            markersize=5,
            linestyle="-",
            label=name,
            lw=1.5,
        )
    ax2.axhline(0.0, color="k", linestyle=":", lw=1, alpha=0.7)
    ax2.set_xlabel(r"$N_{\mathrm{S}^3}$", fontsize=11)
    ax2.set_ylabel(r"$\theta/\theta^{\,*}-1$", fontsize=11)
    ax2.set_xscale("log")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.02,
        0.98,
        "(b)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    plt.tight_layout()
    plt.savefig("test_O_FCC.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig("test_O_FCC.pdf", bbox_inches="tight", facecolor="white")
    print("\nSaved test_O_FCC.png and test_O_FCC.pdf")
    plt.show()


if __name__ == "__main__":
    main()
