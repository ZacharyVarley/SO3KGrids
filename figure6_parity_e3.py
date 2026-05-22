#!/usr/bin/env python3
"""
figure6_parity_e3.py — Compare even vs odd FCC parity for KR(O) using E3 ratio.

Plots:
  y-axis: E3 / E3*
  x-axis: N_S^3 = 2 * |K| * N_FZ, with |K|=24 for Laue O

The two curves correspond to parity masks applied to the centered cubochoric
index lattice (i,j,k in [-h, ..., h]):
  - even parity: (i + j + k) % 2 == 0
  - odd parity:  (i + j + k) % 2 != 0

Usage examples:
  python figure6_parity_e3.py
  python figure6_parity_e3.py --h-min 1 --h-max 20
  python figure6_parity_e3.py --device cpu --no-show
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import torch

from grid_FZ import kr_sample_laue
from orientation_ops import cu2qu, qu_norm, qu_std
from riesz_energy import optimal_constants_S3, riesz_energies_fused

LAUE_O = 11
CARD_O = 24
CU_MAX = 0.5 * math.pi ** (2.0 / 3.0)


@torch.no_grad()
def cubochoric_fcc_grid_parity(h: int, device: torch.device, parity: str) -> torch.Tensor:
    if parity not in {"even", "odd"}:
        raise ValueError("parity must be 'even' or 'odd'")

    u = torch.linspace(-CU_MAX, CU_MAX, 2 * h + 2, device=device, dtype=torch.float64)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")

    idx = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
    if parity == "even":
        mask = ((i + j + k) % 2) == 0
    else:
        mask = ((i + j + k) % 2) != 0

    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


@torch.no_grad()
def build_kr_o_grid_from_parity(h: int, device: torch.device, parity: str) -> torch.Tensor:
    cu = cubochoric_fcc_grid_parity(h, device, parity)
    qu = qu_norm(qu_std(cu2qu(cu)))
    q_fz = qu_norm(qu_std(kr_sample_laue(qu, LAUE_O)))
    return q_fz


@torch.no_grad()
def e3_ratio_for_grid(q_fz: torch.Tensor) -> tuple[int, int, float]:
    n_fz = int(q_fz.shape[0])
    n_s3 = 2 * CARD_O * n_fz
    _, _, e3 = riesz_energies_fused(q_fz, LAUE_O)
    _, _, e3_star = optimal_constants_S3(n_s3)
    ratio = float(e3 / e3_star) if e3_star else float("nan")
    return n_fz, n_s3, ratio


def run_sweep(h_min: int, h_max: int, device: torch.device):
    even_data = []
    odd_data = []

    for h in range(h_min, h_max + 1):
        q_even = build_kr_o_grid_from_parity(h, device, "even")
        n_fz_e, n_s3_e, r_e = e3_ratio_for_grid(q_even)
        even_data.append((h, n_fz_e, n_s3_e, r_e))

        q_odd = build_kr_o_grid_from_parity(h, device, "odd")
        n_fz_o, n_s3_o, r_o = e3_ratio_for_grid(q_odd)
        odd_data.append((h, n_fz_o, n_s3_o, r_o))

        print(
            f"h={h:2d} | even: N_FZ={n_fz_e:6d}, N_S^3={n_s3_e:8d}, E3/E3*={r_e:.6f} "
            f"| odd: N_FZ={n_fz_o:6d}, N_S^3={n_s3_o:8d}, E3/E3*={r_o:.6f}"
        )

    return even_data, odd_data


def plot_results(even_data, odd_data, save_stem: str, show: bool):
    x_even = [row[2] for row in even_data]
    y_even = [row[3] for row in even_data]
    x_odd = [row[2] for row in odd_data]
    y_odd = [row[3] for row in odd_data]

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(x_even, y_even, "o-", lw=1.8, ms=4.5, label="FCC parity even")
    plt.plot(x_odd, y_odd, "s--", lw=1.6, ms=4.0, label="FCC parity odd")
    plt.xlabel(r"$N_{S^3}$")
    plt.ylabel(r"$E_3/E_3^*$")
    plt.title("KR(O): E3 ratio vs N_S^3 for FCC parity choices")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = f"{save_stem}.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[export] {path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def parse_args():
    p = argparse.ArgumentParser(description="Compare FCC even/odd parity via E3 ratio vs N_S^3")
    p.add_argument("--h-min", type=int, default=1, help="Minimum h (inclusive)")
    p.add_argument("--h-max", type=int, default=15, help="Maximum h (inclusive)")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection",
    )
    p.add_argument("--save-stem", type=str, default="figure6_parity_e3")
    p.add_argument("--no-show", action="store_true", help="Do not display interactive window")
    return p.parse_args()


def main():
    args = parse_args()

    if args.h_min < 1:
        raise ValueError("h-min must be >= 1")
    if args.h_max < args.h_min:
        raise ValueError("h-max must be >= h-min")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Sweeping h = {args.h_min}..{args.h_max}")

    even_data, odd_data = run_sweep(args.h_min, args.h_max, device)
    plot_results(even_data, odd_data, args.save_stem, show=(not args.no_show))


if __name__ == "__main__":
    main()
