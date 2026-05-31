"""Torch grid builders for the SO(3) sampler benchmark (SPOT excluded)."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.grid_FZ import kr_sample_laue
from src.grid_SO3 import so3_cubochoric_grid, so3_hardish, so3_super_fibonacci
from src.orientation_ops import qu_norm, qu_std
from src.so3_sampling.hopf import grid_hopf as grid_hopf_numpy

LAUE_ID_SO3 = 1
LAUE_ID_O = 11
CARD_O = 24


def _fcc_cubochoric_grid(h: int, device: torch.device) -> torch.Tensor:
    cu_max = 0.5 * torch.pi ** (2.0 / 3.0)
    u = torch.linspace(-cu_max, cu_max, 2 * h + 2, device=device, dtype=torch.float64)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")
    ix = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(ix, ix, ix, indexing="ij")
    mask = ((i + j + k) % 2) == 0
    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


def grid_hopf(n: int, device: torch.device | None = None) -> torch.Tensor:
    q = grid_hopf_numpy(n)
    return torch.from_numpy(np.asarray(q, dtype=np.float64)).to(
        device or torch.device("cpu")
    )


def grid_hardish(n: int, device: torch.device) -> torch.Tensor:
    return so3_hardish(n, device, canonicalize=True)


def grid_cubochoric(h: int, device: torch.device) -> torch.Tensor:
    q = so3_cubochoric_grid(h, device)
    return qu_norm(qu_std(q))


def grid_fibonacci_all(n: int, device: torch.device) -> torch.Tensor:
    q = so3_super_fibonacci(n, device, reject_invalid=False)
    return qu_norm(qu_std(q))


def grid_fcc_kr(h: int, device: torch.device) -> torch.Tensor:
    from src.orientation_ops import cu2qu

    cu = _fcc_cubochoric_grid(h, device)
    q0 = qu_norm(qu_std(cu2qu(cu)))
    return kr_sample_laue(q0, LAUE_ID_O)
