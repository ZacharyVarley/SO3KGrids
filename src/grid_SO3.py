"""
Logic for generating grids over the full rotation group SO(3).

The main approaches are:
    1) Super-Fibonacci grid
    2) Marsaglia sampling
    3) Shoemake sampling

"""

import torch
from torch import Tensor
from src.orientation_ops import cu2qu, om2qu


@torch.jit.script
def _vdcorput_base2_bitwise(n: int, device: torch.device) -> Tensor:
    """
    vdC(2, k) for k = 1..n using bit reversal.
    Returns float64 tensor of shape (n,).
    """
    if n <= 0:
        return torch.empty((0,), device=device, dtype=torch.float64)

    x = torch.arange(1, n + 1, device=device, dtype=torch.int64)
    r = torch.zeros_like(x)

    # Use a fixed width (63) to avoid signed int64 overflow issues.
    # Reversed bits are interpreted as the binary fraction bits.
    for _ in range(63):
        r = (r << 1) | (x & 1)
        x = x >> 1

    return r.to(torch.float64) / 9223372036854775808.0  # 2^63

@torch.jit.script
def _vdcorput_sequence(n: int, base: int, device: torch.device) -> Tensor:
    """
    Van der Corput sequence values for indices 1..n (inclusive) in a given base.

    Returns:
        Tensor: shape (n,), float64
    """
    if n <= 0:
        return torch.empty((0,), device=device, dtype=torch.float64)

    x = torch.arange(1, n + 1, device=device, dtype=torch.int64)
    out = torch.zeros((n,), device=device, dtype=torch.float64)

    factor = 1.0 / float(base)
    denom = factor

    while torch.any(x > 0):
        digit = torch.remainder(x, base)
        out += digit.to(torch.float64) * denom
        x = torch.floor_divide(x, base)
        denom *= factor

    return out


@torch.jit.script
def so3_hardish(
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    canonicalize: bool = True,
) -> Tensor:
    """
    Deterministic low-discrepancy sampling of SO(3) using the Halton-Arvo /
    Diaconis-Shahshahani variant described by Beltrán and Ferizović ("HArDiSh").

    Construction:
        - x1(k) = vdC(base=3, k)
        - x2(k) = vdC(base=2, k)
        - x3(k) = k, k = 1..n
        - M_k = -H_k R_k
          with H_k = I - 2 v_k v_k^T and R_k a z-axis rotation by 2π x1(k)

    Returns:
        Tensor: (n, 4) quaternions
    """
    if n <= 0:
        return torch.empty((0, 4), device=device, dtype=dtype)

    # Halton components (paper uses base-3 for x1 and base-2 for x2)
    x1 = _vdcorput_sequence(n, 3, device)  # shape (n,)
    # x2 = _vdcorput_sequence(n, 2, device)  # shape (n,)
    x2 = _vdcorput_base2_bitwise(n, device)  # shape (n,)

    # x3 = 1..n, normalized form of the paper's vector v
    k = torch.arange(1, n + 1, device=device, dtype=torch.float64)
    frac = k / float(n)  # k / N

    theta1 = 2.0 * torch.pi * x1
    theta2 = 2.0 * torch.pi * x2

    c1 = torch.cos(theta1)
    s1 = torch.sin(theta1)
    c2 = torch.cos(theta2)
    s2 = torch.sin(theta2)

    # v = (cos(2πx2)*sqrt(k/N), sin(2πx2)*sqrt(k/N), sqrt(1-k/N))
    rk = torch.sqrt(frac)
    zk = torch.sqrt(1.0 - frac)
    v = torch.stack([c2 * rk, s2 * rk, zk], dim=1)  # (n, 3)

    # Householder part: H = I - 2 v v^T
    eye = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0)  # (1,3,3)
    vvT = v.unsqueeze(2) * v.unsqueeze(1)  # (n,3,3)
    H = eye - 2.0 * vvT

    # Rotation about z-axis by 2πx1
    R = torch.zeros((n, 3, 3), device=device, dtype=torch.float64)
    R[:, 0, 0] = c1
    R[:, 0, 1] = s1
    R[:, 1, 0] = -s1
    R[:, 1, 1] = c1
    R[:, 2, 2] = 1.0

    # M = -H R
    M = -torch.bmm(H, R)

    qu = om2qu(M).to(dtype)

    # Canonicalize quaternion sign (q and -q represent same rotation)
    if canonicalize:
        qu = torch.where(qu[..., 0:1] >= 0, qu, -qu)

    return qu


@torch.jit.script
def so3_cubochoric_grid(
    semi_edge_length: int,
    device: torch.device,
    # optionally specify a different z semi-edge length vs x/y one
    z_semi_edge_length: int = 0,
) -> Tensor:
    """
    Generate a cubochoric grid for KR mapping. Centered cubochoric cell centers
    on a (2h+1)^3 grid.

    Args:
        :semi_edge_length (int): semi-edge length of the grid (force odd length)
        :device (torch.device): the device to use for the grid generation

    Returns:
        :torch.Tensor: (N, 4) grid of quaternions in SO(3) (w, x, y, z)
    """
    cu_max = 0.5 * torch.pi ** (2.0 / 3.0)
    cu_xy = torch.linspace(-cu_max, cu_max, 2 * semi_edge_length + 2, device=device)
    cu_xy = cu_xy[:-1]
    cu_xy = cu_xy + 0.5 * (cu_xy[1] - cu_xy[0])
    if z_semi_edge_length == 0:
        cu_z = cu_xy
    else:
        cu_z = torch.linspace(
            -cu_max, cu_max, 2 * z_semi_edge_length + 2, device=device
        )
        cu_z = cu_z[:-1]
        cu_z = cu_z + 0.5 * (cu_z[1] - cu_z[0])
    X, Y, Z = torch.meshgrid(cu_xy, cu_xy, cu_z, indexing="ij")
    return cu2qu(torch.stack([X, Y, Z], dim=-1).reshape(-1, 3))


@torch.jit.script
def so3_super_fibonacci(
    n: int,
    device: torch.device,
    reject_invalid: bool = True,
) -> Tensor:
    """
    Super Fibonacci sampling of orientations.

    Args:
        :n (int): the number of orientations to sample device
        :(torch.device): the device to use

    Returns:
        Tensor: the 3D super Fibonacci sampling quaternions (n, 4)

    Notes:

    Alexa, Marc. "Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO
    (3)." In Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition, pp. 8291-8300. 2022.

    """

    PHI = 2.0**0.5
    # positive real solution to PSI^4 = PSI + 4
    PSI = 1.533751168755204288118041

    # don't use float32 for large numbers of points
    indices = torch.arange(n, device=device, dtype=torch.float64)
    s = indices + 0.5
    t = s / n
    d = 2 * torch.pi * s
    r = torch.sqrt(t)
    R = torch.sqrt(1 - t)
    alpha = d / PHI
    beta = d / PSI
    qu = torch.stack(
        [
            r * torch.sin(alpha),
            r * torch.cos(alpha),
            R * torch.sin(beta),
            R * torch.cos(beta),
        ],
        dim=1,
    )
    if reject_invalid:
        qu = qu[qu[..., 0] >= 0]
    return qu


@torch.jit.script
def so3_marsaglia(
    n: int,
    device: torch.device,
) -> Tensor:
    """
    Generate uniformly distributed elements of SO(3) as quaternions.

    Args:
        :n (int): the number of orientations to sample device
        :(torch.device): the device to use

    Returns:
        Tensor: the 3D Marsaglia sampling quaternions (n, 4)

    Notes:
        - repeatedly draw x1 and y1 from the unit interval [0, 1]
        - only keep x1 and y1 if x1^2 + y1^2 = s1 <= 1
        - do the same for x2 and y2 to get s2 <= 1
        - compute c = sqrt((1 - s1) / s2)
        - set q = (x1, y1, x2 * c, y2 * c)

    Marsaglia, G. (1972). Ann. Math. Stat. 43, 645–646.

    """

    xy1 = torch.rand((n, 2), device=device, dtype=torch.float64)
    xy2 = torch.rand((n, 2), device=device, dtype=torch.float64)

    s1 = xy1[:, 0] ** 2 + xy1[:, 1] ** 2
    s2 = xy2[:, 0] ** 2 + xy2[:, 1] ** 2
    good1 = s1 <= 1
    good2 = s2 <= 1

    while torch.any(~good1):
        n_bad = int(torch.sum(~good1))
        xy1_retry = torch.rand((n_bad, 2), device=device, dtype=torch.float64)
        s1_retry = xy1_retry[:, 0] ** 2 + xy1_retry[:, 1] ** 2
        xy1[~good1] = xy1_retry
        s1[~good1] = s1_retry
        good1 = s1 <= 1

    while torch.any(~good2):
        n_bad = int(torch.sum(~good2))
        xy2_retry = torch.rand((n_bad, 2), device=device, dtype=torch.float64)
        s2_retry = xy2_retry[:, 0] ** 2 + xy2_retry[:, 1] ** 2
        xy2[~good2] = xy2_retry
        s2[~good2] = s2_retry
        good2 = s2 <= 1

    c = torch.sqrt((1 - s1) / s2)
    q = torch.stack([xy1[:, 0], xy1[:, 1], xy2[:, 0] * c, xy2[:, 1] * c], dim=1)
    return q


@torch.jit.script
def so3_shoemake(
    n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate uniformly distributed elements of SO(3) as quaternions.

    Args:
        n (int): the number of orientations to sample
        device (torch.device): the device to use

    Returns:
        torch.Tensor: the 3D random sampling in cubochoric coordinates (n, 4)


    Notes:

    Shoemake, Ken. "Uniform random rotations." Graphics Gems III (IBM Version).
    Morgan Kaufmann, 1992. 124-132.

    """

    # h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))
    u = torch.rand(n, device=device, dtype=dtype)
    v = torch.rand(n, device=device, dtype=dtype)
    w = torch.rand(n, device=device, dtype=dtype)
    h = torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v),
            torch.sqrt(u) * torch.sin(2 * torch.pi * w),
            torch.sqrt(u) * torch.cos(2 * torch.pi * w),
        ],
        dim=1,
    )
    # make positive real part
    h = torch.where(h[..., 0:1] >= 0, h, -h)
    return h
