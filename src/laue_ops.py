import torch
from torch import Tensor
import math

from src.orientation_ops import qu_prod


@torch.jit.script
def laue_elements(laue_id: int) -> Tensor:
    """
    Generators for Laue group specified by the laue_id parameter. The first
    element is always the identity.

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432
    12) Laue I      Icosahedral: 532

    Args:
        laue_id: integer between inclusive [1, 12]

    Returns:
        torch tensor of shape (cardinality, 4) containing the elements of the

    Notes:

    https://en.wikipedia.org/wiki/Space_group

    """

    # Icosahedral constants (values within ~1e-5 treated as same; one digit for reference)
    A = 0.80901699437494756
    B = 0.42532540417601994
    C = 0.30901699437494751
    D = 0.26286555605956663
    E = 0.58778525229247314
    F = 0.16245984811645311
    G = 0.52573111211913370
    H = 0.68819096023558679
    J = 0.85065080835203988  # 2*B
    K = 0.95105651629515364

    # turn off the black formatter
    # fmt: off
    # Original LAUE_I (digits) — kept for reference only
    _LAUE_I_ref = torch.tensor(
        [
        [  1.00000000000000000,  0.00000000000000000,  0.00000000000000000,  0.00000000000000000 ],
        [  0.80901699437494756, -0.42532540417601994, -0.30901699437494751,  0.26286555605956663 ],
        [  0.80901699437494745, -0.42532540417601988,  0.30901699437494751,  0.26286555605956657 ],
        [  0.80901699437494745,  0.00000000000000000,  0.00000000000000000,  0.58778525229247314 ],
        [  0.80901699437494734,  0.16245984811645311, -0.50000000000000000,  0.26286555605956669 ],
        [  0.80901699437494734,  0.16245984811645323,  0.50000000000000000,  0.26286555605956669 ],
        [  0.80901699437494756,  0.52573111211913370,  0.00000000000000000,  0.26286555605956669 ],
        [  0.80901699437494734,  0.42532540417601999, -0.30901699437494740, -0.26286555605956691 ],
        [  0.80901699437494734, -0.16245984811645320,  0.50000000000000000, -0.26286555605956691 ],
        [  0.80901699437494734,  0.00000000000000000,  0.00000000000000000, -0.58778525229247325 ],
        [  0.80901699437494745, -0.16245984811645317, -0.50000000000000000, -0.26286555605956696 ],
        [  0.80901699437494745, -0.52573111211913359,  0.00000000000000000, -0.26286555605956707 ],
        [  0.80901699437494723,  0.42532540417602005,  0.30901699437494751, -0.26286555605956691 ],
        [  0.50000000000000000,  0.42532540417601994,  0.30901699437494751,  0.68819096023558657 ],
        [  0.50000000000000000, -0.16245984811645320,  0.50000000000000000,  0.68819096023558668 ],
        [  0.50000000000000000, -0.16245984811645317, -0.50000000000000000,  0.68819096023558668 ],
        [  0.50000000000000000,  0.42532540417602005, -0.30901699437494756,  0.68819096023558657 ],
        [  0.50000000000000000, -0.85065080835203999,  0.00000000000000000,  0.16245984811645306 ],
        [  0.50000000000000000, -0.52573111211913370,  0.00000000000000000,  0.68819096023558679 ],
        [  0.50000000000000000,  0.52573111211913370,  0.00000000000000000, -0.68819096023558679 ],
        [  0.50000000000000000,  0.16245984811645317, -0.50000000000000000, -0.68819096023558679 ],
        [  0.50000000000000000,  0.16245984811645314,  0.50000000000000000, -0.68819096023558690 ],
        [  0.50000000000000000, -0.26286555605956685, -0.80901699437494756,  0.16245984811645284 ],
        [  0.50000000000000000, -0.68819096023558679, -0.50000000000000000, -0.16245984811645328 ],
        [  0.50000000000000000,  0.26286555605956680, -0.80901699437494745, -0.16245984811645336 ],
        [  0.50000000000000000,  0.26286555605956685,  0.80901699437494756, -0.16245984811645323 ],
        [  0.50000000000000000, -0.68819096023558679,  0.50000000000000000, -0.16245984811645331 ],
        [  0.50000000000000000, -0.42532540417602005,  0.30901699437494751, -0.68819096023558690 ],
        [  0.50000000000000000, -0.42532540417601994, -0.30901699437494734, -0.68819096023558690 ],
        [  0.50000000000000000,  0.85065080835204010,  0.00000000000000000, -0.16245984811645314 ],
        [  0.50000000000000000,  0.68819096023558690,  0.50000000000000000,  0.16245984811645303 ],
        [  0.50000000000000000,  0.68819096023558701, -0.50000000000000000,  0.16245984811645295 ],
        [  0.50000000000000000, -0.26286555605956685,  0.80901699437494767,  0.16245984811645306 ],
        [  0.30901699437494767, -0.68819096023558668, -0.50000000000000000,  0.42532540417601983 ],
        [  0.30901699437494740,  0.00000000000000000,  0.00000000000000000, -0.95105651629515364 ],
        [  0.30901699437494734,  0.68819096023558679,  0.50000000000000000, -0.42532540417601999 ],
        [  0.30901699437494734,  0.26286555605956680, -0.80901699437494756,  0.42532540417601966 ],
        [  0.30901699437494745,  0.00000000000000000,  0.00000000000000000,  0.95105651629515364 ],
        [  0.30901699437494762, -0.68819096023558690,  0.50000000000000000,  0.42532540417601983 ],
        [  0.30901699437494745,  0.85065080835204021,  0.00000000000000000,  0.42532540417601972 ],
        [  0.30901699437494734,  0.26286555605956685,  0.80901699437494767,  0.42532540417601977 ],
        [  0.30901699437494728, -0.85065080835203988,  0.00000000000000000, -0.42532540417602022 ],
        [  0.30901699437494723, -0.26286555605956691, -0.80901699437494745, -0.42532540417602016 ],
        [  0.30901699437494712,  0.68819096023558690, -0.50000000000000000, -0.42532540417602011 ],
        [  0.30901699437494723, -0.26286555605956696,  0.80901699437494756, -0.42532540417602016 ],
        [  0.00000000000000000, -0.42532540417601994, -0.30901699437494745,  0.85065080835203988 ],
        [  0.00000000000000000, -0.42532540417601994,  0.30901699437494745,  0.85065080835203988 ],
        [  0.00000000000000000, -0.85065080835204010,  0.00000000000000000,  0.52573111211913348 ],
        [  0.00000000000000000,  0.00000000000000000,  1.00000000000000000,  0.00000000000000000 ],
        [  0.00000000000000000,  0.52573111211913370,  0.00000000000000000,  0.85065080835203988 ],
        [  0.00000000000000000,  0.16245984811645323, -0.50000000000000000,  0.85065080835203988 ],
        [  0.00000000000000000, -0.26286555605956691,  0.80901699437494734,  0.52573111211913359 ],
        [  0.00000000000000000, -0.26286555605956663, -0.80901699437494756,  0.52573111211913359 ],
        [  0.00000000000000000,  0.68819096023558690,  0.50000000000000000,  0.52573111211913337 ],
        [  0.00000000000000000, -0.95105651629515353,  0.30901699437494751,  0.00000000000000000 ],
        [  0.00000000000000000, -0.58778525229247292, -0.80901699437494756,  0.00000000000000000 ],
        [  0.00000000000000000,  0.16245984811645328,  0.50000000000000000,  0.85065080835204010 ],
        [  0.00000000000000000,  0.68819096023558690, -0.50000000000000000,  0.52573111211913337 ],
        [  0.00000000000000000, -0.95105651629515364, -0.30901699437494740,  0.00000000000000000 ],
        [  0.00000000000000000, -0.58778525229247325,  0.80901699437494756,  0.00000000000000000 ],
        ],
        dtype=torch.float64,
    )
    # LAUE_I from symbols (values within ~1e-5 treated as same)
    LAUE_I = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [A, -B, -C, D],
            [A, -B, C, D],
            [A, 0.0, 0.0, E],
            [A, F, -0.5, D],
            [A, F, 0.5, D],
            [A, G, 0.0, D],
            [A, B, -C, -D],
            [A, -F, 0.5, -D],
            [A, 0.0, 0.0, -E],
            [A, -F, -0.5, -D],
            [A, -G, 0.0, -D],
            [A, B, C, -D],
            [0.5, B, C, H],
            [0.5, -F, 0.5, H],
            [0.5, -F, -0.5, H],
            [0.5, B, -C, H],
            [0.5, -J, 0.0, F],
            [0.5, -G, 0.0, H],
            [0.5, G, 0.0, -H],
            [0.5, F, -0.5, -H],
            [0.5, F, 0.5, -H],
            [0.5, -D, -A, F],
            [0.5, -H, -0.5, -F],
            [0.5, D, -A, -F],
            [0.5, D, A, -F],
            [0.5, -H, 0.5, -F],
            [0.5, -B, C, -H],
            [0.5, -B, -C, -H],
            [0.5, J, 0.0, -F],
            [0.5, H, 0.5, F],
            [0.5, H, -0.5, F],
            [0.5, -D, A, F],
            [C, -H, -0.5, B],
            [C, 0.0, 0.0, -K],
            [C, H, 0.5, -B],
            [C, D, -A, B],
            [C, 0.0, 0.0, K],
            [C, -H, 0.5, B],
            [C, J, 0.0, B],
            [C, D, A, B],
            [C, -J, 0.0, -B],
            [C, -D, -A, -B],
            [C, H, -0.5, -B],
            [C, -D, A, -B],
            [0.0, -B, -C, J],
            [0.0, -B, C, J],
            [0.0, -J, 0.0, G],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, G, 0.0, J],
            [0.0, F, -0.5, J],
            [0.0, -D, A, G],
            [0.0, -D, -A, G],
            [0.0, H, 0.5, G],
            [0.0, -K, C, 0.0],
            [0.0, -E, -A, 0.0],
            [0.0, F, 0.5, J],
            [0.0, H, -0.5, G],
            [0.0, -K, -C, 0.0],
            [0.0, -E, A, 0.0],
        ],
        dtype=torch.float64,
    )
    # fmt: on

    # sqrt(2) / 2 and sqrt(3) / 2
    R2 = 1.0 / (2.0**0.5)
    R3 = (3.0**0.5) / 2.0

    LAUE_O = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [R2, R2, 0.0, 0.0],
            [R2, -R2, 0.0, 0.0],
            [R2, 0.0, R2, 0.0],
            [R2, 0.0, -R2, 0.0],
            [0.0, R2, 0.0, R2],
            [0.0, -R2, 0.0, R2],
            [0.0, 0.0, R2, R2],
            [0.0, 0.0, -R2, R2],
        ],
        dtype=torch.float64,
    )
    LAUE_T = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float64,
    )

    LAUE_D6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # identity
            [0.5, 0.0, 0.0, R3],  # 120° about Z
            [0.5, 0.0, 0.0, -R3],  # 120° about -Z
            [0.0, 0.0, 0.0, 1.0],  # 180° about Z
            [R3, 0.0, 0.0, 0.5],  # 60° about Z
            [R3, 0.0, 0.0, -0.5],  # 60° about -Z
            [0.0, 1.0, 0.0, 0.0],  # 180° about X
            [0.0, -0.5, R3, 0.0],  # 180° about (-0.5, R3, 0.0) direction
            [0.0, 0.5, R3, 0.0],  # 180° about (0.5, R3, 0.0) direction
            [0.0, R3, 0.5, 0.0],  # 180° about (R3, 0.5, 0.0) direction
            [0.0, -R3, 0.5, 0.0],  # 180° about (-R3, 0.5, 0.0) direction
            [0.0, 0.0, 1.0, 0.0],  # 180° about Y
        ],
        dtype=torch.float64,
    )

    LAUE_C6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
        ],
        dtype=torch.float64,
    )

    LAUE_D3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # identity
            [0.5, 0.0, 0.0, R3],  # 120° about Z
            [0.5, 0.0, 0.0, -R3],  # 120° about -Z
            [0.0, 1.0, 0.0, 0.0],  # 180° about X
            [0.0, -0.5, R3, 0.0],  # 180° about (-0.5, R3, 0.0) direction
            [0.0, 0.5, R3, 0.0],  # 180° about (0.5, R3, 0.0) direction
        ],
        dtype=torch.float64,
    )

    LAUE_C3 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
        ],
        dtype=torch.float64,
    )

    LAUE_D4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C4 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
        ],
        dtype=torch.float64,
    )

    LAUE_D2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    LAUE_C1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    LAUE_GROUPS = [
        LAUE_C1,  #  1 - Triclinic
        LAUE_C2,  #  2 - Monoclinic
        LAUE_D2,  #  3 - Orthorhombic
        LAUE_C4,  #  4 - Tetragonal low
        LAUE_D4,  #  5 - Tetragonal high
        LAUE_C3,  #  6 - Trigonal low
        LAUE_D3,  #  7 - Trigonal high
        LAUE_C6,  #  8 - Hexagonal low
        LAUE_D6,  #  9 - Hexagonal high
        LAUE_T,  #  10 - Cubic low
        LAUE_O,  #  11 - Cubic high
        LAUE_I,  #  12 - Icosahedral
    ]

    return LAUE_GROUPS[laue_id - 1]


@torch.jit.script
def qu_prod_pos_real(a: Tensor, b: Tensor) -> Tensor:
    """
    Return only the magnitude of the real part of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., ) of quaternion product real part magnitudes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    return ow.abs()


@torch.jit.script
def ori_in_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    Determine if the given unit quaternions with positive real part are in the
    orientation fundamental zone of the given Laue group.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        mask of quaternions in fundamental zone of shape (...,)

    Raises:
        ValueError: if the laue_id is not supported

    """
    if laue_id == 12:  # I: icosahedral
        # cheap gate thresholds based on min and max angle in the RFZ
        W_MAX_FZ = 0.95105651629 + 1e-6  # eps in case my values are imprecise
        W_MIN_FZ = 0.92561479341 - 1e-6
        inside_fast = quats[..., 0] > W_MAX_FZ
        outside_fast = quats[..., 0] < W_MIN_FZ
        need = ~(inside_fast | outside_fast)
        if not torch.any(need):
            return inside_fast
        out_bf = inside_fast  # start true where inside fast
        ops_need = laue_elements(laue_id).to(quats)[1:13]  # the 12 5-fold ops
        dots_need = torch.einsum("ij,nj->ni", ops_need, quats[need])
        max_other_need = dots_need.abs().max(dim=-1).values
        mask_need = quats[need][..., 0] >= max_other_need
        out_bf[need] = mask_need
        return out_bf

    if laue_id == 11:
        # O: cubic high
        # max(abs(x,y,z)) < R2M1*abs(w) and sum(abs(x,y,z)) < abs(w)
        xyz_abs = torch.abs(quats[..., 1:])
        return (
            torch.max(xyz_abs, dim=-1).values <= (quats[..., 0] * (2**0.5 - 1))
        ) & (torch.sum(xyz_abs, dim=-1) <= quats[..., 0])
    elif laue_id == 10:
        # T: cubic low
        # sum(abs(x,y,z)) < abs(w)
        return torch.sum(torch.abs(quats[..., 1:]), dim=-1) <= quats[..., 0]
    elif laue_id == 9:
        # D6: hexagonal high
        # m, n = max(abs(x,y)), min(abs(x,y))
        # if m > TAN75 * n then rot = m else rot = R3O2 * m + 0.5 * n
        # if abs(z) < TAN15 * abs(w) and rot < abs(w) then in FZ
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2 + 3**0.5) * n, m, (3**0.5 / 2) * m + 0.5 * n)
        return (torch.abs(quats[..., 3]) <= (2 - 3**0.5) * quats[..., 0]) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 8:
        # C6: hexagonal low
        # if abs(z) < TAN15 * abs(w)
        return torch.abs(quats[..., 3]) <= ((2 - 3**0.5) * quats[..., 0])
    elif laue_id == 7:
        # D3: trigonal high
        # if abs(x) > abs(y) * R3:
        #   rot = abs(x)
        # else:
        # rot = R3O2 * abs(y) + 0.5 * abs(x)
        # FZ: if abs(z) < TAN30 * abs(w) and rot < abs(w)
        rot = torch.where(
            torch.abs(quats[..., 1]) >= torch.abs(quats[..., 2]) * (3**0.5),
            torch.abs(quats[..., 1]),
            (3**0.5 / 2) * torch.abs(quats[..., 2]) + 0.5 * torch.abs(quats[..., 1]),
        )
        return (torch.abs(quats[..., 3]) <= ((1.0 / 3**0.5) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 6:
        # C3: trigonal low
        # FZ: abs(z) < TAN30 * abs(w)
        return torch.abs(quats[..., 3]) <= (1.0 / 3**0.5) * quats[..., 0]
    elif laue_id == 5:
        # D4: tetragonal high
        # m, n = max(abs(x,y)), min(abs(x,y))
        # if m > TAN67_5 * n then rot = m else rot = R2O2 * m + R2O2 * n
        # FZ: abs(z) < TAN22_5 * abs(w) and rot < abs(w)
        x_abs, y_abs = torch.abs(quats[..., 1]), torch.abs(quats[..., 2])
        cond = x_abs > y_abs
        m = torch.where(cond, x_abs, y_abs)
        n = torch.where(cond, y_abs, x_abs)
        rot = torch.where(m > (2**0.5 + 1) * n, m, (1 / 2**0.5) * m + (1 / 2**0.5) * n)
        return (torch.abs(quats[..., 3]) <= ((2**0.5 - 1) * quats[..., 0])) & (
            rot <= quats[..., 0]
        )
    elif laue_id == 4:
        # C4: tetragonal low
        # FZ: abs(z) < TAN22_5 * abs(w)
        return torch.abs(quats[..., 3]) <= (2**0.5 - 1) * quats[..., 0]
    elif laue_id == 3:
        # D2: orthorhombic
        # FZ: max(abs(x,y,z)) < abs(w)
        return torch.max(torch.abs(quats[..., 1:]), dim=-1).values <= quats[..., 0]
    elif laue_id == 2:
        # C2: monoclinic
        # FZ: abs(z) < abs(w)
        return torch.abs(quats[..., 3]) <= quats[..., 0]
    elif laue_id == 1:
        # C1: triclinic
        return torch.full(quats.shape[:-1], True, dtype=torch.bool, device=quats.device)
    else:
        raise ValueError(f"Laue group {laue_id} is not supported")


def laue_equivalents(q: Tensor, laue_id: int) -> Tensor:
    """
    Return a stacked of equivalent quaternions for the given quaternion under
    the given Laue group.

    Args:
        :q: (..., 4) Tensor of unit quaternions
        :laue_id: laue group of quaternions to get equivalents for

    Returns:
        (..., card, 4) stacked quaternions. card = |K| for laue group K.

    """
    ops = laue_elements(laue_id).to(q)  # get to correct device and dtype
    return qu_prod(q[..., None, :], ops)  # (..., 4) -> (..., card, 4)


def laue_dist_to_fz_boundary(q: Tensor, laue_id: int) -> Tensor:
    """
    Distance (rotation-angle units, radians) from unit quaternion q=[w,x,y,z] to
    the boundary of the positive-scalar fundamental zone (FZ+) for Laue group
    laue_id in 1..12.

        d = 2 * min_{p in ∂FZ+} arccos(q·p) = 2 * arccos( max_{p in ∂FZ+} q·p )

    No antipodal identification in distances: uses acos(clamp(q·p)).

    Args:
        :q: (..., 4) Tensor of unit quaternions
        :laue_id: laue group of quaternions to get distance to boundary for

    Returns:
        (...,) Tensor of distances in radians
    """

    if laue_id < 1 or laue_id > 12:
        raise ValueError(f"laue_id must be 1..12, got {laue_id}")

    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    _PI = math.pi
    _SQRT2 = math.sqrt(2.0)
    _SQRT3 = math.sqrt(3.0)
    _SQRT5 = math.sqrt(5.0)

    _INV_SQRT2 = 1.0 / _SQRT2
    _INV_SQRT3 = 1.0 / _SQRT3

    _K = _SQRT2 - 1.0  # tan(pi/8)
    _K2 = _K * _K
    _TAN15 = 2.0 - _SQRT3  # tan(pi/12)
    _TAN30 = _INV_SQRT3  # tan(pi/6)

    _EPS = 1e-12
    _TOL = 1e-10

    # ───────────────────────── C1 ─────────────────────────
    if laue_id == 1:
        return 2.0 * torch.asin(w.abs().clamp(0.0, 1.0))

    # ─────────────────────── Cyclic ───────────────────────
    if laue_id in (2, 4, 6, 8):
        if laue_id == 2:
            k = 1.0
        elif laue_id == 4:
            k = _K
        elif laue_id == 6:
            k = _TAN30
        else:
            k = _TAN15

        r_xy = (x * x + y * y).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        d_circle = torch.acos(r_xy)

        denom = math.sqrt(1.0 + k * k)

        s_p = ((z - k * w).abs() / denom).clamp(0.0, 1.0)
        d_p = torch.asin(s_p)
        ok_p = (w + k * z) >= 0.0

        s_m = ((z + k * w).abs() / denom).clamp(0.0, 1.0)
        d_m = torch.asin(s_m)
        ok_m = (w - k * z) >= 0.0

        half = torch.minimum(
            torch.minimum(
                torch.where(ok_p, d_p, d_circle), torch.where(ok_m, d_m, d_circle)
            ),
            d_circle,
        )
        return 2.0 * half

    # ───────────────────────── D2 ─────────────────────────
    if laue_id == 3:
        ax = x.abs()
        ay = y.abs()
        az = z.abs()
        neg = w.new_full(w.shape, -1.0)

        # Faces: w=x / w=y / w=z
        m1 = 0.5 * (w + ax)
        dot_f1 = (2.0 * m1 * m1 + ay * ay + az * az).clamp_min(0.0).sqrt()
        ok_f1 = (m1 - ay >= -_TOL) & (m1 - az >= -_TOL)

        m2 = 0.5 * (w + ay)
        dot_f2 = (ax * ax + 2.0 * m2 * m2 + az * az).clamp_min(0.0).sqrt()
        ok_f2 = (m2 - ax >= -_TOL) & (m2 - az >= -_TOL)

        m3 = 0.5 * (w + az)
        dot_f3 = (ax * ax + ay * ay + 2.0 * m3 * m3).clamp_min(0.0).sqrt()
        ok_f3 = (m3 - ax >= -_TOL) & (m3 - ay >= -_TOL)

        # Ridges: w=x=y / w=x=z / w=y=z
        r12 = (w + ax + ay) / 3.0
        dot_r12 = (3.0 * r12 * r12 + az * az).clamp_min(0.0).sqrt()
        ok_r12 = r12 - az >= -_TOL

        r13 = (w + ax + az) / 3.0
        dot_r13 = (3.0 * r13 * r13 + ay * ay).clamp_min(0.0).sqrt()
        ok_r13 = r13 - ay >= -_TOL

        r23 = (w + ay + az) / 3.0
        dot_r23 = (3.0 * r23 * r23 + ax * ax).clamp_min(0.0).sqrt()
        ok_r23 = r23 - ax >= -_TOL

        # Vertex: [1,1,1,1]/2
        dot_v = 0.5 * (w + ax + ay + az)

        best = (
            torch.stack(
                [
                    torch.where(ok_f1, dot_f1, neg),
                    torch.where(ok_f2, dot_f2, neg),
                    torch.where(ok_f3, dot_f3, neg),
                    torch.where(ok_r12, dot_r12, neg),
                    torch.where(ok_r13, dot_r13, neg),
                    torch.where(ok_r23, dot_r23, neg),
                    dot_v,
                ],
                dim=-1,
            )
            .max(dim=-1)
            .values.clamp(-1.0, 1.0)
        )

        return 2.0 * torch.acos(best)

    # ───────────────────────── D4 ─────────────────────────
    if laue_id == 5:
        qz = z.abs()
        ax = x.abs()
        ay = y.abs()
        qx = torch.maximum(ax, ay)
        qy = torch.minimum(ax, ay)

        neg = w.new_full(w.shape, -1.0)
        inv_d = 1.0 / (1.0 + _K2)
        inv_s2 = _INV_SQRT2
        s2 = _SQRT2

        # Face cap: z = K w
        w1 = (w + _K * qz) * inv_d
        z1 = _K * w1
        dot_f1 = (
            (w1 * w1 + qx * qx + qy * qy + z1 * z1)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        ok_f1 = (w1 - qx >= -_TOL) & (s2 * w1 - (qx + qy) >= -_TOL)

        # Face side: w = x
        m = 0.5 * (w + qx)
        dot_f2 = (2.0 * m * m + qy * qy + qz * qz).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        ok_f2 = (_K * m - qz >= -_TOL) & (s2 * m - (m + qy) >= -_TOL)

        # Face diagonal: w = (x+y)/sqrt2
        t3 = 0.5 * (w - (qx + qy) * inv_s2)
        w3 = w - t3
        x3 = qx + t3 * inv_s2
        y3 = qy + t3 * inv_s2
        dot_f3 = (
            (w3 * w3 + x3 * x3 + y3 * y3 + qz * qz)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        ok_f3 = (_K * w3 - qz >= -_TOL) & (w3 - x3 >= -_TOL)

        # Ridge cap∩side: {z=K w, x=w}, y free
        den_ts = 2.0 + _K2
        a_ts = (w + qx + _K * qz) / den_ts
        dot_r1 = (den_ts * a_ts * a_ts + qy * qy).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        ok_r1 = s2 * a_ts - (a_ts + qy) >= -_TOL

        # Ridge cap∩diag: {z=K w, w=(x+y)/sqrt2}, (x-y) free
        a_td = (w + (qx + qy) * inv_s2 + _K * qz) / (2.0 + _K2)
        b_td = 0.5 * (qx - qy)
        dot_r2 = (
            ((2.0 + _K2) * a_td * a_td + 2.0 * b_td * b_td)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        x_r2 = a_td * inv_s2 + b_td
        ok_r2 = a_td - x_r2 >= -_TOL

        # Ridge side∩diag: {x=w, y=K w}, z free
        a_tz = (w + qx + _K * qy) / (2.0 + _K2)
        dot_r3 = (
            ((2.0 + _K2) * a_tz * a_tz + qz * qz).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        )
        ok_r3 = _K * a_tz - qz >= -_TOL

        # Vertex: [sqrt2+1, sqrt2+1, 1, 1] / (2*sqrt(2+sqrt2))
        inv_vden = 1.0 / (2.0 * math.sqrt(2.0 + _SQRT2))
        vw = (_SQRT2 + 1.0) * inv_vden
        vx = vw
        vy = 1.0 * inv_vden
        vz = 1.0 * inv_vden
        dot_v = (w * vw + qx * vx + qy * vy + qz * vz).clamp(-1.0, 1.0)

        best = (
            torch.stack(
                [
                    torch.where(ok_f1, dot_f1, neg),
                    torch.where(ok_f2, dot_f2, neg),
                    torch.where(ok_f3, dot_f3, neg),
                    torch.where(ok_r1, dot_r1, neg),
                    torch.where(ok_r2, dot_r2, neg),
                    torch.where(ok_r3, dot_r3, neg),
                    dot_v,
                ],
                dim=-1,
            )
            .max(dim=-1)
            .values.clamp(-1.0, 1.0)
        )

        return 2.0 * torch.acos(best)

    # ───────────────────── D3 / D6 prism ───────────────────
    if laue_id in (7, 9):
        if laue_id == 7:
            a = _TAN30
            step = _PI / 3.0
        else:
            a = _TAN15
            step = _PI / 6.0

        c1 = 2.0 + a * a
        tv = 1.0 / math.sqrt(2.0 * (1.0 + a * a))

        x0 = x
        y0 = y
        qz = z.abs()

        phi = torch.atan2(y0, x0)
        n = torch.round(phi / step)
        ang = -n * step
        cs = torch.cos(ang)
        sn = torch.sin(ang)

        qx = cs * x0 - sn * y0
        qy = (sn * x0 + cs * y0).abs()

        pv_w = tv
        pv_x = tv
        pv_y = a * tv
        pv_z = a * tv

        # Ridge 1 (free = qy)
        A1 = w + qx + a * qz
        den1 = (A1 * A1 / c1 + qy * qy).clamp_min(_EPS).sqrt()
        t1 = (A1 / c1) / den1
        u1 = qy / den1
        clamp1 = u1 > a * t1
        e1w = torch.where(clamp1, pv_w, t1)
        e1x = torch.where(clamp1, pv_x, t1)
        e1y = torch.where(clamp1, pv_y, u1)
        e1z = torch.where(clamp1, pv_z, a * t1)
        d_e1 = torch.acos((w * e1w + qx * e1x + qy * e1y + qz * e1z).clamp(-1.0, 1.0))

        # Ridge 2 (free = qz)
        A2 = w + qx + a * qy
        den2 = (A2 * A2 / c1 + qz * qz).clamp_min(_EPS).sqrt()
        t2 = (A2 / c1) / den2
        u2 = qz / den2
        clamp2 = u2 > a * t2
        e2w = torch.where(clamp2, pv_w, t2)
        e2x = torch.where(clamp2, pv_x, t2)
        e2y = torch.where(clamp2, pv_y, a * t2)
        e2z = torch.where(clamp2, pv_z, u2)
        d_e2 = torch.acos((w * e2w + qx * e2x + qy * e2y + qz * e2z).clamp(-1.0, 1.0))

        # Cap face: z = a w
        sz = (qz - a * w) / (1.0 + a * a)
        pw = w + a * sz
        px = qx
        py = qy
        pz = qz - sz
        invn = 1.0 / (pw * pw + px * px + py * py + pz * pz).clamp_min(_EPS).sqrt()
        pw = pw * invn
        px = px * invn
        py = py * invn
        pz = pz * invn

        phi_p = torch.atan2(py, px)
        np_ = torch.round(phi_p / step)
        ang_p = -np_ * step
        cp = torch.cos(ang_p)
        sp = torch.sin(ang_p)
        px2 = cp * px - sp * py
        py2 = (sp * px + cp * py).abs()

        d_pz_raw = torch.acos((w * pw + qx * px2 + qy * py2 + qz * pz).clamp(-1.0, 1.0))
        d_pz = torch.where(px2 <= pw + _TOL, d_pz_raw, d_e1)

        # Side face: w = x
        m = 0.5 * (w + qx)
        sw = m
        sx = m
        sy = qy
        sz2 = qz
        invn2 = 1.0 / (sw * sw + sx * sx + sy * sy + sz2 * sz2).clamp_min(_EPS).sqrt()
        sw = sw * invn2
        sx = sx * invn2
        sy = sy * invn2
        sz2 = sz2 * invn2
        d_px_raw = torch.acos((w * sw + qx * sx + qy * sy + qz * sz2).clamp(-1.0, 1.0))

        bad_z = sz2 > a * sw + _TOL
        bad_y = sy > a * sw + _TOL
        d_px = torch.where(
            bad_z & bad_y,
            torch.minimum(d_e1, d_e2),
            torch.where(bad_z, d_e1, torch.where(bad_y, d_e2, d_px_raw)),
        )

        half = torch.minimum(torch.minimum(d_pz, d_px), torch.minimum(d_e1, d_e2))
        return 2.0 * half

    # ───────────────────────── T ──────────────────────────
    if laue_id == 10:
        ax = x.abs()
        ay = y.abs()
        az = z.abs()

        m1 = torch.maximum(ax, torch.maximum(ay, az))
        m3 = torch.minimum(ax, torch.minimum(ay, az))
        m2 = (ax + ay + az) - m1 - m3

        qz = m1
        qx = m2
        qy = m3

        f = w - (qx + qy + qz)
        pw = w - 0.25 * f
        px = qx + 0.25 * f
        py = qy + 0.25 * f
        pz = qz + 0.25 * f

        face_ok = (px >= -_TOL) & (py >= -_TOL) & (pz >= -_TOL)
        dot_face = (
            (pw * pw + px * px + py * py + pz * pz)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        d_face = torch.acos(dot_face)

        aw = w + qx
        bw = w + qz
        xr = (2.0 * aw - bw) / 3.0
        zr = (2.0 * bw - aw) / 3.0
        den = (2.0 * xr * xr + 2.0 * zr * zr + 2.0 * xr * zr).clamp_min(_EPS).sqrt()
        xr = xr / den
        zr = zr / den
        wr = xr + zr
        ridge_ok = (xr >= -_TOL) & (zr >= -_TOL)
        d_ridge = torch.acos((w * wr + qx * xr + qz * zr).clamp(-1.0, 1.0))

        d_vz = torch.acos((_INV_SQRT2 * (w + qz)).clamp(-1.0, 1.0))
        d_vx = torch.acos((_INV_SQRT2 * (w + qx)).clamp(-1.0, 1.0))
        d_end = torch.minimum(d_vx, d_vz)

        d_ridge = torch.where(ridge_ok, d_ridge, d_end)
        half = torch.where(face_ok, torch.minimum(d_face, d_ridge), d_ridge)
        return 2.0 * half

    # ───────────────────────── O ──────────────────────────
    if laue_id == 11:
        ax = x.abs()
        ay = y.abs()
        az = z.abs()

        L = torch.maximum(ax, torch.maximum(ay, az))
        S = torch.minimum(ax, torch.minimum(ay, az))
        M = (ax + ay + az) - L - S

        qx = M
        qy = S
        qz = L

        neg = w.new_full(w.shape, -1.0)
        inv_d = 1.0 / (1.0 + _K2)

        # Face capZ: z = K w
        w1 = (w + _K * qz) * inv_d
        z1 = _K * w1
        dot_f1 = (
            (w1 * w1 + qx * qx + qy * qy + z1 * z1)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        ok_f1 = (w1 - (qx + qy + z1) >= -_TOL) & (_K * w1 - qx >= -_TOL)

        # Face tri: w = x+y+z
        f = w - (qx + qy + qz)
        t = 0.25 * f
        w2 = w - t
        x2 = qx + t
        y2 = qy + t
        z2 = qz + t
        dot_f2 = (
            (w2 * w2 + x2 * x2 + y2 * y2 + z2 * z2)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        ok_f2 = (_K * w2 - z2 >= -_TOL) & (_K * w2 - x2 >= -_TOL)

        # Face capX: x = K w
        w3 = (w + _K * qx) * inv_d
        x3 = _K * w3
        dot_f3 = (
            (w3 * w3 + x3 * x3 + qy * qy + qz * qz)
            .clamp_min(0.0)
            .sqrt()
            .clamp(0.0, 1.0)
        )
        ok_f3 = (_K * w3 - qz >= -_TOL) & (w3 - (x3 + qy + qz) >= -_TOL)

        # Ridge capZ∩capX: {z=K w, x=K w}, y free
        den_t = 1.0 + 2.0 * _K2
        a1 = (w + _K * qx + _K * qz) / den_t
        dot_r1 = (den_t * a1 * a1 + qy * qy).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        ok_r1 = (1.0 - 2.0 * _K) * a1 - qy >= -_TOL

        # Ridge capZ∩tri: {z=K w, w=x+y+z}
        s = 0.5 * (1.0 - _K)
        den_e = 1.0 + 2.0 * (s * s) + _K2
        a2 = (w + s * qx + s * qy + _K * qz) / den_e
        b2 = 0.5 * (qx - qy)
        dot_r2 = (den_e * a2 * a2 + 2.0 * b2 * b2).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        x_r2 = a2 * s + b2
        ok_r2 = _K * a2 - x_r2 >= -_TOL

        # Ridge tri∩capX: {x=K w, w=x+y+z}
        den_e3 = 1.0 + _K2 + 2.0 * (s * s)
        a3 = (w + _K * qx + s * qy + s * qz) / den_e3
        b3 = 0.5 * (qy - qz)
        dot_r3 = (
            (den_e3 * a3 * a3 + 2.0 * b3 * b3).clamp_min(0.0).sqrt().clamp(0.0, 1.0)
        )
        z_r3 = a3 * s - b3
        ok_r3 = _K * a3 - z_r3 >= -_TOL

        # Vertex: [1+sqrt2, 1, sqrt2-1, 1]/(2*sqrt2)
        inv_vden = 1.0 / (2.0 * _SQRT2)
        vw = (1.0 + _SQRT2) * inv_vden
        vx = 1.0 * inv_vden
        vy = (_SQRT2 - 1.0) * inv_vden
        vz = 1.0 * inv_vden
        dot_v = (w * vw + qx * vx + qy * vy + qz * vz).clamp(-1.0, 1.0)

        best = (
            torch.stack(
                [
                    torch.where(ok_f1, dot_f1, neg),
                    torch.where(ok_f2, dot_f2, neg),
                    torch.where(ok_f3, dot_f3, neg),
                    torch.where(ok_r1, dot_r1, neg),
                    torch.where(ok_r2, dot_r2, neg),
                    torch.where(ok_r3, dot_r3, neg),
                    dot_v,
                ],
                dim=-1,
            )
            .max(dim=-1)
            .values.clamp(-1.0, 1.0)
        )

        return 2.0 * torch.acos(best)

    # ───────────────────────── I ──────────────────────────
    if laue_id == 12:
        PI5 = _PI / 5.0
        W72 = 2.0 * _PI / 5.0

        c36 = math.cos(math.radians(36.0))
        s36 = math.sin(math.radians(36.0))
        cos72 = math.cos(math.radians(72.0))
        sin72 = math.sin(math.radians(72.0))

        cos_beta = 1.0 / _SQRT5
        sin_beta = 2.0 / _SQRT5

        v = q[..., 1:]
        vnorm = v.norm(dim=-1)
        is_id = vnorm < _EPS

        # cache 12 axes (device,dtype)
        cache = getattr(laue_dist_to_fz_boundary, "_I_axes_cache", None)
        if cache is None:
            cache = {}
            setattr(laue_dist_to_fz_boundary, "_I_axes_cache", cache)
        key = (q.device, q.dtype)
        axes = cache.get(key, None)
        if axes is None:
            axes = q.new_tensor(
                [
                    [-0.7236067977499790, -0.5257311121191336, 0.4472135954999579],
                    [-0.7236067977499790, 0.5257311121191336, 0.4472135954999579],
                    [0.0, 0.0, 1.0],
                    [0.2763932022500210, -0.8506508083520400, 0.4472135954999579],
                    [0.2763932022500210, 0.8506508083520400, 0.4472135954999579],
                    [0.8944271909999160, 0.0, 0.4472135954999579],
                    [0.7236067977499790, -0.5257311121191336, -0.4472135954999579],
                    [-0.2763932022500210, 0.8506508083520400, -0.4472135954999579],
                    [0.0, 0.0, -1.0],
                    [-0.2763932022500210, -0.8506508083520400, -0.4472135954999579],
                    [-0.8944271909999160, 0.0, -0.4472135954999579],
                    [0.7236067977499790, 0.5257311121191336, -0.4472135954999579],
                ]
            )
            cache[key] = axes

        # n_hat (safe)
        n_hat = torch.where(
            vnorm[..., None] > _EPS,
            v / vnorm[..., None].clamp_min(_EPS),
            torch.zeros_like(v),
        )

        idx = torch.einsum("...i,ji->...j", n_hat, axes).argmax(dim=-1)
        a_ax = axes[idx]

        X = q.new_tensor([1.0, 0.0, 0.0])
        Z = q.new_tensor([0.0, 0.0, 1.0])

        is_posZ = (a_ax[..., 2] - 1.0).abs() < _EPS
        is_negZ = (a_ax[..., 2] + 1.0).abs() < _EPS
        seed = torch.where(
            is_posZ[..., None],
            X.expand_as(a_ax),
            torch.where(
                is_negZ[..., None],
                (-X).expand_as(a_ax),
                torch.where(a_ax[..., 2:3] >= 0.0, Z, -Z).expand_as(a_ax),
            ),
        )

        e0 = seed - (seed * a_ax).sum(-1, keepdim=True) * a_ax
        e0 = e0 / e0.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        e1 = torch.cross(a_ax, e0, dim=-1)

        psi = torch.atan2((v * e1).sum(-1), (v * e0).sum(-1))
        k_fold = torch.round(psi / W72)
        ang = k_fold * W72
        cs = torch.cos(ang)
        sn = torch.sin(ang)

        e0r = cs[..., None] * e0 + sn[..., None] * e1
        e1r = -sn[..., None] * e0 + cs[..., None] * e1
        e1c = torch.where(((v * e1r).sum(-1) < 0.0)[..., None], -e1r, e1r)

        # local coordinates of v in (e0r, e1c, a_ax)
        v0 = (v * e0r).sum(-1)
        v1 = (v * e1c).sum(-1)
        v2 = (v * a_ax).sum(-1)

        # operator quats in this local basis (constants)
        # qa = [c36, 0, 0, s36]
        # qb = [c36, s36*sin_beta, 0, s36*cos_beta]
        qb_x = s36 * sin_beta
        qb_z = s36 * cos_beta
        qc_x = s36 * sin_beta * cos72
        qc_y = s36 * sin_beta * sin72
        qc_z = s36 * cos_beta

        # bisector normals N = e - op (constants) for face(a) and ridge(a,b)
        a0 = 1.0 - c36
        N1_w = a0
        N1_x = 0.0
        N1_y = 0.0
        N1_z = -s36

        N2_w = a0
        N2_x = -qb_x
        N2_y = 0.0
        N2_z = -qb_z

        # ── Face: project onto N1·p=0 ──
        r1 = w * N1_w + v0 * N1_x + v1 * N1_y + v2 * N1_z
        n11 = N1_w * N1_w + N1_x * N1_x + N1_y * N1_y + N1_z * N1_z
        t = r1 / max(n11, _EPS)

        pfw = w - t * N1_w
        pfx = v0 - t * N1_x
        pfy = v1 - t * N1_y
        pfz = v2 - t * N1_z

        invn = (
            1.0 / (pfw * pfw + pfx * pfx + pfy * pfy + pfz * pfz).clamp_min(_EPS).sqrt()
        )
        pfw = pfw * invn
        pfx = pfx * invn
        pfy = pfy * invn
        pfz = pfz * invn

        # feasibility: pfw >= |pf·qb|, |pf·qc|, |pf·qd|
        abs_b = (pfw * c36 + pfx * qb_x + pfz * qb_z).abs()
        abs_c = (pfw * c36 + pfx * qc_x + pfy * qc_y + pfz * qc_z).abs()
        abs_d = (pfw * c36 + pfx * qc_x - pfy * qc_y + pfz * qc_z).abs()
        face_ok = (pfw >= abs_b - _TOL) & (pfw >= abs_c - _TOL) & (pfw >= abs_d - _TOL)

        d_face = torch.acos((w * pfw + v0 * pfx + v1 * pfy + v2 * pfz).clamp(-1.0, 1.0))

        # ── Ridge: project onto N1·p=0 ∧ N2·p=0 ──
        r2 = w * N2_w + v0 * N2_x + v1 * N2_y + v2 * N2_z

        a11 = n11
        a22 = N2_w * N2_w + N2_x * N2_x + N2_y * N2_y + N2_z * N2_z
        a12 = N1_w * N2_w + N1_x * N2_x + N1_y * N2_y + N1_z * N2_z
        det = max(a11 * a22 - a12 * a12, _EPS)

        c1 = (a22 * r1 - a12 * r2) / det
        c2 = (a11 * r2 - a12 * r1) / det

        prw = w - c1 * N1_w - c2 * N2_w
        prx = v0 - c1 * N1_x - c2 * N2_x
        pry = v1 - c1 * N1_y - c2 * N2_y
        prz = v2 - c1 * N1_z - c2 * N2_z

        invn2 = (
            1.0 / (prw * prw + prx * prx + pry * pry + prz * prz).clamp_min(_EPS).sqrt()
        )
        prw = prw * invn2
        prx = prx * invn2
        pry = pry * invn2
        prz = prz * invn2

        abs_c_r = (prw * c36 + prx * qc_x + pry * qc_y + prz * qc_z).abs()
        abs_d_r = (prw * c36 + prx * qc_x - pry * qc_y + prz * qc_z).abs()
        ridge_ok = (prw >= abs_c_r - _TOL) & (prw >= abs_d_r - _TOL)

        d_ridge = torch.acos(
            (w * prw + v0 * prx + v1 * pry + v2 * prz).clamp(-1.0, 1.0)
        )

        # ── Vertices: only TWO constants in this local basis ──
        # v123 = unit ⟂(N1,N2,N3), v124 = unit ⟂(N1,N2,N4), with fixed signs (abs dot handles ±)
        vw = 0.9256147934109580
        vx = 0.1858740172300923
        vy = 0.1350453783688632
        vz = 0.3007504775037727

        dot123 = (w * vw + v0 * vx + v1 * vy + v2 * vz).abs().clamp(0.0, 1.0)
        dot124 = (w * vw + v0 * vx - v1 * vy + v2 * vz).abs().clamp(0.0, 1.0)
        d_vert = torch.acos(torch.maximum(dot123, dot124))

        d_ridge = torch.where(ridge_ok, d_ridge, d_vert)
        d_face = torch.where(face_ok, d_face, d_ridge)
        half = torch.minimum(d_face, d_ridge)

        half = torch.where(is_id, torch.full_like(w, PI5), half)
        return 2.0 * half

    raise RuntimeError("unreachable")


@torch.jit.script
def ori_to_fz_laue(quats: Tensor, laue_id: int) -> Tensor:
    """
    This function moves the given quaternions to the fundamental zone of the
    given Laue group. This computes the orientation fundamental zone, not the
    misorientation fundamental zone.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        orientations in fundamental zone of shape (..., 4)

    Notes:

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432
    12) Laue I      Icosahedral: 532

    """
    # get the important shapes
    data_shape = quats.shape
    N = torch.prod(torch.tensor(data_shape[:-1]))
    laue_group = laue_elements(laue_id).to(quats.dtype).to(quats.device)
    card = laue_group.shape[0]

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    )

    # find the quaternion with the largest w value
    row_maximum_indices = torch.argmax(equivalent_quaternions_real, dim=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = qu_prod(quats.reshape(N, 4), laue_group[row_maximum_indices])

    return output.reshape(data_shape)
