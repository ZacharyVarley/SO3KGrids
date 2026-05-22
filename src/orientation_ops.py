import torch
from torch import Tensor


@torch.jit.script
def qu_std(qu: Tensor) -> Tensor:
    """
    Standardize unit quaternion to have non-negative real part.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(qu[..., 0:1] >= 0, qu, -qu)


@torch.jit.script
def qu_conj(qu: Tensor) -> Tensor:
    """
    Conjugate of a quaternion.
    """
    qu[..., 1:] = -qu[..., 1:]
    return qu


@torch.jit.script
def qu_norm(qu: Tensor) -> Tensor:
    """
    Normalize quaternions to unit norm.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of normalized quaternions.
    """
    return qu / torch.norm(qu, dim=-1, keepdim=True)


@torch.jit.script
def qu_prod_raw(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


@torch.jit.script
def qu_prod(a: Tensor, b: Tensor) -> Tensor:
    """
    Quaternion multiplication, then make real part non-negative.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b Tensor shape (..., 4) of the quaternion product.

    """
    ab = qu_prod_raw(a, b)
    return qu_std(ab)


@torch.jit.script
def om2qu(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).

    Notes:

    Farrell, J.A., 2015. Computation of the Quaternion from a Rotation Matrix.
    University of California, 2.

    "Converting a Rotation Matrix to a Quaternion" by Mike Day, Insomniac Games

    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m10 = matrix[..., 1, 0]

    mask_A = m22 < 0
    mask_B = m00 > m11
    mask_C = m00 < -m11
    branch_1 = mask_A & mask_B
    branch_2 = mask_A & ~mask_B
    branch_3 = ~mask_A & mask_C
    branch_4 = ~mask_A & ~mask_C

    branch_1_t = 1 + m00[branch_1] - m11[branch_1] - m22[branch_1]
    branch_1_t_rsqrt = 0.5 * torch.rsqrt(branch_1_t)
    branch_2_t = 1 - m00[branch_2] + m11[branch_2] - m22[branch_2]
    branch_2_t_rsqrt = 0.5 * torch.rsqrt(branch_2_t)
    branch_3_t = 1 - m00[branch_3] - m11[branch_3] + m22[branch_3]
    branch_3_t_rsqrt = 0.5 * torch.rsqrt(branch_3_t)
    branch_4_t = 1 + m00[branch_4] + m11[branch_4] + m22[branch_4]
    branch_4_t_rsqrt = 0.5 * torch.rsqrt(branch_4_t)

    qu = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    qu[branch_1, 1] = branch_1_t * branch_1_t_rsqrt
    qu[branch_1, 2] = (m01[branch_1] + m10[branch_1]) * branch_1_t_rsqrt
    qu[branch_1, 3] = (m20[branch_1] + m02[branch_1]) * branch_1_t_rsqrt
    qu[branch_1, 0] = (m12[branch_1] - m21[branch_1]) * branch_1_t_rsqrt

    qu[branch_2, 1] = (m01[branch_2] + m10[branch_2]) * branch_2_t_rsqrt
    qu[branch_2, 2] = branch_2_t * branch_2_t_rsqrt
    qu[branch_2, 3] = (m12[branch_2] + m21[branch_2]) * branch_2_t_rsqrt
    qu[branch_2, 0] = (m20[branch_2] - m02[branch_2]) * branch_2_t_rsqrt

    qu[branch_3, 1] = (m20[branch_3] + m02[branch_3]) * branch_3_t_rsqrt
    qu[branch_3, 2] = (m12[branch_3] + m21[branch_3]) * branch_3_t_rsqrt
    qu[branch_3, 3] = branch_3_t * branch_3_t_rsqrt
    qu[branch_3, 0] = (m01[branch_3] - m10[branch_3]) * branch_3_t_rsqrt

    qu[branch_4, 1] = (m12[branch_4] - m21[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 2] = (m20[branch_4] - m02[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 3] = (m01[branch_4] - m10[branch_4]) * branch_4_t_rsqrt
    qu[branch_4, 0] = branch_4_t * branch_4_t_rsqrt

    # guarantee the correct axis signs
    qu[..., 0] = torch.abs(qu[..., 0])
    qu[..., 1] = qu[..., 1].copysign((m21 - m12))
    qu[..., 2] = qu[..., 2].copysign((m02 - m20))
    qu[..., 3] = qu[..., 3].copysign((m10 - m01))

    return qu


@torch.jit.script
def cu2ho(cu: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to homochoric vector representation.

    Args:
        cu: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        Homochoric vectors as tensor of shape (..., 3).
    """

    # Sort components
    indices = torch.argsort(torch.abs(cu), dim=-1, descending=False)
    sorted = torch.gather(cu, -1, indices)
    s, m, b = sorted.unbind(dim=-1)

    # Calculate trigonometric argument and avoid indeterminate forms
    trig_arg_xy = s * torch.pi / (12.0 * m)
    trig_arg_xy[torch.isnan(trig_arg_xy)] = 0

    factor_xy = (
        2 ** (1 / 12)
        * 3 ** (1 / 3)
        * m
        * torch.sqrt(
            (
                4 * b**2 * (torch.cos(trig_arg_xy) - (2**0.5))
                + (2**0.5) * m**2 * (-2 * (2**0.5) * torch.cos(trig_arg_xy) + 3)
            )
            / (torch.cos(trig_arg_xy) - (2**0.5))
        )
        / (torch.pi ** (1 / 3) * b * torch.sqrt(-torch.cos(trig_arg_xy) + (2**0.5)))
    )

    # Compute x_s3, y_s3, and z_s3 using the provided equations
    x_s3 = factor_xy * torch.sin(trig_arg_xy)
    y_s3 = factor_xy * 2**-0.5 * ((2**0.5) * torch.cos(trig_arg_xy) - 1)
    z_s3 = (
        2 * 6 ** (1 / 3) * b**2 * (torch.cos(trig_arg_xy) - (2**0.5))
        + 2 ** (5 / 6)
        * 3 ** (1 / 3)
        * m**2
        * (-2 * (2**0.5) * torch.cos(trig_arg_xy) + 3)
    ) / (2 * torch.pi ** (1 / 3) * b * (torch.cos(trig_arg_xy) - (2**0.5)))

    # Reassemble the vector and undo the argsort
    ho = torch.stack((x_s3, y_s3, z_s3), dim=-1)
    ho = torch.scatter(ho, -1, indices, ho)

    # replace any nans with 0
    ho[torch.isnan(ho)] = 0

    # copy the sign of the original cubochoric vector
    ho.copysign_(cu)

    return ho


@torch.jit.script
def ho2cu(ho: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to cubochoric vector representation.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).

    """

    # inverse steps in reverse order of cu2ho
    # start with an argsort on the magnitudes
    indices = torch.argsort(torch.abs(ho), dim=-1, descending=False)
    sorted = torch.gather(ho, -1, indices)
    x_s3, y_s3, z_s3 = torch.abs(sorted).unbind(dim=-1)

    # step 3 inverse
    r_s = torch.norm(ho, dim=-1, keepdim=False)
    prefactor_xy_s3 = torch.sqrt(2 * r_s / (r_s + z_s3))
    x_s2 = x_s3 * prefactor_xy_s3
    y_s2 = y_s3 * prefactor_xy_s3
    z_s2 = (torch.pi / 6) ** 0.5 * r_s

    # # step 2 inverse from Appendix A eq (29)
    # prefactor_xy_s2 = (torch.pi / 6)**0.5 * torch.sqrt((x_s2**2 + 2 * y_s2**2) * (x_s2**2 + y_s2**2)) / (
    #     (2 ** 0.5) * torch.sqrt(x_s2**2 + 2 * y_s2**2 - (torch.abs(y_s2) * torch.sqrt(x_s2**2 + 2 * y_s2**2)))
    # )
    # the above equation in the publication can be dramatically simplified if you assume x and y are positive:
    # ((x^2 + 2 * y^2) * (x^2 + y^2)) / (x^2 + 2 * y^2 - |y| * sqrt(x^2 + 2*y^2)) is
    # the same as:
    # x^2 + y*(sqrt(x^2 + 2*y^2) + 2*y)
    # for x and y positive
    # these are the inverse squircle functions
    # this also avoids an annoying 0/0 case that slows down the calculation
    prefactor_xy_s2 = (
        (torch.pi / 6) ** 0.5
        * torch.sqrt(x_s2**2 + y_s2 * (torch.sqrt(x_s2**2 + 2 * y_s2**2) + 2 * y_s2))
        / (2**0.5)
    )
    # z_s1 is unchanged from z_s2 while x_s1 and y_s1 are found from inverse squircle function
    x_s1 = (prefactor_xy_s2 * 12.0 * torch.sign(x_s2) / torch.pi) * (
        torch.arccos(
            (
                (x_s2**2 + y_s2 * torch.sqrt(x_s2**2 + 2 * y_s2**2))
                / ((2**0.5) * (x_s2**2 + y_s2**2))
            ).clamp_(-1.0, 1.0)
        )
    )
    y_s1 = prefactor_xy_s2 * torch.sign(y_s2)

    # undo the argsort with an in-place scatter
    cu = torch.empty_like(ho)
    cu.scatter_(-1, indices, torch.stack((x_s1, y_s1, z_s2), dim=-1))
    cu /= (torch.pi / 6) ** (1 / 6)

    # copy the sign of the original homochoric vector
    cu.copysign_(ho)

    # replace any nans with 0
    cu[torch.isnan(cu)] = 0

    return cu


@torch.jit.script
def ho2ax(ho: Tensor, fast: bool = True) -> Tensor:
    """
    Converts a set of homochoric vectors to axis-angle representation.

    Args:
        ho (Tensor): shape (..., 3) homochoric coordinates (x, y, z)
        fast (bool): by default skip Newton iteration for FP64 only

    Returns:
        torch.Tensor: shape (..., 4) axis-angles (x, y, z, angle)


    Notes:

    These are Chebyshev fits on the modified homochoric inverse
    fitted by Zachary Varley on 05/24/2024. The modified homochoric
    inverse ties the square of the homochoric vector back to the
    cosine of the half rotation angle.

    """
    if ho.dtype == torch.float32 or ho.dtype == torch.float16:
        fit_parameters = torch.tensor(
            [
                # 8 terms to reach FP32 machine eps
                1.0000000000000009e00,
                -4.9999943403867775e-01,
                -2.5015165060149020e-02,
                -3.8120131548551729e-03,
                -1.2106188330642162e-03,
                4.9329295993155416e-04,
                -7.0089385526450620e-04,
                3.0979774923589078e-04,
                -7.3023474963298843e-05,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    else:
        # 10 term loss mean abs error 5e-10 instead of 15 term's 1e-11
        # 1 iteration of Newton's method is needed for double precision
        # machine error... 20 terms somehow is stuck at 1e-11 error
        fit_parameters = torch.tensor(
            [
                # 10 term polyfit
                1.0000000000000000e00,
                -4.9999997124013285e-01,
                -2.5001181866044025e-02,
                -3.9144209820521038e-03,
                -8.9320268104539483e-04,
                3.1181024286083695e-05,
                -4.3961032788396477e-04,
                3.9657471727506439e-04,
                -2.6379945050586932e-04,
                9.1185355979587159e-05,
                -1.4875867805692529e-05,
            ],
            dtype=ho.dtype,
            device=ho.device,
        ).to(ho.dtype)

    # ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=True)
    # # makes out of memory error doing all at once
    # s = torch.zeros_like(ho_norm_sq[..., 0])
    # for i in range(len(fit_parameters)):
    #     s += fit_parameters[i] * ho_norm_sq[..., 0] ** i

    ho_norm_sq = torch.sum(ho**2, dim=-1, keepdim=False)
    # makes out of memory error doing all at once
    s = torch.zeros_like(ho_norm_sq)
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq**i

    if ho.dtype == torch.float64 and not fast:
        w = 2 * torch.arccos(torch.clamp(s, -1.0, 1.0))
        # do 1 iteration of Newton's method
        f_w = ((3 / 4) * (w - torch.sin(w))) ** (1 / 3) - torch.sqrt(ho_norm_sq)
        f_p_w = (1 - torch.cos(w)) / (6 ** (2 / 3) * (w - torch.sin(w)) ** (2 / 3))
        update = f_w / f_p_w
        # remove any nans
        update[torch.isnan(update)] = 0
        w -= update
    else:
        w = 2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0))

    ax = torch.concat(
        [
            ho * torch.rsqrt(ho_norm_sq).unsqueeze(-1),
            w.unsqueeze(-1),
        ],
        dim=-1,
    )
    rot_is_identity = torch.abs(ho_norm_sq) < 1e-6
    # set the identity rotation
    ax[rot_is_identity] = 0
    ax[rot_is_identity, ..., 2] = 1.0
    return ax


@torch.jit.script
def qu2rf(qu: Tensor) -> Tensor:
    """
    Converts a set of quaternions to Rodrigues-Frank vector representation.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Tensor of shape (..., 3) Rodrigues-Frank vectors
    """
    # RF is the axis * tan(angle/2)
    # qu = [cos(angle/2), sin(angle/2) * axis]
    # So simply take qu[..., 1:] / qu[..., 0]
    return qu[..., 1:] / qu[..., 0:1]


@torch.jit.script
def rf2qu(rf: Tensor) -> Tensor:
    """
    Converts a set of Rodrigues-Frank vectors to quaternions.

    Args:
        rf: shape (..., 3) Rodrigues-Frank vectors

    Returns:
        Tensor of shape (..., 4) quaternions in form (w, x, y, z)
    """
    # the norm is tan(angle/2)
    # w is cos(angle/2)
    # w = cos(0.5 * 2 * atan(norm(rf)))
    # w = cos(atan(norm(rf))) = 1 / sqrt(1 + norm(rf)**2)
    w = 1.0 / torch.sqrt(1.0 + torch.norm(rf, dim=-1, keepdim=True) ** 2)
    return torch.concat([w, rf * w], dim=-1)


@torch.jit.script
def ax2qu(ax: Tensor) -> Tensor:
    """
    Converts axis-angle representation to quaternion representation.

    Args:
        ax (Tensor): shape (..., 4) axis-angle in the format (x, y, z, angle).

    Returns:
        torch.Tensor: shape (..., 4) quaternions in the format (w, x, y, z).
    """
    qu = torch.empty_like(ax)
    cos_half_ang = torch.cos(ax[..., 3] / 2.0)
    sin_half_ang = torch.sin(ax[..., 3:4] / 2.0)
    qu[..., 0] = cos_half_ang
    qu[..., 1:] = ax[..., :3] * sin_half_ang
    return qu


@torch.jit.script
def ho2qu(homochoric_vectors: Tensor) -> Tensor:
    """
    Converts homochoric vector representation to quaternions.

    Args:
        homochoric_vectors: Homochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(homochoric_vectors))


@torch.jit.script
def qu2ho(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to homochoric coordinates.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4).

    Returns:
        Homochoric coordinates as tensor of shape (..., 3).
    """
    if qu.size(-1) != 4:
        raise ValueError(f"Invalid quaternion shape {qu.shape}.")
    ho = torch.empty_like(qu[..., :3])
    # get the angle
    angle = 2 * torch.acos(qu[..., 0:1].clamp_(min=-1.0, max=1.0))
    # get the unit vector
    unit = qu[..., 1:] / torch.norm(qu[..., 1:], dim=-1, keepdim=True)
    ho = unit * (3.0 * (angle - torch.sin(angle)) / 4.0) ** (1 / 3)
    # fix the case where the angle is zero
    ho[(angle.squeeze(-1) < 1e-8)] = 0.0
    return ho


@torch.jit.script
def cu2qu(cubochoric_vectors: Tensor) -> Tensor:
    """
    Converts cubochoric vector representation to quaternions.

    Args:
        cubochoric_vectors: Cubochoric vectors as tensor of shape (..., 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    return ax2qu(ho2ax(cu2ho(cubochoric_vectors)))


@torch.jit.script
def qu2cu(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to cubochoric vectors.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Cubochoric vectors as tensor of shape (..., 3).
    """
    return ho2cu(qu2ho(quaternions))
