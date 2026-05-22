"""
Logic for generating grids over the fundamental zone of a given Laue group.

Supported approaches:
    - FZ rejection of cubochoric grids
        - simple and fastest but results in collisions at FZ boundaries
        - good for seeding optimization algorithms like Thomson relaxation
    - FZ KR mapper of cubochoric / super-Fibonacci / Marsaglia / Shoemake grids
        - more complex but similar speed and avoids FZ boundary collisions

Further Notes:
    - For T, O, I:
        - FZ is roughly isotropic in homochoric coords
        - cubochoric grids can be SC/FCC/BCC lattices (cubic lattices)
    - For C2, C3, C4, C6, D3, D4, D6:
        - anisotropic in homochoric coords (z range less than x/y range)
        - KR mappings preserve relative volume but squish in z-direction
        - cubochoric grids need to compensate for this z-anisotropy
        - use tetragonal lattices with c > a = b to pre-rarify z-sampling rate
        - avoid isotropic super-Fibonacci / Marsaglia / Shoemake grids

"""

from src.grid_SO3 import so3_cubochoric_grid
from src.orientation_ops import cu2qu, qu_std, qu_norm, qu2cu, ho2cu, cu2ho, ho2qu
from src.laue_ops import ori_in_fz_laue

from mappings.ho2hoFZ_C_k import ho2ho_C2, ho2ho_C3, ho2ho_C4, ho2ho_C6
from mappings.ho2hoFZ_D_k import ho2ho_D2, ho2ho_D3, ho2ho_D4, ho2ho_D6
from mappings.ho2hoFZ_T import ho2ho_T
from mappings.ho2hoFZ_O import ho2ho_O
from mappings.ho2hoFZ_I import ho2ho_I

import torch
import math


@torch.jit.script
def cu_rej_grid(semi_edge_length: int, laue_id: int, device: torch.device):
    """
    Generate a 3D grid sampling in cubochoric coordinates. Orientations are
    rejected if they are not in the fundamental zone of the given Laue group.
    Quaternions are returned as unit quaternions with positive scalar part as
    (w, x, y, z).

    Args:
        :semi_edge_length (int): semi-edge length of the grid (force odd length)
        :laue_id (int): the Laue group ID for the crystal system
        :device (torch.device): the device to use for the grid generation

    Returns:
        :torch.Tensor: (N, 4) tensor of quaternions in the fundamental zone

    """
    # generate the grid
    cu = torch.linspace(
        -0.5 * torch.pi ** (2 / 3),
        0.5 * torch.pi ** (2 / 3),
        2 * semi_edge_length + 2,  # add extra point at the opposite faces
        device=device,
    )
    # remove the added last point to make the spacing correct
    cu = cu[:-1]

    # add half the spacing to the cu
    cu = cu + 0.5 * (cu[1] - cu[0])

    if laue_id == 1:  # Triclinic
        cu_x, cu_y, cu_z = cu, cu, cu
    elif laue_id == 2:  # Monoclinic C2
        # trim cu_z to points with |cu_z| <= 0.79
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.79]
    elif laue_id == 3:  # Orthorhombic D2
        # trim all points with |cu| <= 0.79
        cu = cu[cu.abs() <= 0.79]
        cu_x, cu_y, cu_z = cu, cu, cu
    elif laue_id == 4:  # Tetragonal Low C4
        # trim cu_z to points with |cu_z| <= 0.49
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.49]
    elif laue_id == 5:  # Tetragonal High D4
        # x and y to 0.66 and z to 0.49
        cu_x = cu[cu.abs() <= 0.66]
        cu_y = cu_x
        cu_z = cu[cu.abs() <= 0.49]
    elif laue_id == 6:  # Trigonal Low C3
        # z to 0.61
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.61]
    elif laue_id == 7:  # Trigonal High D3
        # x and y to 0.7 and z to 0.61
        cu_x = cu[cu.abs() <= 0.7]
        cu_y = cu_x
        cu_z = cu[cu.abs() <= 0.61]
    elif laue_id == 8:  # Hexagonal Low C6
        # z to 0.35
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.35]
    elif laue_id == 9:  # Hexagonal High D6
        # x and y to 0.64 and z to 0.35
        cu_x = cu[cu.abs() <= 0.64]
        cu_y = cu_x
        cu_z = cu[cu.abs() <= 0.35]
    elif laue_id == 10:  # Cubic Low T
        # xyz to 0.61
        cu_x = cu[cu.abs() <= 0.61]
        cu_y = cu_x
        cu_z = cu_x
    elif laue_id == 11:  # Cubic High O
        # xyz to 0.44
        cu_x = cu[cu.abs() <= 0.45]
        cu_y = cu_x
        cu_z = cu_x
    elif laue_id == 12:  # Icosahedral I
        # xyz to 0.30971041941148164 box
        cu_x = cu[cu.abs() <= 0.32]
        cu_y = cu_x
        cu_z = cu_x
    else:
        raise ValueError(
            f"Laue group {laue_id} is not an integer in [1, 12] + \n"
            + "Supported Laue groups are: \n"
            + "1) Triclinic C1 \n"
            + "2) Monoclinic C2 \n"
            + "3) Orthorhombic D2 \n"
            + "4) Tetragonal Low C4 \n"
            + "5) Tetragonal High D4 \n"
            + "6) Trigonal Low C3 \n"
            + "7) Trigonal High D3 \n"
            + "8) Hexagonal Low C6 \n"
            + "9) Hexagonal High D6 \n"
            + "10) Cubic Low T \n"
            + "11) Cubic High O \n"
            + "12) Icosahedral I \n"
        )

    cu = torch.stack(torch.meshgrid(cu_x, cu_y, cu_z, indexing="ij"), dim=-1).reshape(
        -1, 3
    )

    # map from cubochoric to quaternion representation
    qu = cu2qu(cu)

    # standardize and normalize the quaternion representation
    qu = qu_std(qu)
    qu = qu_norm(qu)

    # filter out the quaternions that are not in the fundamental zone
    qu = qu[ori_in_fz_laue(qu, laue_id)]

    return qu


def rejection_sample_laue(qu: torch.Tensor, laue_id: int) -> torch.Tensor:
    """
    Rejection-sample orientations in the Laue fundamental zone.

    Parameters
    ----------
    qu:
        (..., 4) tensor of quaternions (w, x, y, z). They do not need to be
        pre-normalized or have w >= 0; both are enforced here.
    laue_id:
        Integer Laue group ID in [1, 12].

    Returns
    -------
    Tensor
        (N_fz, 4) tensor of unit quaternions in the Laue FZ with w >= 0.
    """
    qu_flat = qu.reshape(-1, 4)
    qu_flat = qu_std(qu_flat)
    qu_flat = qu_norm(qu_flat)

    mask = ori_in_fz_laue(qu_flat, laue_id)
    qu_fz = qu_flat[mask]
    return qu_fz


def _kr_map_homochoric(ho: torch.Tensor, laue_id: int) -> torch.Tensor:
    """
    Internal helper: apply the appropriate homochoric → FZ KR map for a Laue group.
    """
    if laue_id == 1:  # Triclinic C1: identity
        ho_fz = ho
    elif laue_id == 2:  # Monoclinic C2
        ho_fz = ho2ho_C2(ho)
    elif laue_id == 3:  # Orthorhombic D2
        ho_fz = ho2ho_D2(ho)
    elif laue_id == 4:  # Tetragonal Low C4
        ho_fz = ho2ho_C4(ho)
    elif laue_id == 5:  # Tetragonal High D4
        ho_fz = ho2ho_D4(ho)
    elif laue_id == 6:  # Trigonal Low C3
        ho_fz = ho2ho_C3(ho)
    elif laue_id == 7:  # Trigonal High D3
        ho_fz = ho2ho_D3(ho)
    elif laue_id == 8:  # Hexagonal Low C6
        ho_fz = ho2ho_C6(ho)
    elif laue_id == 9:  # Hexagonal High D6
        ho_fz = ho2ho_D6(ho)
    elif laue_id == 10:  # Cubic Low T
        ho_fz = ho2ho_T(ho)
    elif laue_id == 11:  # Cubic High O
        ho_fz = ho2ho_O(ho)
    elif laue_id == 12:  # Icosahedral I (532)
        ho_fz = ho2ho_I(ho)
    else:
        raise ValueError("Laue group {0} is not an integer in [1, 12]".format(laue_id))

    return ho_fz


def kr_sample_laue(qu: torch.Tensor, laue_id: int) -> torch.Tensor:
    """
    Map a set of quaternions into the Laue fundamental zone using KR homochoric maps.

    This keeps all input orientations but transports them into the FZ for the
    chosen Laue group via:
        qu → ho → ho_FZ (KR) → qu_FZ.

    Parameters
    ----------
    qu:
        (..., 4) tensor of quaternions (w, x, y, z). They do not need to be
        pre-normalized or have w >= 0; both are enforced here.
    laue_id:
        Integer Laue group ID in [1, 12].

    Returns
    -------
    Tensor
        (N, 4) tensor of unit quaternions in the Laue FZ with w >= 0.
    """
    qu_flat = qu.reshape(-1, 4)
    qu_flat = qu_std(qu_flat)
    qu_flat = qu_norm(qu_flat)

    # quaternion → homochoric
    cu = qu2cu(qu_flat)
    ho = cu2ho(cu)

    # apply KR mapping in homochoric coordinates
    ho_fz = _kr_map_homochoric(ho, laue_id)

    # back to quaternion representation
    qu_fz = ho2qu(ho_fz)
    qu_fz = qu_std(qu_fz)
    qu_fz = qu_norm(qu_fz)
    return qu_fz


def cu_kr_grid(
    semi_edge_length: int,
    laue_id: int,
    device: torch.device,
    # optionally specify a different z semi-edge length vs x/y one
    z_semi_edge_length: int = 0,
) -> torch.Tensor:
    """
    Generate a 3D grid sampling in cubochoric coordinates and map it into the
    Laue fundamental zone using KR homochoric maps.

    Args:
        :semi_edge_length (int): semi-edge length of the grid (force odd length)
        :laue_id (int): the Laue group ID for the crystal system
        :device (torch.device): the device to use for the grid generation
        :z_semi_edge_length (int): overrides the z semi-edge length (optional)

    Returns:
        :torch.Tensor: (N, 4) tensor of quaternions in the fundamental zone
    """
    qu = so3_cubochoric_grid(
        semi_edge_length,
        device,
        z_semi_edge_length=z_semi_edge_length,
    )
    # renormalize and canonicalize the quaternions
    qu = qu_std(qu)
    qu = qu_norm(qu)
    # map into the Laue fundamental zone
    qu = kr_sample_laue(qu, laue_id)
    return qu
