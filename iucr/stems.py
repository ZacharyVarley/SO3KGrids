"""Ordered, content-descriptive output stems for IUCR panel exports.

Stems are zero-padded so directory listings match paper order. Legacy
``figureN_*`` names are not used — journal figure numbers may change.
"""

from __future__ import annotations

# Combined multi-panel figures (column / full width)
SO3T_REJECTION_VS_KR_FZ = "01_so3t_rejection_vs_kr_fz"
LAUE_O_PC_VS_FCC_KR_GRIDS = "12_laue_o_pc_vs_fcc_kr_grids"

# Cubochoric KR axial anisotropy (formerly figure 2)
E3_RATIO_VS_NS3 = "02_e3_ratio_vs_ns3"
CUZ_STAR_VS_CUXY = "03_cuz_star_vs_cuxy"

# Nearest-neighbour CDFs (formerly figure 3)
WITNESS_NN_CDF = "04_witness_nn_cdf"
SELF_NN_CDF = "05_self_nn_cdf"

# Grid-method comparison on SO(3)/K (formerly figure 4)
E3_RATIO_GRID_METHODS = "06_e3_ratio_grid_methods"
CR_RATIO_GRID_METHODS = "07_cr_ratio_grid_methods"
GRID_METHODS_LEGEND = "08_grid_methods_legend"
GRID_METHODS_STACKED = "09_grid_methods_stacked"

# Thomson relaxation (formerly figure 5)
THOMSON_E3_RATIO_VS_ITER = "10_thomson_e3_ratio_vs_iter"
COVERING_RADIUS_VS_ITER = "11_covering_radius_vs_iter"

ALL_STEMS_IN_ORDER: tuple[str, ...] = (
    SO3T_REJECTION_VS_KR_FZ,
    E3_RATIO_VS_NS3,
    CUZ_STAR_VS_CUXY,
    WITNESS_NN_CDF,
    SELF_NN_CDF,
    E3_RATIO_GRID_METHODS,
    CR_RATIO_GRID_METHODS,
    GRID_METHODS_LEGEND,
    GRID_METHODS_STACKED,
    THOMSON_E3_RATIO_VS_ITER,
    COVERING_RADIUS_VS_ITER,
    LAUE_O_PC_VS_FCC_KR_GRIDS,
)
