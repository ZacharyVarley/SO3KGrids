#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch

from src.orientation_ops import cu2qu, qu_std, qu_norm, qu2cu, ho2cu, cu2ho, ho2qu
from src.laue_ops import ori_in_fz_laue

NAME = "O_FCC"
LAUE_ID = 11
CARD = 24
PI = math.pi
SQRT2 = math.sqrt(2.0)
_KAPPA = SQRT2 - 1.0

# cubochoric cube semi-edge
CU_MAX = 0.5 * PI ** (2.0 / 3.0)

# homochoric ball radius
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)

PHI_LO_O = 0.25 * PI
PHI_HI_O = 0.50 * PI
C_TOT_O_SECTOR = (PI * PI) / 384.0

EPS = 1e-15


# -----------------------------------------------------------------------------
# Source-side asymptotic FCC truncated-cube azimuth fit
# -----------------------------------------------------------------------------

_BETA0 = SQRT2

# J=8, K=4
_C_LEFT = [
    [
        0.047453237051852566,
        -0.006331582702629438,
        0.000637878399962668,
        -2.3307077034889083e-05,
        3.6996838279462817e-06,
        -8.304101202704694e-08,
        2.000914132342378e-08,
        -2.727698494495133e-10,
        1.0430832870903331e-10,
    ],
    [
        -0.024116600775404777,
        0.020197403368641856,
        -0.003159595738665028,
        0.00012617224069805778,
        -3.491410882567933e-05,
        4.952831545926557e-07,
        -2.793244230778757e-07,
        7.936744764801877e-10,
        -1.92994132067333e-09,
    ],
    [
        -0.12864277576489147,
        -0.014595855267684619,
        0.0045782036275230364,
        -0.00012866265386118545,
        0.00013888432803812773,
        1.0004787575397213e-06,
        1.7784706360515206e-06,
        2.8746272824600837e-08,
        1.662381141008372e-08,
    ],
    [
        0.06854997628429998,
        0.0028392812545379484,
        0.0004759872037685191,
        -0.0004916146983315775,
        -0.0003072167117727139,
        -1.8773917645268276e-05,
        -7.011647766920342e-06,
        -3.564048173898984e-07,
        -1.0958513325162026e-07,
    ],
    [
        -0.018281130109020313,
        -0.019741744302077636,
        -0.006854479041325471,
        0.0014540921167888199,
        0.00044201275143979156,
        8.541945600773342e-05,
        2.015156124387631e-05,
        1.6188675924648987e-06,
        2.0596482011159467e-07,
    ],
]


# -----------------------------------------------------------------------------
# Fixed target-side piecewise Pade inverse CDF for ball-sector -> O-FZ
# -----------------------------------------------------------------------------

_U_BREAK = 0.5667905347333297

_A_L = [
    0.972852840588069,
    0.19114256303868526,
    -0.07200597429184963,
    0.0019036882372386453,
    -6.90762808742289e-05,
    -3.541269544600796e-05,
    1.2802239724117446e-06,
    -4.353221441400241e-08,
    -5.969310864624526e-09,
]

_B_L = [
    1.0,
    0.003306117640028675,
    -0.08447285894067914,
    0.010338177449608622,
    -0.0007327163426372906,
    -4.420643674300553e-06,
    3.672842561317652e-06,
    -3.147154560490602e-07,
    1.105314525177379e-08,
]

_A_R = [
    1.3707109341917165,
    0.1947432679225177,
    0.007269980754321227,
    0.0007151827313080849,
    -5.174273157437798e-05,
    -5.3091738401265805e-06,
    -1.659001355271721e-07,
    2.8004315443311014e-09,
    3.284693250980296e-10,
]

_B_R = [
    1.0,
    -0.0017419236794533773,
    0.002750778936592661,
    0.0007074471694800329,
    -6.934266776154665e-05,
    -7.729908914691581e-07,
]


# -----------------------------------------------------------------------------
# Gauss-Legendre 16
# -----------------------------------------------------------------------------

_GL16_X = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878317,
    -0.7554044083550031,
    -0.6178762444026437,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.09501250983763744,
    0.09501250983763744,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026437,
    0.7554044083550031,
    0.8656312023878317,
    0.9445750230732326,
    0.9894009349916499,
]

_GL16_W = [
    0.027152459411754095,
    0.06225352393864789,
    0.09515851168249278,
    0.12462897125553387,
    0.14959598881657673,
    0.16915651939500254,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.16915651939500254,
    0.14959598881657673,
    0.12462897125553387,
    0.09515851168249278,
    0.06225352393864789,
    0.027152459411754095,
]


# -----------------------------------------------------------------------------
# FCC helpers
# -----------------------------------------------------------------------------


def _fcc_grid_n(h: int) -> int:
    return ((2 * h + 1) ** 3 + 1) // 2


def compute_semi_edge_rej(n_target: int, card: int) -> int:
    return max(1, int(round(((n_target * card) ** (1.0 / 3.0) - 1.0) / 2.0)))


@torch.no_grad()
def _cubochoric_fcc_grid(
    h: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    u = torch.linspace(-CU_MAX, CU_MAX, 2 * h + 2, device=device, dtype=dtype)
    u = u[:-1]
    u = u + 0.5 * (u[1] - u[0])
    x, y, z = torch.meshgrid(u, u, u, indexing="ij")

    idx = torch.arange(-h, h + 1, device=device)
    i, j, k = torch.meshgrid(idx, idx, idx, indexing="ij")
    mask = ((i + j + k) % 2) == 0

    return torch.stack([x[mask], y[mask], z[mask]], dim=-1)


def _nearest_odd_int(x: float) -> int:
    k = int(round((x - 1.0) / 2.0))
    cand0 = 2 * k + 1
    cand1 = cand0 + 2 if x >= cand0 else cand0 - 2
    return cand0 if abs(cand0 - x) <= abs(cand1 - x) else cand1


def beta_fcc(h: int) -> float:
    x = (1.0 + SQRT2) * (h + 0.5)
    m = _nearest_odd_int(x)
    return m / (h + 0.5) - 1.0


# -----------------------------------------------------------------------------
# Basic math helpers
# -----------------------------------------------------------------------------


def _as_tensor(x, device, dtype):
    return torch.as_tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def gl16_integrate(
    func, a, b, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    a_t = _as_tensor(a, device, dtype)
    b_t = _as_tensor(b, device, dtype)
    a_b, b_b = torch.broadcast_tensors(a_t, b_t)

    x = _as_tensor(_GL16_X, device, dtype)
    w = _as_tensor(_GL16_W, device, dtype)

    half = 0.5 * (b_b - a_b)
    mid = 0.5 * (b_b + a_b)

    xa = x.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0) + mid.unsqueeze(0)
    wa = w.reshape(16, *([1] * a_b.ndim)) * half.unsqueeze(0)

    val = func(xa)
    wa_b = wa.reshape(wa.shape + (1,) * (val.ndim - wa.ndim))
    return (val * wa_b).sum(dim=0)


@torch.no_grad()
def cheb_T_value(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    n = coeffs.numel() - 1
    if n < 0:
        return torch.zeros_like(t)
    if n == 0:
        return torch.full_like(t, coeffs[0])
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for k in range(n, 0, -1):
        b0 = 2.0 * t * b1 - b2 + coeffs[k]
        b2, b1 = b1, b0
    return t * b1 - b2 + coeffs[0]


@torch.no_grad()
def cheb_T_value_deriv(coeffs: torch.Tensor, t: torch.Tensor):
    f = cheb_T_value(coeffs, t)
    n = coeffs.numel() - 1
    if n <= 0:
        return f, torch.zeros_like(t)
    c = torch.stack([(j + 1) * coeffs[j + 1] for j in range(n)])
    U0 = torch.ones_like(t)
    if n == 1:
        return f, c[0] * U0
    U1 = 2.0 * t
    d = c[0] * U0 + c[1] * U1
    Ukm2, Ukm1 = U0, U1
    for j in range(2, n):
        Uk = 2.0 * t * Ukm1 - Ukm2
        d = d + c[j] * Uk
        Ukm2, Ukm1 = Ukm1, Uk
    return f, d


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def sph_from_cart(v: torch.Tensor):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = torch.linalg.norm(v, dim=-1)
    ct = torch.where(r > 0, (z / r).clamp(-1.0, 1.0), torch.ones_like(z))
    theta = torch.acos(ct)
    phi = torch.atan2(y, x)
    return r, theta, phi


@torch.no_grad()
def cart_from_sph(r: torch.Tensor, th: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    st, ct = torch.sin(th), torch.cos(th)
    cp, sp = torch.cos(ph), torch.sin(ph)
    return torch.stack([r * st * cp, r * st * sp, r * ct], dim=-1)


@torch.no_grad()
def theta_max_O(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan(1.0 / torch.sin(phi).clamp_min(EPS))


@torch.no_grad()
def theta_switch_O(phi: torch.Tensor) -> torch.Tensor:
    denom = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS)
    return torch.atan(
        torch.sqrt(torch.tensor(2.0, device=phi.device, dtype=phi.dtype)) / denom
    )


@torch.no_grad()
def rho3_from_c(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp_min(EPS)
    return 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))


@torch.no_grad()
def R_from_c(c: torch.Tensor) -> torch.Tensor:
    return rho3_from_c(c).clamp_min(0.0).pow(1.0 / 3.0)


@torch.no_grad()
def c_O_components(theta: torch.Tensor, phi: torch.Tensor):
    c_max = (1.0 / _KAPPA) * torch.cos(theta)
    c_sum = torch.cos(theta) + torch.sin(theta) * (torch.cos(phi) + torch.sin(phi))
    return c_max, c_sum


# -----------------------------------------------------------------------------
# Source truncated-cube azimuth law
# -----------------------------------------------------------------------------


@torch.no_grad()
def phi_break_src(beta: torch.Tensor) -> torch.Tensor:
    return torch.atan(1.0 / (beta - 1.0))


@torch.no_grad()
def c_total_src(beta: torch.Tensor) -> torch.Tensor:
    return 0.5 - ((2.0 - beta) ** 3) / 12.0


@torch.no_grad()
def u_break_src(beta: torch.Tensor) -> torch.Tensor:
    ctot = c_total_src(beta)
    return (ctot - 0.5 * (beta - 1.0)) / ctot


@torch.no_grad()
def _left_t_from_phi(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    phib = phi_break_src(beta)
    return 2.0 * (phi - PHI_LO_O) / (phib - PHI_LO_O) - 1.0


@torch.no_grad()
def _w_left_src(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    device, dtype = phi.device, phi.dtype
    coeffs = _as_tensor(_C_LEFT, device, dtype)
    delta = beta - _BETA0
    t = _left_t_from_phi(phi, beta).clamp(-1.0, 1.0)

    T = [torch.ones_like(t)]
    if coeffs.shape[1] > 1:
        T.append(t)
        for _ in range(1, coeffs.shape[1] - 1):
            T.append(2.0 * t * T[-1] - T[-2])

    poly = torch.zeros_like(t)
    delta_pow = torch.ones_like(t)
    for k in range(coeffs.shape[0]):
        s = torch.zeros_like(t)
        for j in range(coeffs.shape[1]):
            s = s + coeffs[k, j] * T[j]
        poly = poly + delta_pow * s
        delta_pow = delta_pow * delta

    w = 0.5 * (1.0 + t) + (1.0 - t * t) * poly
    return w.clamp(0.0, 1.0)


@torch.no_grad()
def _w_right_src(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return (
        1.0 - torch.cos(phi) / (torch.sin(phi).clamp_min(EPS) * (beta - 1.0))
    ).clamp(0.0, 1.0)


@torch.no_grad()
def u_src_tc_fcc(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    phib = phi_break_src(beta)
    ub = u_break_src(beta)

    u = torch.empty_like(phi)
    left = phi <= phib

    if left.any():
        u[left] = ub[left] * _w_left_src(phi[left], beta[left])

    if (~left).any():
        wr = _w_right_src(phi[~left], beta[~left])
        u[~left] = ub[~left] + (1.0 - ub[~left]) * wr

    return u.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Source truncated-cube radial ceiling and conditional polar CDF
# -----------------------------------------------------------------------------


@torch.no_grad()
def _theta_switch_src(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    sphi = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS)
    return torch.atan(beta / sphi)


@torch.no_grad()
def _F_sum_src(tan_theta: torch.Tensor, sphi: torch.Tensor) -> torch.Tensor:
    q = 1.0 + sphi * tan_theta
    return (-1.0 / q + 0.5 / (q * q)) / (sphi * sphi)


@torch.no_grad()
def R_src_tc_unit(
    theta: torch.Tensor, phi: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    sphi = torch.cos(phi) + torch.sin(phi)
    r_cube = 1.0 / torch.cos(theta).clamp_min(EPS)
    r_cut = (1.0 + beta) / (torch.cos(theta) + torch.sin(theta) * sphi).clamp_min(EPS)
    return torch.minimum(r_cube, r_cut)


@torch.no_grad()
def A_src_tc_unit(phi: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    ths = _theta_switch_src(phi, beta)
    thh = theta_max_O(phi)
    tmax = torch.tan(thh)
    ts = torch.tan(ths)
    sphi = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS)

    only_cube = ths >= thh
    A = torch.empty_like(phi)

    if only_cube.any():
        A[only_cube] = 0.5 * tmax[only_cube] * tmax[only_cube]

    if (~only_cube).any():
        mask = ~only_cube
        A1 = 0.5 * ts[mask] * ts[mask]
        A2 = ((1.0 + beta[mask]) ** 3) * (
            _F_sum_src(tmax[mask], sphi[mask]) - _F_sum_src(ts[mask], sphi[mask])
        )
        A[mask] = A1 + A2

    return A.clamp_min(EPS)


@torch.no_grad()
def G_src_tc_unit(
    theta: torch.Tensor, phi: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    ths = _theta_switch_src(phi, beta)
    sphi = (torch.cos(phi) + torch.sin(phi)).clamp_min(EPS)

    G = torch.empty_like(theta)
    cube = theta <= ths

    if cube.any():
        tt = torch.tan(theta[cube])
        G[cube] = 0.5 * tt * tt

    if (~cube).any():
        mask = ~cube
        ts = torch.tan(ths[mask])
        tt = torch.tan(theta[mask])
        G1 = 0.5 * ts * ts
        G2 = ((1.0 + beta[mask]) ** 3) * (
            _F_sum_src(tt, sphi[mask]) - _F_sum_src(ts, sphi[mask])
        )
        G[mask] = G1 + G2

    return G.clamp_min(0.0)


# -----------------------------------------------------------------------------
# Fixed target-side inverse azimuth map for O-FZ
# -----------------------------------------------------------------------------


@torch.no_grad()
def _u_to_t_piece(u: torch.Tensor, u0: float, u1: float) -> torch.Tensor:
    return 2.0 * (u - u0) / (u1 - u0) - 1.0


@torch.no_grad()
def phi_and_dphi_du_from_pade_piece(
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = u.device, u.dtype
    u = u.clamp(0.0, 1.0)

    A_L = _as_tensor(_A_L, device, dtype)
    B_L = _as_tensor(_B_L, device, dtype)
    A_R = _as_tensor(_A_R, device, dtype)
    B_R = _as_tensor(_B_R, device, dtype)

    use_left = u <= _U_BREAK
    phi = torch.empty_like(u)
    dphi_du = torch.empty_like(u)

    if use_left.any():
        uu = u[use_left]
        t = _u_to_t_piece(uu, 0.0, _U_BREAK)
        P, dPdt = cheb_T_value_deriv(A_L, t)
        d = B_L.clone()
        d[0] = 0.0
        Q0, dQdt = cheb_T_value_deriv(d, t)
        Q = 1.0 + Q0
        phi_u = P / Q
        dphi_dt = (dPdt * Q - P * dQdt) / (Q * Q)
        dphi_du_u = (2.0 / _U_BREAK) * dphi_dt
        phi[use_left] = phi_u
        dphi_du[use_left] = dphi_du_u.clamp_min(EPS)

    if (~use_left).any():
        uu = u[~use_left]
        t = _u_to_t_piece(uu, _U_BREAK, 1.0)
        P, dPdt = cheb_T_value_deriv(A_R, t)
        d = B_R.clone()
        d[0] = 0.0
        Q0, dQdt = cheb_T_value_deriv(d, t)
        Q = 1.0 + Q0
        phi_u = P / Q
        dphi_dt = (dPdt * Q - P * dQdt) / (Q * Q)
        dphi_du_u = (2.0 / (1.0 - _U_BREAK)) * dphi_dt
        phi[~use_left] = phi_u
        dphi_du[~use_left] = dphi_du_u.clamp_min(EPS)

    phi = phi.clamp(PHI_LO_O, PHI_HI_O)
    return phi, dphi_du


# -----------------------------------------------------------------------------
# Target-side polar inversion for O-FZ
# -----------------------------------------------------------------------------


@torch.no_grad()
def _G_tar_O(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    device, dtype = theta.device, theta.dtype
    ths = theta_switch_O(phi)
    thh = theta_max_O(phi)

    def f_pc(th):
        cmax = (1.0 / _KAPPA) * torch.cos(th)
        return rho3_from_c(cmax) * torch.sin(th)

    G = torch.empty_like(theta)
    only_pc = ths >= thh

    if only_pc.any():
        G[only_pc] = gl16_integrate(f_pc, 0.0, theta[only_pc], device, dtype)

    if (~only_pc).any():
        mask = ~only_pc
        thp = theta[mask]
        ths_loc = ths[mask]
        phi_loc = phi[mask]
        trig_phi = torch.cos(phi_loc) + torch.sin(phi_loc)

        pc_branch = thp <= ths_loc
        sum_branch = ~pc_branch
        G_loc = torch.empty_like(thp)

        if pc_branch.any():
            G_loc[pc_branch] = gl16_integrate(f_pc, 0.0, thp[pc_branch], device, dtype)

        if sum_branch.any():
            thp2 = thp[sum_branch]
            ths2 = ths_loc[sum_branch]

            def f_sum_local(th):
                cs = torch.cos(th) + torch.sin(th) * trig_phi[sum_branch]
                return rho3_from_c(cs) * torch.sin(th)

            G1 = gl16_integrate(f_pc, 0.0, ths2, device, dtype)
            G2 = gl16_integrate(f_sum_local, ths2, thp2, device, dtype)
            G_loc[sum_branch] = G1 + G2

        G[mask] = G_loc

    return G


# -----------------------------------------------------------------------------
# Core mapping: cubochoric truncated cube -> homochoric O-FZ
# -----------------------------------------------------------------------------


@torch.no_grad()
def cu2ho_O_FCC(cu: torch.Tensor, h: int, newton_iters: int = 8) -> torch.Tensor:
    """
    Map cubochoric FCC points already lying in the snapped truncated cube
    to the homochoric octahedral fundamental zone.
    """
    device, dtype = cu.device, cu.dtype
    beta = _as_tensor(beta_fcc(h), device, dtype)

    v0 = cu.clone()
    sgn = torch.sign(v0)
    sgn[sgn == 0] = 1.0
    v = v0.abs()
    vals, idxs = torch.sort(v, dim=-1)
    v_sorted = vals

    inv_idx = torch.empty_like(idxs)
    base = torch.arange(3, device=device).view(*([1] * (idxs.ndim - 1)), 3)
    inv_idx.scatter_(-1, idxs, base.expand_as(idxs))

    r, th, ph = sph_from_cart(v_sorted)

    beta_full = torch.full_like(ph, beta)
    u_src = u_src_tc_fcc(ph, beta_full)

    ph_p, dphi_du = phi_and_dphi_du_from_pade_piece(u_src)
    A_tar = (C_TOT_O_SECTOR / dphi_du).clamp_min(EPS)

    A_src = A_src_tc_unit(ph, beta_full)
    G_src = G_src_tc_unit(th, ph, beta_full)
    y_src = (G_src / A_src).clamp(0.0, 1.0)

    th_hi_tar = theta_max_O(ph_p)
    th_p = torch.acos((1.0 - y_src * (1.0 - torch.cos(th_hi_tar))).clamp(-1.0, 1.0))

    for _ in range(newton_iters):
        th_p = torch.max(torch.zeros_like(th_p), torch.min(th_p, th_hi_tar))
        G = _G_tar_O(th_p, ph_p)
        Gn = (G / A_tar).clamp(0.0, 1.0)

        cmax, csum = c_O_components(th_p, ph_p)
        c_act = torch.maximum(cmax, csum).clamp_min(EPS)
        fn = (rho3_from_c(c_act) * torch.sin(th_p) / A_tar).clamp_min(EPS)

        step = (Gn - y_src) / fn
        th_new = th_p - step
        big = step.abs() > 0.25
        th_new = torch.where(big, 0.75 * th_p + 0.25 * th_new, th_new)
        th_new = torch.max(torch.zeros_like(th_new), torch.min(th_new, th_hi_tar))

        if torch.max((th_new - th_p).abs()).item() < 1e-12:
            th_p = th_new
            break
        th_p = th_new

    R_src_phys = CU_MAX * R_src_tc_unit(th, ph, beta_full)
    cmax, csum = c_O_components(th_p, ph_p)
    rho = R_from_c(torch.maximum(cmax, csum))
    r_p = rho * (r / R_src_phys)

    mapped_sorted = cart_from_sph(r_p, th_p, ph_p)
    mapped = torch.gather(mapped_sorted, dim=-1, index=inv_idx)
    mapped = mapped * sgn
    return mapped


# -----------------------------------------------------------------------------
# Public grid generator
# -----------------------------------------------------------------------------


@torch.no_grad()
def kr_grid_O_FCC(
    h: int | None = None,
    n_target: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_type: str = "qu",
    check_fz: bool = False,
):
    """
    Generate an FCC-based KR grid for the octahedral fundamental zone.

    Args:
        h:
            FCC semi-edge index. If None, computed from n_target.
        n_target:
            Optional target count proxy. Used only if h is None.
        device:
            Torch device.
        dtype:
            Torch dtype.
        return_type:
            "qu", "ho", or "cu".
        check_fz:
            If True, filter by ori_in_fz_laue after quaternion conversion.

    Returns:
        Tensor of orientations in the requested representation.
    """
    if h is None:
        if n_target is None:
            raise ValueError("Provide either h or n_target.")
        h = compute_semi_edge_rej(n_target, CARD)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cu = _cubochoric_fcc_grid(h, device=device, dtype=dtype)

    beta = beta_fcc(h)
    tau = CU_MAX * (1.0 + beta)

    keep = cu.abs().sum(dim=-1) <= (tau + 1e-12)
    cu_tc = cu[keep]

    ho = cu2ho_O_FCC(cu_tc, h=h)

    if return_type.lower() == "ho":
        out = ho
    elif return_type.lower() == "cu":
        out = ho2cu(ho)
    elif return_type.lower() == "qu":
        out = qu_std(qu_norm(ho2qu(ho)))
        if check_fz:
            try:
                mask = ori_in_fz_laue(out, LAUE_ID)
                out = out[mask]
            except TypeError:
                pass
    else:
        raise ValueError("return_type must be one of {'qu', 'ho', 'cu'}.")

    return out
