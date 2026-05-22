import math
from typing import Tuple
import torch

PI = math.pi
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)
_DTYPE = torch.float64

# ---------- spherical/cartesian ----------
def _sph_from_cart(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = v.unbind(-1)
    r = torch.linalg.norm(v, dim=-1)
    ct = torch.where(r > 0, z / torch.clamp(r, min=1e-30), torch.ones_like(r))
    theta = torch.acos(ct.clamp(-1.0, 1.0))
    phi = torch.atan2(y, x)
    return r, theta, phi

def _cart_from_sph(r: torch.Tensor, th: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    st, ct = torch.sin(th), torch.cos(th)
    cp, sp = torch.cos(ph), torch.sin(ph)
    return torch.stack([r * st * cp, r * st * sp, r * ct], dim=-1)

# ---------- D_k geometry ----------
def _a_polar(k: int) -> float:
    return 1.0 / math.tan(PI / (2.0 * k))

def _rho_bound(thp: torch.Tensor, php: torch.Tensor, k: int) -> torch.Tensor:
    a = _a_polar(k)
    b = torch.cos(php)
    c = torch.maximum(
        torch.as_tensor(a, dtype=thp.dtype, device=thp.device) * torch.cos(thp),
        b * torch.sin(thp),
    ).clamp_min(1e-15)
    rho3 = 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))
    return rho3.clamp_min(0.0).pow(1.0 / 3.0)

def _F_pc(theta: torch.Tensor, a: float) -> torch.Tensor:
    cth = torch.cos(theta)
    return 1.5 * (0.5 * PI * (1.0 - cth) - math.atan(a) + cth * torch.atan(a * cth))

def _F_lat(theta: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bs = b * torch.sin(theta)
    term1 = 0.5 * PI - torch.atan(bs)
    sqrt1pb2 = torch.sqrt(1.0 + b * b)
    term2 = (b / sqrt1pb2) * torch.atan2(sqrt1pb2 * torch.sin(theta), torch.cos(theta))
    return 1.5 * (0.5 * PI - torch.cos(theta) * term1 - term2)

def _theta_switch(phi_p: torch.Tensor, a: float) -> torch.Tensor:
    b = torch.cos(phi_p)
    return torch.atan2(torch.as_tensor(a, dtype=phi_p.dtype, device=phi_p.device), b)

def _A_phi(phi_p: torch.Tensor, k: int) -> torch.Tensor:
    a = _a_polar(k)
    ths = _theta_switch(phi_p, a)
    b = torch.cos(phi_p)
    return _F_pc(ths, a) + (_F_lat(0.5 * PI * torch.ones_like(phi_p), b) - _F_lat(ths, b))

def _G_theta(theta: torch.Tensor, phi_p: torch.Tensor, k: int) -> torch.Tensor:
    a = _a_polar(k)
    ths = _theta_switch(phi_p, a)
    b = torch.cos(phi_p)
    out = torch.empty_like(theta)
    mask = theta <= ths
    if mask.any():
        out[mask] = _F_pc(theta[mask], a)
    if (~mask).any():
        out[~mask] = _F_pc(ths[~mask], a) + (_F_lat(theta[~mask], b[~mask]) - _F_lat(ths[~mask], b[~mask]))
    return out

def _f_theta(theta: torch.Tensor, phi_p: torch.Tensor, k: int) -> torch.Tensor:
    a = _a_polar(k)
    b = torch.cos(phi_p)
    c = torch.maximum(
        torch.as_tensor(a, dtype=theta.dtype, device=theta.device) * torch.cos(theta),
        b * torch.sin(theta),
    ).clamp_min(1e-15)
    rho3 = 1.5 * (torch.atan(1.0 / c) - c / (1.0 + c * c))
    return rho3 * torch.sin(theta)

# ---------- Chebyshev φ'(u) ----------
_CHEB_NUM = {
    2: [0.39627987780106344, 0.39338452161224874, -0.003544966435596199, -0.0006823304169040724, -3.604149307300239e-05, -3.1605464484494636e-06, 2.0432777448919986e-07, 5.053742895615855e-08, 7.523441384495686e-09, 5.248732909091277e-10, -2.2865932040122213e-11, -1.2314903279531874e-11, -2.0242943434689516e-12, -1.6294199721876062e-13, 3.866040574806242e-15],
    3: [0.26372079493664446, 0.2621214219744662, -0.0019209446532809178, -0.00032215106358461057, -5.042526775859698e-07, 1.1411079542434066e-07, 4.183849390993462e-08, 2.789976730130904e-09, -6.884027295928439e-11, -1.2483951830916921e-11, -1.1964284939950267e-12, -2.0890865865872803e-14, 6.360346416223058e-15],
    4: [0.19740759198647356, 0.1965230199482321, -0.0010592628708664318, -0.0001736206447833958, 1.2067879204577828e-06, 1.413140039530892e-07, 4.969033409020158e-09, 2.3319140405071373e-10, -2.3209109765292826e-11, -1.2843902376725979e-12, 1.0043923509978181e-14],
    6: [0.13130528077331868, 0.13096612621072035, -0.00040606488851938056, -6.647953265805797e-05, 4.781749066832917e-07, 4.723681792250029e-08, -1.5943494316860916e-10, -1.5278450598504546e-11, -6.980104484333946e-13, -2.7112824751072206e-14],
}
_PHI_MAX = {2: 0.7853981633974483, 3: 0.5235987755982988, 4: 0.39269908169872414, 6: 0.2617993877991494}

def _cheb_eval(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if coeffs.numel() == 1:
        return torch.full_like(t, coeffs[0])
    b1 = torch.zeros_like(t)
    b2 = torch.zeros_like(t)
    for idx in range(coeffs.numel() - 1, 0, -1):  # k = n..1
        c_k = coeffs[idx]
        b0 = 2.0 * t * b1 - b2 + c_k
        b2, b1 = b1, b0
    return t * b1 - b2 + coeffs[0]

def _phi_from_u_pade(u: torch.Tensor, k: int) -> torch.Tensor:
    t = (2.0 * u - 1.0).clamp(-1.0, 1.0)
    coeffs = torch.tensor(_CHEB_NUM[k], dtype=u.dtype, device=u.device)
    phi = _cheb_eval(coeffs, t)
    return phi.clamp(0.0, _PHI_MAX[k])

# ---------- Newton for θ' ----------
def _invert_theta_newton(y: torch.Tensor, phi_p: torch.Tensor, k: int, max_iter: int = 11) -> torch.Tensor:
    A = _A_phi(phi_p, k).clamp_min(1e-30)
    theta = torch.acos(torch.clamp(1.0 - y, -1.0, 1.0))
    for _ in range(max_iter):
        Gn = _G_theta(theta, phi_p, k) / A
        fn = (_f_theta(theta, phi_p, k) / A).clamp_min(1e-14)
        step = (Gn - y) / fn
        # theta_new = (theta - step).clamp(0.0, halfpi)
        # theta = theta_new
        theta.copy_((theta - step).clamp(0.0, 0.5 * PI))
    return theta

# ---------- XY plane ops ----------
def _rotate_xy(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    ca, sa = torch.cos(ang), torch.sin(ang)
    x, y = v.unbind(-1)[:2]
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    out = v.clone()
    out[:, 0] = xr
    out[:, 1] = yr
    return out

def _reflect_about_axis(v: torch.Tensor, alpha: float, mask: torch.Tensor) -> torch.Tensor:
    ca2, sa2 = math.cos(2.0 * alpha), math.sin(2.0 * alpha)
    x, y = v.unbind(-1)[:2]
    xr = ca2 * x + sa2 * y
    yr = sa2 * x - ca2 * y
    out = v.clone()
    out[mask, 0] = xr[mask]
    out[mask, 1] = yr[mask]
    return out

# ---------- Core mapper ----------
@torch.no_grad()
def _ho2ho_Dk(ho: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    assert ho.dim() == 2 and ho.size(-1) == 3, "ho must be (N,3) tensor"
    v = ho.to(dtype=_DTYPE).clone()

    # Fold to θ∈[0,π/2]
    sgn_z = torch.sign(v[:, 2])
    sgn_z[sgn_z == 0] = 1.0
    v[:, 2] = v[:, 2].abs()

    sector_half = PI / k
    wedge_max = PI / (2 * k)

    # rotate to sector (-sector_half, +sector_half]
    _, _, phi0 = _sph_from_cart(v)
    krot = torch.floor((phi0 + sector_half) / (2.0 * sector_half))
    v = _rotate_xy(v, -krot * (2.0 * sector_half))

    # reflect into [0, wedge_max]
    _, _, phi1 = _sph_from_cart(v)
    mask_neg = phi1 < 0
    if mask_neg.any():
        v = _reflect_about_axis(v, 0.0, mask_neg)
    _, _, phi2 = _sph_from_cart(v)
    mask_over = phi2 > wedge_max
    if mask_over.any():
        v = _reflect_about_axis(v, wedge_max, mask_over)

    # KR inside wedge
    r, th, ph = _sph_from_cart(v)
    u = (ph / wedge_max).clamp(0.0, 1.0)
    ph_p = _phi_from_u_pade(u, k)
    y = 1.0 - torch.cos(th)
    th_p = _invert_theta_newton(y, ph_p, k)
    rho = _rho_bound(th_p, ph_p, k)
    r_p = (1.0 - eps) * rho * (r / H_MAX)
    mapped = _cart_from_sph(r_p, th_p, ph_p)

    # Undo reflections/rotations, restore sign
    if mask_over.any():
        mapped = _reflect_about_axis(mapped, wedge_max, mask_over)
    if mask_neg.any():
        mapped = _reflect_about_axis(mapped, 0.0, mask_neg)
    mapped = _rotate_xy(mapped, krot * (2.0 * sector_half))
    mapped[:, 2] = torch.copysign(mapped[:, 2], sgn_z)
    return mapped

# ---------- Public API ----------
def ho2ho_D2(ho: torch.Tensor) -> torch.Tensor: return _ho2ho_Dk(ho, 2)
def ho2ho_D3(ho: torch.Tensor) -> torch.Tensor: return _ho2ho_Dk(ho, 3)
def ho2ho_D4(ho: torch.Tensor) -> torch.Tensor: return _ho2ho_Dk(ho, 4)
def ho2ho_D6(ho: torch.Tensor) -> torch.Tensor: return _ho2ho_Dk(ho, 6)

# ---------- Smoke test ----------
if __name__ == "__main__":
    torch.set_default_dtype(_DTYPE)
    torch.manual_seed(0)
    N = 20000
    xyz = torch.randn(N, 3, dtype=_DTYPE)
    r = torch.linalg.norm(xyz, dim=-1, keepdim=True).clamp_min(1e-12)
    xyz = xyz / r
    u = torch.rand(N, 1, dtype=_DTYPE)
    rad = (u * (H_MAX ** 3)) ** (1.0 / 3.0)
    ho = xyz * rad
    for k in (2,3,4,6):
        out = _ho2ho_Dk(ho, k)
        assert torch.isfinite(out).all()
        assert (torch.linalg.norm(out, dim=-1) <= H_MAX + 1e-9).all()
    print("Self-test passed.")
