import math
import torch
from torch import Tensor


@torch.jit.script
def _chebval_T(t: Tensor, coeffs: Tensor) -> Tensor:
    """
    Evaluate sum_k coeffs[k] * T_k(t) using Clenshaw.
    coeffs expected as 1D tensor [c0, c1, ..., c_{K-1}].
    """
    K = coeffs.numel()
    if K == 1:
        return coeffs[0].expand_as(t)

    b_kp1 = torch.zeros_like(t)
    b_kp2 = torch.zeros_like(t)
    # iterate reversed over c1..c_{K-1}
    # (TorchScript: for-range over Python ints is supported)
    for idx in range(K - 1, 0, -1):
        a = coeffs[idx]
        b_k = 2.0 * t * b_kp1 - b_kp2 + a
        b_kp2 = b_kp1
        b_kp1 = b_k
    return coeffs[0] + t * b_kp1 - b_kp2


@torch.jit.script
def _omega_max_from_cos(c: Tensor, tau: Tensor) -> Tensor:
    c_safe = torch.clamp(c, min=1e-16)
    return 2.0 * torch.atan(tau / c_safe)


@torch.jit.script
def _sin_omega_from_cos_theta(c: Tensor, tau: Tensor) -> Tensor:
    return (2.0 * tau * c) / (c * c + tau * tau)


@torch.jit.script
def _R_of_theta(theta: Tensor, tau: Tensor) -> Tensor:
    c = torch.cos(theta)
    omega = _omega_max_from_cos(c, tau)
    sin_omega = _sin_omega_from_cos_theta(c, tau)
    R3 = 0.75 * (omega - sin_omega)
    return torch.clamp(R3, min=0.0).pow(1.0 / 3.0)


@torch.jit.script
def _theta_fz_from_rational(theta: Tensor, num: Tensor, den: Tensor) -> Tensor:
    # y in [0,1], t in [-1,1]
    y = 1.0 - torch.cos(theta)
    t = 2.0 * y - 1.0

    num_val = _chebval_T(t, num)

    # den encodes Q(t) = 1 + sum_{k>=1} b_k T_k(t).
    # Build a coeff vector whose T0 term is 0 so chebval gives Σ_{k>=1} b_k T_k.
    den_coeffs = torch.zeros_like(den)
    if den.numel() > 1:
        den_coeffs = den_coeffs.index_copy(0, torch.arange(1, den.numel(), device=den.device), den[1:])
    den_val = 1.0 + _chebval_T(t, den_coeffs)

    g = num_val / den_val

    # do in place and jit friendly
    y_clamped = torch.clamp_min_(y, 1e-15)
    sqrt_y = torch.sqrt(y_clamped)
    return sqrt_y * g


@torch.jit.script
def _map_ho_to_fz(h: Tensor, tau: Tensor, num: Tensor, den: Tensor) -> Tensor:
    """
    Core mapper: homochoric h (...,3) -> mapped h_FZ (...,3)
    """
    x = h[..., 0]
    y = h[..., 1]
    z = h[..., 2]

    zsign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    za = torch.abs(z)

    rho = torch.linalg.norm(h, dim=-1)
    xy = torch.hypot(x, y)

    theta = torch.atan2(xy, za)                 # in [0, pi/2]
    theta_fz = _theta_fz_from_rational(theta, num, den)
    R = _R_of_theta(theta_fz, tau)
    rho_p = rho * (R / 1.3306700394914688) # ball radius

    az = torch.atan2(y, x)
    s = torch.sin(theta_fz)
    c = torch.cos(theta_fz)

    out_x = rho_p * s * torch.cos(az)
    out_y = rho_p * s * torch.sin(az)
    out_z = rho_p * c * zsign

    out = torch.stack((out_x, out_y, out_z), dim=-1)

    # # Preserve exact zero at origin
    # mask0 = (rho == 0.0).unsqueeze(-1)
    # if mask0.any():
    #     out = torch.where(mask0, torch.zeros_like(out), out)

    return out


# ------------------------------
# Public, TorchScript-compiled API
# ------------------------------

@torch.jit.script
def ho2ho_C2(h: Tensor) -> Tensor:
    # tau, num, den literals; then match dtype/device via .to(h)
    tau = torch.tensor(1.0)
    num = torch.tensor([
        1.7737848743975086e00, 4.8055082084235462e-01, -4.2901615601460463e-01,
        3.5067723530850670e-01, -2.5078415551551952e-01, 1.4245817838614297e-02,
        2.0364985682640563e-01, 7.3494423671352616e-02, 7.4055714754984337e-03,
        -2.6791200292731122e-06, -1.3705227262528180e-05
    ])
    den = torch.tensor([
        1.0000000000000000e00, 4.1494079856907912e-01, -2.1735212942622864e-01,
        1.5858436443785079e-01, -1.1731723315019479e-01, 1.6329335772084991e-03,
        1.1851852745686715e-01, 5.0425562752368529e-02, 6.3676661436295858e-03,
        5.5532866871394670e-05, -1.8803246198828434e-05
    ])
    return _map_ho_to_fz(h, tau.to(h), num.to(h), den.to(h))


@torch.jit.script
def ho2ho_C3(h: Tensor) -> Tensor:
    tau = torch.tensor(1.0 / math.sqrt(3.0))
    num = torch.tensor([
        1.8258508913178009e00, 1.0204210503954347e00, -6.8990113854728552e-01,
        2.8365299613149925e-01, 1.7567952865394379e-01, -3.1873401624280262e-01,
        -2.2715481819120306e-01, -5.2585665924886747e-02, -4.3354111038882591e-03,
        -3.5816570541181354e-05, 3.6250774202656139e-06
    ])
    den = torch.tensor([
        1.0000000000000000e00, 7.2739743922667077e-01, -2.8657823943183158e-01,
        1.0022667454700472e-01, 9.6960768233829570e-02, -1.6807702882407449e-01,
        -1.4360331365310397e-01, -4.0424831299393671e-02, -4.3922845051649244e-03,
        -8.8411935275716294e-05, 6.4825070887252555e-06
    ])
    return _map_ho_to_fz(h, tau.to(h), num.to(h), den.to(h))


@torch.jit.script
def ho2ho_C4(h: Tensor) -> Tensor:
    tau = torch.tensor(math.sqrt(2.0) - 1.0)
    num = torch.tensor([
        2.0638830467685660e00, 2.6021401239522907e-01, -1.2186320073474909e-01,
        5.5221717032306668e-02, -4.2520303912717834e-02, 3.0236893037655817e-02,
        -2.9012807237718201e-02, 1.6415416344340801e-02, -4.3515734269519143e-02,
        2.2695699364858511e-02, -2.2550554009399114e-02, 1.0886164310905866e-03,
        1.7816049007855050e-02, -3.0923682555146758e-04, -2.8394264832619254e-02,
        -5.3085979265811899e-02, -3.4727668250554601e-02, -9.1350981843724040e-03,
        -8.3214638244406738e-04, 9.5344026331649555e-07, 1.2203423886000405e-06
    ])
    den = torch.tensor([
        1.0000000000000000e00, 5.0285292503462342e-01, -9.7682933036777386e-02,
        2.6524209648882916e-02, -1.5967794739660854e-02, 8.2740600918967833e-03,
        -8.4219496026539543e-03, -3.3279979553858394e-05, -1.5859151539722664e-02,
        3.9761685229014994e-03, -7.7305903613928651e-03, -6.8929659704462581e-04,
        9.3824990360368652e-03, -6.2969309144329875e-04, -1.8185327299804658e-02,
        -3.0889735741043020e-02, -2.1869520857486984e-02, -6.9416967896511351e-03,
        -8.8759920534314980e-04, -1.8630446847254365e-05, 2.0084648627427544e-06
    ])
    return _map_ho_to_fz(h, tau.to(h), num.to(h), den.to(h))


@torch.jit.script
def ho2ho_C6(h: Tensor) -> Tensor:
    tau = torch.tensor(2.0 - math.sqrt(3.0))
    num = torch.tensor([
        2.0791971633618513e00, 5.4572868416551912e-01, -3.3878248222116070e-01,
        2.5194939508555830e-01, -2.1640338620750996e-01, 1.9307970107561104e-01,
        -1.6750994994148374e-01, 1.2930791464816124e-01, -7.2433309310857660e-02,
        7.9094380324021239e-03, 6.2273851101229023e-02, -1.0405187160310844e-01,
        9.8843104227216927e-02, -2.1019188535004646e-02, -8.4291747319314506e-02,
        7.9501326731339361e-02, 8.4591828839222991e-02, 1.7150398056128016e-02,
        -1.3735157685798671e-03, -5.2349932177729337e-04, -2.0176275817922699e-05
    ])
    den = torch.tensor([
        1.0000000000000000e00, 6.4478059179981684e-01, -1.3942100975789928e-01,
        6.1704157630504450e-02, -4.3641053539025823e-02, 3.6876921485187543e-02,
        -3.2528514191822989e-02, 2.6018436502876999e-02, -1.3762001764068915e-02,
        2.8421427339296650e-04, 1.8762998314057977e-02, -3.0576085328707852e-02,
        3.1881795401240434e-02, -6.6283972799896819e-03, -3.5726723266283830e-02,
        3.6686209833398514e-02, 4.9742917859944558e-02, 1.4894263218343567e-02,
        1.2815175100273083e-04, -4.3855328224429044e-04, -3.5502169839994451e-05
    ])
    return _map_ho_to_fz(h, tau.to(h), num.to(h), den.to(h))
