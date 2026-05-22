# -*- coding: utf-8 -*-
"""
Rational fits for the Ck (k=2,3,4,6) equal-volume map with √y preconditioning + Lawson loop.

For each group Ck we set τ = tan(π/k) and use:
  ω_max(θ; τ) = 2 arctan( τ sec θ )
  sin ω(θ; τ) = 2 τ cos θ / (cos^2 θ + τ^2)
Then R(θ; τ)^3 = (3/4)[ ω_max(θ; τ) - sin ω(θ; τ) ] and
equal-volume 1 - cos θ = G(θ') / G(π/2), G(t)=∫_0^t R(u; τ)^3 sin u du.

We fit g(y) = θ'(y)/√y on y∈[0,1] with a rational Chebyshev P/Q (Lawson to push L∞),
then θ' ≈ √y * P/Q. We do this for C2, C3, C4, C6 and print coefficients + accuracy.

Dependencies: numpy, scipy
"""

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import least_squares

PI = math.pi
H_MAX = (3.0 * PI / 4.0) ** (1.0 / 3.0)
DOMAIN_THETA = (0.0, 0.5 * PI)
TINY = 1e-300

# ----------------------------
# Group specs (name -> tau, laue_id)
# ----------------------------
GROUPS = [
    ("C2", 1.0, 2),  # monoclinic low
    ("C3", 1.0 / np.sqrt(3.0), 6),  # trigonal low
    ("C4", np.sqrt(2.0) - 1.0, 4),  # tetragonal low
    ("C6", 2.0 - np.sqrt(3.0), 8),  # hexagonal low
]


# ----------------------------
# Closed-form boundary & G (parametrized by τ)
# ----------------------------
def _omega_max(theta, tau):
    c = np.cos(theta)
    sec = np.where(c > 1e-16, 1.0 / c, np.inf)
    return 2.0 * np.arctan(tau * sec)


def _sin_omega_from_cos_theta(theta, tau):
    c = np.cos(theta)
    return (2.0 * tau * c) / (c * c + tau * tau)


def R_of_theta(theta, tau):
    omega = _omega_max(theta, tau)
    sin_omega = _sin_omega_from_cos_theta(theta, tau)
    R3 = 0.75 * (omega - sin_omega)
    return np.power(R3, 1.0 / 3.0)


def _integrand(theta, tau):
    omega = _omega_max(theta, tau)
    sin_omega = _sin_omega_from_cos_theta(theta, tau)
    R3 = 0.75 * (omega - sin_omega)
    return R3 * np.sin(theta)


class GIntegrator:
    def __init__(self, tau: float, n_gl: int = 128):
        self.tau = float(tau)
        x, w = leggauss(n_gl)
        self.x = x[None, :]
        self.w = w[None, :]
        self._G_tot = None

    def G(self, t):
        t = np.asarray(t, dtype=np.float64)
        orig = t.shape
        tf = t.reshape(-1)
        T = tf[:, None]
        U = 0.5 * T * (self.x + 1.0)
        F = _integrand(U, self.tau)
        vals = 0.5 * tf * np.sum(self.w * F, axis=1)
        return vals.reshape(orig)

    @property
    def G_tot(self):
        if self._G_tot is None:
            self._G_tot = self.G(0.5 * PI)
        return self._G_tot


def theta_to_theta_fz_exact(theta, Gint: GIntegrator, tol=1e-14, maxit=64):
    theta = np.asarray(theta, dtype=np.float64)
    orig = theta.shape
    th = theta.reshape(-1)

    y = (1.0 - np.cos(th)) * Gint.G_tot
    lo = np.zeros_like(th)
    hi = np.full_like(th, 0.5 * PI)

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        Gmid = Gint.G(mid)
        go_left = Gmid > y
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid)
        if np.max(hi - lo) < tol:
            break

    return (0.5 * (lo + hi)).reshape(orig)


# ----------------------------
# Preconditioning constant (depends on τ)
# ----------------------------
def c0_leading_constant(Gint: GIntegrator):
    # R(0; τ)^3 = (3/4) * [ 2 arctan(τ) - 2τ/(1+τ^2) ]
    tau = Gint.tau
    R0_cubed = 0.75 * (2.0 * np.arctan(tau) - 2.0 * tau / (1.0 + tau * tau))
    return math.sqrt(2.0 * Gint.G_tot / R0_cubed)


# ----------------------------
# Exact ho -> ho_Ck (for validation)
# ----------------------------
def ho_to_hoCk_exact(h, Gint: GIntegrator):
    h = np.asarray(h, dtype=np.float64)
    out = np.empty_like(h)

    x, y, z = h[..., 0], h[..., 1], h[..., 2]
    zsign = np.sign(z)
    zsign[zsign == 0.0] = 1.0
    za = np.abs(z)

    rho = np.linalg.norm(h, axis=-1)
    xy = np.hypot(x, y)
    theta = np.arctan2(xy, za)  # ∈ [0, π/2]

    theta_fz = theta_to_theta_fz_exact(theta, Gint)
    R = R_of_theta(theta_fz, Gint.tau)
    rho_p = rho * (R / H_MAX)

    az = np.arctan2(y, x)
    s, c = np.sin(theta_fz), np.cos(theta_fz)

    out[..., 0] = rho_p * s * np.cos(az)
    out[..., 1] = rho_p * s * np.sin(az)
    out[..., 2] = rho_p * c * zsign
    return out


# ----------------------------
# Chebyshev utilities on y (shared)
# ----------------------------
def y_to_t(y):
    return 2.0 * y - 1.0


def cheb_vander_t(t, deg):
    return cheb.chebvander(t, deg)


def eval_rational_g(y, num_coef, den_coef):
    t = y_to_t(y)
    num = cheb.chebval(t, num_coef)
    if len(den_coef) > 1:
        c = np.zeros_like(den_coef)
        c[1:] = den_coef[1:]  # denom starts at T1
        den = 1.0 + cheb.chebval(t, c)
    else:
        den = 1.0
    return num / den


def theta_fz_from_rational(theta, num_coef, den_coef, c0=None):
    y = 1.0 - np.cos(theta)
    s = np.sqrt(np.maximum(y, TINY))
    g = eval_rational_g(y, num_coef, den_coef)
    if c0 is not None:
        g = np.where(y <= 1e-24, c0, g)
    return s * g


# ----------------------------
# SK + Lawson for g(y)
# ----------------------------
def ratfit_g_SK_Lawson(
    y, g, n, m, sk_iters=10, lawson_iters=6, eps_law=1e-14, verbose=False
):
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    t = y_to_t(y)

    Tn = cheb_vander_t(t, n)  # (N, n+1)
    Tm = cheb_vander_t(t, m)[:, 1:]  # (N, m)
    N = y.size

    a = np.zeros(n + 1)
    b = np.zeros(m)
    wL = np.ones(N)
    for L in range(1, lawson_iters + 1):
        Q_prev = np.ones(N)
        for it in range(1, sk_iters + 1):
            Wsk = 1.0 / np.maximum(1e-16, np.abs(Q_prev))
            W = wL * Wsk
            A = np.hstack([Tn, -(g[:, None] * Tm)])
            Aw = A * W[:, None]
            yw = g * W
            sol, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            a = sol[: n + 1]
            b = sol[n + 1 :]
            P = Tn @ a
            Q = 1.0 + (Tm @ b)
            r = P - g * Q
            Q_prev = Q

        r_abs = np.abs(r)
        wL_new = 1.0 / (r_abs + eps_law)
        wL_new /= np.median(wL_new)
        wL = np.clip(wL_new, 0.1, 10.0)
        if verbose:
            sup = r_abs.max()
            rms = math.sqrt(np.mean(r_abs * r_abs))
            print(f"  Lawson {L:2d}: sup|r|={sup:.3e}  rms|r|={rms:.3e}")

    num_coef = a.copy()
    den_coef = np.empty(m + 1)
    den_coef[0] = 1.0
    den_coef[1:] = b
    return num_coef, den_coef


# ----------------------------
# Nonlinear refinement for g(y)
# ----------------------------
def refine_rational_g(
    y, g_true, num_init, den_init, n, m, lam_pole=1e-3, q_floor=1e-7, grid_pen=4096
):
    y = np.asarray(y, dtype=np.float64)
    t = y_to_t(y)
    Tn = cheb.chebvander(t, n)
    Tm = cheb.chebvander(t, m)[:, 1:]

    def pack(a, b):
        return np.concatenate([a, b])

    def unpack(p):
        return p[: n + 1], p[n + 1 :]

    p0 = pack(num_init, den_init[1:])

    xx = np.cos(PI * (np.arange(grid_pen) + 0.5) / grid_pen)
    Td = cheb.chebvander(xx, m)[:, 1:]  # T1..Tm

    def fun(p):
        a, b = unpack(p)
        P = Tn @ a
        Q = 1.0 + (Tm @ b)
        r = P - g_true * Q
        Qg = 1.0 + (Td @ b)
        pen = np.maximum(0.0, q_floor - np.abs(Qg))
        return np.hstack([r, lam_pole * pen])

    res = least_squares(
        fun, p0, method="trf", ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=100000
    )
    a, b = unpack(res.x)
    den = np.empty(m + 1)
    den[0] = 1.0
    den[1:] = b
    return a, den, res


# ----------------------------
# Degree search for a given group (via its Gintegrator)
# ----------------------------
def search_rational_precond(
    Gint: GIntegrator,
    tol=2e-12,
    deg_grid=None,
    n_samp=4096,
    sk_iters=8,
    lawson_iters=5,
    lam_pole=1e-2,
    q_floor=1e-7,
    grid_pen=4096,
    eps_law=1e-14,
    verbose=True,
):
    if deg_grid is None:
        deg_grid = [
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (6, 4),
            (7, 5),
            (8, 6),
            (9, 7),
            (10, 8),
            (12, 9),
        ]

    # Chebyshev–Lobatto nodes in θ -> to y
    k = np.arange(n_samp)
    theta_train = 0.5 * PI * 0.5 * (1.0 - np.cos(PI * k / (n_samp - 1)))
    y_train = 1.0 - np.cos(theta_train)

    theta_fz = theta_to_theta_fz_exact(theta_train, Gint)
    s = np.sqrt(np.maximum(y_train, TINY))
    g_true = theta_fz / s
    g_true[y_train <= 1e-24] = c0_leading_constant(Gint)

    best = None
    for n, m in deg_grid:
        num0, den0 = ratfit_g_SK_Lawson(
            y_train,
            g_true,
            n,
            m,
            sk_iters=sk_iters,
            lawson_iters=lawson_iters,
            eps_law=eps_law,
            verbose=False,
        )
        num, den, _ = refine_rational_g(
            y_train,
            g_true,
            num0,
            den0,
            n,
            m,
            lam_pole=lam_pole,
            q_floor=q_floor,
            grid_pen=grid_pen,
        )

        # dense grid assess
        xx = np.cos(PI * (np.arange(4096) + 0.5) / 4096)
        th_grid = (xx + 1.0) * 0.25 * PI
        th_true = theta_to_theta_fz_exact(th_grid, Gint)
        th_pred = theta_fz_from_rational(
            th_grid, num, den, c0=c0_leading_constant(Gint)
        )
        err = np.abs(th_pred - th_true)
        sup = np.max(err)
        rms = math.sqrt(np.mean(err * err))

        # denom health on t-grid
        c = np.zeros_like(den)
        c[1:] = den[1:]
        minQ = np.min(np.abs(1.0 + cheb.chebval(xx, c)))

        if verbose:
            print(
                f"[try] n={n:2d}, m={m:2d}   sup={sup:.3e}, rms={rms:.3e}, min|Q|={minQ:.3e}"
            )

        cand = dict(num=num, den=den, sup=sup, rms=rms, minQ=minQ, n=n, m=m)
        if best is None or sup < best["sup"]:
            best = cand
        if sup <= tol and minQ >= 1e-3:
            return cand
    return best


# ----------------------------
# Sampling & reporting
# ----------------------------
def sample_uniform_ho(n, rng=np.random.default_rng(0)):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.random(n)
    r = H_MAX * (u ** (1.0 / 3.0))
    return v * r[:, None]


def report_theta_errors(theta, exact, approx, tag="θ'"):
    err = np.abs(approx - exact)
    q = lambda p: np.quantile(err, p)
    print(
        f"{tag} error (rad):  max={err.max():.3e}  mean={err.mean():.3e}  rms={np.sqrt(np.mean(err**2)):.3e}  p99={q(0.99):.3e}"
    )
    errd = err * 180.0 / PI
    qd = lambda p: np.quantile(errd, p)
    print(
        f"{tag} error (deg):  max={errd.max():.3e}  mean={errd.mean():.3e}  rms={np.sqrt(np.mean(errd**2)):.3e}  p99={qd(0.99):.3e}"
    )


def report_cartesian_errors(H_exact, H_approx, tag="ho Cartesian"):
    diff = H_approx - H_exact
    e = np.linalg.norm(diff, axis=1)
    q = lambda p: np.quantile(e, p)
    print(
        f"{tag} abs:  max={e.max():.3e}  mean={e.mean():.3e}  rms={np.sqrt(np.mean(e**2)):.3e}  p99={q(0.99):.3e}"
    )
    er = e / H_MAX
    qr = lambda p: np.quantile(er, p)
    print(
        f"{tag} rel:  max={er.max():.3e}  mean={er.mean():.3e}  rms={np.sqrt(np.mean(er**2)):.3e}  p99={qr(0.99):.3e}"
    )


# ----------------------------
# Main: loop over C2, C3, C4, C6
# ----------------------------
if __name__ == "__main__":
    # Tunables (kept from your script; feel free to dial them down)
    N_GL = 128
    RAT_TOL = 2e-11
    SK_ITERS = 100
    LAWSON_ITERS = 20
    N_TRAIN = 512
    N_VAL = 8000
    N_TEST = 20000

    print(f"H_MAX = {H_MAX:.16f}")

    COEFFS = {}  # store num/den per group

    for name, tau, laue_id in GROUPS:
        print("\n" + "=" * 30 + f"  {name}  (tau={tau:.16e})  " + "=" * 30)
        Gint = GIntegrator(tau=tau, n_gl=N_GL)
        c0 = c0_leading_constant(Gint)
        print(f"G_tot = {Gint.G_tot:.16e}   c0 = {c0:.16e}")

        # degree grid (you can reuse your custom grid)
        DEG_GRID = [
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            # (6, 6),
            # (7, 7),
            # (8, 8),
            # (9, 9),
            # (10, 5),
            (10, 10),
            (15, 15),
            (20, 20),
            # (6, 4),
            # (7, 5),
            # (8, 6),
            # (9, 7),
            # (10, 8),
            # (12, 9),
        ]

        best = search_rational_precond(
            Gint,
            tol=RAT_TOL,
            deg_grid=DEG_GRID,
            n_samp=N_TRAIN,
            sk_iters=SK_ITERS,
            lawson_iters=LAWSON_ITERS,
            lam_pole=1e-3,
            q_floor=1e-7,
            grid_pen=4096 * 8,
            eps_law=1e-14,
            verbose=True,
        )

        print(f"\n=== {name} (preconditioned) fit selected ===")
        print(f"Degrees: n={best['n']}, m={best['m']}")
        print(
            f"Sup error: {best['sup']:.3e}   RMS: {best['rms']:.3e}   min|Q|: {best['minQ']:.3e}"
        )

        num = best["num"]
        den = best["den"]
        COEFFS[name] = dict(tau=tau, laue_id=laue_id, num=num, den=den)

        np.set_printoptions(precision=18, suppress=False, linewidth=180)
        print("\nNumerator Chebyshev coefficients a[0..n] over t=2y-1:")
        print(num)
        print("\nDenominator Chebyshev coefficients (Q(t)=1+Σ_{k=1}^m b_k T_k(t)):")
        print(den)

        # Validation on random θ
        rng = np.random.default_rng(123)
        theta_val = 0.5 * PI * rng.random(N_VAL)
        theta_exact = theta_to_theta_fz_exact(theta_val, Gint)
        theta_rat = theta_fz_from_rational(theta_val, num, den, c0=c0)
        print("\nValidation vs exact θ' (random θ):")
        report_theta_errors(theta_val, theta_exact, theta_rat, tag=f"θ' {name}")

        # End-to-end Cartesian test
        H = sample_uniform_ho(N_TEST, rng=np.random.default_rng(42))
        H_exact = ho_to_hoCk_exact(H, Gint)

        def ho_to_hoCk_rational(h):
            h = np.asarray(h, dtype=np.float64)
            out = np.empty_like(h)
            x, y, z = h[..., 0], h[..., 1], h[..., 2]
            zsign = np.sign(z)
            zsign[zsign == 0.0] = 1.0
            za = np.abs(z)
            rho = np.linalg.norm(h, axis=-1)
            xy = np.hypot(x, y)
            theta = np.arctan2(xy, za)
            theta_fz = theta_fz_from_rational(theta, num, den, c0=c0)
            R = R_of_theta(theta_fz, Gint.tau)
            rho_p = rho * (R / H_MAX)
            az = np.arctan2(y, x)
            s, c = np.sin(theta_fz), np.cos(theta_fz)
            out[..., 0] = rho_p * s * np.cos(az)
            out[..., 1] = rho_p * s * np.sin(az)
            out[..., 2] = rho_p * c * zsign
            return out

        H_rat = ho_to_hoCk_rational(H)
        print(f"\nEnd-to-end ho -> ho_{name} (rational) vs exact:")
        report_cartesian_errors(H_exact, H_rat, tag=f"ho_{name} (rational)")

    print("\nAll group coefficients collected in COEFFS (by name). Done.")
