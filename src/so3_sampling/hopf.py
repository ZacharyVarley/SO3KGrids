"""Hopf / ISOI grid on SO(3) (Yershova et al.). Ported from SO3Grids."""
from __future__ import annotations

import math
import numpy as np


def qu_std(qu: np.ndarray) -> np.ndarray:
    out = np.where(qu[..., 0:1] >= 0, qu, -qu)
    return np.ascontiguousarray(out)


def qu_norm(qu: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(qu, axis=-1, keepdims=True)
    return qu / np.maximum(n, 1e-12)


# --- Hopf grid (ISOI / Yershova et al.) ---
# Base sequence: 72 points = 12 HEALPix base pixels × 6 S1 angles
# From SO3_sequence/seq.txt: (healpix_pixel, s1_index) pairs
_SEQ_BASE = np.array(
    [
        [6, 0],
        [6, 3],
        [6, 1],
        [6, 4],
        [6, 2],
        [6, 5],
        [4, 0],
        [4, 3],
        [4, 1],
        [4, 4],
        [4, 2],
        [4, 5],
        [1, 0],
        [1, 3],
        [1, 1],
        [1, 4],
        [1, 2],
        [1, 5],
        [11, 0],
        [11, 3],
        [11, 1],
        [11, 4],
        [11, 2],
        [11, 5],
        [9, 0],
        [9, 3],
        [9, 1],
        [9, 4],
        [9, 2],
        [9, 5],
        [3, 0],
        [3, 3],
        [3, 1],
        [3, 4],
        [3, 2],
        [3, 5],
        [5, 0],
        [5, 3],
        [5, 1],
        [5, 4],
        [5, 2],
        [5, 5],
        [7, 0],
        [7, 3],
        [7, 1],
        [7, 4],
        [7, 2],
        [7, 5],
        [10, 0],
        [10, 3],
        [10, 1],
        [10, 4],
        [10, 2],
        [10, 5],
        [0, 0],
        [0, 3],
        [0, 1],
        [0, 4],
        [0, 2],
        [0, 5],
        [2, 0],
        [2, 3],
        [2, 1],
        [2, 4],
        [2, 2],
        [2, 5],
        [8, 0],
        [8, 3],
        [8, 1],
        [8, 4],
        [8, 2],
        [8, 5],
    ],
    dtype=np.int64,
)

_S1_ANGLES_DEG = np.array([30, 90, 150, 210, 270, 330], dtype=np.float64)
_S1_ANGLES_RAD = np.deg2rad(_S1_ANGLES_DEG)

_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)

# HEALPix find_point offsets: (d_healpix, d_s1_sign) for each (base_healpix, position)
# position 0..7; d_s1_sign is ±1 for s1 += d_s1_sign * (interval/2)
_FIND_POINT_OFFSETS = {
    (6,): [(3, -1), (0, 1), (3, 1), (0, -1), (2, -1), (1, 1), (2, 1), (1, -1)],
    (7,): [(3, -1), (0, 1), (3, 1), (0, -1), (2, -1), (1, 1), (2, 1), (1, -1)],
    (3, 1, 9, 11): [(3, -1), (0, 1), (3, 1), (0, -1), (1, -1), (2, 1), (1, 1), (2, -1)],
    (2, 0, 10, 8): [(0, -1), (3, 1), (0, 1), (3, -1), (1, -1), (2, 1), (1, 1), (2, -1)],
    (4, 5): [(0, -1), (3, 1), (0, 1), (3, -1), (2, -1), (1, 1), (2, 1), (1, -1)],
}
_FIND_POINT_OFFSET_BY_BASE = {
    base_healpix: offsets
    for base_group, offsets in _FIND_POINT_OFFSETS.items()
    for base_healpix in base_group
}


def _mk_pix2xy() -> tuple[np.ndarray, np.ndarray]:
    """HEALPix nested: pixel index -> (x, y) on face. Port of mk_pix2xy.c."""
    pix2x = np.zeros(1024, dtype=np.int32)
    pix2y = np.zeros(1024, dtype=np.int32)
    for kpix in range(1024):
        jpix = kpix
        ix, iy = 0, 0
        ip = 1
        while jpix != 0:
            ix += (jpix % 2) * ip
            jpix //= 2
            iy += (jpix % 2) * ip
            jpix //= 2
            ip *= 2
        pix2x[kpix] = ix
        pix2y[kpix] = iy
    return pix2x, pix2y


_PIX2X, _PIX2Y = _mk_pix2xy()


def _pix2ang_nest(nside: int, ipix: int) -> tuple[float, float]:
    """HEALPix nested: pixel -> (theta, phi). Port of pix2ang_nest.c."""
    npface = nside * nside
    face_num = ipix // npface
    ipf = ipix % npface
    ip_low = ipf % 1024
    ip_trunc = ipf // 1024
    ip_med = ip_trunc % 1024
    ip_hi = ip_trunc // 1024
    ix = 1024 * _PIX2X[ip_hi] + 32 * _PIX2X[ip_med] + _PIX2X[ip_low]
    iy = 1024 * _PIX2Y[ip_hi] + 32 * _PIX2Y[ip_med] + _PIX2Y[ip_low]
    jrt = ix + iy
    jpt = ix - iy
    fn = float(nside)
    fact1 = 1.0 / (3.0 * fn * fn)
    fact2 = 2.0 / (3.0 * fn)
    nl4 = 4 * nside
    jr = _JRLL[face_num] * nside - jrt - 1
    nr = nside
    z = (2 * nside - jr) * fact2
    kshift = (jr - nside) % 2
    if jr < nside:
        nr = jr
        z = 1.0 - nr * nr * fact1
        kshift = 0
    elif jr > 3 * nside:
        nr = nl4 - jr
        z = -1.0 + nr * nr * fact1
        kshift = 0
    theta = math.acos(np.clip(z, -1.0, 1.0))
    jp = (_JPLL[face_num] * nr + jpt + 1 + kshift) // 2
    if jp > nl4:
        jp -= nl4
    if jp < 1:
        jp += nl4
    phi = (jp - (kshift + 1) * 0.5) * (0.5 * math.pi / nr)
    return theta, phi


def _get_find_point_offset(base_healpix: int, position: int) -> tuple[int, float]:
    """Return (d_healpix, d_s1) for given base cell and position 0..7."""
    offsets = _FIND_POINT_OFFSET_BY_BASE.get(base_healpix)
    if offsets is None:
        return (0, 0.0)
    return offsets[position]


def _find_point(
    base_healpix: int,
    point: int,
    level: int,
    healpix_point: int,
    s1_point_deg: float,
) -> tuple[float, float, float]:
    """Recursive subdivision for Hopf grid. Returns (theta, phi, psi)."""
    while True:
        position = point % 8
        interval = 30.0 / level
        dh, d_s1_sign = _get_find_point_offset(base_healpix, position)
        healpix_point += dh
        s1_point_deg += d_s1_sign * (interval / 2.0)
        quo = point // 8
        if quo == 0:
            nside = 2**level
            theta, phi = _pix2ang_nest(nside, healpix_point)
            psi = math.radians(s1_point_deg)
            return theta, phi, psi
        point = quo - 1
        level += 1
        healpix_point *= 4


def _hopf2quat(theta: float, phi: float, psi: float) -> np.ndarray:
    """Hopf (theta, phi, psi) -> quaternion (w, x, y, z). Port of hopf2quat.C.
    ISOI outputs (x,y,z,w); we use (w,x,y,z), so reorder: w=x4, x=x1, y=x2, z=x3."""
    ct2, st2 = math.cos(theta / 2), math.sin(theta / 2)
    cp2, sp2 = math.cos(psi / 2), math.sin(psi / 2)
    x1 = ct2 * cp2
    x2 = st2 * math.sin(phi + psi / 2)
    x3 = st2 * math.cos(phi + psi / 2)
    x4 = ct2 * sp2
    # ISOI (x,y,z,w) -> our (w,x,y,z): w=x4, x=x1, y=x2, z=x3
    return np.array([x4, x1, x2, x3], dtype=np.float64)


def _make_hopf_base_quats() -> np.ndarray:
    """Base 72-point Hopf block in ISOI order, converted to (w, x, y, z)."""
    qu = np.empty((len(_SEQ_BASE), 4), dtype=np.float64)
    for i, (hp, s1_idx) in enumerate(_SEQ_BASE):
        theta, phi = _pix2ang_nest(1, int(hp))
        qu[i] = _hopf2quat(theta, phi, _S1_ANGLES_RAD[int(s1_idx)])
    return qu


_HOPF_BASE_QUATS = _make_hopf_base_quats()


def grid_hopf(n: int) -> np.ndarray:
    """
    Hopf grid (ISOI / Yershova et al.).
    SO(3) ≅ S² × S¹: HEALPix on S², 6-fold on S¹, with recursive subdivision.
    Matches the ISOI / SO3_sequence ordering and returns quaternions with w >= 0.
    """
    if n <= 0:
        return np.empty((0, 4), dtype=np.float64)

    limit = min(n, len(_HOPF_BASE_QUATS))
    qu = np.empty((n, 4), dtype=np.float64)
    qu[:limit] = _HOPF_BASE_QUATS[:limit]

    for i in range(n - limit):
        base_idx = i % len(_SEQ_BASE)
        cur_point = i // len(_SEQ_BASE)
        hp, s1_idx = _SEQ_BASE[base_idx]
        point_healpix = 4 * int(hp)
        point_s1_deg = float(_S1_ANGLES_DEG[int(s1_idx)])
        theta, phi, psi = _find_point(
            int(hp), cur_point, 1, point_healpix, point_s1_deg
        )
        qu[limit + i] = _hopf2quat(theta, phi, psi)

    return qu_norm(qu_std(qu))