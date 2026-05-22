"""
Covering radius of a set of quaternions on S^3 with antipodal symmetry:

Larsen, Peter Mahler, and Søren Schmidt. "Improved orientation sampling for
indexing diffraction patterns of polycrystalline materials." Applied
Crystallography 50.6 (2017): 1571-1582.

The naive covering radius is computed by:
    - forming the full set of quaternions by applying the Laue group to the
      original set
    - adding the antipodal quaternions to the set
    - Computing the convex hull of the set of the full set of quaternions
    - Computing the circumradius of the convex hull
    - Returning the maximum circumradius

Mechanism:
    - For each candidate image q_op = q ⊗ op, compute a conservative LOWER BOUND
      on geodesic distance (on S^3) from q_op's orientation-class to the Laue FZ
      boundary, using the actual FZ inequalities (the same ones used in
      ori_in_fz_laue). 
    - Include q_op in the buffer iff:
        outside_FZ(q_op_std) AND dist_lb_S3(q_op_std) <= buffer_deg/2  (in
        radians) where q_op_std = qu_std(q_op) (canonical representative, w>=0).

Cyclic groups C2/C3/C4/C6 (laue_id 2,6,4,8 in your mapping) can have antipodal
adjacency such that the covering radius is found within a simplex with some
vertices from P and some from -P. We therefore also test antipodal candidates
(-q_op) for inclusion, but still evaluate the FZ distance using the canonical
representative qu_std(-q_op). If it qualifies, we include the ACTUAL antipodal
quaternion (-q_op) into the hull input (without adding full -P).

Unfortunately, we have to use the CPU to efficiently compute the convex hull via
Delabella or whatever the underlying implementation is from scipy and this is
very expensive for large sets of quaternions. There may be improvements such as
finding an upper bound on the radius then using a ball search on a kd-tree to
find enough neighbors per point that the convex hull simplex will get caught and
then brute-force computing the circumradii of all K choose 4 simplices. Where K
is the number of neighbors that were in the ball search for each point. 

"""

import math
import numpy as np
import torch
from torch import Tensor
from pykeops.torch import LazyTensor
from scipy.spatial import ConvexHull

from laue_ops import laue_elements, ori_in_fz_laue
from orientation_ops import qu_prod, cu2qu, qu_std, qu_norm, ho2ax


# -----------------------------
# KeOps: max |dot| reductions
# -----------------------------
def keops_max_abs_dot(A: Tensor, B: Tensor) -> Tensor:
    Ai = LazyTensor(A[:, None, :])
    Bj = LazyTensor(B[None, :, :])
    dot = (Ai * Bj).sum(dim=2).abs()
    return dot.max(dim=1).view(-1)


# -----------------------------
# FZ grid
# -----------------------------
def cu_rej_grid(edge_length: int, laue_id: int, device: torch.device) -> Tensor:
    cu = torch.linspace(
        -0.5 * torch.pi ** (2.0 / 3.0),
        0.5 * torch.pi ** (2.0 / 3.0),
        2 * edge_length + 2,
        device=device,
        dtype=torch.float64,
    )
    cu = cu + 0.5 * (cu[1] - cu[0])
    cu = cu[:-1]

    if laue_id == 1:
        cu_x, cu_y, cu_z = cu, cu, cu
    elif laue_id == 2:
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.79]
    elif laue_id == 3:
        c = cu[cu.abs() <= 0.79]
        cu_x, cu_y, cu_z = c, c, c
    elif laue_id == 4:
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.49]
    elif laue_id == 5:
        cxy = cu[cu.abs() <= 0.66]
        cz = cu[cu.abs() <= 0.49]
        cu_x, cu_y, cu_z = cxy, cxy, cz
    elif laue_id == 6:
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.61]
    elif laue_id == 7:
        cxy = cu[cu.abs() <= 0.7]
        cz = cu[cu.abs() <= 0.61]
        cu_x, cu_y, cu_z = cxy, cxy, cz
    elif laue_id == 8:
        cu_x, cu_y, cu_z = cu, cu, cu[cu.abs() <= 0.35]
    elif laue_id == 9:
        cxy = cu[cu.abs() <= 0.64]
        cz = cu[cu.abs() <= 0.35]
        cu_x, cu_y, cu_z = cxy, cxy, cz
    elif laue_id == 10:
        c = cu[cu.abs() <= 0.61]
        cu_x, cu_y, cu_z = c, c, c
    elif laue_id == 11:
        c = cu[cu.abs() <= 0.45]
        cu_x, cu_y, cu_z = c, c, c
    elif laue_id == 12:
        c = cu[cu.abs() <= 0.32]
        cu_x, cu_y, cu_z = c, c, c
    else:
        raise ValueError("laue_id must be in [1,12]")

    cu3 = torch.stack(torch.meshgrid(cu_x, cu_y, cu_z, indexing="ij"), dim=-1).reshape(-1, 3)
    qu = cu2qu(cu3)
    qu = qu_norm(qu_std(qu))
    qu = qu[ori_in_fz_laue(qu, int(laue_id))]
    return qu


# -----------------------------
# Distance LOWER BOUND to FZ boundary (S^3 radians), using the actual FZ inequalities
# Only intended as a *rejection test* for buffering: if lb > buf, discard safely.
# For inside points, returns 0.
# Input quats must be unit and in canonical chart (w>=0), matching ori_in_fz_laue assumptions.
# -----------------------------
@torch.no_grad()
def fz_boundary_distance_lb_s3(quats_std: Tensor, laue_id: int) -> Tensor:
    q = quats_std
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    device = q.device
    dtype = q.dtype

    # inside mask via your exact predicate
    inside = ori_in_fz_laue(q, int(laue_id))
    out = torch.zeros(q.shape[:-1], device=device, dtype=dtype)
    if torch.all(inside):
        return out

    need = ~inside
    wn, xn, yn, zn = w[need], x[need], y[need], z[need]

    # sign helpers for active halfspaces (avoid sign(0)=0; use +1)
    sx = torch.where(xn >= 0, torch.ones_like(xn), -torch.ones_like(xn))
    sy = torch.where(yn >= 0, torch.ones_like(yn), -torch.ones_like(yn))
    sz = torch.where(zn >= 0, torch.ones_like(zn), -torch.ones_like(zn))

    # For a halfspace a·q >= 0:
    # if s = (a·q)/||a|| < 0, then distance to boundary (a·x=0) is asin(-s).
    def plane_dist(a0, a1, a2, a3, ww, xx, yy, zz):
        # a0..a3 can be python floats or tensors; force tensors on correct device/dtype
        if not torch.is_tensor(a0): a0 = torch.tensor(float(a0), device=device, dtype=dtype)
        if not torch.is_tensor(a1): a1 = torch.tensor(float(a1), device=device, dtype=dtype)
        if not torch.is_tensor(a2): a2 = torch.tensor(float(a2), device=device, dtype=dtype)
        if not torch.is_tensor(a3): a3 = torch.tensor(float(a3), device=device, dtype=dtype)

        dot = a0 * ww + a1 * xx + a2 * yy + a3 * zz
        norm = torch.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3).clamp_min(1e-18)
        s = dot / norm
        v = (-s).clamp(0.0, 1.0)
        return torch.asin(v)

    if laue_id == 1:
        out[need] = 0.0
        return out

    elif laue_id == 2:
        # C2: |z| <= w  => (w - z)>=0 and (w + z)>=0
        d1 = plane_dist(1.0, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d2 = plane_dist(1.0, 0.0, 0.0, +1.0, wn, xn, yn, zn)
        out[need] = torch.maximum(d1, d2)
        return out

    elif laue_id == 3:
        # D2: max(|x|,|y|,|z|) <= w  => w ± x >=0, w ± y >=0, w ± z >=0
        d = []
        d.append(plane_dist(1.0, -1.0, 0.0, 0.0, wn, xn, yn, zn))
        d.append(plane_dist(1.0, +1.0, 0.0, 0.0, wn, xn, yn, zn))
        d.append(plane_dist(1.0, 0.0, -1.0, 0.0, wn, xn, yn, zn))
        d.append(plane_dist(1.0, 0.0, +1.0, 0.0, wn, xn, yn, zn))
        d.append(plane_dist(1.0, 0.0, 0.0, -1.0, wn, xn, yn, zn))
        d.append(plane_dist(1.0, 0.0, 0.0, +1.0, wn, xn, yn, zn))
        out[need] = torch.stack(d, dim=0).max(dim=0).values
        return out

    elif laue_id == 4:
        # C4: |z| <= (sqrt2-1) w
        k = (2.0 ** 0.5) - 1.0
        d1 = plane_dist(k, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d2 = plane_dist(k, 0.0, 0.0, +1.0, wn, xn, yn, zn)
        out[need] = torch.maximum(d1, d2)
        return out

    elif laue_id == 5:
        # D4: |z| <= (sqrt2-1) w  AND  rot(x,y) <= w (piecewise linear)
        k = (2.0 ** 0.5) - 1.0
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)
        tan67_5 = (2.0 ** 0.5) + 1.0

        ax, ay = xn.abs(), yn.abs()
        cond = ax > ay  # m=ax if cond else ay
        m = torch.where(cond, ax, ay)
        n = torch.where(cond, ay, ax)
        branch = m > tan67_5 * n

        # coefficients for rot = c_x*|x| + c_y*|y| with pointwise branches
        # if cond:
        #   rot1=ax -> (1,0)
        #   rot2=inv_sqrt2*ax + inv_sqrt2*ay -> (inv_sqrt2, inv_sqrt2)
        # else:
        #   rot1=ay -> (0,1)
        #   rot2=inv_sqrt2*ay + inv_sqrt2*ax -> (inv_sqrt2, inv_sqrt2)
        cx_rot1 = torch.where(cond, torch.ones_like(ax), torch.zeros_like(ax))
        cy_rot1 = torch.where(cond, torch.zeros_like(ay), torch.ones_like(ay))
        cx_rot2 = inv_sqrt2 * torch.ones_like(ax)
        cy_rot2 = inv_sqrt2 * torch.ones_like(ay)

        cx = torch.where(branch, cx_rot1, cx_rot2)
        cy = torch.where(branch, cy_rot1, cy_rot2)

        # active plane: w - cx*sx*x - cy*sy*y >= 0
        d_rot = plane_dist(1.0, -cx * sx, -cy * sy, 0.0, wn, xn, yn, zn)
        d_z1 = plane_dist(k, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d_z2 = plane_dist(k, 0.0, 0.0, +1.0, wn, xn, yn, zn)

        out[need] = torch.stack([d_rot, d_z1, d_z2], dim=0).max(dim=0).values
        return out

    elif laue_id == 6:
        # C3: |z| <= (1/sqrt3) w
        k = 1.0 / (3.0 ** 0.5)
        d1 = plane_dist(k, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d2 = plane_dist(k, 0.0, 0.0, +1.0, wn, xn, yn, zn)
        out[need] = torch.maximum(d1, d2)
        return out

    elif laue_id == 7:
        # D3: |z| <= tan30*w (1/sqrt3) AND rot<=w with:
        # if |x| >= sqrt3*|y|: rot=|x| else rot=0.5|x| + (sqrt3/2)|y|
        k = 1.0 / (3.0 ** 0.5)
        sqrt3 = 3.0 ** 0.5

        ax, ay = xn.abs(), yn.abs()
        branch = ax >= sqrt3 * ay

        cx = torch.where(branch, torch.ones_like(ax), 0.5 * torch.ones_like(ax))
        cy = torch.where(branch, torch.zeros_like(ay), (sqrt3 / 2.0) * torch.ones_like(ay))

        d_rot = plane_dist(1.0, -cx * sx, -cy * sy, 0.0, wn, xn, yn, zn)
        d_z1 = plane_dist(k, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d_z2 = plane_dist(k, 0.0, 0.0, +1.0, wn, xn, yn, zn)

        out[need] = torch.stack([d_rot, d_z1, d_z2], dim=0).max(dim=0).values
        return out

    elif laue_id == 8:
        # C6: |z| <= tan15*w where tan15 = 2 - sqrt3
        k = 2.0 - (3.0 ** 0.5)
        d1 = plane_dist(k, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d2 = plane_dist(k, 0.0, 0.0, +1.0, wn, xn, yn, zn)
        out[need] = torch.maximum(d1, d2)
        return out

    elif laue_id == 9:
        # D6: |z| <= tan15*w AND rot<=w with:
        # m=max(|x|,|y|), n=min(|x|,|y|)
        # if m > tan75*n (2+sqrt3): rot=m else rot=(sqrt3/2)m + 0.5 n
        k = 2.0 - (3.0 ** 0.5)
        sqrt3 = 3.0 ** 0.5
        tan75 = 2.0 + sqrt3

        ax, ay = xn.abs(), yn.abs()
        cond = ax > ay
        m = torch.where(cond, ax, ay)
        n = torch.where(cond, ay, ax)
        branch = m > tan75 * n

        # coefficients for each branch/cond
        # cond true: m=ax,n=ay
        cx1_t = torch.ones_like(ax)
        cy1_t = torch.zeros_like(ay)
        cx2_t = (sqrt3 / 2.0) * torch.ones_like(ax)
        cy2_t = 0.5 * torch.ones_like(ay)

        # cond false: m=ay,n=ax
        cx1_f = torch.zeros_like(ax)
        cy1_f = torch.ones_like(ay)
        cx2_f = 0.5 * torch.ones_like(ax)
        cy2_f = (sqrt3 / 2.0) * torch.ones_like(ay)

        cx_rot1 = torch.where(cond, cx1_t, cx1_f)
        cy_rot1 = torch.where(cond, cy1_t, cy1_f)
        cx_rot2 = torch.where(cond, cx2_t, cx2_f)
        cy_rot2 = torch.where(cond, cy2_t, cy2_f)

        cx = torch.where(branch, cx_rot1, cx_rot2)
        cy = torch.where(branch, cy_rot1, cy_rot2)

        d_rot = plane_dist(1.0, -cx * sx, -cy * sy, 0.0, wn, xn, yn, zn)
        d_z1 = plane_dist(k, 0.0, 0.0, -1.0, wn, xn, yn, zn)
        d_z2 = plane_dist(k, 0.0, 0.0, +1.0, wn, xn, yn, zn)

        out[need] = torch.stack([d_rot, d_z1, d_z2], dim=0).max(dim=0).values
        return out

    elif laue_id == 10:
        # T: |x|+|y|+|z| <= w  (active plane uses pointwise signs)
        d_sum = plane_dist(1.0, -sx, -sy, -sz, wn, xn, yn, zn)
        out[need] = d_sum
        return out

    elif laue_id == 11:
        # O: max(|x|,|y|,|z|) <= (sqrt2-1) w  AND  |x|+|y|+|z| <= w
        k = (2.0 ** 0.5) - 1.0
        d_sum = plane_dist(1.0, -sx, -sy, -sz, wn, xn, yn, zn)

        # component max constraints: (k*w - sx*x)>=0 etc
        d_x = plane_dist(k, -sx, 0.0, 0.0, wn, xn, yn, zn)
        d_y = plane_dist(k, 0.0, -sy, 0.0, wn, xn, yn, zn)
        d_z = plane_dist(k, 0.0, 0.0, -sz, wn, xn, yn, zn)

        out[need] = torch.stack([d_sum, d_x, d_y, d_z], dim=0).max(dim=0).values
        return out

    elif laue_id == 12:
        # I: use Voronoi-type condition from your ori_in_fz_laue:
        # inside if w >= max_other over 12 five-fold ops (subset used in your fast check).
        # Distance to the bisector set (where arccos(w)=arccos(max_other)) has lower bound:
        #   d_lb >= 0.5 * (arccos(w) - arccos(max_other))  for outside points.
        ops_need = laue_elements(laue_id).to(q)[1:13]  # 12 five-fold ops
        dots = torch.einsum("ij,nj->ni", ops_need, q[need])
        max_other = dots.abs().max(dim=-1).values.clamp(0.0, 1.0)
        a = torch.acos(wn.clamp(0.0, 1.0))
        b = torch.acos(max_other)
        # outside means wn < max_other => a > b
        d_lb = 0.5 * (a - b).clamp_min(0.0)
        out[need] = d_lb
        return out

    else:
        raise ValueError(f"laue_id {laue_id} not supported")


# -----------------------------
# Efficient covering radius: buffer points by true "degrees from FZ boundary"
# -----------------------------
@torch.no_grad()
def covering_radius(
    quats_fz: Tensor,
    laue_id: int,
    *,
    edge_length_check: int = 20,
    grid_eps_factor: float = 2.0,
    qhull_options: str = "QJ",
) -> float:
    device = quats_fz.device
    dtype = torch.float64

    # core in FZ
    q = qu_norm(qu_std(quats_fz.to(device=device, dtype=dtype))).contiguous()
    N_core = q.shape[0]

    ops = laue_elements(int(laue_id)).to(device=device, dtype=dtype).contiguous()
    G = ops.shape[0]

    # ---- Step 1: reference grid spacing estimate (SO(3) angle)
    grid = cu_rej_grid(int(edge_length_check), int(laue_id), device=device).to(dtype=dtype).contiguous()
    grid = qu_norm(qu_std(grid))
    N_grid = grid.shape[0]

    ho_edge = ((math.pi ** 2) / (G * N_grid)) ** (1.0 / 3.0)
    ho_toy = torch.tensor([[0.0, 0.0, ho_edge]], device=device, dtype=dtype)
    theta_grid = float(ho2ax(ho_toy)[0, 3].item())  # SO(3) radians

    # ---- Step 2: upper bound from grid to nearest core point (SO(3) radians)
    max_dot = keops_max_abs_dot(grid, q)
    smallest_max_dot = float(max_dot.min().item())
    max_theta = 2.0 * math.acos(max(0.0, min(1.0, smallest_max_dot)))  # SO(3)

    theta_buf = (max_theta + theta_grid) * grid_eps_factor  # SO(3)
    theta_buf = min(theta_buf, math.pi)                   # SO(3) can't exceed pi
    buf_s3 = 0.5 * theta_buf                              # S^3 buffer radius

    print(
        f"theta_grid: {theta_grid * 180.0 / math.pi:.8f} deg, "
        f"max_theta: {max_theta * 180.0 / math.pi:.8f} deg, "
        f"theta_buf: {theta_buf * 180.0 / math.pi:.8f} deg"
    )

    # ---- Step 3: augment by boundary-distance test (normal images)
    augmented = [q]

    for op in ops[1:]:
        q_op = qu_norm(qu_prod(q, op))          # actual candidate in S^3
        q_std = qu_norm(qu_std(q_op))           # canonical for FZ tests

        inside = ori_in_fz_laue(q_std, int(laue_id))
        d_lb = fz_boundary_distance_lb_s3(q_std, int(laue_id))

        take = (~inside) & (d_lb <= buf_s3)
        if torch.any(take):
            augmented.append(q_op[take])

    # ---- Step 3b: cyclic Laue groups can be adjacent to antipodes; test antipodal candidates too
    # Your mapping: 1=C1 (trivial group), 2=C2, 4=C4, 6=C3, 8=C6

    if int(laue_id) == 1:
        # just include the antipodes
        augmented.append(-q)

    if int(laue_id) in (2, 4, 6, 8):
        for op in ops:
            q_op = qu_norm(qu_prod(q, op))
            q_ant = -q_op                       # actual antipodal candidate to include
            augmented.append(q_ant)

    # if int(laue_id) in (2, 4, 6, 8):
    #     for op in ops:
    #         q_op = qu_norm(qu_prod(q, op))
    #         q_ant = -q_op                       # actual antipodal candidate to include
    #         q_std = qu_norm(qu_std(q_ant))      # canonical for FZ tests

    #         inside = ori_in_fz_laue(q_std, int(laue_id))
    #         d_lb = fz_boundary_distance_lb_s3(q_std, int(laue_id))

    #         take = (~inside) & (d_lb <= buf_s3)
    #         if torch.any(take):
    #             augmented.append(q_ant[take])

    augmented = torch.cat(augmented, dim=0).contiguous()
    print(f"Buffer size: {augmented.shape[0] - N_core} added to {N_core} core points")

    # ---- Step 4: ConvexHull on CPU; NO qu_std; NO full antipodes
    P = qu_norm(augmented).contiguous()
    P_np = P.detach().cpu().contiguous().numpy()

    """
    apparently adding a point on the interior of the 3-sphere 
    removes degeneracies in the convex hull and this is important:

    Alexa, Marc. "Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO
    (3)." In Proceedings of the IEEE/CVF Conference on Computer Vision and
    Pattern Recognition, pp. 8291-8300. 2022.
    
    The authors states: "An important trick is to add one point in the interior
    of the sphere. This has no effect on the convex hull, but avoids the
    computationally costly handling of the degenerate situation that all
    simplices share the same circumsphere."
    """

    # q_interior = np.array([[1e-5, 1e-6, 1e-7, 1e-8]])
    # q_interior = q_interior / np.linalg.norm(q_interior)
    # P_np = np.concatenate([q_interior, P_np], axis=0)
    # hull = ConvexHull(P_np, qhull_options=qhull_options)

    # just use the original P_np
    hull = ConvexHull(P_np, qhull_options=qhull_options)

    # ---- Step 5: GPU processing of equations + facet filtering touching core
    simplices = torch.from_numpy(hull.simplices).to(device=device, dtype=torch.int64)
    keep = (simplices < N_core).any(dim=1)

    eq = torch.from_numpy(hull.equations).to(device=device, dtype=dtype)  # (F,5) [n b]
    n = eq[:, :4]
    b = eq[:, 4]
    h = b.abs() / n.norm(dim=1).clamp_min(1e-18)
    h = h.clamp(0.0, 1.0)
    # the covering radius is half of the usual alpha angle so no *2.0
    theta_facets = torch.acos(h)  # SO(3) radians
    theta = theta_facets[keep].max().item()
    return float(theta)


# -----------------------------
# Naive covering radius: full orbit + antipodes in hull (reference)
# -----------------------------
@torch.no_grad()
def covering_radius_naive(
    quats_fz: Tensor,
    laue_id: int,
    *,
    qhull_options: str = "QJ",
) -> float:
    device = quats_fz.device
    dtype = torch.float64

    q = qu_norm(qu_std(quats_fz.to(device=device, dtype=dtype))).contiguous()
    ops = laue_elements(int(laue_id)).to(device=device, dtype=dtype).contiguous()

    imgs = [qu_norm(qu_prod(q, ops[h])) for h in range(ops.shape[0])]
    P = torch.cat(imgs, dim=0).contiguous()
    P = torch.cat([P, -P], dim=0).contiguous()  # required for naive reference

    P_np = P.detach().cpu().contiguous().numpy()
    hull = ConvexHull(P_np, qhull_options=qhull_options)

    eq = torch.from_numpy(hull.equations).to(device=device, dtype=dtype)
    n = eq[:, :4]
    b = eq[:, 4]
    h = b.abs() / n.norm(dim=1).clamp_min(1e-18)
    h = h.clamp(0.0, 1.0)
    # the covering radius is half of the usual alpha angle so no *2.0
    theta = torch.acos(h).max().item()
    return float(theta)


def _deg(x: float) -> float:
    return x * 180.0 / math.pi

def covering_radius_star_deg(N: int) -> float:
    if N < 100:
        print("Asymptotic approximation is only float32 accurate for N > 100")
    else:
        if N < 1000:
            print("Asymptotic approximation is only float64 accurate for N > 1000")
    
    """
    theta*(N) asymptotic series (radians):
  1.889832101454757e+00 * N^(-0.333333)
  5.739307635124498e-01 * N^(-1)
  5.810692175392054e-01 * N^(-1.66667)
  8.597894141163933e-01 * N^(-2.33333)
  1.490131881944242e+00 * N^(-3)
  2.791922498626377e+00 * N^(-3.66667)
  5.481645765756285e+00 * N^(-4.33333)
  1.112163594341394e+01 * N^(-5)
    """
    Nfp = float(N)
    out = 1.889832101454757e+00 * Nfp**(-(1.0/3)) + \
        5.739307635124498e-01 * Nfp**(-1.0) + \
        5.810692175392054e-01 * Nfp**(-5.0/3) + \
        8.597894141163933e-01 * Nfp**(-7.0/3) + \
        1.490131881944242e+00 * Nfp**(-3.0) + \
        2.791922498626377e+00 * Nfp**(-11.0/3) + \
        5.481645765756285e+00 * Nfp**(-13.0/3) + \
        1.112163594341394e+01 * Nfp**(-5.0)
    return out * (180.0 / math.pi)



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check that theta_star of 120 is 0.388140 and of 1920 is 0.152352
    print(f"theta_star of  120 is {covering_radius_star_deg(120):.8f} vs {_deg(0.388140):.8f}...")
    print(f"theta_star of 1920 is  {covering_radius_star_deg(1920):.8f} vs  {_deg(0.152352):.8f}...")
    # for laue_id in range(2, 13):
    for laue_id in [11, 12]:
        print("-" * 100)
        print(f"Laue {laue_id:2d} | device={device.type}")

        quats_fz = cu_rej_grid(20, laue_id, device=device)

        th_eff_rad = covering_radius(
            quats_fz,
            laue_id,
            edge_length_check=100,
            grid_eps_factor=2.0,
            qhull_options="QJ",
        )
        th_eff_deg = _deg(th_eff_rad)

        G = laue_elements(laue_id).shape[0]
        # theta* is based on total number of quaternions on S^3 not just FZ
        # We can apply all of the operators |G| and then join negation {P, -P}
        theta_star_deg = covering_radius_star_deg(2.0 * G * quats_fz.shape[0])

        print(f"  theta*    = {theta_star_deg:.8f} deg")
        print(f"  eff theta = {th_eff_deg:.8f} deg")

        # th_ref = covering_radius_naive(quats_fz, laue_id, qhull_options="QJ")
        # # th_ref = 0.0
        # err = abs(th_eff - th_ref)
        # print(f"  ref theta = {_deg(th_ref):.8f} deg | abs error = {_deg(err):.8f} deg")
