"""
Shared utilities for UAV multi-agent simulation environments.

This module contains the duplicated parts between:
- `envs/sim_env_pid.py`
- `envs/sim_env_mpc.py`

It includes:
- 3D feasibility / geometry helpers (half-space projection utilities)
- laser scan -> collision plane selection (`get_collision_plane`)
- sampling start/goal positions for the 3D map
- reward shaping helpers (safety penalty / tetrahedral utilities)
- a lightweight 3D visualizer (`QuadSim3D`)
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np

enable_realtime = os.environ.get("ENABLE_REALTIME_DISPLAY", "False").lower() == "true"
if enable_realtime:
    # Enable interactive plotting if real-time mode is requested.
    plt.ion()

# Optional QP solver dependency.
try:
    import casadi as ca
except ImportError:
    ca = None
    print("Warning: casadi not available, project_goal_to_planes_qp will use fallback method")


def _vol6(a, b, c, d):
    """Return 6 * oriented volume det([b-a, c-a, d-a])."""
    return np.linalg.det(np.stack([b - a, c - a, d - a], axis=1))


def point_in_tetrahedron_3d(P, A, B, C, D, tol=1e-9):
    """
    Check whether point P is inside tetrahedron ABCD (including boundary).

    Returns:
        (inside: bool, error_scale: float)
    """
    P, A, B, C, D = map(lambda x: np.asarray(x, float).reshape(3), [P, A, B, C, D])
    V = np.abs(_vol6(A, B, C, D))  # 6x volume
    if V < tol:
        return False, V / 6.0

    v0 = np.abs(_vol6(P, B, C, D))
    v1 = np.abs(_vol6(A, P, C, D))
    v2 = np.abs(_vol6(A, B, P, D))
    v3 = np.abs(_vol6(A, B, C, P))

    s = v0 + v1 + v2 + v3 - V
    close_sum = s <= tol * (V + 1.0)
    return close_sum, s / 6


def barycentric_in_tetrahedron_3d(P, A, B, C, D, tol=1e-12):
    """
    Compute barycentric coordinates of P w.r.t tetrahedron ABCD.

    Returns:
        (inside: bool, weights: np.ndarray | None)
    """
    P, A, B, C, D = map(lambda x: np.asarray(x, float).reshape(3), [P, A, B, C, D])
    V = _vol6(A, B, C, D)
    if abs(V) < tol:
        return False, None

    vA = _vol6(P, B, C, D)
    vB = _vol6(A, P, C, D)
    vC = _vol6(A, B, P, D)
    vD = _vol6(A, B, C, P)

    wA, wB, wC, wD = vA / V, vB / V, vC / V, vD / V
    w = np.array([wA, wB, wC, wD])
    inside = (np.all(w >= -tol) and abs(w.sum() - 1.0) <= tol)
    return inside, w


def batch_points_in_tetrahedron_3d(Ps, A, B, C, D, tol=1e-12):
    """
    Batched tetrahedron inclusion test for points Ps (N,3).
    Returns a boolean mask of length N.
    """
    Ps = np.asarray(Ps, float).reshape(-1, 3)
    A, B, C, D = map(lambda x: np.asarray(x, float).reshape(3), [A, B, C, D])

    V = _vol6(A, B, C, D)
    if abs(V) < tol:
        return np.zeros(len(Ps), dtype=bool)

    BA, CA, DA = (B - A), (C - A), (D - A)
    M = np.stack([BA, CA, DA], axis=1)  # 3x3
    Minv = np.linalg.inv(M)

    rhs = (Ps - A) @ Minv.T  # (N,3) -> wB,wC,wD
    wB, wC, wD = rhs[:, 0], rhs[:, 1], rhs[:, 2]
    wA = 1.0 - (wB + wC + wD)
    W = np.stack([wA, wB, wC, wD], axis=1)
    inside = (W >= -tol).all(axis=1)
    return inside


class QuadSim3D:
    """
    Minimal 3D visualizer.

    - Uses `world_frame()` from quadrotor objects to render geometry
    - Updates arm/heading lines and trajectory history
    """

    def __init__(self, dt=0.01, Tmax=5.0, animation_frequency=50):
        self.dt = dt
        self.Tmax = Tmax
        self.t = 0.0

        # In real-time mode we animate more frequently (smoother but heavier).
        if enable_realtime:
            animation_frequency = max(animation_frequency, 100)
        self.animation_rate = 1.0 / animation_frequency

        self.lines = None
        self.agent_lines = []  # each item: [l1,l2,l3,l4]
        self.ax = None

    def init_plot(self, ax=None):
        """Create 3D axis and initialize line handles."""
        if ax is None:
            fig = plt.figure(figsize=(13, 11))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_position([0.01, 0.05, 0.99, 0.99])

        # four line handles
        l1, = ax.plot([], [], [], "-", c="red", zorder=10)
        l2, = ax.plot([], [], [], "-", c="blue", zorder=10)
        l3, = ax.plot([], [], [], "-", c="green", markevery=2, zorder=-10)
        l4, = ax.plot([], [], [], ".", c="black", markersize=2, zorder=5)

        self.lines = [l1, l2, l3, l4]
        self.ax = ax

    def add_agent_plot(self, base_color=None):
        """Create a dedicated set of line handles for one quadrotor."""
        if base_color is None:
            base_color = f"C{len(self.agent_lines) % 10}"

        l1, = self.ax.plot([], [], [], "-", c=base_color, zorder=10)
        l2, = self.ax.plot([], [], [], "-", c="blue", zorder=10)
        l3, = self.ax.plot([], [], [], "-", c="green", zorder=10)
        l4, = self.ax.plot(
            [],
            [],
            [],
            "-",
            color="green",
            linewidth=2,
            alpha=0.95,
            solid_capstyle="round",
            zorder=5,
        )

        self.agent_lines.append([l1, l2, l3, l4])

    def update_plot(self, frames, paths):
        """Update line geometries + trajectories for all rendered quads."""
        traj_colors = ["b", "g", "y", "c", "r", "m", "k"]
        for i in range(len(self.agent_lines)):
            F = np.asarray(frames[i])
            lines = self.agent_lines[i]

            # update arm/heading lines
            lines_data = [
                F[:, [0, 2]],
                F[:, [1, 3]],
                F[:, [4, 5]],
            ]
            for line, line_data in zip(lines[:3], lines_data):
                x, y, z = line_data
                line.set_data(x, y)
                line.set_3d_properties(z)

            # update trajectories
            hist = np.asarray(paths[i])
            if hist.size >= 3:
                ln_traj = lines[-1]
                ln_traj.set_data(hist[:, 0], hist[:, 1])
                ln_traj.set_3d_properties(hist[:, 2])
                ln_traj.set_color(traj_colors[i % len(traj_colors)])


def sellect(scan):
    """
    Select top-k closest hit points from a scan dict.

    Note: currently unused in most pipelines, kept for completeness.
    """
    hit_dists = scan["hit_dists"]
    valid = np.where(np.isfinite(hit_dists))[0]
    k = min(8, valid.size)

    order = np.argsort(hit_dists[valid])
    topk_idx = valid[order[:k]]

    return {
        "topk_dists": hit_dists[topk_idx],
        "topk_points": scan["hit_points"][topk_idx],
        "topk_dirs": scan["dirs_world"][topk_idx],
        "topk_ids": scan["hit_ids"][topk_idx],
        "topk_results": [scan["results"][i] for i in topk_idx],
    }


def bin_dirs_into_16(dirs, eps=1e-12):
    """Bin direction vectors into 16 groups (used for collision plane extraction)."""
    V = np.asarray(dirs, dtype=float)

    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    thr = np.array(0.5)  # cos(60°)

    elev_bin = np.floor(x / thr).astype(int)
    elev_bin = np.clip(elev_bin, 0, 1)

    theta = np.arctan2(z, y)
    theta = np.mod(theta, 2 * np.pi)
    theta_bin = np.floor(theta / (np.pi / 4)).astype(int)
    theta_bin = np.clip(theta_bin, 0, 7)

    bucket_id = elev_bin * 8 + theta_bin
    groups = [np.where(bucket_id == k)[0] for k in range(16)]

    def deg(a):
        return int(round(a))

    labels = []
    for eb in range(2):
        elev_lo = eb * 45
        elev_hi = (eb + 1) * 45
        for tb in range(8):
            th_lo = tb * 45
            th_hi = (tb + 1) * 45
            labels.append(
                f"|x|∈[{deg(elev_lo)},{deg(elev_hi)})°, yz∈[{deg(th_lo)},{deg(th_hi)})°"
            )

    return groups, labels


def plot_dirs_by_bucket(
    ax, p0, dirs, groups, labels, radius=30.0, scan=None, lw=1.2, alpha=0.9, cmap_name="tab20"
):
    """Visualize direction bins as 3D rays."""
    p0 = np.asarray(p0, float)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 16, endpoint=False))

    for k in range(16):
        idxs = groups[k]
        if len(idxs) == 0:
            continue
        color = colors[k]

        for i in idxs:
            if scan is not None and scan.get("hit_mask", None) is not None:
                if scan["hit_mask"][i]:
                    p1 = scan["hit_points"][i]
                else:
                    p1 = p0 + radius * dirs[i]
            else:
                p1 = p0 + radius * dirs[i]

            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=color,
                lw=lw,
                alpha=alpha,
            )

    ax.scatter(p0[0], p0[1], p0[2], s=40, c="k", marker="o", label="origin")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, ncol=2)


def plane_from_ray_dir(radius, scan, chose=None):
    """Convert scan rays to half-space planes of the form a*x + b*y + c*z + d = 0."""
    hit_idx = np.where(scan["hit_dists"] < radius)[0]
    P = scan["hit_points"][hit_idx]
    N = scan["dirs_world"][hit_idx]
    d = -np.sum(N * P, axis=1, keepdims=True)
    planes = np.hstack([N, d])
    return planes


def get_collision_plane(groups, scan, radius, l, p0=None):
    """
    Extract per-bucket half-space planes from laser scan.

    Returns:
        planes: list[np.ndarray(4,)]
        debug: dict with min_dist/min_idx/end_point per bucket
    """
    N_BUCKETS = len(groups)
    hit_mask = np.asarray(scan["hit_mask"], bool)
    hit_dists = np.asarray(scan["hit_dists"], float)
    hit_points = np.asarray(scan["hit_points"], float)
    dirs_world = np.asarray(scan["dirs_world"], float)

    min_dist = np.full(N_BUCKETS, radius, dtype=float)
    min_idx = np.full(N_BUCKETS, -1, dtype=int)
    end_point = np.full((N_BUCKETS, 3), np.nan, dtype=float)
    planes = []

    for k in range(N_BUCKETS):
        idxs = groups[k]
        if len(idxs) == 0:
            continue

        d_bucket = hit_dists[idxs]
        j_rel = np.argmin(d_bucket)
        j = int(idxs[j_rel])

        min_dist[k] = hit_dists[j]
        min_idx[k] = j
        end_point[k] = hit_points[j]

        if hit_mask[j]:
            P = hit_points[j]
            N = dirs_world[j]
            d = -float(np.dot(N, P)) + l
            planes.append(np.array([N[0], N[1], N[2], d], dtype=float))

    return planes, {"min_dist": min_dist, "min_idx": min_idx, "end_point": end_point}


def is_inside_planes(x, planes):
    """Check whether a point x satisfies all half-space constraints a·x + d <= 0."""
    if not planes:
        return True
    x = np.asarray(x, float)
    A = np.asarray([p[:3] for p in planes], float)
    b = np.asarray([p[3] for p in planes], float)
    return np.all(A.dot(x) + b <= 1e-9)


def project_goal_to_planes(goal, planes, bounds=None, z_bounds=None, iters=5):
    """
    Sequential projection (POCS) of a goal onto the intersection of half-spaces.
    """
    x = np.asarray(goal, float).copy()

    if not planes:
        if bounds is not None:
            if len(bounds) == 2:
                x[0] = np.clip(x[0], bounds[0], bounds[1])
                x[1] = np.clip(x[1], bounds[0], bounds[1])
            else:
                x[0] = np.clip(x[0], bounds[0], bounds[1])
                x[1] = np.clip(x[1], bounds[2], bounds[3])
        if z_bounds is not None:
            x[2] = np.clip(x[2], z_bounds[0], z_bounds[1])
        return x

    if bounds is not None:
        if len(bounds) == 2:
            x[0] = np.clip(x[0], bounds[0], bounds[1])
            x[1] = np.clip(x[1], bounds[0], bounds[1])
        else:
            x[0] = np.clip(x[0], bounds[0], bounds[1])
            x[1] = np.clip(x[1], bounds[2], bounds[3])
    if z_bounds is not None:
        x[2] = np.clip(x[2], z_bounds[0], z_bounds[1])

    if is_inside_planes(x, planes):
        return x

    for _ in range(iters):
        for p in planes:
            n = np.asarray(p[:3], float)
            d = float(p[3])
            viol = float(n.dot(x) + d)
            if viol > 0.0:
                nn = np.linalg.norm(n)
                if nn > 1e-9:
                    x = x - (viol / nn**2) * n

        # optional clipping
        if bounds is not None:
            if len(bounds) == 2:
                x[0] = np.clip(x[0], bounds[0], bounds[1])
                x[1] = np.clip(x[1], bounds[0], bounds[1])
            else:
                x[0] = np.clip(x[0], bounds[0], bounds[1])
                x[1] = np.clip(x[1], bounds[2], bounds[3])
        if z_bounds is not None:
            x[2] = np.clip(x[2], z_bounds[0], z_bounds[1])

        if is_inside_planes(x, planes):
            break

    return x


def clip_actions_to_bounds(actions, bounds=None, z_bounds=None):
    """
    Clip actions only by bounds (no planes projection).
    """
    a = np.asarray(actions, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError(f"actions last dim must be 3, got shape={a.shape}")

    out = a.copy()

    if bounds is not None:
        if len(bounds) == 2:
            xmin, xmax = bounds
            out[..., 0] = np.clip(out[..., 0], xmin, xmax)
            out[..., 1] = np.clip(out[..., 1], xmin, xmax)
        elif len(bounds) == 4:
            xmin, xmax, ymin, ymax = bounds
            out[..., 0] = np.clip(out[..., 0], xmin, xmax)
            out[..., 1] = np.clip(out[..., 1], ymin, ymax)
        else:
            raise ValueError(f"bounds must be len 2 or 4, got len={len(bounds)}")

    if z_bounds is not None:
        zmin, zmax = z_bounds
        out[..., 2] = np.clip(out[..., 2], zmin, zmax)

    return out


def project_goal_to_planes_qp(
    goal,
    planes,
    bounds=None,
    z_bounds=None,
    mu=1e3,
    eps=1e-6,
    solver_prefs=("qpoases", "osqp", "qrqp"),
):
    """
    Solve a QP projection problem onto half-space constraints (via casadi).
    Falls back to geometric projection if casadi is unavailable.
    """
    if ca is None:
        return project_goal_to_planes(goal, planes, bounds, z_bounds, iters=10)

    x0 = np.asarray(goal, dtype=float).reshape(3)
    m = len(planes)

    H = ca.DM.zeros(3 + m, 3 + m)
    H[0:3, 0:3] = 2.0 * ca.DM.eye(3)
    if m > 0:
        H[3:, 3:] = eps * ca.DM.eye(m)

    g = ca.DM.zeros(3 + m, 1)
    g[0:3] = -2.0 * ca.DM(x0)
    if m > 0:
        g[3:] = mu * ca.DM.ones(m, 1)

    if m > 0:
        A = ca.DM([p[:3] for p in planes])
        dvec = ca.DM([p[3] for p in planes]).reshape((m, 1))
        Aqp = ca.hcat([A, -ca.DM.eye(m)])
        lba = -ca.inf * ca.DM.ones(m, 1)
        uba = -dvec
    else:
        Aqp, lba, uba = None, None, None

    lbx = -ca.inf * ca.DM.ones(3 + m, 1)
    ubx = ca.inf * ca.DM.ones(3 + m, 1)

    if bounds is not None:
        if len(bounds) == 2:
            lbx[0] = bounds[0]
            ubx[0] = bounds[1]
            lbx[1] = bounds[0]
            ubx[1] = bounds[1]
        else:
            lbx[0] = bounds[0]
            ubx[0] = bounds[1]
            lbx[1] = bounds[2]
            ubx[1] = bounds[3]

    if z_bounds is not None:
        lbx[2] = z_bounds[0]
        ubx[2] = z_bounds[1]

    if m > 0:
        lbx[3:] = 0.0

    for name in solver_prefs:
        try:
            qp_dict = {"h": H, "g": g}
            if Aqp is not None:
                qp_dict["a"] = Aqp

            S = ca.qpsol("proj", name, qp_dict)
            x0_full = ca.vcat([ca.DM(x0), ca.DM.zeros(m, 1)])
            if Aqp is not None:
                sol = S(lbx=lbx, ubx=ubx, lba=lba, uba=uba, x0=x0_full)
            else:
                sol = S(lbx=lbx, ubx=ubx, x0=x0_full)

            x = np.array(sol["x"]).reshape(-1)
            return x[:3]
        except Exception:
            continue

    # If all QP solvers fail, fall back to geometric POCS.
    return project_goal_to_planes(goal, planes, bounds, z_bounds, iters=10)


def sample_start_goal_3d(L=100.0, H=50.0, margin=20.0, different_edge=True, opposite_only=True):
    """Sample a start point and a goal on the 3D box edges (used by UAV environments)."""
    start, e1 = sample_point_on_edge_3d(L, H, margin)

    if opposite_only:
        mapping = {0: 1, 1: 0, 2: 3, 3: 2}
        e2 = mapping[e1]

        t = np.random.uniform(margin, L - margin)
        z = np.random.uniform(0.0, H)
        if e2 == 0:
            goal = np.array([0.0, t, z])
        elif e2 == 1:
            goal = np.array([L, t, z])
        elif e2 == 2:
            goal = np.array([t, 0.0, z])
        else:
            goal = np.array([t, L, z])
    else:
        goal, e2 = sample_point_on_edge_3d(L, H, margin)
        if different_edge:
            while e2 == e1:
                goal, e2 = sample_point_on_edge_3d(L, H, margin)

    return start, goal


def sample_point_on_edge_3d(L=100.0, H=50.0, margin=0.0):
    """
    Sample a point on one of the 4 edges of a 3D map.

    edge_id:
        0=left (x=0), 1=right (x=L), 2=bottom (y=0), 3=top (y=L)
    """
    edge = np.random.randint(4)
    t = np.random.uniform(margin, L - margin)
    z = np.random.uniform(0.0, H)

    if edge == 0:
        p = np.array([0.0, t, z])
    elif edge == 1:
        p = np.array([L, t, z])
    elif edge == 2:
        p = np.array([t, 0.0, z])
    else:
        p = np.array([t, L, z])
    return p, edge


def reward_tetra_shape_only(p, s=0.05, eps=1e-6):
    """Tetrahedral shape reward component based only on edge lengths."""
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.linalg.norm(p[i] - p[j]))
    L = np.array(edges, dtype=float)
    Lm = np.mean(L) + eps
    E_edge = np.mean((L / Lm - 1.0) ** 2)
    return -np.tanh(E_edge / s)


def inter_uav_safety_penalty(p, d_long=16.0, s=2.0, power=1, eps=1e-6):
    """Global pairwise safety penalty (negative reward)."""
    p = np.asarray(p, dtype=float).reshape(-1, 3)
    N = p.shape[0]
    pen_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(p[i] - p[j]) + eps
            err = max(d_long - dij, 0.0)
            if power == 2:
                err = err * err
            pen_pairs.append(np.tanh(err / s))
    return -float(np.sum(pen_pairs) / 4)


def inter_uav_safety_penalty_per_agent(
    p, d_safe=5.0, s=4.0, power=1, eps=1e-6, reduce="mean"
):
    """Per-agent safety penalty: penalize distances below d_safe (negative reward)."""
    p = np.asarray(p, dtype=float).reshape(-1, 3)
    N = p.shape[0]
    pen = np.zeros(N, dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(p[i] - p[j]) + eps
            err = max(d_safe - dij, 0.0)
            if err <= 0.0:
                continue
            if power == 2:
                err = err * err

            w = np.tanh(err / s)
            pen[i] += w
            pen[j] += w

    if reduce == "mean" and N > 1:
        pen = pen / (N - 1)
    elif reduce in ("sum", "none"):
        pass
    else:
        if N > 1:
            pen = pen / (N - 1)

    return -pen


def one_hot_stage(tag):  # tag in {1,2,3,4}
    """Encode discrete stage id (1..4) into one-hot (length=4)."""
    v = np.zeros(4, dtype=np.float32)
    v[tag - 1] = 1.0
    return v

