"""
Plot what the RL pilot is doing: fly a few flights with a saved checkpoint and draw them.

For each flight:
    top view and side view of the path with the obstacles it actually had (dark) and the ones
    thinned out for this curriculum stage (light), start, goal, and where / how the flight ended;
    the points its far scans hit (red dots);
    the waypoints it was flying to (numbered purple diamonds: the strategist's picks for a stage-2
    run, the random practice waypoints for a stage-1 flier);
    time histories of speed, climb angle, bank angle, thrust, and the lidar distance straight
    ahead and the closest of all rays.
The agent flies its mean action (no exploration noise), on this computer (CPU is fine).

First copy the run's files from Modal (checkpoint.pt / best.pt land in supervised-learning-main/rl_pilot/<name>/):
    modal run supervised-learning-main/rl_pilot.py --name pilot_v2 --download-only
Then (from the fixed-wing-rl folder):
    python supervised-learning-main/plot_pilot.py --name pilot_v2                    6 flights, current curriculum stage
    python supervised-learning-main/plot_pilot.py --name pilot_v2 --n 10 --full      full environments (validation split)
    python supervised-learning-main/plot_pilot.py --name pilot_v2 --checkpoint best --show
    python supervised-learning-main/plot_pilot.py --name flier_v1 --to-goal          a flier along the line to the goal
A flier round trained with a strategist (--strategy) is flown on that strategist's waypoints (download it too).
A stage-2 (strategy) run also needs its flier's files downloaded (rl_pilot/<flier>/checkpoint.pt, as
in training, else best.pt); the flier run is taken from the strategy run's settings.
Plots are saved to supervised-learning-main/rl_pilot/<name>/plots/.
Needs a checkpoint from this version of rl_pilot.py (OBS_VERSION 11: waypoint flier / strategist, no map).
Older runs: plot them with the plot_pilot.py / rl_pilot.py of their version.
"""
import argparse
import json
import math
import time
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import os
os.environ["FW_NO_MODAL_DEFS"] = "1"
import torch                                                  # noqa: E402
import build_cache as bc                                      # noqa: E402
import rl_pilot as rp                                         # noqa: E402

STATUS_COLOR = {"goal": "#1baf7a", "crash": "#e34948", "out_of_bounds": "#eb6834", "stall": "#6250d6",
                "timeout": "#eda100", "missed": "#e87ba4", "flying": "#6b6a66"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e2dd"


def use_legacy_rays():
    """The lidar before the forward-only change: +-105 deg fan, tilted fans, up/down, sphere tracing."""
    def ray_dirs(chi, device):
        az = [math.radians(a) for a in torch.linspace(-105, 105, 15).tolist()]
        azt = [math.radians(a) for a in torch.linspace(-45, 45, 7).tolist()]
        el = math.radians(8.0)
        dirs = [(a, 0.0) for a in az] + [(a, el) for a in azt] + [(a, -el) for a in azt]
        A = torch.tensor([d[0] for d in dirs], device=device)
        E = torch.tensor([d[1] for d in dirs], device=device)
        ang = chi[:, None] + A[None]
        v = torch.stack([torch.cos(E) * torch.cos(ang), torch.cos(E) * torch.sin(ang), torch.sin(E).expand_as(ang)], -1)
        up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(len(chi), 1, 3)
        return torch.cat([v, up, -up], 1)

    def cast_rays(pos, chi, env_ids, W, drop=None):
        dirs = ray_dirs(chi, pos.device)
        t = torch.zeros(dirs.shape[:2], device=pos.device)
        for _ in range(10):
            h = rp.sdf(pos[:, None] + t[..., None] * dirs, env_ids, W, drop=drop)
            t = torch.clamp(t + torch.clamp(h, min=0.0), max=rp.RAY_RANGE_M)
        return t / rp.RAY_RANGE_M

    rp.ray_dirs, rp.cast_rays = ray_dirs, cast_rays
    rp.N_RAYS = 31
    rp.N_RAYS_FAN = 15


def drop_inputs(lo, hi, n_keep):
    """Older checkpoints: remove inputs lo:hi of the current observation (added later)."""
    orig = rp.Sim.obs

    def obs(self):
        o, m = orig(self)
        return torch.cat([o[:, :lo], o[:, hi:]], -1), m
    rp.Sim.obs = obs
    rp.OBS_DIM = n_keep


FLY_KW = {}                                                          # strategy= / wp_mode=, set in main()


def fly_recorded(agent, W, maps, af, env_rows, keep_frac, seed):
    """rl_pilot.fly with recording: the path, the lidar along it, the map it built, what its far
    scans hit, the waypoints, and the thinned-out obstacles."""
    return rp.fly(agent, None, W, maps, None, af, env_rows, "cpu", keep_frac=keep_frac, seed=seed, record=True,
                  **FLY_KW)


def occupancy(W, e, pts, drop=None):
    """Inside (True) an inflated obstacle or outside the box, for points (M, 3) in the flight frame."""
    P = torch.as_tensor(pts, dtype=torch.float32)[None]
    dr = None if drop is None else drop[None]
    return (rp.sdf(P, torch.tensor([e]), W, chunk=4096, drop=dr)[0] < 0).numpy()


def plot_flight(fl, W, e, env_index, af, out, show, stage_txt):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, U, t = fl["X"], fl["U"], fl["t"]
    goal = W["goal"][e].numpy()
    xs = np.arange(-500, max(goal[0], X[:, 0].max()) + 700, 25.0)
    ys = np.arange(min(-1500, X[:, 1].min() - 300), max(1500, X[:, 1].max() + 300), 25.0)
    zs = np.arange(-W["origin"][e, 2].item() - 10, (W["hi"][2] - W["origin"][e, 2]).item() + 10, 8.0)
    order = np.argsort(X[:, 0])
    z_at = lambda x: np.interp(x, X[order, 0], X[order, 2])
    y_at = lambda x: np.interp(x, X[order, 0], X[order, 1])
    XX, YY = np.meshgrid(xs, ys)
    top = np.stack([XX.ravel(), YY.ravel(), z_at(XX.ravel())], 1)
    XZ, ZZ = np.meshgrid(xs, zs)
    side = np.stack([XZ.ravel(), y_at(XZ.ravel()), ZZ.ravel()], 1)
    has_drop = bool(fl["drop"].any())
    info_txt = f"   waypoints reached {fl.get('wp_reached', 0)}, passed {fl.get('wp_missed', 0)}"
    slices = []
    for pts, shape in ((top, XX.shape), (side, XZ.shape)):
        all_ = occupancy(W, e, pts).reshape(shape)
        kept = occupancy(W, e, pts, fl["drop"]).reshape(shape) if has_drop else all_
        slices.append((all_, kept))

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 1.1, 1, 1], hspace=0.55, wspace=0.18)
    col = STATUS_COLOR.get(fl["status"], INK)
    km = 1e-3
    for v, (ax, (all_, kept)) in enumerate(zip([fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])], slices)):
        img = np.ones(all_.shape + (4,))
        img[all_] = matplotlib.colors.to_rgba("#e3e1dc")             # thinned out for this stage
        img[kept] = matplotlib.colors.to_rgba("#a9a7a1")             # obstacles the flight had
        ext = [xs[0] * km, xs[-1] * km, (ys[0] if v == 0 else zs[0]) * km, (ys[-1] if v == 0 else zs[-1]) * km]
        ax.imshow(img, origin="lower", extent=ext, aspect="auto" if v else "equal", interpolation="nearest")
        j = 1 if v == 0 else 2
        for bi, h in enumerate(fl.get("far_hits", [])):              # what the far scans hit
            if len(h):
                ax.plot(h[:, 0] * km, h[:, j] * km, ".", ms=1.5, color="#e34948", alpha=0.5,
                        label="far-scan hits" if bi == 0 else None)
        ax.plot(X[:, 0] * km, X[:, j] * km, color=col, lw=2.2, label="flown")
        wps = [w_ for w_ in fl.get("waypoints", []) if not w_[2]]          # (time, point, final)
        for wi, (_, p_, _) in enumerate(wps):
            ax.plot(p_[0] * km, p_[j] * km, "D", ms=6, color="#7a5af5", mec="white", mew=1,
                    label="waypoints" if wi == 0 else None)
            if v == 0:
                ax.annotate(str(wi + 1), (p_[0] * km, p_[j] * km), xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color="#7a5af5")
        if v == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left", markerscale=1.5)
        ax.plot(0, 0, "o", ms=8, color=INK, mec="white", mew=2)
        ax.plot(goal[0] * km, goal[j] * km, "s", ms=9, color=INK, mec="white", mew=2)
        ax.plot(X[-1, 0] * km, X[-1, j] * km, "X" if fl["status"] != "goal" else "o", ms=12, color=col, mec="white", mew=1.5)
        if v == 0:                                                   # heading ticks every 20 s
            for k in range(0, len(t), 40):
                ax.annotate("", xy=((X[k, 0] + 250 * math.cos(X[k, 5])) * km, (X[k, 1] + 250 * math.sin(X[k, 5])) * km),
                            xytext=(X[k, 0] * km, X[k, 1] * km), arrowprops=dict(arrowstyle="->", color=col, lw=1))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title("Top view (obstacles at the flight's altitude; red dots = far-scan hits; purple = waypoints)" if v == 0 else
                     "Side view (obstacles at the flight's lateral position)", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("along start-goal line (km)", fontsize=8, color=MUTED)
        ax.set_ylabel("cross track (km)" if v == 0 else "height rel. start (km)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
    # time histories
    V, gam, nv, nh, T_ = X[:, 3], np.degrees(X[:, 4]), U[:, 1], U[:, 2], U[:, 0]
    bank = np.degrees(np.arctan2(nh, np.maximum(nv, 1e-6)))
    T_av = af["THRUST_PLAN_FRACTION"] * (af["T_MAX"] + af["THRUST_SLOPE"] * V)
    rays = fl["rays"]
    mid = rp.N_RAYS_FAN // 2 if hasattr(rp, "N_RAYS_FAN") else 7
    panels = [
        (gs[2, 0], [(V, "speed (m/s)", INK)], [(af["V_MIN"], "V min"), (af["V_MAX"], "V max")]),
        (gs[2, 1], [(gam, "climb angle (deg)", "#2a78d6")], [(af["GAMMA_MAX_DEG"], ""), (-af["GAMMA_MAX_DEG"], "")]),
        (gs[3, 0], [(bank, "bank (deg)", "#eb6834"), (100 * T_ / T_av, "thrust (% available)", "#1baf7a")],
         [(af["BANK_MAX_DEG"], ""), (-af["BANK_MAX_DEG"], "")]),
        (gs[3, 1], [(rays[:, mid], "lidar straight ahead (m)", INK), (rays[:, :rp.N_RAYS_FAN].min(1), "closest horizontal ray (m)", "#e34948")], []),
    ]
    for spec, series, hlines in panels:
        ax = fig.add_subplot(spec)
        for y, lab, c in series:
            ax.plot(t, y, color=c, lw=1.4, label=lab)
        for y, lab in hlines:
            ax.axhline(y, color=MUTED, lw=0.8, ls="--")
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(color=GRID, lw=0.6)
    fig.suptitle(f"Environment {env_index}: {fl['status'].replace('_', ' ')} after {t[-1]:.0f} s{info_txt}   ({stage_txt})",
                 x=0.06, ha="left", fontsize=13, color=col if fl["status"] != "goal" else INK)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def lidar_points(fl, W, e, every=2):
    """Where the scans hit OBSTACLES (box walls and ground left out): short scan every `every` steps,
    and the far scans."""
    X, rays = fl["X"], fl["rays"]
    k = np.arange(0, len(X), every)
    dirs = rp.ray_dirs(torch.as_tensor(X[k, 5], dtype=torch.float32), "cpu").numpy()        # (K, R, 3)
    d = rays[k]
    hit = d < rp.RAY_RANGE_M - 1.0
    short = (X[k, None, :3] + dirs * d[..., None])[hit]
    far = [h for h in fl.get("far_hits", []) if len(h)]
    far = np.concatenate(far) if far else np.zeros((0, 3))

    def on_obstacle(P):
        if not len(P):
            return P
        c = rp.sdf(torch.as_tensor(P, dtype=torch.float32)[None], torch.tensor([e]), W, chunk=4096,
                   drop=torch.as_tensor(fl["drop"])[None], walls=False)[0].numpy()
        return P[c < 15.0]
    return on_obstacle(short), on_obstacle(far)


def course_limits(fl, W, e):
    """Common 3-D limits (flight frame, m): the course box around the path and the goal."""
    X, goal = fl["X"], W["goal"][e].numpy()
    oz = W["origin"][e, 2].item()
    lo = np.array([-500.0, min(-1500.0, X[:, 1].min() - 300), W["lo"][2].item() - oz])
    hi = np.array([max(goal[0], X[:, 0].max()) + 700, max(1500.0, X[:, 1].max() + 300), W["hi"][2].item() - oz])
    return lo, hi


def in_box(W, e, P):
    """Points (M, 3, flight frame) inside the environment's box."""
    yaw = W["yaw"][e].item()
    c, s_ = math.cos(yaw), math.sin(yaw)
    pw = np.stack([P[:, 0] * c - P[:, 1] * s_, P[:, 0] * s_ + P[:, 1] * c, P[:, 2]], 1) + W["origin"][e].numpy()
    return ((pw >= W["lo"].numpy()) & (pw <= W["hi"].numpy())).all(1)


def obstacle_points(fl, W, e, lo, hi, step=(40.0, 40.0, 25.0)):
    """Points on the outer shell of the obstacles this flight had (thinned ones left out)."""
    axes = [np.arange(lo[k], hi[k] + 1e-6, step[k]) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    G = G[in_box(W, e, G)]
    c = rp.sdf(torch.as_tensor(G, dtype=torch.float32)[None], torch.tensor([e]), W, chunk=8192,
               drop=torch.as_tensor(fl["drop"])[None], walls=False)[0].numpy()
    return G[(c < 0) & (c > -1.5 * max(step))]                     # a shell: enough to see the shapes


def obstacle_mesh(fl, W, e, lo, hi, step=(30.0, 30.0, 15.0)):
    """Surfaces of the obstacles this flight had (marching cubes on the clearance field), as
    (vertices (V, 3) m, faces (F, 3)); None without scikit-image."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None
    axes = [np.arange(lo[k] - step[k], hi[k] + 2 * step[k], step[k]) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    c = rp.sdf(torch.as_tensor(G, dtype=torch.float32)[None], torch.tensor([e]), W, chunk=8192,
               drop=torch.as_tensor(fl["drop"])[None], walls=False)[0].numpy()
    c = np.where(in_box(W, e, G), c, np.maximum(c, step[0]))       # nothing outside the box
    vol = c.reshape([len(a_) for a_ in axes])
    if vol.min() >= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=step)
    return v + np.array([a_[0] for a_ in axes]), f


def _axes3d(plt, lo, hi, z_scale, figsize=(14, 7.5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    km = 1e-3
    ax.set_xlim(lo[0] * km, hi[0] * km); ax.set_ylim(lo[1] * km, hi[1] * km); ax.set_zlim(lo[2] * km, hi[2] * km)
    span = np.maximum(hi - lo, 1.0)
    ax.set_box_aspect((span[0], span[1], span[2] * z_scale))
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel("along start-goal line (km)", fontsize=8, color=MUTED)
    ax.set_ylabel("cross track (km)", fontsize=8, color=MUTED)
    ax.set_zlabel(f"height (km, x{z_scale:g})", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    return fig, ax


def _ends(ax, fl, goal, col):
    km = 1e-3
    X = fl["X"]
    ax.scatter([0], [0], [0], s=60, c=INK, depthshade=False)
    ax.scatter([goal[0] * km], [goal[1] * km], [goal[2] * km], marker="s", s=70, c=INK, depthshade=False)
    ax.scatter([X[-1, 0] * km], [X[-1, 1] * km], [X[-1, 2] * km], marker="X" if fl["status"] != "goal" else "o",
               s=110, c=col, edgecolors="white", depthshade=False)


def _title(fl, env_index, stage_txt, what):
    return (f"Environment {env_index}: {fl['status'].replace('_', ' ')} after {fl['t'][-1]:.0f} s   "
            f"waypoints reached {fl.get('wp_reached', 0)}, passed {fl.get('wp_missed', 0)}   ({stage_txt})\n{what}")


def plot_obstacles_3d(fl, W, e, env_index, out, show, stage_txt, z_scale=4.0):
    """3-D view of the obstacles this flight had (nothing else but start and goal)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = course_limits(fl, W, e)
    goal = W["goal"][e].numpy()
    km = 1e-3
    fig, ax = _axes3d(plt, lo, hi, z_scale)
    mesh = obstacle_mesh(fl, W, e, lo, hi)
    if mesh is None:                                               # (no scikit-image: points instead)
        print("  (pip install scikit-image for solid obstacle surfaces)")
        O = obstacle_points(fl, W, e, lo, hi)
        if len(O):
            ax.scatter(O[:, 0] * km, O[:, 1] * km, O[:, 2] * km, s=2, c="#a9a7a1", alpha=0.5, depthshade=True)
    elif len(mesh[1]):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        v, f = mesh
        tri = v[f] * km                                            # (F, 3, 3)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0] * 1.0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        light = np.array([-0.4, -0.5, 0.77]); light /= np.linalg.norm(light)
        shade = 0.55 + 0.45 * np.abs(n @ light)                     # simple matte shading
        base = np.array(matplotlib.colors.to_rgb("#b9b6ae"))
        cols = np.clip(base[None] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors="none", linewidths=0))
    ax.scatter([0], [0], [0], s=60, c=INK, depthshade=False)
    ax.scatter([goal[0] * km], [goal[1] * km], [goal[2] * km], marker="s", s=70, c=INK, depthshade=False)
    ax.set_title(_title(fl, env_index, stage_txt, "obstacles this flight had (inflated by the clearance; thinned-out ones left out)"),
                 loc="left", fontsize=11, color=INK)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def plot_histories(fl, af, env_index, out, show, stage_txt):
    """Speed, climb angle, bank and thrust, and the lidar ranges over the flight."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, U, t = fl["X"], fl["U"], fl["t"]
    V, gam, nv, nh, T_ = X[:, 3], np.degrees(X[:, 4]), U[:, 1], U[:, 2], U[:, 0]
    bank = np.degrees(np.arctan2(nh, np.maximum(nv, 1e-6)))
    T_av = af["THRUST_PLAN_FRACTION"] * (af["T_MAX"] + af["THRUST_SLOPE"] * V)
    rays = fl["rays"]
    mid = rp.N_RAYS_FAN // 2
    panels = [
        ([(V, "speed (m/s)", INK)], [af["V_MIN"], af["V_MAX"]]),
        ([(gam, "climb angle (deg)", "#2a78d6")], [af["GAMMA_MAX_DEG"], -af["GAMMA_MAX_DEG"]]),
        ([(bank, "bank (deg)", "#eb6834"), (100 * T_ / T_av, "thrust (% available)", "#1baf7a")],
         [af["BANK_MAX_DEG"], -af["BANK_MAX_DEG"]]),
        ([(rays[:, mid], "lidar straight ahead (m)", INK), (rays[:, :rp.N_RAYS_FAN].min(1), "closest horizontal ray (m)", "#e34948")], []),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14, 6.5))
    for ax, (series, hl) in zip(axs.ravel(), panels):
        for y, lab, c in series:
            ax.plot(t, y, color=c, lw=1.4, label=lab)
        for y in hl:
            ax.axhline(y, color=MUTED, lw=0.8, ls="--")
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(color=GRID, lw=0.6)
    col = STATUS_COLOR.get(fl["status"], INK)
    fig.suptitle(_title(fl, env_index, stage_txt, "flight data"), x=0.06, ha="left", fontsize=11,
                 color=col if fl["status"] != "goal" else INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def plot_flight_3d(fl, W, e, env_index, out, show, stage_txt, z_scale=4.0):
    """3-D view with no obstacles drawn: only what the lidar saw (short scan orange, far scans red),
    the path, the waypoints, start and goal. Heights are exaggerated z_scale x."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, t = fl["X"], fl["t"]
    goal = W["goal"][e].numpy()
    short, far = lidar_points(fl, W, e)
    col = STATUS_COLOR.get(fl["status"], INK)
    km = 1e-3
    lo, hi = course_limits(fl, W, e)
    fig, ax = _axes3d(plt, lo, hi, z_scale)
    if len(far):
        ax.scatter(far[:, 0] * km, far[:, 1] * km, far[:, 2] * km, s=1.5, c="#e34948", alpha=0.35,
                   depthshade=False, label="far-scan hits")
    if len(short):
        ax.scatter(short[:, 0] * km, short[:, 1] * km, short[:, 2] * km, s=2.5, c="#eb8a34", alpha=0.55,
                   depthshade=False, label="short-scan hits")
    ax.plot(X[:, 0] * km, X[:, 1] * km, X[:, 2] * km, color=col, lw=2.4, label="flown")
    wps = [w_ for w_ in fl.get("waypoints", []) if not w_[2]]
    if wps:
        P = np.array([p_ for _, p_, _ in wps])
        ax.scatter(P[:, 0] * km, P[:, 1] * km, P[:, 2] * km, marker="D", s=40, c="#7a5af5",
                   edgecolors="white", depthshade=False, label="waypoints")
        for wi, p_ in enumerate(P):
            ax.text(p_[0] * km, p_[1] * km, p_[2] * km + 0.03, str(wi + 1), fontsize=7, color="#7a5af5")
    _ends(ax, fl, goal, col)
    ax.legend(fontsize=8, frameon=False, loc="upper left", markerscale=3)
    ax.set_title(_title(fl, env_index, stage_txt, "obstacles not drawn: only what the lidar saw, the path and the waypoints"),
                 loc="left", fontsize=11, color=col if fl["status"] != "goal" else INK)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def make_video(fl, W, e, env_index, out, stage_txt, speed=1.0, fps=10):
    """Top-down animation of one flight: the aircraft, its short scan every step, its far scans (flash,
    then their hits fade over the FAR_KEEP scans the strategist remembers), the waypoints, the goal.
    speed 1 = real time. MP4 with ffmpeg (or imageio-ffmpeg), else an animated GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    X, t = fl["X"], fl["t"]
    goal = W["goal"][e].numpy()
    km = 1e-3
    # background: the obstacles this flight had, at the flight's altitude along the course
    xs = np.arange(-500, max(goal[0], X[:, 0].max()) + 700, 25.0)
    ys = np.arange(min(-1500, X[:, 1].min() - 300), max(1500, X[:, 1].max() + 300), 25.0)
    order = np.argsort(X[:, 0])
    XX, YY = np.meshgrid(xs, ys)
    top = np.stack([XX.ravel(), YY.ravel(), np.interp(XX.ravel(), X[order, 0], X[order, 2])], 1)
    occ = occupancy(W, e, top, fl["drop"]).reshape(XX.shape)
    img = np.ones(occ.shape + (4,)); img[occ] = matplotlib.colors.to_rgba("#bdbab3")
    ext = [xs[0] * km, xs[-1] * km, ys[0] * km, ys[-1] * km]
    # far scans: when, from where, toward the goal; their hits
    far_t = np.array(fl.get("far_hits_t", []), float)
    far_h = fl.get("far_hits", [])
    scan_times = np.unique(far_t) if len(far_t) else np.zeros(0)
    keep_s = rp.FAR_KEEP * rp.FAR_EVERY_S
    wps = fl.get("waypoints", [])
    fan = rp.N_RAYS_FAN
    col = STATUS_COLOR.get(fl["status"], INK)

    fig, ax = plt.subplots(figsize=(13, 13 * (ext[3] - ext[2]) / (ext[1] - ext[0]) + 0.6))
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", zorder=0)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(matplotlib.patches.Rectangle((goal[0] * km - 0.06, goal[1] * km - 0.06), 0.12, 0.12,
                                              facecolor="white", edgecolor=INK, lw=2, zorder=6))
    trail, = ax.plot([], [], color=col, lw=1.6, alpha=0.8, zorder=4)
    craft, = ax.plot([], [], "o", ms=9, color=INK, zorder=7)
    short_lines = [ax.plot([], [], color="#d9822b", lw=1.1, alpha=0.85, zorder=5)[0] for _ in range(fan)]
    short_hits, = ax.plot([], [], "o", ms=4, color="#d9822b", zorder=6)
    far_cone, = ax.plot([], [], color="#c0392b", lw=1.0, alpha=0.0, zorder=3)
    far_dots = ax.scatter([], [], s=7, c=[], cmap="Reds", vmin=0, vmax=1, zorder=3)
    wp_mark, = ax.plot([], [], "D", ms=9, color="#7a5af5", mec="white", mew=1.2, zorder=6)
    clock = ax.text(0.01, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=11, color=MUTED)

    T_end = t[-1]
    n_frames = int(T_end / speed * fps) + 1

    def at(tt):
        k = int(min(np.searchsorted(t, tt, side="right") - 1, len(t) - 1))
        k = max(k, 0)
        if k + 1 < len(t):
            w = (tt - t[k]) / max(t[k + 1] - t[k], 1e-9)
            p = X[k, :3] * (1 - w) + X[k + 1, :3] * w
        else:
            p = X[k, :3]
        return k, p

    def frame(i):
        tt = min(i * speed / fps, T_end)
        k, p = at(tt)
        trail.set_data(np.r_[X[:k + 1, 0], p[0]] * km, np.r_[X[:k + 1, 1], p[1]] * km)
        craft.set_data([p[0] * km], [p[1] * km])
        dirs = rp.ray_dirs(torch.tensor([X[k, 5]], dtype=torch.float32), "cpu")[0].numpy()[:fan]
        d = fl["rays"][k, :fan]
        ends = p[None, :2] + dirs[:, :2] * d[:, None]
        for j, ln in enumerate(short_lines):
            ln.set_data([p[0] * km, ends[j, 0] * km], [p[1] * km, ends[j, 1] * km])
        hit = d < rp.RAY_RANGE_M - 1
        short_hits.set_data(ends[hit, 0] * km, ends[hit, 1] * km)
        # far scans: the newest flashes for 1 s; hits stay while remembered, fading with age
        pts, alpha = [], []
        for ts in scan_times[scan_times <= tt]:
            age = tt - ts
            if age > keep_s:
                continue
            for h, th in zip(far_h, far_t):
                if th == ts and len(h):
                    pts.append(h[:, :2]); alpha.append(np.full(len(h), 1.0 - 0.75 * age / keep_s))
        if pts:
            P = np.concatenate(pts); A = np.concatenate(alpha)
            far_dots.set_offsets(P * km); far_dots.set_array(A); far_dots.set_alpha(None)
            far_dots.set_sizes(np.full(len(P), 7.0)); far_dots.set_facecolor(
                [matplotlib.colors.to_rgba("#c0392b", a_) for a_ in A])
        else:
            far_dots.set_offsets(np.zeros((0, 2)))
        recent = scan_times[(scan_times <= tt) & (scan_times > tt - 1.0)]
        if len(recent):
            k0, p0 = at(recent[-1])
            az = math.atan2(goal[1] - p0[1], goal[0] - p0[0])
            a_ = np.radians(np.linspace(-rp.FAR_AZ[1], rp.FAR_AZ[1], 25)) + az
            R = rp.FAR_RANGE_M
            cx = np.r_[p0[0], p0[0] + R * np.cos(a_), p0[0]]; cy = np.r_[p0[1], p0[1] + R * np.sin(a_), p0[1]]
            far_cone.set_data(cx * km, cy * km); far_cone.set_alpha(0.7 * (1 - (tt - recent[-1])))
        else:
            far_cone.set_alpha(0.0)
        cur = [w_ for w_ in wps if w_[0] <= tt and not w_[2]]
        if cur and not (len(wps) and [w_ for w_ in wps if w_[0] <= tt][-1][2]):
            wp_mark.set_data([cur[-1][1][0] * km], [cur[-1][1][1] * km])
        else:
            wp_mark.set_data([], [])
        clock.set_text(f"{tt:5.1f} s")
        return []

    fig.tight_layout(pad=0.3)
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    ff = None
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    if animation.writers.is_available("ffmpeg"):
        out = out.with_suffix(".mp4")
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=100)
    else:
        out = out.with_suffix(".gif")
        print("  (no ffmpeg: saving a GIF; pip install imageio-ffmpeg for MP4)")
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    return out


class _Timed:
    """Wraps a network: times and counts its decisions (mean() calls), passes everything else on."""
    def __init__(self, net):
        self.net, self.s, self.calls = net, 0.0, 0

    def mean(self, *a_):
        t0 = time.perf_counter()
        out = self.net.mean(*a_)
        self.s += time.perf_counter() - t0
        self.calls += 1
        return out

    def __getattr__(self, k):
        return getattr(self.net, k)


def time_flights(agent, W, af, rows, keep, seed, cache):
    """Fly each environment on its own (one flight, as on the aircraft) and report where the time goes."""
    clock = {"obs": 0.0, "strat_obs": 0.0, "step": 0.0}
    orig = {k: getattr(rp.Sim, k) for k in clock}

    def wrap(k):
        def f(self, *a_, **kw):
            t0 = time.perf_counter()
            out = orig[k](self, *a_, **kw)
            clock[k] += time.perf_counter() - t0
            return out
        return f
    for k in clock:
        setattr(rp.Sim, k, wrap(k))
    fl_net = _Timed(agent)
    st_net = _Timed(FLY_KW["strategy"]) if FLY_KW.get("strategy") is not None else None
    kw = dict(FLY_KW, strategy=st_net) if st_net else dict(FLY_KW)
    rp.fly(fl_net, None, W, None, None, af, [int(rows[0])], "cpu", keep_frac=keep, seed=seed, **kw)   # warm-up
    for k in clock:
        clock[k] = 0.0
    fl_net.s = fl_net.calls = 0
    if st_net:
        st_net.s = st_net.calls = 0
    print(f"\ntiming {len(rows)} flights one at a time on this CPU ({torch.get_num_threads()} threads):")
    tot = {"wall": 0.0, "sim_s": 0.0}
    with torch.no_grad():
        for e in rows:
            c0 = dict(clock); f0, n0 = fl_net.s, fl_net.calls
            s0, m0 = (st_net.s, st_net.calls) if st_net else (0.0, 0)
            t0 = time.perf_counter()
            fl = rp.fly(fl_net, None, W, None, None, af, [int(e)], "cpu", keep_frac=keep, seed=seed, **kw)[0]
            wall = time.perf_counter() - t0
            tot["wall"] += wall; tot["sim_s"] += fl["t"][-1]
            st_txt = f", strategist {st_net.calls - m0:3d} decisions {1e3 * (st_net.s - s0):6.0f} ms" if st_net else ""
            print(f"  env {int(cache['env_index'][e]):5d}: {fl['status']:8s} {fl['t'][-1]:5.0f} s flown in {wall:6.2f} s  "
                  f"(flier {fl_net.calls - n0:4d} decisions {1e3 * (fl_net.s - f0):6.0f} ms{st_txt}, "
                  f"sensing {1e3 * (clock['obs'] - c0['obs'] + clock['strat_obs'] - c0['strat_obs']):6.0f} ms, "
                  f"dynamics {1e3 * (clock['step'] - c0['step']):6.0f} ms)")
    for k in clock:
        setattr(rp.Sim, k, orig[k])
    n = max(fl_net.calls, 1)
    print(f"\nper flight on average: {tot['wall'] / len(rows):.2f} s of compute for {tot['sim_s'] / len(rows):.0f} s of flight "
          f"({tot['sim_s'] / max(tot['wall'], 1e-9):.0f}x faster than real time)")
    print(f"  flier network   {1e3 * fl_net.s / n:7.3f} ms per decision (every {rp.DT} s)")
    if st_net and st_net.calls:
        print(f"  strategist net  {1e3 * st_net.s / st_net.calls:7.3f} ms per decision (at least every {rp.STRAT_EVERY_S:.0f} s)")
    print(f"  sensing         {1e3 * (clock['obs'] + clock['strat_obs']) / n:7.3f} ms per step (short scan every step, far scan every "
          f"{rp.FAR_EVERY_S:.0f} s, strategist inputs)")
    print(f"  dynamics+reward {1e3 * clock['step'] / n:7.3f} ms per step (simulator only: not needed on the aircraft)")
    print("  (sensing here is ray casting against the known obstacles: on the aircraft the lidar does that part)")


def main():
    ap = argparse.ArgumentParser(description="Fly and plot the RL pilot from a saved checkpoint.")
    ap.add_argument("--name", required=True, help="run name (folder supervised-learning-main/rl_pilot/<name>)")
    ap.add_argument("--checkpoint", default="checkpoint", choices=["checkpoint", "best"])
    ap.add_argument("--dataset", default=rp.DATASET)
    ap.add_argument("--run", default=rp.RUN)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--full", action="store_true", help="full validation environments instead of the current stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--failures", action="store_true", help="only save plots of flights that did not reach the goal")
    ap.add_argument("--to-goal", action="store_true", help="stage-1 flier: waypoints along the straight line to the goal (as in its solver evaluations)")
    ap.add_argument("--flier-checkpoint", default="checkpoint", choices=["checkpoint", "best"],
                    help="stage-2 run: which checkpoint of its flier to use (falls back to the other)")
    ap.add_argument("--successes", action="store_true",
                    help="keep flying until --n flights have reached the goal, and plot only those")
    ap.add_argument("--3d", dest="three_d", action="store_true",
                    help="instead of the 2-D plot, three per flight: the obstacles in 3-D (*_obstacles.png), the lidar "
                         "hits, path and waypoints in 3-D, same view (*_lidar.png), and the flight data (*_data.png)")
    ap.add_argument("--z-scale", type=float, default=4.0, help="--3d: height exaggeration")
    ap.add_argument("--video", action="store_true",
                    help="no plots: a top-down video per flight (aircraft, short and far scans, waypoints), MP4 or GIF")
    ap.add_argument("--speed", type=float, default=1.0, help="--video: playback speed (1 = real time)")
    ap.add_argument("--fps", type=int, default=10, help="--video: frames per second")
    ap.add_argument("--keep", type=float, default=None,
                    help="fly at this obstacle share (0..1) instead of the checkpoint's curriculum position")
    ap.add_argument("--pool", type=float, default=None,
                    help="draw environments from this least-cluttered share of the training set (0..1) instead")
    ap.add_argument("--time", action="store_true",
                    help="no plots: fly --n flights one at a time on this CPU and report the compute time per flight "
                         "and per decision (networks, sensing, dynamics)")
    ap.add_argument("--threads", type=int, default=0, help="--time: CPU threads for torch (0 = torch's default)")
    a = ap.parse_args()

    run_dir = _HERE / "rl_pilot" / a.name
    ck_path = run_dir / f"{a.checkpoint}.pt"
    if not ck_path.exists():
        raise SystemExit(f"{ck_path} not found. Copy it from Modal first:\n  modal run "
                         f"supervised-learning-main/rl_pilot.py --name {a.name} --download-only")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    version = (ck.get("settings") or {}).get("OBS_VERSION")
    if version != rp.OBS_VERSION:
        raise SystemExit(f"checkpoint is from an older version of the pilot (OBS_VERSION {version}); plot it with "
                         f"the plot_pilot.py and rl_pilot.py of that version (this one is {rp.OBS_VERSION}).")
    st = ck.get("state", {})
    sett = ck.get("settings") or {}
    rp.ACTION_MODE = sett.get("ACTION_MODE", "rate")                 # earlier runs commanded rates
    rp.CLEAR_RADIUS_M = sett.get("CLEAR_RADIUS_M", 0.0)              # and kept every obstacle
    if "TAU_GAMMA_S" in sett:
        rp.TAU_GAMMA_S = sett["TAU_GAMMA_S"]

    env_root = _HERE / "datasets" / a.dataset
    cache = bc.load_cache(env_root / "cache" / f"{a.run}.npz")
    scale = cache["summary"]["length_scale_m"]
    env0 = json.loads((env_root / "envs" / "shard_00000.json").read_text())[0]
    af = rp.airframe(_HERE.parent)
    W = rp.build_world(cache, scale, env0["bounds_lo"], env0["bounds_hi"], env0["vehicle_radius"],
                       env0["vehicle_radius"] + env0["margin"], "cpu")
    stage = (ck.get("cfg") or {}).get("stage") or "flier"
    if stage == "strategy":                                              # the strategist + its frozen flier
        strat = rp.make_agent("cpu", rp.STRAT_DIM, (0.0, 0.0, 0.0), rp.FAR_SHAPE)
        strat.load_state_dict(ck["agent"])
        strat.eval()
        fname = ck["cfg"]["flier"]
        fdir = _HERE / "rl_pilot" / fname
        cands = [fdir / f"{a.flier_checkpoint}.pt", fdir / ("checkpoint.pt" if a.flier_checkpoint == "best" else "best.pt")]
        fpath = next((p for p in cands if p.exists()), None)
        if fpath is None:
            raise SystemExit(f"the flier run '{fname}' is not here ({fdir}). Download it too:\n  modal run "
                             f"supervised-learning-main/rl_pilot.py --name {fname} --download-only")
        agent = rp.make_agent("cpu")
        agent.load_state_dict(torch.load(fpath, map_location="cpu", weights_only=False)["agent"])
        agent.eval()
        FLY_KW.update(strategy=strat)
        print(f"stage-2 run: strategist {a.name} ({a.checkpoint}.pt) with flier {fname} ({fpath.name})")
    else:
        agent = rp.make_agent("cpu")
        agent.load_state_dict(ck["agent"])
        agent.eval()
        sname = (ck.get("cfg") or {}).get("strategy")
        if sname and not a.to_goal:                                      # a flier round trained with a strategist
            sdir = _HERE / "rl_pilot" / sname
            spath = next((p for p in (sdir / "checkpoint.pt", sdir / "best.pt") if p.exists()), None)   # as in training
            if spath is None:
                raise SystemExit(f"this flier trained with strategist '{sname}', which is not here ({sdir}). "
                                 f"Download it too, or add --to-goal:\n  modal run supervised-learning-main/rl_pilot.py "
                                 f"--name {sname} --download-only")
            strat = rp.make_agent("cpu", rp.STRAT_DIM, (0.0, 0.0, 0.0), rp.FAR_SHAPE)
            strat.load_state_dict(torch.load(spath, map_location="cpu", weights_only=False)["agent"])
            strat.eval()
            FLY_KW.update(strategy=strat)
            print(f"flier {a.name} on the waypoints of strategist {sname} ({spath.name}), as it trained")
        else:
            FLY_KW.update(wp_mode="line" if a.to_goal else "random")
            print(f"flier {a.name}: " + ("waypoints along the line to the goal" if a.to_goal else "on random practice waypoints"))

    rng = np.random.default_rng(a.seed)
    if a.full:
        val = np.nonzero(cache["split"] == "val")[0]
        pool = val
        rows = rng.choice(val, min(a.n, len(val)), replace=False)
        keep = 1.0
        stage_txt = "full validation environment"
    else:
        import csv
        with open(env_root / "index.csv", newline="") as f:
            occ_of = {int(r["index"]): float(r.get("occupancy") or 0) for r in csv.DictReader(f)}
        tr = np.nonzero(cache["split"] == "train")[0]
        tr = tr[np.argsort([occ_of.get(int(cache["env_index"][e]), 0) for e in tr], kind="stable")]
        pool_frac = a.pool if a.pool is not None else st.get("pool_frac", rp.CURRICULUM_START)
        pool = tr[:max(1, int(round(pool_frac * len(tr))))]
        keep = a.keep if a.keep is not None else st.get("keep", rp._OLD_KEEP_STAGES[st["keep_idx"]] if "keep_idx" in st else 0.0)
        rows = rng.choice(pool, min(a.n, len(pool)), replace=False)
        stage_txt = (f"{keep:.0%} of obstacles, least-cluttered {pool_frac:.0%} of environments"
                     + ("" if a.keep is None and a.pool is None else " (set with --keep / --pool)"))
    maps = None
    if a.time:
        if a.threads > 0:
            torch.set_num_threads(a.threads)
        time_flights(agent, W, af, rows, keep, a.seed, cache)
        return
    if a.successes:                                   # fly batches until enough flights reached the goal
        print(f"{a.checkpoint}.pt from update {st.get('update', '?')}; flying until {a.n} reach the goal ({stage_txt}) ...")
        got_rows, got, all_flights, tries = [], [], [], 0
        while len(got) < a.n and tries < 10:
            batch = rng.choice(pool, min(max(2 * a.n, 10), len(pool)), replace=False)
            fl_b = fly_recorded(agent, W, maps, af, batch, keep, a.seed + tries)
            all_flights += fl_b
            for e, fl in zip(batch, fl_b):
                if fl["status"] == "goal" and len(got) < a.n:
                    got_rows.append(e); got.append(fl)
            tries += 1
            print(f"  batch {tries}: {sum(f['status'] == 'goal' for f in fl_b)} of {len(fl_b)} reached the goal")
        rows, flights = np.asarray(got_rows, dtype=int), got
    else:
        print(f"{a.checkpoint}.pt from update {st.get('update', '?')}; flying {len(rows)} flights ({stage_txt}) ...")
        flights = fly_recorded(agent, W, maps, af, rows, keep, a.seed)
        all_flights = flights
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (e, fl) in enumerate(zip(rows, flights)):
        idx = int(cache["env_index"][e])
        if a.failures and fl["status"] == "goal":
            continue
        out = out_dir / f"{'full' if a.full else 'stage'}_{i:02d}_env{idx:05d}.png"
        if a.video:
            v_ = make_video(fl, W, int(e), idx, out.with_name(out.stem + "_video"), stage_txt, a.speed, a.fps)
            print(f"  video: {v_}")
            continue
        if a.three_d:
            plot_obstacles_3d(fl, W, int(e), idx, out.with_name(out.stem + "_obstacles.png"), a.show, stage_txt, a.z_scale)
            plot_flight_3d(fl, W, int(e), idx, out.with_name(out.stem + "_lidar.png"), a.show, stage_txt, a.z_scale)
            plot_histories(fl, af, idx, out.with_name(out.stem + "_data.png"), a.show, stage_txt)
        else:
            plot_flight(fl, W, int(e), idx, af, out, a.show, stage_txt)
        X = fl["X"]
        print(f"  env {idx:5d}: {fl['status']:13s} after {fl['t'][-1]:5.0f} s ({fl['wp_reached']} waypoints reached), {X[-1, 0] / 1000:5.2f} of "
              f"{W['goal'][e, 0].item() / 1000:5.2f} km along, speed {X[-1, 3]:4.1f} m/s  -> {out}")
    st_counts = {}
    for fl in all_flights:
        st_counts[fl["status"]] = st_counts.get(fl["status"], 0) + 1
    print("summary:", ", ".join(f"{k} {v}" for k, v in st_counts.items()))

    # where the failures happen
    fails = [(fl, e) for fl, e in zip(flights, rows) if fl["status"] != "goal"]
    if fails:
        print("\nfailures by distance remaining to the goal when the flight ended:")
        bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]
        for lo_, hi_ in bins:
            sel = [fl for fl, e in fails if lo_ <= np.linalg.norm(W["goal"][e].numpy() - fl["X"][-1, :3]) / 1000 < hi_]
            if sel:
                kinds = {}
                for fl in sel:
                    kinds[fl["status"]] = kinds.get(fl["status"], 0) + 1
                bank_sat = np.mean([np.abs(np.degrees(np.arctan2(fl["U"][-11:, 2], np.maximum(fl["U"][-11:, 1], 1e-6)))).max()
                                    >= af["BANK_MAX_DEG"] - 1 for fl in sel])
                print(f"  {lo_:>2}-{hi_ if hi_ < 99 else '..':>2} km from goal: {len(sel):3d} ({', '.join(f'{k} {v}' for k, v in kinds.items())}); "
                      f"at full bank in the last 5 s: {100 * bank_sat:.0f}%")
        near = sum(np.linalg.norm(W["goal"][e].numpy() - fl["X"][-1, :3]) < 1000 for fl, e in fails)
        print(f"  -> {near} of {len(fails)} failures ended within 1 km of the goal")


if __name__ == "__main__":
    main()