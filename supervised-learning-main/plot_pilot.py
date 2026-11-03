"""
Plot what the RL pilot is doing: fly a few flights with a saved checkpoint and draw them.

For each flight:
    top view and side view of the path with the obstacles it actually had (dark) and the ones
    thinned out for this curriculum stage (light), start, goal, and where / how the flight ended;
    time histories of speed, climb angle, bank angle, thrust, and the lidar distance straight
    ahead and the closest of all rays.
The agent flies its mean action (no exploration noise), on this computer (CPU is fine).

First copy the run's files from Modal (checkpoint.pt / best.pt land in supervised-learning-main/rl_pilot/<name>/):
    modal run supervised-learning-main/rl_pilot.py --name pilot_v2 --download-only
Then (from the fixed-wing-rl folder):
    python supervised-learning-main/plot_pilot.py --name pilot_v2                    6 flights, current curriculum stage
    python supervised-learning-main/plot_pilot.py --name pilot_v2 --n 10 --full      full environments (validation split)
    python supervised-learning-main/plot_pilot.py --name pilot_v2 --checkpoint best --show
Plots are saved to supervised-learning-main/rl_pilot/<name>/plots/.
Works with checkpoints from before the forward-only lidar change too (detected automatically).
"""
import argparse
import json
import math
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


def drop_projected_miss(n_rays):
    """Checkpoints from before the projected-miss inputs: remove those two inputs."""
    orig = rp.Sim.obs

    def obs(self):
        o, m = orig(self)
        return torch.cat([o[:, :13], o[:, 15:]], -1), m
    rp.Sim.obs = obs
    rp.OBS_DIM = 13 + n_rays


def fly_recorded(agent, W, maps, af, env_rows, keep_frac, seed):
    """Like rl_pilot.fly, but also keeps the thinned-out obstacles and the lidar readings."""
    n = len(env_rows)
    sim = rp.Sim(W, maps, None, af, n, "cpu", seed=seed)
    sim.keep_frac = keep_frac
    sim.reset(torch.arange(n), torch.as_tensor(env_rows))
    drop = sim.drop.clone()
    done = torch.zeros(n, dtype=torch.bool)
    status = torch.zeros(n, dtype=torch.long)
    X, U, T, RY = [sim.x.clone()], [sim.u.clone()], [sim.t.clone()], []
    steps = int(rp.TIME_LIMIT_X * sim.tref.max().item() / rp.DT) + 2
    with torch.no_grad():
        for _ in range(steps):
            o, m = sim.obs()
            RY.append(sim.rays.clone())
            act = agent.mean(o, m).clamp(-1, 1)
            xp, up_, tp = sim.x.clone(), sim.u.clone(), sim.t.clone()
            _, d, s, _ = sim.step(act)
            sim.x = torch.where(done[:, None], xp, sim.x)
            sim.u = torch.where(done[:, None], up_, sim.u)
            sim.t = torch.where(done, tp, sim.t)
            status = torch.where(~done & d, s, status)
            done = done | d
            X.append(sim.x.clone()); U.append(sim.u.clone()); T.append(sim.t.clone())
            if done.all():
                break
    RY.append(RY[-1])
    X, U, T, RY = (torch.stack(v, 1).numpy() for v in (X, U, T, RY))
    out = []
    for i in range(n):
        k = int(np.searchsorted(T[i], T[i, -1])) + 1
        out.append({"status": rp.STATUS[int(status[i])], "t": T[i, :k], "X": X[i, :k], "U": U[i, :k],
                    "rays": RY[i, :k] * rp.RAY_RANGE_M, "drop": drop[i]})
    return out


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
        ax.plot(X[:, 0] * km, X[:, j] * km, color=col, lw=2.2)
        ax.plot(0, 0, "o", ms=8, color=INK, mec="white", mew=2)
        ax.plot(goal[0] * km, goal[j] * km, "s", ms=9, color=INK, mec="white", mew=2)
        ax.plot(X[-1, 0] * km, X[-1, j] * km, "X" if fl["status"] != "goal" else "o", ms=12, color=col, mec="white", mew=1.5)
        if v == 0:                                                   # heading ticks every 20 s
            for k in range(0, len(t), 40):
                ax.annotate("", xy=((X[k, 0] + 250 * math.cos(X[k, 5])) * km, (X[k, 1] + 250 * math.sin(X[k, 5])) * km),
                            xytext=(X[k, 0] * km, X[k, 1] * km), arrowprops=dict(arrowstyle="->", color=col, lw=1))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title("Top view (obstacles at the flight's altitude)" if v == 0 else
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
    fig.suptitle(f"Environment {env_index}: {fl['status'].replace('_', ' ')} after {t[-1]:.0f} s   ({stage_txt})",
                 x=0.06, ha="left", fontsize=13, color=col if fl["status"] != "goal" else INK)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


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
    ap.add_argument("--successes", action="store_true",
                    help="keep flying until --n flights have reached the goal, and plot only those")
    a = ap.parse_args()

    run_dir = _HERE / "rl_pilot" / a.name
    ck_path = run_dir / f"{a.checkpoint}.pt"
    if not ck_path.exists():
        raise SystemExit(f"{ck_path} not found. Copy it from Modal first:\n  modal run "
                         f"supervised-learning-main/rl_pilot.py --name {a.name} --download-only")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    n_in = ck["agent"]["actor.0.weight"].shape[1] - 128
    version = (ck.get("settings") or {}).get("OBS_VERSION") or (1 if n_in == 13 + 31 else 2 if n_in == 13 + 29 else None)
    if version == 1:
        use_legacy_rays()
        drop_projected_miss(31)
        print("checkpoint uses the first lidar (+-105 deg, sphere tracing); plotting with that")
    elif version == 2:
        drop_projected_miss(rp.N_RAYS)
        print("checkpoint uses the forward lidar without the projected-miss inputs; plotting with that")
    if n_in != rp.OBS_DIM:
        raise SystemExit(f"checkpoint expects {n_in} inputs, this rl_pilot.py makes {rp.OBS_DIM}")
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
    agent = rp.make_agent("cpu")
    agent.load_state_dict(ck["agent"])
    agent.eval()

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
        pool = tr[:max(1, int(round(st.get("pool_frac", rp.CURRICULUM_START) * len(tr))))]
        keep = rp.KEEP_STAGES[st.get("keep_idx", 0)]
        rows = rng.choice(pool, min(a.n, len(pool)), replace=False)
        stage_txt = f"training stage: {keep:.0%} of obstacles, least-cluttered {st.get('pool_frac', 0.3):.0%} of environments"
    maps = torch.zeros((W["E"], 3, rp.MAP_NX, rp.MAP_NY))
    if a.successes:                                   # fly batches until enough flights reached the goal
        print(f"{a.checkpoint}.pt from update {st.get('update', '?')}; flying until {a.n} reach the goal ({stage_txt}) ...")
        got_rows, got, all_flights, tries = [], [], [], 0
        while len(got) < a.n and tries < 10:
            batch = rng.choice(pool, min(max(2 * a.n, 10), len(pool)), replace=False)
            maps[torch.as_tensor(batch)] = rp.build_maps(W, "cpu", ids_all=torch.as_tensor(batch))
            fl_b = fly_recorded(agent, W, maps, af, batch, keep, a.seed + tries)
            all_flights += fl_b
            for e, fl in zip(batch, fl_b):
                if fl["status"] == "goal" and len(got) < a.n:
                    got_rows.append(e); got.append(fl)
            tries += 1
            print(f"  batch {tries}: {sum(f['status'] == 'goal' for f in fl_b)} of {len(fl_b)} reached the goal")
        rows, flights = np.asarray(got_rows, dtype=int), got
    else:
        maps[torch.as_tensor(rows)] = rp.build_maps(W, "cpu", ids_all=torch.as_tensor(rows))
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
        plot_flight(fl, W, int(e), idx, af, out, a.show, stage_txt)
        X = fl["X"]
        print(f"  env {idx:5d}: {fl['status']:13s} after {fl['t'][-1]:5.0f} s, {X[-1, 0] / 1000:5.2f} of "
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