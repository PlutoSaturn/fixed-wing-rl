#!/usr/bin/env python3
"""
bench_pilot.py - benchmark test: fly one or more trained pilots on the SAME environments at a chosen
difficulty and compare them with each other and with the benchmark solver trajectories.

For every pilot (a run name; a flier round flies with the strategist it trained with, a strategist run
with its flier, a lone stage-1 flier on waypoints along the straight line) it flies --n environments,
each flight keeping a random --keep share of its obstacles (the same subset for every pilot), drawn
from the least-cluttered --pool share of the training environments (or --val: the validation set).
No exploration noise. It reports, per pilot:

    goal / crash / stall / timeout / missed rates
    speed vs the benchmark: benchmark flight time / pilot flight time, over successful flights
        (median, 10th and 90th percentile; 1.0 = as fast as the benchmark solver's trajectory)
    compute: wall time per flight (networks + simulator) on the machine that flew it

The benchmark trajectories were solved with ALL obstacles: at --keep < 1 the pilot flies a thinned
course, so its speed ratio is optimistic there. Outputs (supervised-learning-main/rl_pilot/bench/):
    bench_<tag>.csv   one row per pilot and environment
    bench_<tag>.png   outcome shares and the speed distribution, per pilot

Environments the convex solver FAILED on: build their cache once (build_cache.py --failed), then
    python supervised-learning-main/bench_pilot.py --names flier_r4b --run birkhoff_v1_failed --split all --n 500
("speed vs bench" is then vs the straight-line time at cruise: there is no solver trajectory).

On Modal (a GPU container reads the runs and the dataset straight from the Volume, nothing to
download; the table, CSV and figure are made here). Several names go in one quoted string:
    modal run supervised-learning-main/bench_pilot.py --names "flier_r4b strat_r5" --n 1000 --keep 0.75 --pool 1.0
    add --cpu to fly on a CPU container instead of a GPU

Locally (needs the runs downloaded, and their partner runs):
    modal run supervised-learning-main/rl_pilot.py --name flier_r4b --download-only
    python supervised-learning-main/bench_pilot.py --names flier_r4b --n 200 --keep 0.75 --pool 1.0
    python supervised-learning-main/bench_pilot.py --names flier_r3b flier_r4a flier_r4b --n 200 --keep 1 --pool 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
os.environ.setdefault("FW_NO_MODAL_DEFS", "1")
import torch                                                  # noqa: E402
import build_cache as bc                                      # noqa: E402
import rl_pilot as rp                                         # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OUTCOMES = [("goal", "#1baf7a"), ("crash", "#e34948"), ("out_of_bounds", "#e87ba4"), ("stall", "#eda100"),
            ("timeout", "#6250d6"), ("missed", "#a9a7a1")]


def load_agent(name, runs_root, device="cpu"):
    """-> (flier net, fly() keyword arguments, description) for run `name` (latest weights, like training)."""
    rdir = Path(runs_root) / name
    path = next((p for p in (rdir / "checkpoint.pt", rdir / "best.pt") if p.exists()), None)
    if path is None:
        raise SystemExit(f"run '{name}' not found ({rdir}). Download it:\n  modal run "
                         f"supervised-learning-main/rl_pilot.py --name {name} --download-only")
    ck = torch.load(path, map_location=device, weights_only=False)
    ver = (ck.get("settings") or {}).get("OBS_VERSION")
    if ver != rp.OBS_VERSION:
        raise SystemExit(f"{name}: checkpoint is OBS_VERSION {ver}, this code is {rp.OBS_VERSION}")
    cfg = ck.get("cfg") or {}

    def partner(pname, strat):
        pdir = Path(runs_root) / pname
        p = next((q for q in (pdir / "checkpoint.pt", pdir / "best.pt") if q.exists()), None)
        if p is None:
            raise SystemExit(f"{name} needs its partner run '{pname}' ({pdir}). Download it:\n  modal run "
                             f"supervised-learning-main/rl_pilot.py --name {pname} --download-only")
        net = rp.make_agent(device, rp.STRAT_DIM, (0.0, 0.0, 0.0), rp.FAR_SHAPE) if strat else rp.make_agent(device)
        net.load_state_dict(torch.load(p, map_location=device, weights_only=False)["agent"])
        return net.eval()
    if (cfg.get("stage") or "flier") == "strategy":
        strat = rp.make_agent(device, rp.STRAT_DIM, (0.0, 0.0, 0.0), rp.FAR_SHAPE)
        strat.load_state_dict(ck["agent"])
        return partner(cfg["flier"], False), {"strategy": strat.eval()}, f"strategist + flier {cfg['flier']}"
    flier = rp.make_agent(device)
    flier.load_state_dict(ck["agent"])
    if cfg.get("strategy"):
        return flier.eval(), {"strategy": partner(cfg["strategy"], True)}, f"flier + strategist {cfg['strategy']}"
    return flier.eval(), {"wp_mode": "line"}, "flier on waypoints along the line to the goal"


def compute(o, data_root, runs_root, device="cpu", project=None):
    """Fly every pilot in o["names"] on the same environments. o: names, n, keep, pool, val, seed, batch,
    dataset, run, tag. -> (results {name: desc, status, speed (list), wall}, per_env rows, meta). Plain
    Python types only, so a Modal container can return it."""
    torch.set_grad_enabled(False)
    env_root = Path(data_root) / o["dataset"]
    cache = bc.load_cache(env_root / "cache" / f"{o['run']}.npz")
    scale = cache["summary"]["length_scale_m"]
    bench_tf = cache["lab_sigma"][cache["lab_off"][:-1]].astype(np.float64)
    env0 = json.loads((env_root / "envs" / "shard_00000.json").read_text())[0]
    af = rp.airframe(project) if project is not None else rp.airframe()
    W = rp.build_world(cache, scale, env0["bounds_lo"], env0["bounds_hi"], env0["vehicle_radius"],
                       env0["vehicle_radius"] + env0["margin"], device)
    rng = np.random.default_rng(o["seed"])
    failed_only = bool(cache["summary"].get("failed_only"))
    split = o.get("split") or "train"
    if o["val"]:
        pool = np.nonzero(cache["split"] == "val")[0]
        keep = 1.0
        where = "validation environments, all obstacles"
    else:
        with open(env_root / "index.csv", newline="") as f:
            occ_of = {int(r["index"]): float(r.get("occupancy") or 0) for r in csv.DictReader(f)}
        tr = np.arange(len(cache["split"])) if split == "all" else np.nonzero(cache["split"] == split)[0]
        if len(tr) == 0:
            raise SystemExit(f"no '{split}' environments in this cache (use --split all)")
        tr = tr[np.argsort([occ_of.get(int(cache["env_index"][e]), 0) for e in tr], kind="stable")]
        pool = tr[:max(1, int(round(o["pool"] * len(tr))))]
        keep = o["keep"]
        what = {"train": "training", "val": "validation", "test": "test", "all": "all"}[split]
        where = f"{keep:.0%} of obstacles, least-cluttered {o['pool']:.0%} of {what} environments"
    if failed_only:
        where += " the SOLVER FAILED on"
    rows = rng.choice(pool, min(o["n"], len(pool)), replace=False)
    print(f"{len(rows)} environments: {where}; the same environments and obstacle subsets for every pilot "
          f"(on {device})", flush=True)

    solver_status = {}                                   # failed-env cache: why the solver failed, per env
    if failed_only:
        res_csv = env_root / "solutions" / o["run"].removesuffix("_failed") / "results.csv"
        if res_csv.exists():
            with open(res_csv, newline="") as f:
                solver_status = {int(r["index"]): r["status"] for r in csv.DictReader(f)}
    results, per_env = {}, []
    for name in o["names"]:
        flier, kw, desc = load_agent(name, runs_root, device)
        t0 = time.perf_counter()
        flights = []
        for b in range(0, len(rows), o["batch"]):        # same seed per batch -> same obstacle subsets
            flights += rp.fly(flier, None, W, None, bench_tf, af, rows[b:b + o["batch"]].tolist(), device,
                              keep_frac=keep, seed=o["seed"] + b, **kw)
        wall = (time.perf_counter() - t0) / len(rows)
        st = [f["status"] for f in flights]
        speed = [float(bench_tf[e] / f["t"][-1]) for e, f in zip(rows, flights) if f["status"] == "goal"]
        results[name] = dict(desc=desc, status=st, speed=speed, wall=float(wall))
        print(f"  {name}: done, goal {100 * st.count('goal') / len(st):.1f}%", flush=True)
        for e, f in zip(rows, flights):
            per_env.append({"pilot": name, "env_index": int(cache["env_index"][e]), "status": f["status"],
                            "flight_time_s": round(float(f["t"][-1]), 2),
                            "bench_flight_time_s": round(float(bench_tf[e]), 2),
                            "speed_vs_bench": round(float(bench_tf[e] / f["t"][-1]), 4) if f["status"] == "goal" else "",
                            **({"solver_status": solver_status.get(int(cache["env_index"][e]), "")} if solver_status else {})})
    tag = o.get("tag") or ("val" if o["val"] else f"k{int(round(100 * keep))}_p{int(round(100 * o['pool']))}") \
        + f"_n{len(rows)}"
    if failed_only:
        tag = "failed_" + tag
    meta = dict(where=where, keep=float(keep), n=int(len(rows)), tag=tag, device=str(device), failed_only=failed_only)
    return results, per_env, meta


def report(results, per_env, meta, out_dir):
    """Print the table and save bench_<tag>.csv / .png in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag, where, keep = meta["tag"], meta["where"], meta["keep"]
    for r in results.values():
        r["speed"] = np.asarray(r["speed"], float)
    # ---- table
    head = f"{'pilot':16s} {'goal':>6s} {'crash':>6s} {'stall':>6s} {'timeout':>7s} {'missed':>7s}   " \
           f"{'speed vs bench (median, p10-p90)':34s} {'s/flight':>8s}"
    print(f"\n(compute: {meta['device']})")
    print(head + "\n" + "-" * len(head))
    for name, r in results.items():
        n = len(r["status"])
        pct = lambda k: 100 * sum(s == k for s in r["status"]) / n
        sp = r["speed"]
        sp_txt = f"{np.median(sp):.3f}  ({np.percentile(sp, 10):.3f}-{np.percentile(sp, 90):.3f})" if len(sp) else "n/a"
        print(f"{name:16s} {pct('goal'):5.1f}% {pct('crash') + pct('out_of_bounds'):5.1f}% {pct('stall'):5.1f}% "
              f"{pct('timeout'):6.1f}% {pct('missed'):6.1f}%   {sp_txt:34s} {r['wall']:8.2f}")
    for name, r in results.items():
        print(f"  {name}: {r['desc']}")
    if per_env and "solver_status" in per_env[0]:            # goal rate by why the solver failed
        print("\ngoal rate by the solver's failure status:")
        for name in results:
            by = {}
            for r in per_env:
                if r["pilot"] == name:
                    by.setdefault(r["solver_status"] or "?", []).append(r["status"] == "goal")
            print(f"  {name:16s} " + "   ".join(f"{k} {100 * np.mean(v):5.1f}% (n={len(v)})" for k, v in sorted(by.items())))
    if meta.get("failed_only"):
        print("\nnote: the solver found no trajectory for these environments; 'speed vs bench' here is the "
              "straight-line time at cruise / the pilot's flight time")
    elif keep < 1:
        print("\nnote: the benchmark trajectories were solved with all obstacles; at --keep < 1 the speed ratio is optimistic")

    with open(out_dir / f"bench_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(per_env[0].keys()))
        w.writeheader()
        w.writerows(per_env)

    # ---- figure: outcome shares (stacked bars) and speed distribution (box per pilot)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.9 + 0.55 * max(3, len(names))),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(names))[::-1]
    for yi, name in zip(y, names):
        n = len(results[name]["status"])
        left = 0.0
        for k, c in OUTCOMES:
            v = 100 * sum(s == k for s in results[name]["status"]) / n
            if v > 0:
                ax1.barh(yi, v, left=left, color=c, height=0.62, edgecolor="white", linewidth=2)
                if v >= 6:
                    ax1.text(left + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax1.set_yticks(y, names, fontsize=9)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("share of flights (%)", color=MUTED, fontsize=9)
    ax1.set_title("Outcome", loc="left", fontsize=10.5, color=INK, pad=24)
    from matplotlib.patches import Patch
    ax1.legend([Patch(color=c) for _, c in OUTCOMES], [k.replace("_", " ") for k, _ in OUTCOMES],
               ncol=6, fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.06))
    data = [results[nm]["speed"] for nm in names]
    ok = [i for i, d in enumerate(data) if len(d)]
    if ok:
        bp = ax2.boxplot([data[i] for i in ok], positions=[y[i] for i in ok], vert=False, widths=0.5,
                         patch_artist=True, showfliers=False, whis=(10, 90))
        for i, box in zip(ok, bp["boxes"]):
            box.set(facecolor=SERIES[i % len(SERIES)], alpha=0.35, edgecolor=SERIES[i % len(SERIES)])
        for el in ("whiskers", "caps", "medians"):
            for ln in bp[el]:
                ln.set(color=INK if el == "medians" else MUTED, linewidth=1.6 if el == "medians" else 1)
    ax2.axvline(1.0, color="#a9a7a1", ls="--", lw=1.2)
    ax2.set_yticks(y, ["" for _ in names])
    ax2.set_xlabel("benchmark flight time / pilot's  (1.0 = as fast; box: p25-p75, whiskers: p10-p90)",
                   color=MUTED, fontsize=9)
    ax2.set_title("Speed vs the benchmark solve (successful flights)", loc="left", fontsize=10.5, color=INK, pad=24)
    for ax in (ax1, ax2):
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(f"Benchmark: {meta['n']} environments, {where}", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / f"bench_{tag}.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nsaved {out_dir / f'bench_{tag}.csv'} and {out_dir / f'bench_{tag}.png'}")



def options(names, n=100, keep=1.0, pool=1.0, val=False, seed=0, batch=50, dataset=None, run=None, tag=None,
            split="train"):
    if split not in ("train", "val", "test", "all"):
        raise SystemExit("--split must be train, val, test or all")
    return dict(names=list(names), n=int(n), keep=float(keep), pool=float(pool), val=bool(val), seed=int(seed),
                batch=int(batch), dataset=dataset or rp.DATASET, run=run or rp.RUN, tag=tag or None, split=split)


def main():
    ap = argparse.ArgumentParser(description="Benchmark trained pilots at a chosen difficulty (locally).")
    ap.add_argument("--names", nargs="+", required=True, help="run names to compare")
    ap.add_argument("--n", type=int, default=100, help="environments to fly")
    ap.add_argument("--keep", type=float, default=1.0, help="share of each environment's obstacles kept (0..1)")
    ap.add_argument("--pool", type=float, default=1.0, help="least-cluttered share of the training environments")
    ap.add_argument("--val", action="store_true", help="fly the validation environments instead (all obstacles)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50, help="flights simulated together")
    ap.add_argument("--dataset", default=rp.DATASET)
    ap.add_argument("--run", default=rp.RUN)
    ap.add_argument("--tag", default=None, help="name for the output files")
    ap.add_argument("--split", default="train", help="environments to draw from: train, val, test or all")
    a = ap.parse_args()
    o = options(a.names, a.n, a.keep, a.pool, a.val, a.seed, a.batch, a.dataset, a.run, a.tag, a.split)
    report(*compute(o, _HERE / "datasets", _HERE / "rl_pilot", "cpu", _HERE.parent), _HERE / "rl_pilot" / "bench")


# --------------------------------------------------------------------------
# Modal: `modal run bench_pilot.py ...` flies on a Modal container, reading the Volume
# --------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("fixed-wing-rl-bench")
    bench_volume = modal.Volume.from_name(rp.MODAL_VOLUME)
    bench_image = (modal.Image.debian_slim(python_version=rp.PYTHON_VERSION)
                   .pip_install("numpy", "torch").env({"FW_NO_MODAL_DEFS": "1"})
                   .add_local_dir(str(rp.PROJECT / "aircraftmodel"), "/root/fixed-wing-rl/aircraftmodel"))
    for _f in ("rl_pilot.py", "build_cache.py", "eval_warmstart.py"):
        bench_image = bench_image.add_local_file(str(_HERE / _f), f"/root/{_f}")

    def _remote(o):
        bench_volume.reload()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return compute(o, f"{rp.VOLUME_MOUNT}/datasets", f"{rp.VOLUME_MOUNT}/rl_pilot", device)

    @app.function(image=bench_image, gpu=rp.GPU, cpu=4.0, memory=16384, timeout=4 * 3600,
                        volumes={rp.VOLUME_MOUNT: bench_volume})
    def bench_remote_gpu(o):
        return _remote(o)

    @app.function(image=bench_image, cpu=4.0, memory=16384, timeout=4 * 3600,
                        volumes={rp.VOLUME_MOUNT: bench_volume})
    def bench_remote_cpu(o):
        return _remote(o)

    @app.local_entrypoint()
    def modal_main(names: str, n: int = 100, keep: float = 1.0, pool: float = 1.0, val: bool = False,
                   seed: int = 0, batch: int = 256, dataset: str = "", run: str = "", tag: str = "",
                   cpu: bool = False, split: str = "train"):
        o = options(names.replace(",", " ").split(), n, keep, pool, val, seed, batch, dataset, run, tag, split)
        print(f"benchmarking {', '.join(o['names'])} on Modal ({'CPU' if cpu else rp.GPU})...", flush=True)
        results, per_env, meta = (bench_remote_cpu if cpu else bench_remote_gpu).remote(o)
        report(results, per_env, meta, _HERE / "rl_pilot" / "bench")


if __name__ == "__main__":
    main()