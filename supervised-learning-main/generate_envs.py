"""
Generate a large environment dataset for supervised learning, on Modal or locally.

Calls environment/envgen.py (unchanged) to make the environments; this file only
handles seeding, sharding, parallel execution, saving, resume and bookkeeping.

Output (inside this supervised-learning-main folder, and mirrored to a Modal
Volume when run on Modal):

    supervised-learning-main/datasets/<DATASET>/
        manifest.json            settings, seeds, code hashes; checked on resume
        envs/shard_00000.json    list of environments, same format envgen saves
        meta/shard_00000.json    per-environment rows, failures and timing for that shard
        index.csv                one row per environment (rebuilt at the end of every run)

Any shard file can be passed straight to the solvers (give the path from the
fixed-wing-rl folder; it is outside localstore), e.g.
    python -m solvers.scp_birkhoff --envs supervised-learning-main/datasets/c172_envs_v1/envs/shard_00003.json

Run (from the fixed-wing-rl folder):
    modal run supervised-learning-main/generate_envs.py --n 5000 --dataset c172_envs_v1
    python supervised-learning-main/generate_envs.py --local --n 20 --dataset smoke_test

Rerunning the same command resumes: finished shards are skipped. Raising --n
extends a dataset. Changing any generator setting requires a new --dataset name.
"""

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET          = "c172_envs_v1"   # folder name under OUTPUT_DIR
OUTPUT_DIR       = None             # None = supervised-learning-main/datasets
N_ENVIRONMENTS   = 1000             # total environments in the dataset
SHARD_SIZE       = 100              # environments per shard (= per Modal call)
BASE_SEED        = 0                # environment i uses a seed derived from (BASE_SEED, i)
LOCAL_WORKERS    = None             # --local: processes to use (None = all CPU cores)

# envgen settings that differ from envgen's own defaults (the C172 setup).
# Anything not listed here keeps the value from envgen's USER SETTINGS block.
ENV_OVERRIDES = {
    "size": (12000.0, 2500.0, 600.0),
    "vehicle_radius": 6.0,
    "margin": 30.0,
    "occupancy_range": (0.05, 0.12),
    "endpoint_mode": "ends",
}

# train / val / test split, assigned from each environment's fingerprint
SPLIT_FRACTIONS  = {"train": 0.8, "val": 0.1, "test": 0.1}

# Modal
MODAL_APP_NAME   = "fixed-wing-envgen"
MODAL_VOLUME     = "fixed-wing-rl-data"   # None = do not mirror results to a Modal Volume
MAX_CONTAINERS   = 100              # shards generated at the same time
CPU_PER_SHARD    = 1.0
SHARD_TIMEOUT_S  = 3600
SHARD_RETRIES    = 2                # re-run a shard if its container dies
PYTHON_VERSION   = "3.12"
NUMPY_SPEC       = "numpy"          # pin (e.g. "numpy==2.2.6") to match your local numpy exactly

# ==========================================================================
import argparse
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# The project folder: locally the parent of this folder; in a Modal container,
# where envgen is mounted under /root/fixed-wing-rl.
_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if (_HERE.parent / "environment").is_dir() else Path("/root/fixed-wing-rl")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

VOLUME_MOUNT = "/data"


# --------------------------------------------------------------------------
# generation (runs on Modal workers or local processes)
# --------------------------------------------------------------------------
def env_seed(base_seed, index):
    """Independent, reproducible seed for environment `index`."""
    import numpy as np
    return int(np.random.SeedSequence([int(base_seed), int(index)]).generate_state(1)[0])


def make_config(overrides):
    from environment.envgen import EnvConfig
    return EnvConfig(**overrides)


def generate_shard(task):
    """Generate the environments in one shard. Returns plain data (no files written)."""
    from environment.envgen import EnvironmentGenerator

    cfg = make_config(task["overrides"])
    envs, rows, failures = [], [], []
    t_shard = time.perf_counter()
    for i in task["indices"]:
        seed = env_seed(task["base_seed"], i)
        t0 = time.perf_counter()
        try:
            env = EnvironmentGenerator(cfg, seed=seed).generate()
        except Exception as e:                        # envgen gave up (max tries) or broke
            failures.append({"index": i, "seed": seed, "error": f"{type(e).__name__}: {e}"})
            continue
        dt = time.perf_counter() - t0
        env.metrics["dataset"] = {"name": task["dataset"], "index": i, "seed": seed,
                                  "generation_time_s": round(dt, 3)}
        fp = env.fingerprint()                        # metrics are not part of the fingerprint
        envs.append(env.to_dict())
        rows.append(index_row(env, fp, i, seed, task["shard"], dt))
    return {"shard": task["shard"], "indices": list(task["indices"]), "envs": envs,
            "rows": rows, "failures": failures,
            "time_s": round(time.perf_counter() - t_shard, 2)}


def split_for(fingerprint):
    """Deterministic split from the fingerprint, so adding data never reshuffles."""
    u = int(hashlib.sha1(fingerprint.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLIT_FRACTIONS.items():
        acc += frac
        if u <= acc:
            return name
    return list(SPLIT_FRACTIONS)[-1]


OBSTACLE_KINDS = ("spheroid", "oriented_cylinder", "cylinder", "box", "wall")


def index_row(env, fp, index, seed, shard, gen_time):
    m = env.metrics
    counts = m.get("obstacle_counts", {})
    row = {"fingerprint": fp, "index": index, "shard": shard, "seed": seed,
           "split": split_for(fp),
           "occupancy_target": round(m["occupancy_target"], 5),
           "occupancy": round(m["occupancy_fraction"], 5),
           "inflated_occupancy": round(m["inflated_occupancy_fraction"], 5),
           "n_obstacles": len(env.obstacles),
           "n_pieces": m["n_constraint_pieces"],
           "straight_line_m": round(m["straight_line_distance"], 2),
           "line_blocked": m["straight_line_blocked"],
           "generation_time_s": round(gen_time, 3)}
    for k in OBSTACLE_KINDS:
        row[f"n_{k}"] = counts.get(k, 0)
    return row


# --------------------------------------------------------------------------
# saving, resume and bookkeeping (runs on your machine)
# --------------------------------------------------------------------------
def dataset_dir(name):
    d = (Path(OUTPUT_DIR) if OUTPUT_DIR else _HERE / "datasets") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def shard_name(k):
    return f"shard_{k:05d}.json"


def write_json_atomic(path, obj):
    """Write to a temporary file, then rename: a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_manifest(dataset, n, shard_size, base_seed, overrides):
    cfg = make_config(overrides)
    return {
        "dataset": dataset,
        "n_environments": n,
        "shard_size": shard_size,
        "base_seed": base_seed,
        "seed_rule": "numpy SeedSequence([base_seed, index]).generate_state(1)[0]",
        "split_fractions": SPLIT_FRACTIONS,
        "env_overrides": overrides,
        "env_config": json.loads(json.dumps(dataclasses.asdict(cfg))),   # tuples -> lists
        "envgen_sha1": file_sha1(PROJECT / "environment" / "envgen.py"),
        "git_commit": git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# settings that must match for a resumed / extended dataset to stay consistent
_MUST_MATCH = ("shard_size", "base_seed", "env_config", "split_fractions")


def check_or_write_manifest(root, manifest):
    path = root / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        bad = [k for k in _MUST_MATCH if old.get(k) != manifest[k]]
        if bad:
            raise SystemExit(
                f"Dataset '{manifest['dataset']}' already exists with different {', '.join(bad)}.\n"
                f"Use a new --dataset name (or delete {root}) rather than mixing settings.")
        if old.get("envgen_sha1") != manifest["envgen_sha1"]:
            print(f"WARNING: envgen.py has changed since this dataset was started "
                  f"({old.get('envgen_sha1')} -> {manifest['envgen_sha1']}). New shards may "
                  f"differ from old ones. Consider a new --dataset name.")
        manifest = {**old, "n_environments": max(old["n_environments"], manifest["n_environments"]),
                    "updated": manifest["created"]}
    write_json_atomic(path, manifest)
    return manifest


def plan_shards(root, dataset, n, shard_size, base_seed, overrides):
    """Every shard needed for n environments, minus the ones already complete."""
    todo, done = [], 0
    for k in range((n + shard_size - 1) // shard_size):
        indices = list(range(k * shard_size, min((k + 1) * shard_size, n)))
        meta = root / "meta" / shard_name(k)
        if (root / "envs" / shard_name(k)).exists() and meta.exists():
            if json.loads(meta.read_text()).get("indices") == indices:
                done += 1
                continue                       # complete (a shorter last shard gets redone)
        todo.append({"dataset": dataset, "shard": k, "indices": indices,
                     "base_seed": base_seed, "overrides": overrides})
    return todo, done


def save_shard(root, result):
    k = result["shard"]
    write_json_atomic(root / "envs" / shard_name(k), result["envs"])
    meta = {key: result[key] for key in ("shard", "indices", "rows", "failures", "time_s")}
    write_json_atomic(root / "meta" / shard_name(k), meta)


def build_index(root, n):
    """index.csv from every shard on disk; returns a summary dict."""
    rows, failures, seen, dupes = [], [], set(), []
    for meta in sorted((root / "meta").glob("shard_*.json")):
        m = json.loads(meta.read_text())
        for r in m["rows"]:
            if r["index"] >= n:
                continue
            if r["fingerprint"] in seen:
                dupes.append(r["fingerprint"])
            seen.add(r["fingerprint"])
            rows.append(r)
        failures += [f for f in m["failures"] if f["index"] < n]
    rows.sort(key=lambda r: r["index"])
    if rows:
        with open(root / "index.csv.tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(root / "index.csv.tmp", root / "index.csv")
    write_json_atomic(root / "failures.json", failures)
    return {"rows": rows, "failures": failures, "duplicates": dupes}


def print_summary(root, summary, n, wall_s):
    import numpy as np
    rows = summary["rows"]
    print(f"\nDataset: {root}")
    print(f"  environments: {len(rows)} of {n} requested, {len(summary['failures'])} failed"
          + (f", {len(summary['duplicates'])} DUPLICATE fingerprints" if summary["duplicates"] else ""))
    if not rows:
        return
    splits = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print("  splits: " + ", ".join(f"{k} {v}" for k, v in splits.items()))
    for key, label in (("occupancy", "fill"), ("inflated_occupancy", "inflated fill"),
                       ("n_obstacles", "obstacles"), ("generation_time_s", "gen time (s)")):
        v = np.array([r[key] for r in rows], float)
        print(f"  {label:14s} min {v.min():8.3f}   median {np.median(v):8.3f}   max {v.max():8.3f}")
    walls = np.array([r["n_wall"] for r in rows])
    print("  walls per env: " + ", ".join(f"{w}: {np.mean(walls == w):.0%}" for w in np.unique(walls)))
    print(f"  wall-clock this run: {wall_s / 60:.1f} min")


def report(result, n_done, n_total, t0):
    print(f"  shard {result['shard']:5d}: {len(result['envs'])} envs, "
          f"{len(result['failures'])} failed, {result['time_s'] / 60:.1f} min   "
          f"[{n_done}/{n_total} shards, {(time.time() - t0) / 60:.1f} min elapsed]", flush=True)


def prepare(dataset, n, shard_size, base_seed):
    root = dataset_dir(dataset)
    manifest = build_manifest(dataset, n, shard_size, base_seed, ENV_OVERRIDES)
    check_or_write_manifest(root, manifest)
    todo, done = plan_shards(root, dataset, n, shard_size, base_seed, ENV_OVERRIDES)
    print(f"Dataset '{dataset}': {n} environments in shards of {shard_size}; "
          f"{done} shards already done, {len(todo)} to generate.")
    return root, todo


# --------------------------------------------------------------------------
# local mode (multiprocessing)
# --------------------------------------------------------------------------
def run_local(dataset, n, shard_size, base_seed, workers):
    from multiprocessing import Pool
    root, todo = prepare(dataset, n, shard_size, base_seed)
    t0 = time.time()
    if todo:
        workers = min(workers or os.cpu_count() or 1, len(todo))
        with Pool(workers) as pool:
            for k, result in enumerate(pool.imap_unordered(generate_shard, todo), 1):
                save_shard(root, result)
                report(result, k, len(todo), t0)
    print_summary(root, build_index(root, n), n, time.time() - t0)


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------
try:
    import modal
except ImportError:          # local mode works without Modal installed
    modal = None

if modal is not None:
    app = modal.App(MODAL_APP_NAME)
    image = (modal.Image.debian_slim(python_version=PYTHON_VERSION)
             .pip_install(NUMPY_SPEC)
             .add_local_dir(str(PROJECT / "environment"), "/root/fixed-wing-rl/environment",
                            ignore=["__pycache__", "*.pyc"]))
    volume = (modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
              if MODAL_VOLUME else None)

    @app.function(image=image, cpu=CPU_PER_SHARD, timeout=SHARD_TIMEOUT_S,
                  retries=SHARD_RETRIES, max_containers=MAX_CONTAINERS,
                  volumes={VOLUME_MOUNT: volume} if volume is not None else {})
    def generate_shard_remote(task):
        result = generate_shard(task)
        if volume is not None:              # mirror for the (Modal) solving stage
            base = Path(VOLUME_MOUNT) / "datasets" / task["dataset"]
            save_shard(base, result)
            volume.commit()
        return result

    @app.local_entrypoint()
    def main(n: int = N_ENVIRONMENTS, dataset: str = DATASET,
             shard_size: int = SHARD_SIZE, seed: int = BASE_SEED):
        root, todo = prepare(dataset, n, shard_size, seed)
        t0 = time.time()
        failed_shards = []
        for k, result in enumerate(generate_shard_remote.map(todo, order_outputs=False,
                                                             return_exceptions=True), 1):
            if isinstance(result, BaseException):
                failed_shards.append(repr(result))
                print(f"  a shard failed after retries: {result!r}", flush=True)
                continue
            save_shard(root, result)
            report(result, k, len(todo), t0)
        summary = build_index(root, n)
        if volume is not None:              # keep the volume's bookkeeping in sync
            with volume.batch_upload(force=True) as batch:
                for name in ("manifest.json", "index.csv", "failures.json"):
                    if (root / name).exists():
                        batch.put_file(str(root / name), f"/datasets/{dataset}/{name}")
            print(f"Mirrored to Modal Volume '{MODAL_VOLUME}' at /datasets/{dataset}")
        print_summary(root, summary, n, time.time() - t0)
        if failed_shards:
            print(f"\n{len(failed_shards)} shard(s) failed; rerun the same command to retry them.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate an environment dataset. Use "
                                 "'modal run' for the cloud, or --local here.")
    ap.add_argument("--local", action="store_true", help="run on this computer (multiprocessing)")
    ap.add_argument("--n", type=int, default=N_ENVIRONMENTS)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--workers", type=int, default=LOCAL_WORKERS)
    a = ap.parse_args()
    if not a.local:
        raise SystemExit("For Modal use:  modal run supervised-learning-main/generate_envs.py "
                         "--n ... --dataset ...\nFor this computer add --local.")
    run_local(a.dataset, a.n, a.shard_size, a.seed, a.workers)