"""
Plot the warm-start network's K predicted routes against the solver's optimal route.

For each chosen environment: a top view and a side view with the obstacles, the
solver's best route (black), its other valid routes (dashed gray), and the network's
K predictions (colored; the most confident one is drawn thicker). The legend gives
each prediction's confidence, its length relative to the optimal route (a stand-in
for flight time), and how much of it is inside an inflated obstacle.

Everything comes from the training cache and the model file; nothing is solved.

Run (from the fixed-wing-rl folder):
    python supervised-learning-main/plot_warmstart.py                          5 random validation envs
    python supervised-learning-main/plot_warmstart.py --n 10 --split test
    python supervised-learning-main/plot_warmstart.py --env 1234 --show        a specific env index
Plots are saved to supervised-learning-main/models/plots/<model name>/.
"""

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET     = "c172_envs_v1"
RUN         = "birkhoff_v1"
MODEL       = "warmstart_v2"         # file in supervised-learning-main/models (without .pt)
SPLIT       = "val"
N_ENVS      = 5
SEED        = 0
CLEARANCE_M = 36.0                   # inflation drawn around the obstacles (the solver's clearance)
GRID_M      = 20.0                   # resolution of the obstacle slices, m

# ==========================================================================
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE.parent), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_cache as bc                                   # noqa: E402
import train_warmstart as tw                               # noqa: E402

PRED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e2dd"
OBS, OBS_INF = "#a9a7a1", "#e3e1dc"


def route_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def occupancy(tok, pts, d):
    """True where points (N, 3) are inside any obstacle token inflated by d (normalized units)."""
    t = torch.from_numpy(tok)[None]
    mask = torch.zeros(t.shape[:2], dtype=torch.bool)
    out = []
    for chunk in np.array_split(pts, max(1, len(pts) // 20000)):
        q = torch.from_numpy(chunk.astype(np.float32))[None, None]
        out.append((tw.obstacle_h(q, t, mask, d)[0, 0].min(-1).values < 0).numpy())
    return np.concatenate(out)


def plot_env(model, cache, e, scale, out_dir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = cache
    tok = c["tok"][c["tok_off"][e]:c["tok_off"][e + 1]]
    ctx = c["ctx"][e]
    l0, l1 = c["lab_off"][e], c["lab_off"][e + 1]
    labels = c["lab_x"][l0:l1, :, :3]
    best = labels[np.argmax(c["lab_best"][l0:l1])]
    with torch.no_grad():
        paths, sig, logit = model(torch.from_numpy(tok)[None], torch.zeros(1, len(tok), dtype=torch.bool),
                                  torch.from_numpy(ctx)[None])
    paths = paths[0].numpy()
    conf = torch.softmax(logit[0], 0).numpy()
    top = int(np.argmax(conf))
    d_inf = CLEARANCE_M / scale

    # quality of each prediction
    L_best = route_length(best)
    inside = []
    for p in paths:
        inside.append(100 * occupancy(tok, p, d_inf).mean())
    err = [np.linalg.norm(p - best, axis=1).mean() * scale for p in paths]

    # obstacle slices: top view at the optimal route's altitude, side view at its lateral position
    xlo, xhi, ylo, yhi = ctx[10], ctx[11], ctx[8], ctx[9]
    floor, ceil = ctx[6], ctx[7]
    res = GRID_M / scale
    xs = np.arange(xlo, xhi, res) + res / 2
    ys = np.arange(ylo, yhi, res) + res / 2
    zs = np.arange(floor, ceil, res / 2) + res / 4
    o = np.argsort(best[:, 0])
    z_at = lambda x: np.interp(x, best[o, 0], best[o, 2])
    y_at = lambda x: np.interp(x, best[o, 0], best[o, 1])
    XX, YY = np.meshgrid(xs, ys)
    top_pts = np.stack([XX.ravel(), YY.ravel(), z_at(XX.ravel())], 1)
    XZ, ZZ = np.meshgrid(xs, zs)
    side_pts = np.stack([XZ.ravel(), y_at(XZ.ravel()), ZZ.ravel()], 1)
    slices = []
    for pts, shape in ((top_pts, XX.shape), (side_pts, XZ.shape)):
        slices.append((occupancy(tok, pts, 0.0).reshape(shape), occupancy(tok, pts, d_inf).reshape(shape)))

    km = scale / 1000.0                                      # normalized -> km
    fig, all_axes = plt.subplots(3, 1, figsize=(14, 9.5),
                                 gridspec_kw=dict(height_ratios=[2.4, 1.4, 0.9], hspace=0.45))
    axes, leg_ax = all_axes[:2], all_axes[2]
    for v, ax in enumerate(axes):
        real, infl = slices[v]
        img = np.zeros(real.shape + (4,))
        img[infl] = matplotlib.colors.to_rgba(OBS_INF)
        img[real] = matplotlib.colors.to_rgba(OBS)
        j = 1 if v == 0 else 2
        ext = [xlo * km, xhi * km, (ylo if v == 0 else floor) * km, (yhi if v == 0 else ceil) * km]
        ax.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest", zorder=0)
        for k, lab in enumerate(labels):
            if not np.array_equal(lab, best):
                ax.plot(lab[:, 0] * km, lab[:, j] * km, "--", color=MUTED, lw=1.2, zorder=2,
                        label="other valid solver routes" if k == 0 or (k == 1 and np.array_equal(labels[0], best)) else None)
        ax.plot(best[:, 0] * km, best[:, j] * km, color=INK, lw=3.0, zorder=3, label="solver optimum")
        for k in np.argsort(-conf):
            p = paths[k]
            lab = (f"prediction {k + 1}: conf {conf[k]:.0%}, length {route_length(p) / L_best:.3f}x, "
                   f"{inside[k]:.0f}% inside, {err[k]:.0f} m from optimum" + ("  (top-1)" if k == top else ""))
            ax.plot(p[:, 0] * km, p[:, j] * km, color=PRED_COLORS[k % len(PRED_COLORS)],
                    lw=2.6 if k == top else 1.6, alpha=1.0 if k == top else 0.85, zorder=5 if k == top else 4,
                    label=lab if v == 0 else None)
        ax.plot(0, 0, "o", ms=8, color=INK, mec="white", mew=2, zorder=6)
        ax.plot(best[-1, 0] * km, best[-1, j] * km, "s", ms=8, color=INK, mec="white", mew=2, zorder=6)
        ax.set_xlim(xlo * km, xhi * km)
        ax.set_ylim(ext[2], ext[3])
        if v == 0:
            ax.set_aspect("equal")
        ax.set_xlabel("along track (km)", color=MUTED, fontsize=9)
        ax.set_ylabel("cross track (km)" if v == 0 else "altitude rel. start (km)", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title("Top view" if v == 0 else "Side view (vertical stretched)", loc="left", color=INK, fontsize=11)
    leg_ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    leg_ax.legend(h, l, loc="upper left", ncol=1, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle(f"Environment {int(c['env_index'][e])} ({c['split'][e]}): {len(paths)} predicted routes vs the solver",
                 x=0.125, ha="left", fontsize=13, color=INK)
    fig.text(0.125, 0.005, f"Obstacles sliced at the optimal route's altitude (top) and lateral position (side). "
             f"Dark gray: obstacles; light gray: inflated by {CLEARANCE_M:.0f} m. Circle = start, square = goal. "
             f"Length is relative to the optimal route.", fontsize=8, color=MUTED)
    out = out_dir / f"env_{int(c['env_index'][e]):05d}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"  env {int(c['env_index'][e]):5d}: top-1 length {route_length(paths[top]) / L_best:.3f}x, "
          f"{inside[top]:.0f}% inside, {err[top]:.0f} m from optimum; best of {len(paths)}: "
          f"{min(err):.0f} m  -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot predicted routes against the solver's optimum.")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--n", type=int, default=N_ENVS)
    ap.add_argument("--env", type=int, nargs="*", default=None, help="environment index(es) to plot")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--show", action="store_true", help="open each plot in a window as well")
    a = ap.parse_args()

    cache = bc.load_cache(tw.cache_path(a.dataset, a.run))
    scale = cache["summary"]["length_scale_m"]
    model, ck = tw.load_model(_HERE / "models" / f"{a.model}.pt")
    if a.env:
        rows = [int(np.nonzero(cache["env_index"] == i)[0][0]) for i in a.env
                if np.any(cache["env_index"] == i)]
        if not rows:
            raise SystemExit("None of those environment indices are in the cache (only solved ones are).")
    else:
        pool = np.nonzero(cache["split"] == a.split)[0]
        rows = np.random.default_rng(a.seed).choice(pool, size=min(a.n, len(pool)), replace=False)
    out_dir = _HERE / "models" / "plots" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {a.model} (epoch {ck.get('epoch')}), {len(rows)} environments:")
    for e in rows:
        plot_env(model, cache, int(e), scale, out_dir, a.show)


if __name__ == "__main__":
    main()