"""
Train the learned warm start: obstacles + start/goal  ->  K candidate routes + flight times.

Reads the cache written by build_cache.py (normalized frame: start at the origin, goal
along +x, lengths in turn radii). The network:

    obstacle tokens (17 + Fourier(centre))  ┐
    context (12 + Fourier(goal))            ├─ transformer encoder ─┬─ K route queries (decoder)
                                            ┘                        │      │  confidence, flight time
                                                                     └─ 64 point queries per route
                                                                        (decoder: each point attends
                                                                         to the obstacles near it)
                                                                              │
                                                                        offset of each path point

Several routes are predicted because environments often have more than one valid
answer (left or right of an obstacle). Training uses a winner-takes-all loss: every
label is pulled toward the prediction closest to it, so different queries specialize
in different routes instead of averaging them into an obstacle.

Path points are offsets from the straight start-goal line, scaled by 4 tau (1 - tau),
so every predicted path starts exactly at the start and ends exactly at the goal.

Losses: route shape (winner-takes-all), an obstacle penalty (predicted points inside an
inflated obstacle, computed from the obstacle tokens), flight time, and which route is
best (confidence). An exponential moving average (EMA) of the weights is evaluated and
saved, which smooths out epoch-to-epoch noise.

Output: supervised-learning-main/models/<NAME>.pt  (weights, settings, normalization)
        supervised-learning-main/models/<NAME>_log.csv  (per-epoch metrics)

Run (from the fixed-wing-rl folder):
    modal run supervised-learning-main/train_warmstart.py --dataset c172_envs_v1 --run birkhoff_v1
    python supervised-learning-main/train_warmstart.py --dataset c172_envs_v1 --run birkhoff_v1   (this computer)
    add --epochs 5 for a quick check, --name to keep several models
On Modal the cache is uploaded to the Volume (skipped if already there), training runs on a
GPU, and the model and log are copied back to supervised-learning-main/models/ (they are also
kept on the Volume under /models).
"""

# ==========================================================================
# USER SETTINGS
# ==========================================================================
DATASET       = "c172_envs_v1"
RUN           = "birkhoff_v1"        # which cache (= solver run) to train on
NAME          = "warmstart_v2"       # model name (file in supervised-learning-main/models)
DATASETS_DIR  = None                 # None = supervised-learning-main/datasets

# ---- model
K_ROUTES      = 5                    # candidate routes predicted per environment
D_MODEL       = 128
N_HEADS       = 4
ENC_LAYERS    = 4                    # obstacle/context encoder
DEC_LAYERS    = 2                    # route decoder (K queries)
POINT_LAYERS  = 2                    # point decoder (64 queries per route, attend to obstacles)
N_FREQ        = 10                   # Fourier frequencies per coordinate
FOURIER_WL    = (0.1, 64.0)          # shortest / longest wavelength, turn radii (~40 m ... 27 km)
DROPOUT       = 0.1

# ---- training
EPOCHS        = 150
BATCH         = 32
LR            = 3e-4
WARMUP        = 0.1                  # fraction of training spent warming the learning rate up
WEIGHT_DECAY  = 1e-4
EMA_DECAY     = 0.999                # averaged weights used for evaluation and saving (0 = off)
MIRROR_AUG    = True                 # randomly mirror left/right (a mirrored solution is still valid)
TRAIN_EVAL_ENVS = 600                # also measure error on this many training envs (spots underfitting)

# ---- loss
BEST_WEIGHT   = 2.0                  # extra weight on each environment's best-cost route
SPREAD_WEIGHT = 0.05                 # small pull of every query toward the best route (keeps all queries useful)
OBS_WEIGHT    = 5.0                  # penalty on predicted points inside an inflated obstacle (0 = off)
CLEARANCE_M   = 25.0                 # obstacles are inflated by this in that penalty, m. Below the solver's
                                     #   36 m because its own routes may dip up to 7.5 m into the margin
                                     #   between check points; at 25 m no solver route counts as "inside"
SIGMA_WEIGHT  = 0.1                  # flight-time loss, on ((pred - true) / SIGMA_SCALE)^2
SIGMA_SCALE   = 0.02                 #   (flight time / straight-line time typically varies by ~0.02-0.05)
CONF_WEIGHT   = 0.2                  # "which route is best" classification loss
SEED          = 0
DEVICE        = None                 # None = cuda if available, else cpu

# Modal (`modal run supervised-learning-main/train_warmstart.py ...`)
MODAL_APP_NAME = "fixed-wing-train"
MODAL_VOLUME   = "fixed-wing-rl-data" # the cache is uploaded here; models are saved here too
GPU            = "L4"                # e.g. "T4" (cheapest), "L4", "A10G", "A100"
TRAIN_TIMEOUT_S = 8 * 3600
PYTHON_VERSION = "3.12"
PIP_PACKAGES   = ["numpy", "torch"]

# ==========================================================================
import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent
for p in (str(PROJECT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_cache as bc                               # noqa: E402

N_PTS = 64


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class Routes:
    """The cache, split by environment, with batching and mirror augmentation."""

    def __init__(self, cache, split, limit=None):
        self.c = cache
        self.envs = np.nonzero(cache["split"] == split)[0][:limit]
        self.max_labels = int(np.max(np.diff(cache["lab_off"]))) if len(cache["lab_off"]) > 1 else 1

    def __len__(self):
        return len(self.envs)

    def batches(self, batch, shuffle, rng, mirror):
        order = rng.permutation(self.envs) if shuffle else self.envs
        for i in range(0, len(order), batch):
            yield self.collate(order[i:i + batch], rng if mirror else None)

    def collate(self, rows, rng=None):
        c = self.c
        B = len(rows)
        n_tok = c["tok_off"][rows + 1] - c["tok_off"][rows]
        T = int(max(n_tok.max(), 1))
        L = self.max_labels
        tok = np.zeros((B, T, bc.TOKEN_DIM), np.float32)
        tok_mask = np.ones((B, T), bool)                    # True = padding
        lab_p = np.zeros((B, L, N_PTS, 3), np.float32)
        lab_s = np.zeros((B, L), np.float32)
        lab_mask = np.zeros((B, L), bool)                   # True = a real label
        lab_best = np.zeros((B, L), bool)
        ctx = c["ctx"][rows].copy()
        for b, e in enumerate(rows):
            t0, t1 = c["tok_off"][e], c["tok_off"][e + 1]
            tok[b, :t1 - t0] = c["tok"][t0:t1]
            tok_mask[b, :t1 - t0] = False
            l0, l1 = c["lab_off"][e], c["lab_off"][e + 1]
            n = min(l1 - l0, L)
            lab_p[b, :n] = c["lab_x"][l0:l0 + n, :, :3]
            lab_s[b, :n] = c["lab_sigma_n"][l0:l0 + n]
            lab_mask[b, :n] = True
            lab_best[b, :n] = c["lab_best"][l0:l0 + n]
        if rng is not None:
            flip = rng.random(B) < 0.5
            if flip.any():
                tok[flip], ctx[flip], lab_p[flip] = mirror(tok[flip], ctx[flip], lab_p[flip])
        t = lambda a: torch.from_numpy(a)
        return {"tok": t(tok), "tok_mask": t(tok_mask), "ctx": t(ctx), "lab_p": t(lab_p),
                "lab_s": t(lab_s), "lab_mask": t(lab_mask), "lab_best": t(lab_best)}


def mirror(tok, ctx, lab_p):
    """Reflect y -> -y. Obstacles: centre y and both orientation axes' y flip. Context:
    goal y, start heading and the lateral bounds flip. Paths: y flips."""
    tok, ctx, lab_p = tok.copy(), ctx.copy(), lab_p.copy()
    tok[..., 5] *= -1                                    # centre y
    tok[..., 11] *= -1                                   # R[:,0] y
    tok[..., 14] *= -1                                   # axis y
    ctx[:, 1] *= -1                                      # goal y
    ctx[:, 5] *= -1                                      # start heading
    ctx[:, [8, 9]] = -ctx[:, [9, 8]]                     # y_lo, y_hi
    lab_p[..., 1] *= -1
    return tok, ctx, lab_p


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Fourier(nn.Module):
    """sin/cos of coordinates at geometrically spaced wavelengths: lets the network
    reason about positions at every scale from tens of metres to the whole route."""

    def __init__(self, n_in, n_freq=N_FREQ, wl=FOURIER_WL):
        super().__init__()
        wavelengths = torch.logspace(math.log10(wl[1]), math.log10(wl[0]), n_freq)
        self.register_buffer("f", 2 * math.pi / wavelengths)
        self.dim = 2 * n_in * n_freq

    def forward(self, x):
        z = x[..., None] * self.f                        # (..., n_in, n_freq)
        return torch.cat([torch.sin(z), torch.cos(z)], -1).flatten(-2)


class WarmStartNet(nn.Module):
    def __init__(self, k=K_ROUTES, d=D_MODEL, heads=N_HEADS, enc_layers=ENC_LAYERS,
                 dec_layers=DEC_LAYERS, point_layers=POINT_LAYERS, dropout=DROPOUT,
                 n_pts=N_PTS, n_freq=N_FREQ, wl=FOURIER_WL):
        super().__init__()
        self.k, self.n_pts = k, n_pts
        self.ff = Fourier(3, n_freq, wl)
        self.ft = Fourier(1, n_freq, (1 / 64, 2.0))      # along-route parameter tau in [0, 1]
        mlp = lambda i: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))
        self.tok_in = mlp(bc.TOKEN_DIM + self.ff.dim)
        self.ctx_in = mlp(bc.CTX_DIM + self.ff.dim)
        self.pt_in = mlp(self.ff.dim + self.ft.dim)
        layer = lambda kind: kind(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), enc_layers,
                                             enable_nested_tensor=False)
        self.route_dec = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), dec_layers)
        self.point_dec = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), point_layers)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.route_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))   # flight time, confidence
        self.point_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))   # offset of each point
        nn.init.zeros_(self.point_head[-1].weight)
        nn.init.zeros_(self.point_head[-1].bias)                                        # start from the straight line
        tau = torch.linspace(0, 1, n_pts)
        self.register_buffer("tau", tau)
        self.register_buffer("bump", 4 * tau * (1 - tau))

    def forward(self, tok, tok_mask, ctx):
        """-> paths (B, K, 64, 3) in the normalized frame, sigma_n (B, K), logits (B, K)."""
        B, K, P = tok.shape[0], self.k, self.n_pts
        goal = ctx[:, :3]
        c = self.ctx_in(torch.cat([ctx, self.ff(goal)], -1)).unsqueeze(1)
        x = torch.cat([c, self.tok_in(torch.cat([tok, self.ff(tok[..., 4:7])], -1))], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tok.device), tok_mask], 1)
        mem = self.encoder(x, src_key_padding_mask=mask)
        # K route queries
        r = self.route_dec(self.queries.unsqueeze(0).expand(B, -1, -1) + c, mem,
                           memory_key_padding_mask=mask)                         # (B, K, d)
        out = self.route_head(r)
        # 64 point queries per route: where the point would be on the straight line, which point it is,
        # and which route it belongs to; each attends to the obstacle memory
        line = self.tau[None, :, None] * goal[:, None, :]                         # (B, P, 3)
        pq = self.pt_in(torch.cat([self.ff(line), self.ft(self.tau[:, None]).expand(B, -1, -1)], -1))
        q = (pq[:, None] + r[:, :, None]).reshape(B * K, P, -1)
        mem_k = mem.repeat_interleave(K, 0)
        mask_k = mask.repeat_interleave(K, 0)
        h = self.point_dec(q, mem_k, memory_key_padding_mask=mask_k).reshape(B, K, P, -1)
        dp = self.point_head(h)
        paths = line[:, None] + self.bump[None, None, :, None] * dp
        sigma_n = 1.0 + F.softplus(out[..., 0] - 2.0)                           # flight time / straight-line time
        return paths, sigma_n, out[..., 1]


class WarmStartNetV1(nn.Module):
    """The first architecture (one MLP head per route). Kept so v1 models still load."""
    def __init__(self, k=5, d=128, heads=4, enc_layers=4, dec_layers=2, dropout=0.1, n_pts=64):
        super().__init__()
        self.k, self.n_pts = k, n_pts
        mlp = lambda i: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))
        self.tok_in = mlp(bc.TOKEN_DIM)
        self.ctx_in = mlp(bc.CTX_DIM)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, n_pts * 3 + 2))
        tau = torch.linspace(0, 1, n_pts)
        self.register_buffer("tau", tau)
        self.register_buffer("bump", 4 * tau * (1 - tau))

    def forward(self, tok, tok_mask, ctx):
        """-> paths (B, K, 64, 3) in the normalized frame, sigma_n (B, K), logits (B, K)."""
        B = tok.shape[0]
        c = self.ctx_in(ctx).unsqueeze(1)                              # context as an extra token
        x = torch.cat([c, self.tok_in(tok)], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tok.device), tok_mask], 1)
        mem = self.encoder(x, src_key_padding_mask=mask)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + c
        h = self.decoder(q, mem, memory_key_padding_mask=mask)
        out = self.head(h)
        dp = out[..., :self.n_pts * 3].reshape(B, self.k, self.n_pts, 3)
        goal = ctx[:, :3]
        line = self.tau[None, None, :, None] * goal[:, None, None, :]
        paths = line + self.bump[None, None, :, None] * dp
        sigma_n = 1.0 + F.softplus(out[..., -2]) * 0.5                 # flight time / straight-line time, > 1
        return paths, sigma_n, out[..., -1]


# --------------------------------------------------------------------------
# obstacle penalty (differentiable, from the tokens)
# --------------------------------------------------------------------------
def obstacle_h(p, tok, tok_mask, d):
    """Signed clearance of points p (B, K, P, 3) from every obstacle token (B, T, 17),
    inflated by d (normalized units). Same formulas as envgen; negative = inside.
    Returns (B, K, P, T), with +large for padding tokens."""
    c, s = tok[..., 4:7], tok[..., 7:10].clamp_min(1e-4)
    r0, a = tok[..., 10:13], tok[..., 13:16]
    R = torch.stack([r0, torch.cross(a, r0, dim=-1), a], -1)                   # (B, T, 3, 3), columns
    q = torch.einsum("bkpi,btij->bkptj", p, R) - torch.einsum("bti,btij->btj", c, R)[:, None, None]
    sph, cyl, box = tok[..., 0], tok[..., 1], tok[..., 2]
    smin = s.min(-1).values
    h_sph = ((q / s[:, None, None]).norm(dim=-1) - (1 + d / smin[:, None, None])) * smin[:, None, None]
    h_cyl = torch.maximum(q[..., :2].norm(dim=-1) - (s[..., 0] + d)[:, None, None],
                          q[..., 2].abs() - (s[..., 2] + d)[:, None, None])
    h_box = (q.abs() - (s + d)[:, None, None]).max(-1).values
    h = (sph[:, None, None] * h_sph + cyl[:, None, None] * h_cyl + box[:, None, None] * h_box)
    return h.masked_fill(tok_mask[:, None, None], 1e3)


def obstacle_penalty(paths, tok, tok_mask, d):
    """Mean intrusion depth into inflated obstacles over all predicted points (normalized units)."""
    inside = F.relu(-obstacle_h(paths, tok, tok_mask, d))
    return inside.sum(-1).mean()


# --------------------------------------------------------------------------
# loss and metrics
# --------------------------------------------------------------------------
def route_cost(pred, lab):
    """Mean squared distance between every prediction and every label: (B, K, L)."""
    return ((pred[:, :, None] - lab[:, None]) ** 2).sum(-1).mean(-1)


def loss_fn(out, batch, d_clear):
    paths, sig, logit = out
    C = route_cost(paths, batch["lab_p"])                              # (B, K, L)
    m = batch["lab_mask"].float()
    w = m * (1.0 + (BEST_WEIGHT - 1.0) * batch["lab_best"].float())
    C_masked = C + (1 - m[:, None]) * 1e6
    best_k = C_masked.argmin(1)                                        # (B, L) nearest prediction per label
    c_min = C.gather(1, best_k[:, None]).squeeze(1)
    pos = (w * c_min).sum() / w.sum()
    # every query is pulled a little toward the best route, so no query is left unused
    best_l = batch["lab_best"].float().argmax(1)
    spread = C.gather(2, best_l[:, None, None].expand(-1, C.shape[1], 1)).mean()
    # predicted points must stay out of the inflated obstacles
    obs = obstacle_penalty(paths, batch["tok"], batch["tok_mask"], d_clear) if OBS_WEIGHT else paths.new_zeros(())
    # flight time of the matched prediction for each label
    s_pred = sig.gather(1, best_k)
    s_loss = (w * ((s_pred - batch["lab_s"]) / SIGMA_SCALE) ** 2).sum() / w.sum()
    # confidence: which query matches the best route
    target = best_k.gather(1, best_l[:, None]).squeeze(1)
    conf = F.cross_entropy(logit, target)
    total = (pos + SPREAD_WEIGHT * spread + OBS_WEIGHT * obs + SIGMA_WEIGHT * s_loss
             + CONF_WEIGHT * conf)
    return total, {"pos": pos.item(), "obs": obs.item(), "sigma": s_loss.item(), "conf": conf.item()}


@torch.no_grad()
def evaluate(model, data, scale, device, d_clear, batch=32):
    """minADE to the best route (m), top-1 error (m), coverage of all routes (m),
    flight-time error (%), confidence accuracy, and how many predicted points of the
    top-1 route are inside an inflated obstacle."""
    model.eval()
    rng = np.random.default_rng(0)
    best_err, all_err, top1_err, sig_err, inside, conf_hit, n_env = [], [], [], [], [], 0, 0
    for b in data.batches(batch, False, rng, mirror=False):
        b = {k: v.to(device) for k, v in b.items()}
        paths, sig, logit = model(b["tok"], b["tok_mask"], b["ctx"])
        d = (paths[:, :, None] - b["lab_p"][:, None]).norm(dim=-1).mean(-1) * scale   # (B, K, L) metres
        best_l = b["lab_best"].float().argmax(1)
        d_best = d.gather(2, best_l[:, None, None].expand(-1, d.shape[1], 1)).squeeze(2)   # (B, K)
        best_err += d_best.min(1).values.tolist()
        top = logit.argmax(1)
        top1_err += d_best.gather(1, top[:, None]).squeeze(1).tolist()
        conf_hit += int((top == d_best.argmin(1)).sum())
        dl = (d + (~b["lab_mask"])[:, None] * 1e9).min(1).values                    # (B, L)
        all_err += dl[b["lab_mask"]].tolist()
        k_best = d_best.argmin(1)
        s_true = b["lab_s"].gather(1, best_l[:, None]).squeeze(1)
        s_hat = sig.gather(1, k_best[:, None]).squeeze(1)
        sig_err += ((s_hat - s_true).abs() / s_true * 100).tolist()
        top_path = paths[torch.arange(len(top)), top][:, None]                       # (B, 1, P, 3)
        h = obstacle_h(top_path, b["tok"], b["tok_mask"], d_clear).min(-1).values    # (B, 1, P)
        inside += (h < 0).float().mean((1, 2)).tolist()
        n_env += len(best_l)
    model.train()
    return {"minADE_best_m": float(np.mean(best_err)), "top1_ADE_best_m": float(np.mean(top1_err)),
            "minADE_all_m": float(np.mean(all_err)), "sigma_err_pct": float(np.mean(sig_err)),
            "conf_acc": conf_hit / max(n_env, 1), "top1_inside_pct": 100 * float(np.mean(inside))}


# --------------------------------------------------------------------------
# prediction (used by the solver integration)
# --------------------------------------------------------------------------
def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = WarmStartNet if "point_layers" in ck["model_args"] else WarmStartNetV1   # v1 models still load
    model = arch(**ck["model_args"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict(model, ck, env, start_state, device="cpu"):
    """K candidate routes for an Environment, in WORLD metres, ordered by confidence.
    Returns (paths (K, 64, 3), flight_times_s (K,), confidence (K,))."""
    scale, vc = ck["length_scale_m"], ck["v_cruise"]
    origin, yaw = bc.frame_of(env.start_pos, env.goal_pos)
    tok = torch.from_numpy(bc.tokens_for(env, origin, yaw, scale))[None].to(device)
    ctx = torch.from_numpy(bc.context_for(env, np.asarray(start_state), origin, yaw, scale, vc))[None].to(device)
    mask = torch.zeros(tok.shape[:2], dtype=torch.bool, device=device)
    paths, sig_n, logit = model(tok, mask, ctx)
    order = torch.argsort(logit[0], descending=True)
    p = bc.to_world(paths[0, order].cpu().numpy(), origin, yaw, scale)
    dist = np.linalg.norm(env.goal_pos - env.start_pos)
    sigma = sig_n[0, order].cpu().numpy() * dist / vc
    return p, sigma, torch.softmax(logit[0, order], 0).cpu().numpy()


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def cache_path(dataset, run):
    root = Path(DATASETS_DIR) if DATASETS_DIR else _HERE / "datasets"
    return root / dataset / "cache" / f"{run}.npz"


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)


def main(dataset, run, name, epochs, device, cache_file=None, out_dir=None, on_save=None):
    """Train and save the best model. `on_save(ck_path, log_path)` is called after every
    improvement (used on Modal to commit the Volume, so a crash keeps the best model)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cache = bc.load_cache(cache_file or cache_path(dataset, run))
    scale = cache["summary"]["length_scale_m"]
    d_clear = CLEARANCE_M / scale
    train, val = Routes(cache, "train"), Routes(cache, "val")
    train_eval = Routes(cache, "train", limit=TRAIN_EVAL_ENVS)
    if len(train) == 0:
        raise SystemExit("No training environments in the cache.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_args = dict(k=K_ROUTES, d=D_MODEL, heads=N_HEADS, enc_layers=ENC_LAYERS,
                      dec_layers=DEC_LAYERS, point_layers=POINT_LAYERS, dropout=DROPOUT,
                      n_pts=N_PTS, n_freq=N_FREQ, wl=FOURIER_WL)
    model = WarmStartNet(**model_args).to(device)
    ema = EMA(model, EMA_DECAY) if EMA_DECAY else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(epochs * math.ceil(len(train) / BATCH), 20)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=min(max(WARMUP, 3 / steps), 0.5))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Training {name}: {len(train)} train / {len(val)} val environments, "
          f"{int(cache['summary']['labels'])} routes, K={K_ROUTES}, {n_par / 1e6:.2f} M parameters, {device}")

    out_dir = Path(out_dir) if out_dir else _HERE / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path, log_path = out_dir / f"{name}.pt", out_dir / f"{name}_log.csv"
    best, t0 = float("inf"), time.time()
    with open(log_path, "w", newline="") as flog:
        log = None
        for ep in range(1, epochs + 1):
            tot, parts, nb = 0.0, {"pos": 0.0, "obs": 0.0, "sigma": 0.0, "conf": 0.0}, 0
            for b in train.batches(BATCH, True, rng, mirror=MIRROR_AUG):
                b = {k: v.to(device) for k, v in b.items()}
                loss, info = loss_fn(model(b["tok"], b["tok_mask"], b["ctx"]), b, d_clear)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if ema is not None:
                    ema.update(model)
                tot += loss.item(); nb += 1
                for k in parts: parts[k] += info[k]
            net = ema.model if ema is not None else model
            row = {"epoch": ep, "loss": tot / nb, **{k: v / nb for k, v in parts.items()},
                   "time_s": round(time.time() - t0, 1)}
            report = ep == 1 or ep % 5 == 0 or ep == epochs
            if len(val):
                row.update(evaluate(net, val, scale, device, d_clear))
            if report and len(train_eval):
                tr = evaluate(net, train_eval, scale, device, d_clear)
                row.update({"train_minADE_best_m": tr["minADE_best_m"], "train_top1_ADE_best_m": tr["top1_ADE_best_m"]})
            if log is None:
                fields = list(row) + [f for f in ("train_minADE_best_m", "train_top1_ADE_best_m") if f not in row]
                log = csv.DictWriter(flog, fieldnames=fields, restval="")
                log.writeheader()
            log.writerow(row); flog.flush()
            score = row.get("minADE_best_m", row["loss"])
            if score < best:
                best = score
                torch.save({"state": net.state_dict(), "model_args": model_args,
                            "length_scale_m": scale, "v_cruise": cache["summary"]["v_cruise"],
                            "dataset": dataset, "run": run, "epoch": ep, "val": row,
                            "settings": {k: v for k, v in globals().items() if k.isupper()
                                         and isinstance(v, (int, float, str, bool, tuple, type(None)))}}, ck_path)
                if on_save is not None:
                    on_save(ck_path, log_path)
            if report:
                msg = f"  epoch {ep:4d}  loss {row['loss']:.4f}"
                if "minADE_best_m" in row:
                    msg += (f"  val: best-route {row['minADE_best_m']:6.1f} m (top-1 {row['top1_ADE_best_m']:6.1f} m), "
                            f"all routes {row['minADE_all_m']:6.1f} m, flight time {row['sigma_err_pct']:4.1f}%, "
                            f"confidence {row['conf_acc']:.0%}, top-1 inside obstacles {row['top1_inside_pct']:4.1f}%")
                if "train_minADE_best_m" in row:
                    msg += f" | train best-route {row['train_minADE_best_m']:6.1f} m"
                print(msg + f"  [{row['time_s']:.0f} s]", flush=True)
    print(f"\nBest model -> {ck_path}  (val best-route error {best:.1f} m)\nLog -> {log_path}")
    return ck_path, log_path


# --------------------------------------------------------------------------
# Modal
# --------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    VOLUME_MOUNT = "/data"
    app = modal.App(MODAL_APP_NAME)
    image = (modal.Image.debian_slim(python_version=PYTHON_VERSION)
             .pip_install(*PIP_PACKAGES)
             .add_local_file(str(_HERE / "build_cache.py"), "/root/build_cache.py"))
    volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)

    @app.function(image=image, gpu=GPU, timeout=TRAIN_TIMEOUT_S, volumes={VOLUME_MOUNT: volume})
    def train_remote(dataset, run, name, epochs):
        import torch as _t
        print(f"GPU: {_t.cuda.get_device_name(0) if _t.cuda.is_available() else 'none (CPU)'}", flush=True)
        volume.reload()
        cache_file = Path(VOLUME_MOUNT) / "datasets" / dataset / "cache" / f"{run}.npz"
        out_dir = Path(VOLUME_MOUNT) / "models"
        state = {"last": 0.0}

        def commit(ck, log):                    # keep the best model safe, at most once a minute
            if time.time() - state["last"] > 60:
                volume.commit()
                state["last"] = time.time()

        ck, log = main(dataset, run, name, epochs, None, cache_file=cache_file,
                       out_dir=out_dir, on_save=commit)
        volume.commit()
        return Path(ck).read_bytes(), Path(log).read_bytes()

    @app.local_entrypoint()
    def modal_main(dataset: str = DATASET, run: str = RUN, name: str = NAME, epochs: int = EPOCHS):
        local = cache_path(dataset, run)
        if not local.exists():
            raise SystemExit(f"Cache not found: {local}\nRun build_cache.py first.")
        remote = f"/datasets/{dataset}/cache/{run}.npz"
        size = local.stat().st_size
        try:
            have = {e.path.lstrip("/"): e.size for e in volume.listdir(f"/datasets/{dataset}/cache")}
        except Exception:
            have = {}
        if have.get(remote.lstrip("/")) == size:
            print(f"Cache already on Volume '{MODAL_VOLUME}' ({size / 1e6:.0f} MB), skipping upload.")
        else:
            print(f"Uploading cache to Volume '{MODAL_VOLUME}' ({size / 1e6:.0f} MB)...", flush=True)
            with volume.batch_upload(force=True) as batch:
                batch.put_file(str(local), remote)
                batch.put_file(str(local.with_suffix(".json")), remote[:-4] + ".json")
        print(f"Training on Modal ({GPU})...", flush=True)
        ck_bytes, log_bytes = train_remote.remote(dataset, run, name, epochs)
        out = _HERE / "models"
        out.mkdir(exist_ok=True)
        (out / f"{name}.pt").write_bytes(ck_bytes)
        (out / f"{name}_log.csv").write_bytes(log_bytes)
        print(f"Saved {out / (name + '.pt')} and {out / (name + '_log.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the learned warm start.")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()
    main(a.dataset, a.run, a.name, a.epochs, a.device)