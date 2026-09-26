#!/usr/bin/env python3
"""
envgen.py - Procedural 3D obstacle environments for trajectory optimization.

Generates bounded box workspaces filled with obstacles, plus a start and goal
state. It does NOT check whether a solution exists; that is left to a
downstream solver (e.g. SCP), which accepts or rejects each environment.

--------------------------------------------------------------------------
Mathematical description of every constraint an environment defines
--------------------------------------------------------------------------
Let p = (x, y, z) be the vehicle position at any trajectory node, r the
vehicle radius, and d = r + margin the required clearance.

(1) Workspace bounds (linear):
        lo_k + r  <=  p_k  <=  hi_k - r,            k in {x, y, z}

(2) Boundary conditions (linear equalities):
        p(0) = p_start,   v(0) = v_start,   p(T) = p_goal,   v(T) = 0

(3) Obstacle keep-out constraints, one per convex obstacle piece:
        h_i(p; d) >= 0
    Spheroid  (center c, semi-axes a, rotation R, a_min = min(a)):
        h = || diag(1/a) R^T (p - c) ||_2  -  (1 + d / a_min)
    Cylinder  (vertical, standing on the floor; axis center c_xy, radius rho, top z_top):
        h = max( ||(x, y) - c_xy||_2 - (rho + d),  z - (z_top + d) )
        full-height cylinders (z_top = ceiling) drop the second term:
        h = ||(x, y) - c_xy||_2 - (rho + d)
    Box       (axis-aligned; center c, half-extents e):
        h = max_k ( |p_k - c_k| - (e_k + d) )
    Wall      (slab with a rectangular opening): the union of up to four boxes,
              so it contributes one box constraint per piece.

Properties (why these forms are used):
  * Conservative: h_i(p; d) >= 0 guarantees the vehicle's sphere of radius d
    does not touch the obstacle. (Spheroid: the ellipsoid scaled by
    1 + d/a_min contains the obstacle grown by d. Cylinder/box: the max-form
    keeps square edges, which contain the rounded grown shape.)
  * Convex: every h_i is convex in p (a norm of an affine map, or a max of
    convex functions). So its first-order linearization about any p_ref,
        h_i(p_ref) + g_i(p_ref) . (p - p_ref) >= 0,
    is a half-space that implies h_i(p) >= 0. Each SCP iteration can use
    these half-spaces directly; they never admit a colliding point.
  * h_i = 0 with d = 0 is exactly the obstacle surface, so h_i(p; 0) < 0
    exactly means "inside", which is how filled volume is measured.
  * Cylinder and box constraints are non-smooth at edges (the max switches
    branch); g_i is a valid subgradient there.
Actuator limits and vehicle dynamics are properties of the vehicle model,
not the environment, and belong in the SCP formulation.

--------------------------------------------------------------------------
Generation procedure
--------------------------------------------------------------------------
  1. Sample start and goal positions (minimum separation enforced).
  2. Draw a fill target uniformly from OCCUPANCY_RANGE.
  3. Propose obstacles of random type, log-uniform size, uniform position.
     Keep one only if start and goal stay clear of it (with extra margin) and
     it does not overshoot the fill target. Filled volume is a union over
     Monte Carlo points, so overlaps are not double-counted. Proposed sizes
     shrink when placements keep failing, so small obstacles fill the gaps.
  4. Optionally reject environments whose straight start-goal line is free.

Usage
    Edit the USER SETTINGS block below, then run:
        python envgen.py
    or override from the command line:
        python envgen.py --size 10 10 5 --n 100 --seed 0 --out envs.json --plot
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np


# ==========================================================================
# USER SETTINGS - edit these. Everything below this block uses them.
# (Command-line flags, where available, override the matching setting.)
# ==========================================================================

# ---- run / output --------------------------------------------------------
N_ENVIRONMENTS     = 10             # how many environments to generate
SEED               = None           # int for reproducible output, None for random
OUTPUT_FILE        = "envs.json"
PLOT_FIRST         = False          # show a 3D plot of the first environment

# ---- workspace and vehicle -----------------------------------------------
WORKSPACE_SIZE     = (10.0, 10.0, 5.0)   # box extent (Lx, Ly, Lz)
WORKSPACE_ORIGIN   = (0.0, 0.0, 0.0)     # lower corner of the box
VEHICLE_RADIUS     = 0.25
SAFETY_MARGIN      = 0.10                # extra clearance beyond the vehicle radius

# ---- fill target (fraction of box volume, drawn uniformly per environment) -
OCCUPANCY_RANGE     = (0.20, 0.60)
OCCUPANCY_TOLERANCE = 0.01          # max allowed overshoot of the target
OCCUPANCY_SAMPLES   = 20000         # Monte Carlo points for estimating fill

# ---- obstacle types (relative probability of proposing each) -------------
TYPE_WEIGHTS = {
    "spheroid": 0.5,
    "cylinder": 0.5,
    # "box": 0.3,                   # <- uncomment to enable boxes
    # "wall": 0.05,                 # <- uncomment to enable walls
}

# ---- obstacle sizes (fractions of the smallest box dimension; log-uniform) -
SPHEROID_RADIUS    = (0.03, 0.30)   # equatorial semi-axis
SPHEROID_ASPECT    = (0.5, 2.0)     # polar / equatorial ratio
CYLINDER_RADIUS    = (0.03, 0.20)
CYLINDER_HEIGHT    = (0.3, 1.0)     # fraction of box height (Lz), partial-height cylinders
CYLINDER_FULL_HEIGHT_PROB = 0.5     # chance a cylinder spans floor to ceiling (smooth constraint)
BOX_HALF_EXTENT    = (0.03, 0.20)
WALL_THICKNESS     = (0.02, 0.05)
WALL_OPENING       = (0.10, 0.30)   # extra opening beyond the minimum passable width

# ---- size shrinking when the box gets crowded -----------------------------
SHRINK_AFTER       = 30             # consecutive failed placements before shrinking
SHRINK_FACTOR      = 0.8
MIN_SIZE_SCALE     = 0.15           # never shrink below this fraction of the base size

# ---- start / goal ---------------------------------------------------------
ENDPOINT_MODE        = "random"     # "random": anywhere in the box
                                    # "ends":   start near one end of the longest axis,
                                    #           goal near the other end (corridor flights)
END_ZONE_FRAC        = 0.08         # "ends" mode: depth of each end zone / box length
START_GOAL_MIN_FRAC  = 0.5          # "random" mode: min start-goal distance / inner-box diagonal
ENDPOINT_CLEARANCE   = 2.0          # start/goal clearance, multiples of (radius + margin)
START_SPEED_MAX      = 0.5          # initial speed upper bound (goal is at rest)
REQUIRE_BLOCKED_LINE = True         # reject envs whose straight start-goal line is free
LINE_CHECK_SPACING   = 0.25         # blocked-line test sample spacing, fraction of (radius + margin)

# ---- retry limits -----------------------------------------------------------
MAX_PROPOSALS      = 20000          # obstacle proposals per environment attempt
MAX_ENV_TRIES      = 50             # environment attempts before giving up


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _random_rotation(rng):
    """Uniformly random 3x3 rotation matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _fmt(v):
    return "[" + ", ".join(f"{x:.3f}" for x in np.atleast_1d(v)) + "]"


# --------------------------------------------------------------------------
# obstacles
# --------------------------------------------------------------------------
class Obstacle:
    """Every obstacle defines a convex constraint function h(p; d), with
    h >= 0 meaning a sphere of radius d centered at p is clear of it.
    `p` may be a single point (3,) or a batch (N, 3)."""
    kind = "obstacle"
    n_features = 0

    def constraint(self, p, d=0.0):
        raise NotImplementedError

    def gradient(self, p, d=0.0):
        """(Sub)gradient of h with respect to p."""
        raise NotImplementedError

    def pieces(self):
        """Convex pieces, each contributing one constraint. Most obstacles
        are a single piece; walls are several boxes."""
        return [self]

    def formula(self, d):
        """The constraint written out with this obstacle's numbers."""
        raise NotImplementedError

    def features(self):
        """Fixed-length parameter vector (for learning)."""
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError


@dataclass
class Spheroid(Obstacle):
    """Ellipsoid with semi-axes `radii` (two equal -> spheroid), rotated by R.
    h = ||diag(1/a) R^T (p - c)||_2 - (1 + d / a_min)"""
    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    kind = "spheroid"
    n_features = 15

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.radii = np.asarray(self.radii, float)
        self.rotation = np.asarray(self.rotation, float)

    def _body(self, p):
        q = (np.asarray(p, float) - self.center) @ self.rotation      # R^T (p - c)
        return q, np.linalg.norm(q / self.radii, axis=-1)

    def constraint(self, p, d=0.0):
        _, k = self._body(p)
        return k - (1.0 + d / self.radii.min())

    def gradient(self, p, d=0.0):
        q, k = self._body(p)
        k = np.maximum(np.asarray(k), 1e-9)
        return ((q / self.radii**2) @ self.rotation.T) / k[..., None]

    def formula(self, d):
        return (f"||diag(1/a) R^T (p - c)||_2 >= {1 + d / self.radii.min():.4f}   "
                f"c={_fmt(self.center)}, a={_fmt(self.radii)}, R=stored rotation")

    def features(self):
        return np.r_[self.center, self.radii, self.rotation.ravel()]

    def to_dict(self):
        return {"type": self.kind, "center": self.center.tolist(),
                "radii": self.radii.tolist(), "rotation": self.rotation.tolist()}


@dataclass
class Cylinder(Obstacle):
    """Vertical cylinder standing on the floor.
    h = max(||(x,y) - c_xy||_2 - (rho + d), z - (z_top + d)),
    or just the first term if full_height."""
    center_xy: np.ndarray
    radius: float
    z_top: float
    full_height: bool = False
    kind = "cylinder"
    n_features = 5

    def __post_init__(self):
        self.center_xy = np.asarray(self.center_xy, float)

    def _terms(self, p, d):
        p = np.asarray(p, float)
        v = p[..., :2] - self.center_xy
        n = np.linalg.norm(v, axis=-1)
        return v, n, n - (self.radius + d), p[..., 2] - (self.z_top + d)

    def constraint(self, p, d=0.0):
        _, _, radial, top = self._terms(p, d)
        return radial if self.full_height else np.maximum(radial, top)

    def gradient(self, p, d=0.0):
        v, n, radial, top = self._terms(p, d)
        g_rad = np.concatenate([v / np.maximum(n, 1e-9)[..., None], np.zeros_like(n)[..., None]], axis=-1)
        if self.full_height:
            return g_rad
        g_top = np.zeros_like(g_rad)
        g_top[..., 2] = 1.0
        return np.where((radial >= top)[..., None], g_rad, g_top)

    def formula(self, d):
        base = f"||(x, y) - c_xy||_2 - {self.radius + d:.4f}"
        loc = f"c_xy={_fmt(self.center_xy)}"
        if self.full_height:
            return f"{base} >= 0   {loc} (full height)"
        return f"max({base}, z - {self.z_top + d:.4f}) >= 0   {loc}"

    def features(self):
        return np.r_[self.center_xy, self.radius, self.z_top, float(self.full_height)]

    def to_dict(self):
        return {"type": self.kind, "center_xy": self.center_xy.tolist(),
                "radius": float(self.radius), "z_top": float(self.z_top),
                "full_height": bool(self.full_height)}


@dataclass
class Box(Obstacle):
    """Axis-aligned box.  h = max_k (|p_k - c_k| - (e_k + d))"""
    center: np.ndarray
    half_extents: np.ndarray
    kind = "box"
    n_features = 6

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.half_extents = np.asarray(self.half_extents, float)

    @classmethod
    def from_bounds(cls, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return cls(0.5 * (lo + hi), 0.5 * (hi - lo))

    def constraint(self, p, d=0.0):
        q = np.abs(np.asarray(p, float) - self.center) - (self.half_extents + d)
        return q.max(axis=-1)

    def gradient(self, p, d=0.0):
        diff = np.asarray(p, float) - self.center
        q = np.abs(diff) - (self.half_extents + d)
        k = np.argmax(q, axis=-1)
        g = np.zeros_like(diff)
        s = np.where(np.take_along_axis(diff, np.asarray(k)[..., None], -1) >= 0, 1.0, -1.0)
        np.put_along_axis(g, np.asarray(k)[..., None], s, axis=-1)
        return g

    def formula(self, d):
        return (f"max_k(|p_k - c_k| - (e_k + {d:.3f})) >= 0   "
                f"c={_fmt(self.center)}, e={_fmt(self.half_extents)}")

    def features(self):
        return np.r_[self.center, self.half_extents]

    def to_dict(self):
        return {"type": self.kind, "center": self.center.tolist(),
                "half_extents": self.half_extents.tolist()}


@dataclass
class Wall(Obstacle):
    """Slab normal to `axis` at `position`, spanning the whole workspace, with a
    rectangular opening. Represented as up to four boxes around the opening,
    each contributing its own box constraint. opening_center / opening_size
    are in the two in-plane axes, in increasing axis order."""
    axis: int
    position: float
    thickness: float
    opening_center: np.ndarray
    opening_size: np.ndarray
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    kind = "wall"
    n_features = 10

    def __post_init__(self):
        self.opening_center = np.asarray(self.opening_center, float)
        self.opening_size = np.asarray(self.opening_size, float)
        self.bounds_lo = np.asarray(self.bounds_lo, float)
        self.bounds_hi = np.asarray(self.bounds_hi, float)
        self.boxes = self._build_boxes()

    def _build_boxes(self):
        k = self.axis
        i, j = [a for a in range(3) if a != k]
        lo, hi = self.bounds_lo, self.bounds_hi
        gi0, gi1 = np.clip(self.opening_center[0] + np.array([-0.5, 0.5]) * self.opening_size[0], lo[i], hi[i])
        gj0, gj1 = np.clip(self.opening_center[1] + np.array([-0.5, 0.5]) * self.opening_size[1], lo[j], hi[j])
        spans = [((lo[i], gi0), (lo[j], hi[j])), ((gi1, hi[i]), (lo[j], hi[j])),
                 ((gi0, gi1), (lo[j], gj0)), ((gi0, gi1), (gj1, hi[j]))]
        boxes = []
        for (a0, a1), (b0, b1) in spans:
            if a1 - a0 < 1e-9 or b1 - b0 < 1e-9:
                continue
            blo, bhi = np.empty(3), np.empty(3)
            blo[k], bhi[k] = self.position - self.thickness / 2, self.position + self.thickness / 2
            blo[i], bhi[i], blo[j], bhi[j] = a0, a1, b0, b1
            boxes.append(Box.from_bounds(blo, bhi))
        return boxes

    def pieces(self):
        return list(self.boxes)

    def constraint(self, p, d=0.0):
        # clear of the wall  <=>  clear of every piece
        return np.min([b.constraint(p, d) for b in self.boxes], axis=0)

    def gradient(self, p, d=0.0):
        raise NotImplementedError("Use the individual pieces (wall.pieces()).")

    def formula(self, d):
        return "; ".join(b.formula(d) for b in self.boxes)

    def features(self):
        return np.r_[np.eye(3)[self.axis], self.position, self.thickness,
                     self.opening_center, self.opening_size]

    def to_dict(self):
        return {"type": self.kind, "axis": int(self.axis), "position": float(self.position),
                "thickness": float(self.thickness),
                "opening_center": self.opening_center.tolist(),
                "opening_size": self.opening_size.tolist(),
                "bounds_lo": self.bounds_lo.tolist(), "bounds_hi": self.bounds_hi.tolist()}


OBSTACLE_TYPES = {c.kind: c for c in (Spheroid, Cylinder, Box, Wall)}


def obstacle_from_dict(d):
    d = dict(d)
    return OBSTACLE_TYPES[d.pop("type")](**d)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
@dataclass
class Environment:
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    obstacles: list
    start_pos: np.ndarray
    start_vel: np.ndarray
    goal_pos: np.ndarray
    goal_vel: np.ndarray
    vehicle_radius: float
    margin: float
    metrics: dict = field(default_factory=dict)

    @property
    def clearance(self):
        """d = vehicle_radius + margin, the inflation used in every h_i."""
        return self.vehicle_radius + self.margin

    def pieces(self):
        """All convex constraint pieces, flattened (walls expand into boxes)."""
        return [q for o in self.obstacles for q in o.pieces()]

    def position_bounds(self):
        """(lower, upper) box bounds on the vehicle position: constraint (1)."""
        r = self.vehicle_radius
        return self.bounds_lo + r, self.bounds_hi - r

    def constraint_values(self, p, d=None):
        """h_i(p; d) for every piece. Shape (m,) for one point, (m, N) for N points."""
        d = self.clearance if d is None else d
        return np.array([q.constraint(p, d) for q in self.pieces()])

    def linearized_constraints(self, p_ref):
        """First-order (conservative) linearization of every obstacle constraint
        about reference positions p_ref, shape (N, 3). Returns A (N, m, 3) and
        b (N, m) such that the SCP subproblem enforces  A[n] @ p_n >= b[n]."""
        p_ref = np.atleast_2d(np.asarray(p_ref, float))
        d = self.clearance
        A = np.stack([q.gradient(p_ref, d) for q in self.pieces()], axis=1)       # (N, m, 3)
        h = np.stack([q.constraint(p_ref, d) for q in self.pieces()], axis=1)     # (N, m)
        b = np.einsum("nmk,nk->nm", A, p_ref) - h
        return A, b

    def describe(self):
        """Human-readable statement of the full constraint set."""
        lo, hi = self.position_bounds()
        d = self.clearance
        lines = [f"Clearance d = r + margin = {self.vehicle_radius:.3f} + {self.margin:.3f} = {d:.3f}",
                 "(1) Workspace bounds:"]
        lines += [f"      {lo[k]:.3f} <= {ax} <= {hi[k]:.3f}" for k, ax in enumerate("xyz")]
        lines += ["(2) Boundary conditions:",
                  f"      p(0) = {_fmt(self.start_pos)},  v(0) = {_fmt(self.start_vel)}",
                  f"      p(T) = {_fmt(self.goal_pos)},  v(T) = {_fmt(self.goal_vel)}",
                  f"(3) Obstacle constraints ({len(self.pieces())} convex pieces, all must hold):"]
        lines += [f"      [{i}] {o.kind}: {o.formula(d)}" for i, o in enumerate(self.obstacles)]
        return "\n".join(lines)

    def feature_vector(self, max_per_type=None):
        """Fixed-size vector: boundary conditions, bounds, then a fixed number of
        slots per obstacle type as [present, features...]. Dense environments can
        hold 100+ obstacles per type; occupancy_grid() is often a better input."""
        max_per_type = max_per_type or {"spheroid": 160, "cylinder": 160}
        parts = [self.start_pos, self.start_vel, self.goal_pos, self.goal_vel,
                 self.bounds_lo, self.bounds_hi]
        for kind, n_slots in max_per_type.items():
            feats = [o.features() for o in self.obstacles if o.kind == kind]
            if len(feats) > n_slots:
                raise ValueError(f"{len(feats)} {kind}s exceeds {n_slots} slots")
            dim = OBSTACLE_TYPES[kind].n_features
            for m in range(n_slots):
                parts.append(np.r_[1.0, feats[m]] if m < len(feats) else np.zeros(dim + 1))
        return np.concatenate(parts)

    def occupancy_grid(self, resolution=(32, 32, 16)):
        """Boolean voxel grid (True = inside an obstacle) at voxel centers."""
        axes = [self.bounds_lo[k] + (np.arange(n) + 0.5) * (self.bounds_hi[k] - self.bounds_lo[k]) / n
                for k, n in enumerate(resolution)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = (self.constraint_values(pts, d=0.0) < 0).any(axis=0)
        return inside.reshape(resolution)

    def fingerprint(self):
        """Short hash of the geometry and boundary conditions. Stored with solver
        results so a plan can never be paired with the wrong environment."""
        import hashlib
        d = self.to_dict()
        d.pop("metrics", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self):
        return {"bounds_lo": self.bounds_lo.tolist(), "bounds_hi": self.bounds_hi.tolist(),
                "obstacles": [o.to_dict() for o in self.obstacles],
                "start_pos": self.start_pos.tolist(), "start_vel": self.start_vel.tolist(),
                "goal_pos": self.goal_pos.tolist(), "goal_vel": self.goal_vel.tolist(),
                "vehicle_radius": self.vehicle_radius, "margin": self.margin,
                "metrics": self.metrics}

    @classmethod
    def from_dict(cls, d):
        arr = lambda x: np.asarray(x, float)
        return cls(arr(d["bounds_lo"]), arr(d["bounds_hi"]),
                   [obstacle_from_dict(o) for o in d["obstacles"]],
                   arr(d["start_pos"]), arr(d["start_vel"]),
                   arr(d["goal_pos"]), arr(d["goal_vel"]),
                   d["vehicle_radius"], d["margin"], d.get("metrics", {}))


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Generator settings. Defaults come from the USER SETTINGS block at the top
    of this file; pass keyword arguments to override them in code."""
    size: tuple = WORKSPACE_SIZE
    origin: tuple = WORKSPACE_ORIGIN
    vehicle_radius: float = VEHICLE_RADIUS
    margin: float = SAFETY_MARGIN

    occupancy_range: tuple = OCCUPANCY_RANGE
    occupancy_tolerance: float = OCCUPANCY_TOLERANCE
    occupancy_samples: int = OCCUPANCY_SAMPLES
    type_weights: dict = field(default_factory=lambda: dict(TYPE_WEIGHTS))

    spheroid_radius: tuple = SPHEROID_RADIUS
    spheroid_aspect: tuple = SPHEROID_ASPECT
    cylinder_radius: tuple = CYLINDER_RADIUS
    cylinder_height: tuple = CYLINDER_HEIGHT
    cylinder_full_height_prob: float = CYLINDER_FULL_HEIGHT_PROB
    box_half_extent: tuple = BOX_HALF_EXTENT
    wall_thickness: tuple = WALL_THICKNESS
    wall_opening: tuple = WALL_OPENING

    shrink_after: int = SHRINK_AFTER
    shrink_factor: float = SHRINK_FACTOR
    min_size_scale: float = MIN_SIZE_SCALE

    endpoint_mode: str = ENDPOINT_MODE
    end_zone_frac: float = END_ZONE_FRAC
    start_goal_min_frac: float = START_GOAL_MIN_FRAC
    endpoint_clearance: float = ENDPOINT_CLEARANCE
    start_speed_max: float = START_SPEED_MAX
    require_blocked_line: bool = REQUIRE_BLOCKED_LINE
    line_check_spacing: float = LINE_CHECK_SPACING

    max_proposals: int = MAX_PROPOSALS
    max_env_tries: int = MAX_ENV_TRIES


class EnvironmentGenerator:
    def __init__(self, cfg: EnvConfig = None, seed=None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.lo = np.asarray(c.origin, float)
        self.hi = self.lo + np.asarray(c.size, float)
        self.d = c.vehicle_radius + c.margin
        self.inner_lo = self.lo + self.d
        self.inner_hi = self.hi - self.d
        self.scale = float(min(c.size))
        if np.any(self.inner_hi <= self.inner_lo):
            raise ValueError("Workspace too small for the vehicle radius + margin.")
        self.proposers = {"spheroid": self._propose_spheroid, "cylinder": self._propose_cylinder,
                          "box": self._propose_box, "wall": self._propose_wall}

    # ---- public API ------------------------------------------------------
    def generate(self) -> Environment:
        for _ in range(self.cfg.max_env_tries):
            env = self._try_generate()
            if env is not None:
                return env
        raise RuntimeError("Could not generate an environment; "
                           "lower OCCUPANCY_RANGE or loosen the settings.")

    def generate_many(self, n):
        return [self.generate() for _ in range(n)]

    # ---- internals -------------------------------------------------------
    def _try_generate(self):
        c, rng = self.cfg, self.rng
        target = rng.uniform(*c.occupancy_range)
        start, goal = self._sample_endpoints()

        # Monte Carlo points; `occupied` is the union of all obstacle interiors.
        samples = rng.uniform(self.lo, self.hi, size=(c.occupancy_samples, 3))
        occupied = np.zeros(len(samples), dtype=bool)

        kinds = list(c.type_weights)
        probs = np.array([c.type_weights[k] for k in kinds], float)
        probs /= probs.sum()

        obstacles, size_scale, fails = [], 1.0, 0
        for _ in range(c.max_proposals):
            if occupied.mean() >= target:
                break
            obs = self.proposers[kinds[rng.choice(len(kinds), p=probs)]](start, goal, size_scale)
            if self._endpoints_clear(obs, start, goal):
                new_occ = occupied | (obs.constraint(samples, 0.0) < 0)
                if new_occ.mean() <= target + c.occupancy_tolerance:
                    obstacles.append(obs)
                    occupied, fails = new_occ, 0
                    continue
            fails += 1
            if fails >= c.shrink_after:          # stuck: try smaller obstacles
                size_scale = max(size_scale * c.shrink_factor, c.min_size_scale)
                fails = 0
        else:
            return None                          # ran out of proposals

        if occupied.mean() < target or not obstacles:
            return None
        line_blocked = self._line_blocked(start, goal, obstacles)
        if c.require_blocked_line and not line_blocked:
            return None

        u = rng.normal(size=3)
        start_vel = rng.uniform(0.0, c.start_speed_max) * u / np.linalg.norm(u)

        env = Environment(self.lo.copy(), self.hi.copy(), obstacles, start, start_vel,
                          goal, np.zeros(3), c.vehicle_radius, c.margin)
        env.metrics = self._metrics(env, samples, line_blocked, target, float(occupied.mean()))
        return env

    def _sample_endpoints(self):
        if self.cfg.endpoint_mode == "ends":
            k = int(np.argmax(self.hi - self.lo))            # longest axis
            span = self.inner_hi[k] - self.inner_lo[k]
            zone = self.cfg.end_zone_frac * span
            a = self.rng.uniform(self.inner_lo, self.inner_hi)
            b = self.rng.uniform(self.inner_lo, self.inner_hi)
            a[k] = self.rng.uniform(self.inner_lo[k], self.inner_lo[k] + zone)
            b[k] = self.rng.uniform(self.inner_hi[k] - zone, self.inner_hi[k])
            return a, b
        diag = np.linalg.norm(self.inner_hi - self.inner_lo)
        for _ in range(1000):
            a = self.rng.uniform(self.inner_lo, self.inner_hi)
            b = self.rng.uniform(self.inner_lo, self.inner_hi)
            if np.linalg.norm(b - a) >= self.cfg.start_goal_min_frac * diag:
                return a, b
        raise RuntimeError("START_GOAL_MIN_FRAC too large for this workspace.")

    def _endpoints_clear(self, obs, start, goal):
        ep = self.cfg.endpoint_clearance * self.d
        return bool(np.all(obs.constraint(np.stack([start, goal]), ep) >= 0))

    def _size(self, bounds, s, n=None):
        """Log-uniform size in `bounds` (fraction of min dim), shrunk by factor s."""
        lo, hi = np.log(bounds[0]), np.log(bounds[1])
        return np.exp(self.rng.uniform(lo, hi, size=n)) * self.scale * s

    def _propose_spheroid(self, start, goal, s=1.0):
        c, rng = self.cfg, self.rng
        eq = self._size(c.spheroid_radius, s)
        radii = np.array([eq, eq, eq * rng.uniform(*c.spheroid_aspect)])
        return Spheroid(rng.uniform(self.lo, self.hi), radii, _random_rotation(rng))

    def _propose_cylinder(self, start, goal, s=1.0):
        c, rng = self.cfg, self.rng
        r = self._size(c.cylinder_radius, s)
        xy = rng.uniform(self.lo[:2], self.hi[:2])
        if rng.random() < c.cylinder_full_height_prob:
            return Cylinder(xy, r, self.hi[2], full_height=True)
        z_top = min(self.lo[2] + rng.uniform(*c.cylinder_height) * c.size[2], self.hi[2])
        return Cylinder(xy, r, z_top, full_height=False)

    def _propose_box(self, start, goal, s=1.0):
        half = self._size(self.cfg.box_half_extent, s, n=3)
        return Box(self.rng.uniform(self.lo, self.hi), half)

    def _propose_wall(self, start, goal, s=1.0):
        """Vertical wall between start and goal with a randomly placed opening.
        Walls are large, so they are not shrunk by s."""
        c, rng = self.cfg, self.rng
        dvec = goal - start
        k = int(np.argmax(np.abs(dvec[:2])))                # wall normal: x or y
        i, j = [a for a in range(3) if a != k]
        position = start[k] + rng.uniform(0.3, 0.7) * dvec[k]
        size = 2 * self.d + rng.uniform(*c.wall_opening, size=2) * self.scale
        center = rng.uniform(self.inner_lo[[i, j]], self.inner_hi[[i, j]])
        return Wall(k, position, rng.uniform(*c.wall_thickness) * self.scale,
                    center, size, self.lo, self.hi)

    def _line_blocked(self, start, goal, obstacles):
        n = max(int(np.linalg.norm(goal - start) / (self.cfg.line_check_spacing * self.d)), 2)
        line = np.linspace(start, goal, n)
        return bool(any(np.any(o.constraint(line, self.d) < 0) for o in obstacles))

    def _metrics(self, env, samples, line_blocked, target, achieved):
        counts = {}
        for o in env.obstacles:
            counts[o.kind] = counts.get(o.kind, 0) + 1
        # fraction of the box the vehicle CENTER cannot reach (obstacles grown by d)
        blocked = (env.constraint_values(samples) < 0).any(axis=0)
        return {"occupancy_target": float(target),
                "occupancy_fraction": achieved,
                "inflated_occupancy_fraction": float(blocked.mean()),
                "straight_line_blocked": line_blocked,
                "straight_line_distance": float(np.linalg.norm(env.goal_pos - env.start_pos)),
                "n_constraint_pieces": len(env.pieces()),
                "obstacle_counts": counts}


# --------------------------------------------------------------------------
# I/O and plotting
# --------------------------------------------------------------------------
def save_environments(envs, path):
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in envs], f)


def load_environments(path):
    with open(path) as f:
        return [Environment.from_dict(d) for d in json.load(f)]


def plot_environment(env, ax=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")

    def draw_box(b, color):
        lo, hi = b.center - b.half_extents, b.center + b.half_extents
        v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        ax.add_collection3d(Poly3DCollection([v[f] for f in faces], alpha=0.35,
                                             facecolor=color, edgecolor="k", linewidth=0.2))

    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
    for o in env.obstacles:
        if o.kind == "spheroid":
            s = np.stack([np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)], -1) * o.radii
            s = s @ o.rotation.T + o.center
            ax.plot_surface(s[..., 0], s[..., 1], s[..., 2], color="tab:orange", alpha=0.45, linewidth=0)
        elif o.kind == "cylinder":
            th, z = np.meshgrid(np.linspace(0, 2 * np.pi, 24), [env.bounds_lo[2], o.z_top])
            ax.plot_surface(o.center_xy[0] + o.radius * np.cos(th), o.center_xy[1] + o.radius * np.sin(th),
                            z, color="tab:blue", alpha=0.45, linewidth=0)
        elif o.kind == "box":
            draw_box(o, "tab:green")
        elif o.kind == "wall":
            for b in o.boxes:
                draw_box(b, "tab:gray")

    ax.scatter(*env.start_pos, c="green", s=50, label="start")
    ax.scatter(*env.goal_pos, c="red", s=50, label="goal")
    ax.set_xlim(env.bounds_lo[0], env.bounds_hi[0])
    ax.set_ylim(env.bounds_lo[1], env.bounds_hi[1])
    ax.set_zlim(env.bounds_lo[2], env.bounds_hi[2])
    ax.set_box_aspect(env.bounds_hi - env.bounds_lo)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left")
    return ax


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate random 3D obstacle environments. "
                                 "Defaults come from the USER SETTINGS block.")
    ap.add_argument("--size", type=float, nargs=3, default=list(WORKSPACE_SIZE), metavar=("LX", "LY", "LZ"))
    ap.add_argument("--n", type=int, default=N_ENVIRONMENTS, help="number of environments")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--vehicle-radius", type=float, default=VEHICLE_RADIUS)
    ap.add_argument("--margin", type=float, default=SAFETY_MARGIN)
    ap.add_argument("--occupancy", type=float, nargs=2, default=list(OCCUPANCY_RANGE), metavar=("LO", "HI"),
                    help="fill target range, fractions of box volume")
    ap.add_argument("--endpoints", choices=["random", "ends"], default=ENDPOINT_MODE)
    ap.add_argument("--out", default=OUTPUT_FILE)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=PLOT_FIRST,
                    help="plot the first environment")
    args = ap.parse_args()

    cfg = EnvConfig(size=tuple(args.size), vehicle_radius=args.vehicle_radius, margin=args.margin,
                    occupancy_range=tuple(args.occupancy), endpoint_mode=args.endpoints)
    envs = EnvironmentGenerator(cfg, seed=args.seed).generate_many(args.n)
    save_environments(envs, args.out)
    print(f"Saved {len(envs)} environments to {args.out}")
    for k, e in enumerate(envs[:5]):
        m = e.metrics
        print(f"  env {k}: {m['obstacle_counts']}, fill {m['occupancy_fraction']:.1%} "
              f"(target {m['occupancy_target']:.1%}), "
              f"inflated fill {m['inflated_occupancy_fraction']:.1%}")
    if args.plot:
        import matplotlib.pyplot as plt
        plot_environment(envs[0])
        plt.show()


if __name__ == "__main__":
    main()