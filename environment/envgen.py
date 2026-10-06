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
    Rotations: spheroids and oriented cylinders are defined upright and turned
    by an angle vector (roll, pitch, yaw) in degrees,
        R = Rz(yaw) Ry(pitch) Rx(roll)          (aerospace Z-Y-X convention)
    and their axis (polar axis / cylinder axis) is a = R e_z.
    Spheroid  (center c, semi-axes a, rotation R, a_min = min(a)):
        h = || diag(1/a) R^T (p - c) ||_2  -  (1 + d / a_min)
    Oriented cylinder  (finite, center c, rotation R, axis a = R e_z,
                        radius rho, half-length L):
        s = a . (p - c)                      (position along the axis)
        r = || (p - c) - s a ||_2            (distance from the axis)
        h = max( r - (rho + d),  |s| - (L + d) )
    Cylinder  (legacy upright pillar standing on the floor; axis center c_xy,
               radius rho, top z_top):
        h = max( ||(x, y) - c_xy||_2 - (rho + d),  z - (z_top + d) )
        full-height cylinders (z_top = ceiling) drop the second term:
        h = ||(x, y) - c_xy||_2 - (rho + d)
    Box       (center c, half-extents e, rotation R from its angle vector):
        h = max_k ( |(R^T (p - c))_k| - (e_k + d) )
    Wall      (finite panel with one rectangular opening, yawed and leaned):
              the union of up to four rotated boxes around the opening, so it
              contributes one box constraint per piece.

Properties (why these forms are used):
  * Conservative: h_i(p; d) >= 0 guarantees the vehicle's sphere of radius d
    does not touch the obstacle. (Spheroid: the ellipsoid scaled by
    1 + d/a_min contains the obstacle grown by d. Cylinders/box: the max-form
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
  3. Propose obstacles of random type, log-uniform size, uniform position and
     a random angle vector (roll, pitch, yaw), each angle drawn from its own
     range, so a sweep can go from upright pillars to horizontal bars. The
     type is drawn per obstacle, so the obstacle count depends only on the
     fill target. Obstacles below MIN_OBSTACLE_RADIUS are discarded, so the
     fill is made of fewer, larger pieces. Walls (panels across the start-goal
     line) are capped per environment, and later obstacles must leave each
     wall's opening passable.
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
    choose obstacle types and orientations:
        python envgen.py --types spheroid=1 oriented_cylinder=1 box=1 wall=0.1 \\
                         --cylinder-rotation 0 75:90 180 --spheroid-rotation 90 90 180
    (each angle: m means [-m, m], lo:hi means [lo, hi], degrees)
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
OUTPUT_FILE        = "envs.json"      # relative names are saved to and read from LOCALSTORE_DIR
LOCALSTORE_DIR     = None             # where generated files go. None = <project root>/localstore,
                                      #   where the project root is the enclosing folder named
                                      #   fixed-wing-rl / fixed_wing_rl. Absolute paths are used as is.
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

# ---- obstacle types -----------------------------------------------------------
# Every obstacle's type is drawn at random from these weights, one obstacle at a
# time, until the fill target is reached. So enabling more types changes the
# MIX, not the number of obstacles: the count is set by OCCUPANCY_RANGE and the
# obstacle sizes. Walls are whole-workspace gates, so at most MAX_WALLS are
# placed per environment; once that many exist, "wall" is dropped from the draw.
TYPE_WEIGHTS = {
    "spheroid": 0.35,
    "oriented_cylinder": 0.35,      # finite cylinder, rotated by CYLINDER_ROTATION_DEG
    "box": 0.29,                    # rotated by BOX_ROTATION_DEG
    "wall": 0.01,                   # vertical gate with one opening (see WALL_* below); at the
                                    #   C172 scale this gives about 13% / 20% / 67% of
                                    #   environments with 0 / 1 / 2 walls
    # "cylinder": 0.2,              # <- uncomment for upright pillars standing on the floor
}
MAX_WALLS          = 2

# ---- obstacle sizes (fractions of the smallest box dimension; log-uniform) -
SPHEROID_RADIUS    = (0.03, 0.30)   # equatorial semi-axis
SPHEROID_ASPECT    = (0.5, 2.0)     # polar / equatorial ratio
CYLINDER_RADIUS    = (0.03, 0.20)   # used by both cylinder types
CYLINDER_LENGTH    = (0.5, 3.0)     # oriented cylinders: full length (end cap to end cap)
CYLINDER_HEIGHT    = (0.3, 1.0)     # upright pillars: fraction of box height (Lz), partial height
CYLINDER_FULL_HEIGHT_PROB = 0.5     # upright pillars: chance of spanning floor to ceiling


# ---- obstacle orientations ---------------------------------------------------
# Each obstacle is built upright (axis along +z) and turned by an angle vector
# (roll, pitch, yaw) in degrees: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
#   roll  - about x      pitch - about y      (together: how far it leans)
#   yaw   - about z      (which compass direction it leans toward)
# For cylinders the axis is the long axis; for spheroids, the polar axis
# (the one scaled by SPHEROID_ASPECT).
# Every entry gives the range one angle is drawn from, uniformly per obstacle:
#     m          ->  [-m, m]          e.g. 90
#     (lo, hi)   ->  [lo, hi]         e.g. (75, 90);  (30, 30) fixes it at 30
# Examples (roll, pitch, yaw):
#     (0, 0, 0)            upright, all identical
#     (15, 15, 180)        mostly upright, leaning up to ~20 deg
#     (0, (75, 90), 180)   near-horizontal bars pointing any way
#     (90, 90, 180)        any orientation (default)
CYLINDER_ROTATION_DEG = (90.0, 90.0, 180.0)
SPHEROID_ROTATION_DEG = (90.0, 90.0, 180.0)

CYLINDER_UPRIGHT_FRAC = 0.2         # share of oriented cylinders forced exactly vertical
                                    #   (the rest use CYLINDER_ROTATION_DEG). A random angle
                                    #   vector is almost never vertical, so without this
                                    #   upright cylinders essentially never appear.
BOX_ROTATION_DEG      = (0.0, 0.0, 180.0)   # upright boxes at any heading (buildings,
                                            #   terrain blocks); widen roll/pitch for OOD sets

# ---- boxes and walls (sizes as fractions of the smallest box dimension) -----
BOX_HALF_EXTENT    = (0.055, 0.33)   # log-uniform per side; sized so a box averages about
                                    #   the volume of a spheroid / cylinder, which keeps
                                    #   the obstacle count unchanged when boxes are mixed in
WALL_THICKNESS     = (0.02, 0.05)
WALL_WIDTH         = (0.3, 0.7)     # panel width, fraction of the workspace's width along the
                                    #   wall line (so a wall never spans the whole corridor)
WALL_HEIGHT        = (0.5, 1.0)     # panel height above the floor, fraction of box height (Lz);
                                    #   panels stand on the floor, so 1.0 reaches the ceiling
WALL_OPENING       = (0.10, 0.30)   # extra opening (width, height) beyond the minimum 2 * (radius + margin)
WALL_YAW_DEG       = 30.0           # the panel faces within +-this of the start->goal heading,
                                    #   so it stands roughly across the flight path
WALL_PITCH_DEG     = 15.0           # and leans up to +-this toward (+) or away from (-) the goal
WALL_GATE_CLEAR    = 0.25           # other obstacles must leave a straight passage through each
                                    #   opening, this far (fraction of smallest box dim) on
                                    #   either side of the wall

# ---- clutter -------------------------------------------------------------------
MIN_OBSTACLE_RADIUS = 0.08          # drop any obstacle smaller than this, measured as the radius
                                    #   of the sphere with the same volume (fraction of the
                                    #   smallest box dimension; 0 keeps everything). Fewer,
                                    #   larger obstacles then make up the same fill.

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
def rotation_from_angles(angles_deg):
    """Rotation matrix for an angle vector (roll, pitch, yaw) in degrees,
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Columns are the rotated x, y, z axes,
    so R[:, 2] is where an upright obstacle's axis ends up."""
    r, p, y = np.radians(np.asarray(angles_deg, float))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def angle_ranges(spec):
    """Normalize a (roll, pitch, yaw) range spec to three (lo, hi) pairs.
    Each entry is a number m (meaning [-m, m]) or a (lo, hi) pair."""
    spec = list(spec)
    if len(spec) != 3:
        raise ValueError(f"rotation spec needs 3 entries (roll, pitch, yaw), got {spec!r}")
    out = []
    for e in spec:
        lo, hi = (-abs(float(e)), abs(float(e))) if np.isscalar(e) else map(float, e)
        if lo > hi:
            raise ValueError(f"rotation range {e!r} has lo > hi")
        out.append((lo, hi))
    return tuple(out)


def sample_angles(rng, spec):
    """Random (roll, pitch, yaw) in degrees, each uniform in its range."""
    return np.array([rng.uniform(lo, hi) for lo, hi in angle_ranges(spec)])


def tilt_deg(axis):
    """Angle between an axis and vertical, in degrees (0 = upright, 90 = flat).
    Axes are treated as unsigned, so the result is always in [0, 90]."""
    a = np.asarray(axis, float)
    return float(np.degrees(np.arccos(np.clip(abs(a[2]) / np.linalg.norm(a), 0.0, 1.0))))


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

    def equivalent_radius(self):
        """Radius of the sphere with the same volume: one size number that
        works for every shape. Used to drop obstacles too small to matter."""
        raise NotImplementedError


def _r_eq(volume):
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


@dataclass
class Spheroid(Obstacle):
    """Ellipsoid with semi-axes `radii` = (a1, a2, a_polar) (two equal ->
    spheroid), built upright and rotated by R.
        h = ||diag(1/a) R^T (p - c)||_2 - (1 + d / a_min)
    Give the orientation as `angles_deg` = (roll, pitch, yaw); R is computed
    from it. Files written before angle vectors existed store the matrix as
    `rotation` instead, and still load (R is then taken as given)."""
    center: np.ndarray
    radii: np.ndarray
    angles_deg: np.ndarray = None
    rotation: np.ndarray = None
    kind = "spheroid"
    n_features = 15

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.radii = np.asarray(self.radii, float)
        if self.angles_deg is not None:
            self.angles_deg = np.asarray(self.angles_deg, float)
            self.rotation = rotation_from_angles(self.angles_deg)
        else:
            self.rotation = np.eye(3) if self.rotation is None else np.asarray(self.rotation, float)

    @property
    def axis(self):
        """Polar axis direction, R e_z."""
        return self.rotation[:, 2]

    def equivalent_radius(self):
        return float(np.prod(self.radii) ** (1.0 / 3.0))

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
                f"c={_fmt(self.center)}, a={_fmt(self.radii)}, "
                + (f"angles={_fmt(self.angles_deg)} deg" if self.angles_deg is not None
                   else "R=stored rotation"))

    def features(self):
        return np.r_[self.center, self.radii, self.rotation.ravel()]

    def to_dict(self):
        d = {"type": self.kind, "center": self.center.tolist(), "radii": self.radii.tolist()}
        if self.angles_deg is not None:
            d["angles_deg"] = self.angles_deg.tolist()
        else:                                   # legacy: matrix only, keeps old fingerprints
            d["rotation"] = self.rotation.tolist()
        return d


@dataclass
class OrientedCylinder(Obstacle):
    """Finite solid cylinder with flat end caps, built upright (axis along +z)
    and rotated by the angle vector `angles_deg` = (roll, pitch, yaw):
        R = rotation_from_angles(angles_deg),   a = R e_z
        s = a . (p - c),   r = ||(p - c) - s a||_2
        h = max(r - (rho + d), |s| - (L + d))
    `half_length` is L (center to cap)."""
    center: np.ndarray
    angles_deg: np.ndarray
    radius: float
    half_length: float
    kind = "oriented_cylinder"
    n_features = 8

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.angles_deg = np.asarray(self.angles_deg, float)
        self.radius = float(self.radius)
        self.half_length = float(self.half_length)
        self.rotation = rotation_from_angles(self.angles_deg)
        self.axis = self.rotation[:, 2]

    def _terms(self, p, d):
        q = np.asarray(p, float) - self.center
        s = q @ self.axis
        rvec = q - s[..., None] * self.axis
        r = np.linalg.norm(rvec, axis=-1)
        return rvec, r, s, r - (self.radius + d), np.abs(s) - (self.half_length + d)

    def constraint(self, p, d=0.0):
        _, _, _, radial, axial = self._terms(p, d)
        return np.maximum(radial, axial)

    def gradient(self, p, d=0.0):
        rvec, r, s, radial, axial = self._terms(p, d)
        on_axis = (r < 1e-9)[..., None]                 # any perpendicular is a valid subgradient
        g_rad = np.where(on_axis, self.rotation[:, 0], rvec / np.maximum(r, 1e-9)[..., None])
        g_ax = np.where((s >= 0)[..., None], 1.0, -1.0) * self.axis
        return np.where((radial >= axial)[..., None], g_rad, g_ax)

    def equivalent_radius(self):
        return _r_eq(np.pi * self.radius**2 * 2.0 * self.half_length)

    def endpoints(self):
        """Centers of the two end caps."""
        return self.center - self.half_length * self.axis, self.center + self.half_length * self.axis

    def formula(self, d):
        return (f"max(||(p - c) - (a.(p - c)) a||_2 - {self.radius + d:.4f}, "
                f"|a.(p - c)| - {self.half_length + d:.4f}) >= 0   "
                f"c={_fmt(self.center)}, angles={_fmt(self.angles_deg)} deg, "
                f"a={_fmt(self.axis)} (tilt {tilt_deg(self.axis):.1f} deg)")

    def features(self):
        # The axis (sign fixed so a_z >= 0, since a and -a are the same cylinder)
        # rather than the angles: angles wrap around and many triples give the
        # same cylinder, which makes them a poor input for learning.
        a = -self.axis if self.axis[2] < 0 else self.axis
        return np.r_[self.center, a, self.radius, self.half_length]

    def to_dict(self):
        return {"type": self.kind, "center": self.center.tolist(),
                "angles_deg": self.angles_deg.tolist(),
                "radius": self.radius, "half_length": self.half_length}


@dataclass
class Cylinder(Obstacle):
    """Legacy upright cylinder (pillar) standing on the floor.
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

    def equivalent_radius(self):
        # the floor height is not stored on the pillar; z_top approximates its
        # height above the floor, which is exact for workspaces with z_lo = 0
        return _r_eq(np.pi * self.radius**2 * max(self.z_top, self.radius))

    def to_dict(self):
        return {"type": self.kind, "center_xy": self.center_xy.tolist(),
                "radius": float(self.radius), "z_top": float(self.z_top),
                "full_height": bool(self.full_height)}


@dataclass
class Box(Obstacle):
    """Box with half-extents e, built axis-aligned and rotated by the angle
    vector `angles_deg` = (roll, pitch, yaw), R = rotation_from_angles(...):
        q = R^T (p - c),    h = max_k (|q_k| - (e_k + d))
    Without angles_deg the box is axis-aligned (R = I), as in older files."""
    center: np.ndarray
    half_extents: np.ndarray
    angles_deg: np.ndarray = None
    kind = "box"
    n_features = 15

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.half_extents = np.asarray(self.half_extents, float)
        if self.angles_deg is not None:
            self.angles_deg = np.asarray(self.angles_deg, float)
            self.rotation = rotation_from_angles(self.angles_deg)
        else:
            self.rotation = np.eye(3)

    @classmethod
    def from_bounds(cls, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return cls(0.5 * (lo + hi), 0.5 * (hi - lo))

    def _local(self, p):
        return (np.asarray(p, float) - self.center) @ self.rotation          # R^T (p - c)

    def constraint(self, p, d=0.0):
        return (np.abs(self._local(p)) - (self.half_extents + d)).max(axis=-1)

    def gradient(self, p, d=0.0):
        q = self._local(p)
        k = np.argmax(np.abs(q) - (self.half_extents + d), axis=-1)          # active face
        s = np.where(np.take_along_axis(q, np.asarray(k)[..., None], -1) >= 0, 1.0, -1.0)
        return s * self.rotation.T[k]                                        # +-R[:, k]

    def equivalent_radius(self):
        return _r_eq(8.0 * np.prod(self.half_extents))

    def halfspaces(self):
        """The box as A x <= b (6 rows), used for clipping and meshing."""
        n = np.vstack([self.rotation.T, -self.rotation.T])                  # outward face normals
        b = n @ self.center + np.r_[self.half_extents, self.half_extents]
        return n, b

    def formula(self, d):
        rot = f", angles={_fmt(self.angles_deg)} deg" if self.angles_deg is not None else ""
        return (f"max_k(|R^T(p - c)|_k - (e_k + {d:.3f})) >= 0   "
                f"c={_fmt(self.center)}, e={_fmt(self.half_extents)}{rot}")

    def features(self):
        return np.r_[self.center, self.half_extents, self.rotation.ravel()]

    def to_dict(self):
        d = {"type": self.kind, "center": self.center.tolist(),
             "half_extents": self.half_extents.tolist()}
        if self.angles_deg is not None:
            d["angles_deg"] = self.angles_deg.tolist()
        return d


@dataclass
class Wall(Obstacle):
    """A finite wall panel (thickness x width x height) with one rectangular
    opening, rotated by yaw about z and then leaned by pitch:
        R = rotation_from_angles((0, pitch_deg, yaw_deg)),   columns n, u, v
    n is the panel normal (heading yaw, tipped by pitch), u runs along the
    panel (always horizontal) and v up the panel (vertical when pitch = 0).
    Positive pitch leans the top toward +n, i.e. toward the goal for a wall
    across the flight path.
    `center` is the middle of the panel; `opening_center` = (u, v) and
    `opening_size` = (width, height) of the opening, measured from it.
    The panel is up to four rotated boxes around the opening (one constraint each)."""
    center: np.ndarray
    yaw_deg: float
    pitch_deg: float
    width: float
    height: float
    thickness: float
    opening_center: np.ndarray
    opening_size: np.ndarray
    kind = "wall"
    n_features = 13

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.yaw_deg, self.pitch_deg = float(self.yaw_deg), float(self.pitch_deg)
        self.width, self.height, self.thickness = float(self.width), float(self.height), float(self.thickness)
        self.opening_center = np.asarray(self.opening_center, float)
        self.opening_size = np.asarray(self.opening_size, float)
        self.angles_deg = np.array([0.0, self.pitch_deg, self.yaw_deg])
        self.rotation = rotation_from_angles(self.angles_deg)            # columns n, u, v
        self.boxes = self._build_boxes()

    @property
    def normal(self):
        return self.rotation[:, 0]

    def _build_boxes(self):
        W, H = self.width / 2, self.height / 2
        (ou, ov), (w, h) = self.opening_center, self.opening_size
        g_u0, g_u1 = np.clip([ou - w / 2, ou + w / 2], -W, W)
        g_v0, g_v1 = np.clip([ov - h / 2, ov + h / 2], -H, H)
        spans = [((-W, g_u0), (-H, H)), ((g_u1, W), (-H, H)),           # left, right of the opening
                 ((g_u0, g_u1), (-H, g_v0)), ((g_u0, g_u1), (g_v1, H))]  # below, above it
        boxes = []
        for (a0, a1), (b0, b1) in spans:
            if a1 - a0 < 1e-9 or b1 - b0 < 1e-9:
                continue
            local = np.array([0.0, 0.5 * (a0 + a1), 0.5 * (b0 + b1)])
            boxes.append(Box(self.center + self.rotation @ local,
                             [self.thickness / 2, 0.5 * (a1 - a0), 0.5 * (b1 - b0)],
                             angles_deg=self.angles_deg))
        return boxes

    def opening_world_center(self):
        return self.center + self.rotation @ np.r_[0.0, self.opening_center]

    def equivalent_radius(self):
        return float("inf")                      # walls are never too small to keep

    def gate_line(self, depth, spacing):
        """Points along the wall normal through the middle of the opening, from
        -depth to +depth: the straight passage the generator keeps clear."""
        n = max(int(np.ceil(2 * depth / spacing)) + 1, 2)
        return self.opening_world_center() + np.linspace(-depth, depth, n)[:, None] * self.normal

    def pieces(self):
        return list(self.boxes)

    def constraint(self, p, d=0.0):
        # clear of the wall  <=>  clear of every piece
        return np.min([b.constraint(p, d) for b in self.boxes], axis=0)

    def gradient(self, p, d=0.0):
        raise NotImplementedError("Use the individual pieces (wall.pieces()).")

    def formula(self, d):
        return (f"wall {self.width:.1f} x {self.height:.1f}, yaw {self.yaw_deg:.1f} deg, "
                f"pitch {self.pitch_deg:.1f} deg, opening {_fmt(self.opening_size)}: "
                + "; ".join(b.formula(d) for b in self.boxes))

    def features(self):
        y = np.radians(self.yaw_deg)
        return np.r_[self.center, np.cos(y), np.sin(y), np.radians(self.pitch_deg),
                     self.width, self.height, self.thickness, self.opening_center, self.opening_size]

    def to_dict(self):
        return {"type": self.kind, "center": self.center.tolist(), "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg, "width": self.width, "height": self.height,
                "thickness": self.thickness, "opening_center": self.opening_center.tolist(),
                "opening_size": self.opening_size.tolist()}

    @classmethod
    def from_workspace_gate(cls, opening_world_center, yaw_deg, thickness, opening_size,
                            bounds_lo, bounds_hi):
        """A wall that spans the whole workspace around one opening, as earlier
        versions of envgen.py generated. Used to load their files: inside the
        workspace the result is the same solid."""
        span = 4.0 * float(np.linalg.norm(np.asarray(bounds_hi, float) - np.asarray(bounds_lo, float)))
        return cls(opening_world_center, yaw_deg, 0.0, span, span, thickness, [0.0, 0.0], opening_size)


def _wall_from_old_dict(d):
    """Walls from files written before finite, leaning walls existed."""
    if "axis" in d:                              # oldest: normal along x (axis 0) or y (axis 1)
        if d["axis"] not in (0, 1):
            raise ValueError("only vertical walls (axis 0 or 1) are supported")
        oc = np.asarray(d["opening_center"], float)
        p = d["position"]
        c = np.array([p, oc[0], oc[1]]) if d["axis"] == 0 else np.array([oc[0], p, oc[1]])
        yaw = 0.0 if d["axis"] == 0 else 90.0
    else:                                        # full-width yawed gate
        R = rotation_from_angles((0.0, 0.0, d["yaw_deg"]))
        c = np.asarray(d["center"], float) + R @ np.r_[0.0, d["opening_center"]]
        yaw = d["yaw_deg"]
    return Wall.from_workspace_gate(c, yaw, d["thickness"], d["opening_size"], d["bounds_lo"], d["bounds_hi"])


OBSTACLE_TYPES = {c.kind: c for c in (Spheroid, OrientedCylinder, Cylinder, Box, Wall)}


def obstacle_from_dict(d):
    d = dict(d)
    kind = d.pop("type")
    if kind == "wall" and "width" not in d:              # written before finite walls existed
        return _wall_from_old_dict(d)
    return OBSTACLE_TYPES[kind](**d)


def clip_polytope(A, b, lo, hi):
    """Vertices and faces of {x : A x <= b} intersected with the box [lo, hi],
    or None if that is empty. Faces are polygons (vertex index lists) wound
    counter-clockwise seen from outside. Used to draw and mesh boxes and wall
    pieces only where they lie inside the workspace."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    A = np.vstack([A, np.eye(3), -np.eye(3)])
    b = np.r_[b, hi, -np.asarray(lo, float)]
    norms = np.linalg.norm(A, axis=1)
    # Chebyshev center: a strictly interior point, needed by HalfspaceIntersection
    res = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[A, norms], b_ub=b,
                  bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if res.status != 0 or res.x[3] < 1e-6 * max(1.0, float(np.max(np.abs(hi - lo)))):
        return None
    hull = ConvexHull(HalfspaceIntersection(np.c_[A, -b], res.x[:3]).intersections)
    keep = np.unique(hull.simplices)
    V = hull.points[keep]
    remap = {old: new for new, old in enumerate(keep)}
    # merge the hull's coplanar triangles into one polygon per face
    faces = {}
    for tri, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq / np.linalg.norm(eq[:3]), 6))
        faces.setdefault(key, (eq[:3], set()))[1].update(remap[i] for i in tri)
    F = []
    for normal, idx in faces.values():
        idx = np.array(sorted(idx))
        P = V[idx] - V[idx].mean(axis=0)
        e1 = P[np.argmax(np.linalg.norm(P, axis=1))]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal / np.linalg.norm(normal), e1)
        F.append(list(idx[np.argsort(np.arctan2(P @ e2, P @ e1))]))   # CCW about the outward normal
    return V, F


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
        max_per_type = max_per_type or {"spheroid": 160, "oriented_cylinder": 160, "cylinder": 160,
                                        "box": 160, "wall": 4}
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
    cylinder_length: tuple = CYLINDER_LENGTH
    cylinder_rotation_deg: tuple = CYLINDER_ROTATION_DEG
    spheroid_rotation_deg: tuple = SPHEROID_ROTATION_DEG
    cylinder_upright_frac: float = CYLINDER_UPRIGHT_FRAC
    box_rotation_deg: tuple = BOX_ROTATION_DEG
    max_walls: int = MAX_WALLS
    wall_yaw_deg: float = WALL_YAW_DEG
    wall_pitch_deg: float = WALL_PITCH_DEG
    wall_width: tuple = WALL_WIDTH
    wall_height: tuple = WALL_HEIGHT
    min_obstacle_radius: float = MIN_OBSTACLE_RADIUS
    wall_gate_clear: float = WALL_GATE_CLEAR
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
        self.proposers = {"spheroid": self._propose_spheroid,
                          "oriented_cylinder": self._propose_oriented_cylinder,
                          "cylinder": self._propose_cylinder,
                          "box": self._propose_box, "wall": self._propose_wall}
        unknown = set(self.cfg.type_weights) - set(self.proposers)
        if unknown:
            raise ValueError(f"Unknown obstacle type(s) {sorted(unknown)}; "
                             f"choose from {sorted(self.proposers)}.")
        if sum(self.cfg.type_weights.values()) <= 0:
            raise ValueError("At least one obstacle type needs a positive weight.")
        angle_ranges(self.cfg.cylinder_rotation_deg)        # fail early on a malformed spec
        angle_ranges(self.cfg.spheroid_rotation_deg)
        angle_ranges(self.cfg.box_rotation_deg)

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

        def type_draw(allow_walls):
            kinds = [k for k, w in c.type_weights.items() if w > 0 and (allow_walls or k != "wall")]
            p = np.array([c.type_weights[k] for k in kinds], float)
            return kinds, p / p.sum()
        kinds, probs = type_draw(c.max_walls > 0)

        # Point sets every obstacle must stay clear of: (points, clearance).
        # Start and goal first; each accepted wall adds the passage through its opening.
        keep_clear = [(np.stack([start, goal]), c.endpoint_clearance * self.d)]
        gate_depth = c.wall_gate_clear * self.scale

        min_r = c.min_obstacle_radius * self.scale
        obstacles, size_scale, fails, n_walls = [], 1.0, 0, 0
        for _ in range(c.max_proposals):
            if occupied.mean() >= target:
                break
            kind = kinds[rng.choice(len(kinds), p=probs)]
            obs = self.proposers[kind](start, goal, size_scale)
            if obs is not None and obs.equivalent_radius() < min_r:
                continue                                 # too small to keep; not a placement failure
            if obs is not None and self._clear_of(obs, keep_clear):
                gate = None
                if kind == "wall":                       # its opening must not already be blocked
                    gate = obs.gate_line(gate_depth, 0.5 * self.d)
                    if any(np.any(o.constraint(gate, self.d) < 0) for o in obstacles):
                        gate = False
                if gate is not False:
                    new_occ = occupied | (obs.constraint(samples, 0.0) < 0)
                    if new_occ.mean() <= target + c.occupancy_tolerance:
                        obstacles.append(obs)
                        occupied, fails = new_occ, 0
                        if kind == "wall":
                            keep_clear.append((gate, self.d))
                            n_walls += 1
                            if n_walls >= c.max_walls:
                                kinds, probs = type_draw(False)
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

    @staticmethod
    def _clear_of(obs, keep_clear):
        return all(np.all(obs.constraint(P, d) >= 0) for P, d in keep_clear)

    def _size(self, bounds, s, n=None):
        """Log-uniform size in `bounds` (fraction of min dim), shrunk by factor s."""
        lo, hi = np.log(bounds[0]), np.log(bounds[1])
        return np.exp(self.rng.uniform(lo, hi, size=n)) * self.scale * s

    def _propose_spheroid(self, start, goal, s=1.0):
        c, rng = self.cfg, self.rng
        eq = self._size(c.spheroid_radius, s)
        radii = np.array([eq, eq, eq * rng.uniform(*c.spheroid_aspect)])
        angles = sample_angles(rng, c.spheroid_rotation_deg)
        return Spheroid(rng.uniform(self.lo, self.hi), radii, angles_deg=angles)

    def _propose_oriented_cylinder(self, start, goal, s=1.0):
        c, rng = self.cfg, self.rng
        r = self._size(c.cylinder_radius, s)
        half = 0.5 * self._size(c.cylinder_length, s)
        if rng.random() < c.cylinder_upright_frac:
            angles = np.zeros(3)                     # exactly vertical
        else:
            angles = sample_angles(rng, c.cylinder_rotation_deg)
        return OrientedCylinder(rng.uniform(self.lo, self.hi), angles, r, half)

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
        angles = sample_angles(self.rng, self.cfg.box_rotation_deg)
        return Box(self.rng.uniform(self.lo, self.hi), half, angles_deg=angles)

    def _propose_wall(self, start, goal, s=1.0):
        """A finite wall panel standing on the floor across the start-goal line:
        facing within +-WALL_YAW_DEG of the flight direction, leaning up to
        +-WALL_PITCH_DEG toward or away from the goal, WALL_WIDTH of the
        corridor wide and WALL_HEIGHT of the box tall, with one opening placed
        inside the workspace. Walls are not shrunk by s. Returns None if the
        drawn panel cannot hold its opening (it is then simply redrawn)."""
        c, rng = self.cfg, self.rng
        dvec = goal - start
        heading = np.degrees(np.arctan2(dvec[1], dvec[0]))
        yaw = heading + rng.uniform(-c.wall_yaw_deg, c.wall_yaw_deg)
        pitch = rng.uniform(-c.wall_pitch_deg, c.wall_pitch_deg)
        R = rotation_from_angles((0.0, pitch, yaw))
        u_dir, v_dir = R[:, 1], R[:, 2]                     # along the panel (horizontal), up it
        base = start + rng.uniform(0.3, 0.7) * dvec          # where it crosses the flight path
        base[2] = self.lo[2]                                 # standing on the floor

        # stretch of the wall line (base + t u) inside the inner box
        t_lo, t_hi = -np.inf, np.inf
        for k in range(2):
            if abs(u_dir[k]) > 1e-12:
                a, b = sorted(((self.inner_lo[k] - base[k]) / u_dir[k], (self.inner_hi[k] - base[k]) / u_dir[k]))
                t_lo, t_hi = max(t_lo, a), min(t_hi, b)
        if not t_lo < t_hi:
            return None

        thickness = rng.uniform(*c.wall_thickness) * self.scale
        frame = max(thickness, 0.5 * self.d)                 # solid border kept around the opening
        sink = self.d + thickness                            # bottom edge sits this far below the floor
        width = rng.uniform(*c.wall_width) * (t_hi - t_lo)
        height = rng.uniform(*c.wall_height) * c.size[2] / np.cos(np.radians(pitch)) + sink
        t_c = rng.uniform(t_lo, t_hi)                        # panel middle, along the wall line
        center = base + t_c * u_dir + (height / 2 - sink) * v_dir

        ow, oh = 2 * self.d + rng.uniform(*c.wall_opening, size=2) * self.scale
        # opening along u: inside the panel (with a frame) and inside the workspace
        u_min = max(-width / 2 + frame, t_lo - t_c) + ow / 2
        u_max = min(width / 2 - frame, t_hi - t_c) - ow / 2
        # opening along v: above the floor, below the ceiling, inside the panel
        cos_p = np.cos(np.radians(pitch))
        v_min = max(-height / 2 + sink + frame, (self.inner_lo[2] - center[2]) / cos_p) + oh / 2
        v_max = min(height / 2 - frame, (self.inner_hi[2] - center[2]) / cos_p) - oh / 2
        if u_min > u_max or v_min > v_max:
            return None
        return Wall(center, yaw, pitch, width, height, thickness,
                    [rng.uniform(u_min, u_max), rng.uniform(v_min, v_max)], [ow, oh])

    def _line_blocked(self, start, goal, obstacles):
        n = max(int(np.linalg.norm(goal - start) / (self.cfg.line_check_spacing * self.d)), 2)
        line = np.linspace(start, goal, n)
        return bool(any(np.any(o.constraint(line, self.d) < 0) for o in obstacles))

    def _metrics(self, env, samples, line_blocked, target, achieved):
        counts, tilts = {}, {}
        for o in env.obstacles:
            counts[o.kind] = counts.get(o.kind, 0) + 1
            if o.kind in ("spheroid", "oriented_cylinder"):
                tilts.setdefault(o.kind, []).append(tilt_deg(o.axis))
            elif o.kind == "box":
                tilts.setdefault(o.kind, []).append(tilt_deg(o.rotation[:, 2]))
            elif o.kind == "cylinder":
                tilts.setdefault(o.kind, []).append(0.0)
        # fraction of the box the vehicle CENTER cannot reach (obstacles grown by d)
        blocked = (env.constraint_values(samples) < 0).any(axis=0)
        return {"occupancy_target": float(target),
                "occupancy_fraction": achieved,
                "inflated_occupancy_fraction": float(blocked.mean()),
                "straight_line_blocked": line_blocked,
                "straight_line_distance": float(np.linalg.norm(env.goal_pos - env.start_pos)),
                "n_constraint_pieces": len(env.pieces()),
                "obstacle_counts": counts,
                "mean_tilt_deg": {k: float(np.mean(v)) for k, v in tilts.items()}}


# --------------------------------------------------------------------------
# I/O and plotting
# --------------------------------------------------------------------------
_PROJECT_NAMES = ("fixed-wing-rl", "fixed_wing_rl")


def project_root():
    """The enclosing project folder: the nearest parent of this file named
    fixed-wing-rl / fixed_wing_rl, else the nearest one holding a
    pyproject.toml or .git, else the current working directory."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if d.name.lower() in _PROJECT_NAMES:
            return d
    for d in (here, *here.parents):
        if (d / "pyproject.toml").is_file() or (d / ".git").exists():
            return d
    return Path.cwd()


def localstore_dir():
    """Folder that generated files are written to (created if missing)."""
    from pathlib import Path
    d = Path(LOCALSTORE_DIR) if LOCALSTORE_DIR else project_root() / "localstore"
    d.mkdir(parents=True, exist_ok=True)
    return d


def localstore_path(path):
    """Where to WRITE `path`: absolute paths as given, relative ones inside localstore."""
    from pathlib import Path
    p = Path(path)
    out = p if p.is_absolute() else localstore_dir() / p
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def find_input(path):
    """Where to READ `path`: absolute paths as given; relative ones from
    localstore (falling back to the current folder only if not found there)."""
    from pathlib import Path
    p = Path(path)
    if p.is_absolute():
        return p
    q = localstore_dir() / p
    if q.exists():
        return q
    if p.exists():
        return p
    raise FileNotFoundError(f"'{path}' not found in {localstore_dir()} (or the current folder)")


def save_environments(envs, path):
    """Save to localstore (or to `path` itself if absolute). Returns the full path."""
    out = localstore_path(path)
    with open(out, "w") as f:
        json.dump([e.to_dict() for e in envs], f)
    return str(out)


def load_environments(path):
    """Load from localstore (or from `path` itself if absolute)."""
    with open(find_input(path)) as f:
        return [Environment.from_dict(d) for d in json.load(f)]


def plot_environment(env, ax=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
    # matplotlib >= 3.10 can clip 3D artists to the axes box, so obstacles that
    # extend past the workspace (e.g. long tilted cylinders) are drawn cut off
    import matplotlib
    clip = {"axlim_clip": True} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10) else {}

    def draw_box(b, color):
        mesh = clip_polytope(*b.halfspaces(), env.bounds_lo, env.bounds_hi)    # part inside the box
        if mesh is not None:
            V, F = mesh
            ax.add_collection3d(Poly3DCollection([V[f] for f in F], alpha=0.35,
                                                 facecolor=color, edgecolor="k", linewidth=0.2))

    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
    for o in env.obstacles:
        if o.kind == "spheroid":
            s = np.stack([np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w)], -1) * o.radii
            s = s @ o.rotation.T + o.center
            ax.plot_surface(s[..., 0], s[..., 1], s[..., 2], color="tab:orange", alpha=0.45, linewidth=0, **clip)
        elif o.kind == "oriented_cylinder":
            R = o.rotation
            # many rows along the length so clipping trims a cylinder at the box edge
            # instead of dropping whole side panels
            th, sl = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(-o.half_length, o.half_length, 40))
            pts = (o.center + sl[..., None] * R[:, 2]
                   + o.radius * (np.cos(th)[..., None] * R[:, 0] + np.sin(th)[..., None] * R[:, 1]))
            ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], color="tab:purple", alpha=0.45, linewidth=0, **clip)
            for end in o.endpoints():                          # end caps inside the workspace
                if np.any(end < env.bounds_lo) or np.any(end > env.bounds_hi):
                    continue
                rim = end + o.radius * (np.cos(th[0])[:, None] * R[:, 0] + np.sin(th[0])[:, None] * R[:, 1])
                ax.add_collection3d(Poly3DCollection([rim], alpha=0.45, facecolor="tab:purple", linewidth=0), **clip)
        elif o.kind == "cylinder":
            th, z = np.meshgrid(np.linspace(0, 2 * np.pi, 24), [env.bounds_lo[2], o.z_top])
            ax.plot_surface(o.center_xy[0] + o.radius * np.cos(th), o.center_xy[1] + o.radius * np.sin(th),
                            z, color="tab:blue", alpha=0.45, linewidth=0, **clip)
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
def _angle_arg(text):
    """CLI angle range: 'M' -> M (meaning -M..M), 'LO:HI' -> (LO, HI)."""
    try:
        if ":" in text:
            lo, hi = (float(v) for v in text.split(":"))
            return (lo, hi)
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not M or LO:HI (degrees)")


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
    ap.add_argument("--types", nargs="+", default=None, metavar="TYPE=WEIGHT",
                    help="obstacle types and relative weights, e.g. spheroid=1 oriented_cylinder=1 "
                         f"cylinder=0.5 (types: {', '.join(OBSTACLE_TYPES)}). Default: TYPE_WEIGHTS")
    ap.add_argument("--cylinder-rotation", nargs=3, type=_angle_arg, default=None,
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="oriented-cylinder angle ranges in degrees; each is M (= -M..M) or LO:HI. "
                         "0 0 0 = upright, 0 75:90 180 = near-horizontal, 90 90 180 = any "
                         "(default: CYLINDER_ROTATION_DEG)")
    ap.add_argument("--spheroid-rotation", nargs=3, type=_angle_arg, default=None,
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="spheroid angle ranges, same format (default: SPHEROID_ROTATION_DEG)")
    ap.add_argument("--box-rotation", nargs=3, type=_angle_arg, default=None,
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="box angle ranges, same format (default: BOX_ROTATION_DEG, upright boxes)")
    ap.add_argument("--upright-frac", type=float, default=CYLINDER_UPRIGHT_FRAC,
                    help="share of oriented cylinders that are exactly vertical")
    ap.add_argument("--max-walls", type=int, default=MAX_WALLS, help="most walls per environment")
    ap.add_argument("--wall-yaw", type=float, default=WALL_YAW_DEG,
                    help="walls face within +-this many degrees of the start->goal heading")
    ap.add_argument("--wall-pitch", type=float, default=WALL_PITCH_DEG,
                    help="walls lean up to +-this many degrees toward / away from the goal")
    ap.add_argument("--min-size", type=float, default=MIN_OBSTACLE_RADIUS,
                    help="drop obstacles whose equal-volume sphere radius is below this fraction "
                         "of the smallest box dimension (0 = keep all)")
    ap.add_argument("--out", default=OUTPUT_FILE)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=PLOT_FIRST,
                    help="plot the first environment")
    args = ap.parse_args()

    type_weights = dict(TYPE_WEIGHTS)
    if args.types:
        type_weights = {}
        for item in args.types:
            name, _, w = item.partition("=")
            try:
                type_weights[name] = float(w) if w else 1.0
            except ValueError:
                ap.error(f"bad --types entry {item!r}; use TYPE=WEIGHT")
        type_weights = {k: w for k, w in type_weights.items() if w > 0}
    cfg = EnvConfig(size=tuple(args.size), vehicle_radius=args.vehicle_radius, margin=args.margin,
                    occupancy_range=tuple(args.occupancy), endpoint_mode=args.endpoints,
                    type_weights=type_weights,
                    cylinder_rotation_deg=tuple(args.cylinder_rotation or CYLINDER_ROTATION_DEG),
                    spheroid_rotation_deg=tuple(args.spheroid_rotation or SPHEROID_ROTATION_DEG),
                    box_rotation_deg=tuple(args.box_rotation or BOX_ROTATION_DEG),
                    cylinder_upright_frac=args.upright_frac, max_walls=args.max_walls,
                    wall_yaw_deg=args.wall_yaw, wall_pitch_deg=args.wall_pitch,
                    min_obstacle_radius=args.min_size)
    envs = EnvironmentGenerator(cfg, seed=args.seed).generate_many(args.n)
    saved = save_environments(envs, args.out)
    print(f"Saved {len(envs)} environments to {saved}")
    for k, e in enumerate(envs[:5]):
        m = e.metrics
        print(f"  env {k}: {m['obstacle_counts']}, fill {m['occupancy_fraction']:.1%} "
              f"(target {m['occupancy_target']:.1%}), "
              f"inflated fill {m['inflated_occupancy_fraction']:.1%}, mean tilt "
              + ", ".join(f"{k} {v:.0f} deg" for k, v in m.get("mean_tilt_deg", {}).items()))
    if args.plot:
        import matplotlib.pyplot as plt
        plot_environment(envs[0])
        plt.show()


if __name__ == "__main__":
    main()