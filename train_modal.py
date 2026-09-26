"""PPO training for a JSBSim fixed-wing env on Modal.

Train (prints the run id):
    modal run --detach train_modal.py::main

Restart a run:
    modal run --detach train_modal.py::main --run-id <id>

Generate trajectories from a run's latest complete checkpoint:
    modal run train_modal.py::generate --run-id <id>
"""

import os
import uuid
from typing import Optional

import modal

app = modal.App("fixed-wing-rl")

image = (
    modal.Image
    .debian_slim(python_version="3.11")
    .apt_install("build-essential", "libgl1", "libglib2.0-0")
    .pip_install(
        "jsbsim",
        "gymnasium",
        "stable-baselines3[extra]",
        "torch",
        "numpy",
        "matplotlib",
    )
)

volume = modal.Volume.from_name("fixed-wing-rl-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"
ACTION_REPEAT = 6
CRASH_PENALTY = 100.0

# "Settled" tolerance band -- being within ALL of these simultaneously counts
# as tracking tightly, not just roughly in the right ballpark. The per-step
# bonus below rewards reaching and staying in this band sooner, which is a
# direct proxy for shorter settling time (more of the episode spent settled).
ALT_TOL_FT = 100.0
HEADING_TOL_DEG = 5.0
ROLL_TOL_DEG = 5.0
AIRSPEED_TOL_KTS = 10.0
SETTLE_BONUS = 0.5
THROTTLE_COST_WEIGHT = 0.05  # small per-step penalty proportional to throttle usage
CONTROL_RATE_COST_WEIGHT = 0.02  # penalizes changing control surfaces too fast step-to-step

# Initial-condition randomization ranges -- widened for more variance across
# episodes and pulled out as named options so they're easy to tune later.
INIT_ALT_RANGE_FT = 500.0        # altitude: target +/- this
INIT_SPEED_RANGE_KTS = (100.0, 140.0)  # airspeed: uniform in this range
INIT_HEADING_RANGE_DEG = 45.0    # heading: target +/- this (was 30)
INIT_ROLL_RANGE_DEG = 15.0       # initial bank angle: +/- this (was 10)
INIT_PITCH_RANGE_DEG = 10.0      # initial pitch angle: +/- this (was 5)


def latest_checkpoint(run_dir: str) -> Optional[str]:
    if not os.path.isdir(run_dir):
        return None
    complete = [
        d
        for d in os.listdir(run_dir)
        if d.startswith("step_")
        and d[5:].isdigit()
        and os.path.isfile(os.path.join(run_dir, d, "model.zip"))
        and os.path.isfile(os.path.join(run_dir, d, "vecnormalize.pkl"))
    ]
    if not complete:
        return None
    return os.path.join(run_dir, max(complete, key=lambda d: int(d[5:])))


def save_checkpoint(model, run_dir: str) -> None:
    step_dir = os.path.join(run_dir, f"step_{model.num_timesteps}")
    os.makedirs(step_dir, exist_ok=True)
    model.save(os.path.join(step_dir, "model.zip"))
    model.get_vec_normalize_env().save(os.path.join(step_dir, "vecnormalize.pkl"))


class FixedWingEnv:
    def __init__(self, aircraft="c172p", dt=1 / 60, max_steps=2000):
        import numpy as np
        from gymnasium import spaces

        self._np = np
        self.aircraft = aircraft
        self.dt = dt
        self.max_steps = max_steps
        self.sim = None
        self.step_count = 0
        self.np_random = np.random.default_rng()

        self.target_altitude_ft = 5000.0
        self.target_heading_deg = 90.0
        self.target_airspeed_kts = 120.0
        self.first_settled_step = None  # step index when it first entered the tolerance band
        self.prev_action = None  # for penalizing rapid control-surface changes

        obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )

    def _init_sim(self):
        import jsbsim

        sim = jsbsim.FGFDMExec(None)
        sim.set_debug_level(0)
        sim.load_model(self.aircraft)
        sim.set_dt(self.dt)
        rng = self.np_random
        sim["ic/h-sl-ft"] = self.target_altitude_ft + rng.uniform(-INIT_ALT_RANGE_FT, INIT_ALT_RANGE_FT)
        sim["ic/vc-kts"] = rng.uniform(*INIT_SPEED_RANGE_KTS)
        sim["ic/gamma-deg"] = 0
        sim["ic/psi-true-deg"] = (self.target_heading_deg + rng.uniform(-INIT_HEADING_RANGE_DEG, INIT_HEADING_RANGE_DEG)) % 360
        sim["ic/phi-deg"] = rng.uniform(-INIT_ROLL_RANGE_DEG, INIT_ROLL_RANGE_DEG)
        sim["ic/theta-deg"] = rng.uniform(-INIT_PITCH_RANGE_DEG, INIT_PITCH_RANGE_DEG)
        sim.run_ic()
        sim["propulsion/set-running"] = -1
        sim["fcs/mixture-cmd-norm"] = 1
        return sim

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = self._np.random.default_rng(seed)
        self.sim = self._init_sim()
        self.step_count = 0
        self.first_settled_step = None
        self.prev_action = self._np.array([0.0, 0.0, 0.0, 0.0], dtype=self._np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = self._np.asarray(action, dtype=self._np.float32)
        aileron, elevator, rudder, throttle = action
        self.sim["fcs/aileron-cmd-norm"] = float(aileron)
        self.sim["fcs/elevator-cmd-norm"] = float(elevator)
        self.sim["fcs/rudder-cmd-norm"] = float(rudder)
        self.sim["fcs/throttle-cmd-norm"] = float(throttle)

        for _ in range(ACTION_REPEAT):
            self.sim.run()
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs, throttle)

        # Penalize changing control surfaces too fast step-to-step (a real
        # servo/actuator shouldn't be commanded to slam around erratically).
        # Excludes throttle -- that's already penalized separately above,
        # and throttle changes are naturally slower/smoother on real engines
        # anyway so it doesn't need the same rate penalty as the surfaces.
        control_rate = self._np.abs(action[:3] - self.prev_action[:3]).sum()
        reward -= CONTROL_RATE_COST_WEIGHT * control_rate
        self.prev_action = action.copy()

        is_settled = (
            abs(obs[0]) < ALT_TOL_FT
            and abs(obs[1]) < HEADING_TOL_DEG
            and abs(obs[2]) < ROLL_TOL_DEG
            and abs(obs[7] - self.target_airspeed_kts) < AIRSPEED_TOL_KTS
        )
        if is_settled:
            reward += SETTLE_BONUS
            if self.first_settled_step is None:
                self.first_settled_step = self.step_count

        alt = self.sim["position/h-sl-ft"]
        roll = self.sim["attitude/phi-deg"]
        pitch = self.sim["attitude/theta-deg"]
        aoa = self.sim["aero/alpha-deg"]
        vc = self.sim["velocities/vc-kts"]
        lost_control = (
            abs(roll) > 60 or abs(pitch) > 45 or alt < 500 or alt > 20000
            or aoa > 14 or vc < 45  # approaching/entering a stall
        )
        terminated = bool(lost_control)
        if terminated:
            reward -= CRASH_PENALTY

        truncated = self.step_count >= self.max_steps
        info = {}
        if truncated or terminated:
            info["first_settled_step"] = self.first_settled_step
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        np = self._np
        s = self.sim
        alt = s["position/h-sl-ft"]
        heading = s["attitude/psi-deg"]
        roll = s["attitude/phi-deg"]
        pitch = s["attitude/theta-deg"]
        p = s["velocities/p-rad_sec"]
        q = s["velocities/q-rad_sec"]
        r = s["velocities/r-rad_sec"]
        vc = s["velocities/vc-kts"]
        vsi = s["velocities/h-dot-fps"]
        aoa = s["aero/alpha-deg"]

        alt_err = alt - self.target_altitude_ft
        heading_err = ((heading - self.target_heading_deg + 180) % 360) - 180

        return np.array(
            [alt_err, heading_err, roll, pitch, p, q, r, vc, vsi, aoa], dtype=np.float32
        )

    def _compute_reward(self, obs, throttle):
        alt_err, heading_err, roll, pitch = obs[0], obs[1], obs[2], obs[3]
        vc, vsi = obs[7], obs[8]
        airspeed_err = vc - self.target_airspeed_kts
        err = (
            abs(alt_err) / 1000
            + abs(heading_err) / 90
            + abs(roll) / 45
            + abs(pitch) / 45
            + abs(airspeed_err) / 50
            + abs(vsi) / 20
        )
        base_reward = max(1.0 - err, -1.0)
        throttle_cost = THROTTLE_COST_WEIGHT * abs(throttle)
        return base_reward - throttle_cost


def _make_gym_env(rank=0, log_dir=None, max_steps=None):
    def _init():
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor

        base = FixedWingEnv() if max_steps is None else FixedWingEnv(max_steps=max_steps)

        class _Wrapped(gym.Env):
            metadata = {}

            def __init__(self):
                super().__init__()
                self.observation_space = base.observation_space
                self.action_space = base.action_space

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return base.reset(seed=seed, options=options)

            def step(self, action):
                return base.step(action)

        env = _Wrapped()
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env

    return _init


def _plot_rewards(log_dir, out_path):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3.common.results_plotter import (
        X_TIMESTEPS,
        load_results,
        ts2xy,
    )

    try:
        results = load_results(log_dir)
    except Exception as e:
        print(f"Could not load monitor logs from {log_dir}: {e}")
        return

    if len(results) == 0:
        print("No completed episodes yet, skipping reward plot.")
        return

    x, y = ts2xy(results, X_TIMESTEPS)
    order = np.argsort(x)
    x, y = x[order], y[order]

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=8, alpha=0.35, label="Episode reward")

    window = min(50, max(1, len(y) // 10))
    if len(y) >= window and window > 1:
        rolling = np.convolve(y, np.ones(window) / window, mode="valid")
        plt.plot(
            x[window - 1 :],
            rolling,
            color="tab:red",
            linewidth=2,
            label=f"Rolling mean ({window} episodes)",
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (raw)")
    plt.title("Reward over training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Reward plot saved to {out_path}")


@app.function(
    image=image,
    gpu="A10G",
    cpu=10,
    timeout=60 * 60 * 6,
    volumes={CHECKPOINT_DIR: volume},
    single_use_containers=True,
)
def train(run_id: str, total_timesteps: int = 2_000_000, n_envs: int = 8):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    volume.reload()
    run_dir = f"{CHECKPOINT_DIR}/{run_id}"
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    raw_env = SubprocVecEnv(
        [_make_gym_env(rank=i, log_dir=log_dir) for i in range(n_envs)],
        start_method="fork",
    )

    ckpt = latest_checkpoint(run_dir)
    tb_log = os.path.join(run_dir, "tb_logs")
    if ckpt:
        print(f"Resuming from {ckpt}")
        env = VecNormalize.load(os.path.join(ckpt, "vecnormalize.pkl"), raw_env)
        model = PPO.load(
            os.path.join(ckpt, "model.zip"),
            env=env,
            device="cuda",
            tensorboard_log=tb_log,
        )
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            n_steps=2048,
            batch_size=256,
            tensorboard_log=tb_log,
        )

    # No periodic checkpoint callback anymore -- only the final save below.
    # NOTE: this means if the run is interrupted (timeout, crash, manual
    # cancellation) before model.learn() returns, NOTHING gets saved at all.
    # That's the tradeoff for simplicity -- fine for shorter/reliable runs,
    # riskier for very long ones where a mid-run failure loses everything.
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=ckpt is None,
    )

    save_checkpoint(model, run_dir)
    volume.commit()
    _plot_rewards(log_dir, os.path.join(run_dir, "reward_plot.png"))
    volume.commit()
    return f"Training complete. Checkpoint saved under {run_dir}."


def _plot_trajectories(episodes, out_path, target_altitude_ft=5000.0, target_heading_deg=90.0):
    """Ground track (dead-reckoned, no wind), altitude, heading, roll over
    time, one episode's control inputs, and a histogram of final tracking
    error -- built from the raw obs/actions saved by generate_trajectories."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def derive_fields(obs):
        alt_err, heading_err, roll, pitch = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        vc = obs[:, 7]
        alt = target_altitude_ft + alt_err
        heading = (target_heading_deg + heading_err) % 360
        return alt, heading, roll, pitch, vc

    def dead_reckon(heading_deg, vc_kts, dt):
        vc_fps = vc_kts * 1.68781
        heading_rad = np.deg2rad(heading_deg)
        dx = vc_fps * np.sin(heading_rad) * dt
        dy = vc_fps * np.cos(heading_rad) * dt
        east = np.concatenate([[0], np.cumsum(dx)[:-1]])
        north = np.concatenate([[0], np.cumsum(dy)[:-1]])
        return east, north

    def tracking_error(obs_row):
        return abs(obs_row[0]) / 1000 + abs(obs_row[1]) / 90 + abs(obs_row[2]) / 45

    dt_per_step = ACTION_REPEAT * (1 / 60)
    n_plot = min(10, len(episodes))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_plot, 1)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    for i in range(n_plot):
        obs = np.asarray(episodes[i]["obs"])
        alt, heading, roll, pitch, vc = derive_fields(obs)
        east, north = dead_reckon(heading, vc, dt_per_step)
        ax.plot(east, north, color=colors[i], linewidth=1)
        ax.plot(east[0], north[0], "o", color=colors[i], markersize=4)
    ax.set_title("Approx. ground track (dead-reckoned, no wind)")
    ax.set_xlabel("East (ft)"); ax.set_ylabel("North (ft)")
    ax.axis("equal"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i in range(n_plot):
        obs = np.asarray(episodes[i]["obs"])
        alt, *_ = derive_fields(obs)
        t = np.arange(len(alt)) * dt_per_step
        ax.plot(t, alt, color=colors[i], linewidth=1)
    ax.axhline(target_altitude_ft, color="gray", linestyle="--", label="Target")
    ax.set_title("Altitude over time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (ft)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    for i in range(n_plot):
        obs = np.asarray(episodes[i]["obs"])
        _, heading, *_ = derive_fields(obs)
        t = np.arange(len(heading)) * dt_per_step
        ax.plot(t, heading, color=colors[i], linewidth=1)
    ax.axhline(target_heading_deg, color="gray", linestyle="--", label="Target")
    ax.set_title("Heading over time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading (deg)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i in range(n_plot):
        obs = np.asarray(episodes[i]["obs"])
        _, _, roll, *_ = derive_fields(obs)
        t = np.arange(len(roll)) * dt_per_step
        ax.plot(t, roll, color=colors[i], linewidth=1)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Roll (bank angle) over time"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Roll (deg)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    actions = np.asarray(episodes[0]["actions"])
    t = np.arange(len(actions)) * dt_per_step
    for j, label in enumerate(["aileron", "elevator", "rudder", "throttle"]):
        ax.plot(t, actions[:, j], linewidth=1, label=label)
    ax.set_title("Control inputs (episode 0)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Command (normalized)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    final_errs = [tracking_error(np.asarray(ep["obs"])[-1]) for ep in episodes]
    ax.hist(final_errs, bins=20, color="tab:blue", alpha=0.75)
    ax.set_title(f"Final tracking error across all {len(episodes)} episodes")
    ax.set_xlabel("Error (lower = better)"); ax.set_ylabel("Episode count")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    ep_lengths = [len(ep["obs"]) for ep in episodes]
    max_len = max(ep_lengths)
    n_full_length = sum(1 for l in ep_lengths if l >= max_len)
    print(f"Trajectory plot saved to {out_path}")
    print(f"Mean episode length: {np.mean(ep_lengths):.0f} steps "
          f"({np.mean(ep_lengths) * dt_per_step:.1f}s of simulated flight)")
    print(f"Episodes reaching full length (no loss-of-control): {n_full_length}/{len(episodes)}")
    print(f"Mean final tracking error: {np.mean(final_errs):.3f}")


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={CHECKPOINT_DIR: volume},
)
def generate_trajectories(run_id: str, n_episodes: int = 100, max_steps: int = 2000):
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    volume.reload()
    run_dir = f"{CHECKPOINT_DIR}/{run_id}"
    ckpt = latest_checkpoint(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No complete checkpoint in {run_dir}")

    model = PPO.load(os.path.join(ckpt, "model.zip"), device="cpu")

    raw_env = DummyVecEnv([_make_gym_env(rank=0, log_dir=None, max_steps=max_steps)])
    env = VecNormalize.load(os.path.join(ckpt, "vecnormalize.pkl"), raw_env)
    env.training = False
    env.norm_reward = False

    all_trajectories = []
    for _ in range(n_episodes):
        obs = env.reset()
        traj = {"obs": [], "actions": []}
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            traj["obs"].append(env.get_original_obs()[0].copy())
            traj["actions"].append(np.array(action[0]).copy())
            obs, _, done_arr, _ = env.step(action)
            done = bool(done_arr[0])
        all_trajectories.append(traj)

    out_path = os.path.join(run_dir, "trajectories.npy")
    np.save(out_path, np.array(all_trajectories, dtype=object), allow_pickle=True)

    plot_path = os.path.join(run_dir, "trajectories_plot.png")
    _plot_trajectories(all_trajectories, plot_path)

    volume.commit()
    return f"Saved {n_episodes} trajectories to {out_path} and plot to {plot_path}"


@app.local_entrypoint()
def main(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    run_id: Optional[str] = None,
):
    run_id = run_id or uuid.uuid4().hex[:12]
    call = train.spawn(run_id, total_timesteps=total_timesteps, n_envs=n_envs)
    print(run_id)
    print(call.object_id)
    call.get()


@app.local_entrypoint()
def generate(run_id: str, n_episodes: int = 100, max_steps: int = 2000):
    print(generate_trajectories.remote(run_id, n_episodes=n_episodes, max_steps=max_steps))