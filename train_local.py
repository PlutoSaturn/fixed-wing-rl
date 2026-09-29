
import argparse
import os

import jsbsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
VECNORM_PATH = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")


class FixedWingEnv:
    def __init__(self, aircraft="c172p", dt=1 / 60, max_steps=1000):
        self._np = np
        self.aircraft = aircraft
        self.dt = dt
        self.max_steps = max_steps
        self.sim = None
        self.step_count = 0
        self.target_altitude_ft = 5000.0
        self.target_heading_deg = 90.0

        # 10 features: alt_err, heading_err, roll, pitch, p, q, r, vc, vsi, aoa
        # (dropped raw altitude/heading -- redundant with the error terms, and
        #  raw heading has a 359->0 wraparound discontinuity that confuses the net)
        obs_dim = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
        )

    def _init_sim(self):
        sim = jsbsim.FGFDMExec(None)
        sim.set_debug_level(0)
        sim.load_model(self.aircraft)
        sim.set_dt(self.dt)
        sim["ic/h-sl-ft"] = self.target_altitude_ft
        sim["ic/vc-kts"] = 120
        sim["ic/gamma-deg"] = 0
        sim["ic/psi-true-deg"] = self.target_heading_deg
        sim.run_ic()
        return sim

    def reset(self, seed=None, options=None):
        self.sim = self._init_sim()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        aileron, elevator, rudder, throttle = action
        self.sim["fcs/aileron-cmd-norm"] = float(aileron)
        self.sim["fcs/elevator-cmd-norm"] = float(elevator)
        self.sim["fcs/rudder-cmd-norm"] = float(rudder)
        self.sim["fcs/throttle-cmd-norm"] = float(throttle)
        self.sim.run()
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)

        # Check for loss of control directly from sim state (not from the
        # obs vector) so this stays correct regardless of what's in obs.
        alt = self.sim["position/h-sl-ft"]
        roll = self.sim["attitude/phi-deg"]
        pitch = self.sim["attitude/theta-deg"]
        lost_control = abs(roll) > 60 or abs(pitch) > 45 or alt < 500 or alt > 20000
        terminated = bool(lost_control)
        if terminated:
            reward -= 50.0  # clear penalty so the policy learns to avoid this fast

        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
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

        return np.array([alt_err, heading_err, roll, pitch, p, q, r, vc, vsi, aoa], dtype=np.float32)

    def _compute_reward(self, obs):
        alt_err, heading_err, roll = obs[0], obs[1], obs[2]
        return -(abs(alt_err) / 1000 + abs(heading_err) / 90 + abs(roll) / 45)


def make_gym_env(rank=0, log_dir=LOG_DIR):
    def _init():
        base = FixedWingEnv()

        class Wrapped(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = base.observation_space
                self.action_space = base.action_space

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return base.reset(seed=seed, options=options)

            def step(self, action):
                return base.step(action)

        env = Wrapped()
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}"))
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    resume_path = os.path.join(CHECKPOINT_DIR, f"{args.resume_from}.zip") if args.resume_from else None
    starting_fresh = not (resume_path and os.path.exists(resume_path))
    if starting_fresh and os.path.isdir(LOG_DIR):
        for f in os.listdir(LOG_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(LOG_DIR, f))

    raw_env = SubprocVecEnv([make_gym_env(rank=i) for i in range(args.n_envs)])

    if starting_fresh:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = VecNormalize.load(VECNORM_PATH, raw_env)
        env.training = True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5_000 // args.n_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_fixedwing",
    )

    if not starting_fresh:
        print(f"Resuming from {resume_path}")
        model = PPO.load(resume_path, env=env, device="cpu")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",  # no GPU needed locally for a small MLP policy
            n_steps=256,
            batch_size=64,
        )
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    final_path = os.path.join(CHECKPOINT_DIR, "ppo_fixedwing_final")
    model.save(final_path)
    env.save(VECNORM_PATH)
    print(f"\nDone. Model saved to {final_path}.zip")
    print(f"Normalization stats saved to {VECNORM_PATH}")
    env.close()

    plot_rewards()


def plot_rewards(log_dir=LOG_DIR, out_path=None):
    """Load per-episode rewards logged by Monitor and plot reward vs.
    training progress (cumulative timesteps across all parallel envs).
    Note: these are RAW rewards (Monitor logs before VecNormalize scales
    them for the algorithm), so they're directly comparable across runs."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed, just save to file
    import matplotlib.pyplot as plt
    from stable_baselines3.common.results_plotter import load_results, ts2xy, X_TIMESTEPS

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
        plt.plot(x[window - 1:], rolling, color="tab:red", linewidth=2, label=f"Rolling mean ({window} episodes)")

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (raw)")
    plt.title("Reward over training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = out_path or os.path.join(CHECKPOINT_DIR, "reward_plot.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Reward plot saved to {out_path}")


if __name__ == "__main__":
    main()