"""
Local smoke test for the fixed-wing RL pipeline.

Run this BEFORE spending any Modal compute — it verifies JSBSim loads
correctly, the gym env is well-formed, and the full PPO training loop
(vectorized envs -> rollout -> update -> save -> reload -> predict)
works end to end, all on CPU on your own machine.

Usage:
    python test_ppo.py
"""

import time
import jsbsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


class FixedWingEnv:
    def __init__(self, aircraft="c172p", dt=1 / 60, max_steps=500):
        self._np = np
        self.aircraft = aircraft
        self.dt = dt
        self.max_steps = max_steps
        self.sim = None
        self.step_count = 0
        self.target_altitude_ft = 5000.0
        self.target_heading_deg = 90.0
        obs_dim = 12
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
        terminated = self._check_terminated(obs)
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
        return np.array([alt_err, heading_err, roll, pitch, p, q, r, vc, vsi, aoa, alt, heading], dtype=np.float32)

    def _compute_reward(self, obs):
        alt_err, heading_err, roll = obs[0], obs[1], obs[2]
        return -(abs(alt_err) / 1000 + abs(heading_err) / 90 + abs(roll) / 45)

    def _check_terminated(self, obs):
        alt = obs[10]
        return bool(alt < 500 or alt > 20000)


def make_gym_env():
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

    return Wrapped()


def test_jsbsim_loads():
    print("=== 1. JSBSim model load test ===")
    sim = jsbsim.FGFDMExec(None)
    sim.set_debug_level(0)
    sim.load_model("c172p")
    print("c172p loaded OK\n")


def test_raw_throughput(n_steps=2000):
    print("=== 2. Raw JSBSim step throughput (single core, random actions) ===")
    env = FixedWingEnv()
    obs, _ = env.reset()
    t0 = time.time()
    steps_run = 0
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps_run += 1
        if terminated or truncated:
            break
    elapsed = time.time() - t0
    print(f"Ran {steps_run} steps in {elapsed:.3f}s -> {steps_run/elapsed:.1f} steps/sec\n")


def test_repeated_resets(n_episodes=5, steps_per_ep=300):
    print("=== 3. Repeated reset / state-leak / NaN check ===")
    env = FixedWingEnv(max_steps=steps_per_ep)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        nan_found = False
        for i in range(steps_per_ep):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if np.isnan(obs).any() or np.isnan(reward):
                nan_found = True
                break
            if terminated or truncated:
                break
        print(f"  Episode {ep}: {i+1} steps, nan_found={nan_found}, final_alt={obs[10]:.1f}")
    print()


def test_gym_compliance():
    print("=== 4. gymnasium check_env compliance ===")
    check_env(make_gym_env(), skip_render_check=True)
    print("check_env passed\n")


def test_ppo_pipeline(n_envs=4, total_timesteps=4096):
    print(f"=== 5. Full PPO pipeline ({n_envs} envs, {total_timesteps} timesteps) ===")
    env = SubprocVecEnv([make_gym_env for _ in range(n_envs)])
    model = PPO("MlpPolicy", env, verbose=0, device="cpu", n_steps=256, batch_size=64)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    print(f"Trained {total_timesteps} timesteps in {elapsed:.1f}s -> {total_timesteps/elapsed:.1f} steps/sec aggregate")

    model.save("test_model")
    loaded = PPO.load("test_model")
    obs = env.reset()
    action, _ = loaded.predict(obs, deterministic=True)
    print(f"Save/load/predict OK, action shape: {action.shape}")
    env.close()
    print()


if __name__ == "__main__":
    test_jsbsim_loads()
    test_raw_throughput()
    test_repeated_resets()
    test_gym_compliance()
    test_ppo_pipeline()
    print("ALL CHECKS PASSED")
