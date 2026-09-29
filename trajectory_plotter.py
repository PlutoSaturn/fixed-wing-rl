
import argparse
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
TARGET_ALTITUDE_FT = 5000.0
TARGET_HEADING_DEG = 90.0
ACTION_REPEAT = 6
PHYSICS_DT = 1 / 60
DT_PER_STEP = ACTION_REPEAT * PHYSICS_DT  # seconds of sim time per recorded step
 
 
def load_episodes(path):
    data = np.load(path, allow_pickle=True)
    episodes = []
    for ep in data:
        obs = np.asarray(ep["obs"], dtype=np.float64)        # (T, 10)
        actions = np.asarray(ep["actions"], dtype=np.float64)  # (T, 4)
        episodes.append({"obs": obs, "actions": actions})
    return episodes
 
 
def derive_fields(obs):
    alt_err, heading_err, roll, pitch = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    vc = obs[:, 7]
    alt = TARGET_ALTITUDE_FT + alt_err
    heading = (TARGET_HEADING_DEG + heading_err) % 360
    return alt, heading, roll, pitch, vc
 
 
def dead_reckon(heading_deg, vc_kts, dt=DT_PER_STEP):
    """Approximate ground track by integrating airspeed + heading.
    Ignores wind entirely -- true position would require logging it
    directly from the sim during generate_trajectories()."""
    vc_fps = vc_kts * 1.68781  # knots -> ft/s
    heading_rad = np.deg2rad(heading_deg)
    dx = vc_fps * np.sin(heading_rad) * dt  # east, ft
    dy = vc_fps * np.cos(heading_rad) * dt  # north, ft
    east = np.concatenate([[0], np.cumsum(dx)[:-1]])
    north = np.concatenate([[0], np.cumsum(dy)[:-1]])
    return east, north
 
 
def tracking_error(obs_row):
    alt_err, heading_err, roll = obs_row[0], obs_row[1], obs_row[2]
    return abs(alt_err) / 1000 + abs(heading_err) / 90 + abs(roll) / 45
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories_path")
    parser.add_argument("--max-episodes-plotted", type=int, default=10)
    parser.add_argument("--out", default="trajectories_analysis.png")
    args = parser.parse_args()
 
    episodes = load_episodes(args.trajectories_path)
    print(f"Loaded {len(episodes)} episodes")
 
    n_plot = min(args.max_episodes_plotted, len(episodes))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_plot, 1)))
 
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
 
    # 1. Approximate ground track
    ax = axes[0, 0]
    for i in range(n_plot):
        alt, heading, roll, pitch, vc = derive_fields(episodes[i]["obs"])
        east, north = dead_reckon(heading, vc)
        ax.plot(east, north, color=colors[i], linewidth=1)
        ax.plot(east[0], north[0], "o", color=colors[i], markersize=4)
    ax.set_title("Approx. ground track (dead-reckoned, no wind)")
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
 
    # 2. Altitude over time
    ax = axes[0, 1]
    for i in range(n_plot):
        alt, *_ = derive_fields(episodes[i]["obs"])
        t = np.arange(len(alt)) * DT_PER_STEP
        ax.plot(t, alt, color=colors[i], linewidth=1)
    ax.axhline(TARGET_ALTITUDE_FT, color="gray", linestyle="--", label="Target")
    ax.set_title("Altitude over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (ft)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # 3. Heading over time
    ax = axes[0, 2]
    for i in range(n_plot):
        _, heading, *_ = derive_fields(episodes[i]["obs"])
        t = np.arange(len(heading)) * DT_PER_STEP
        ax.plot(t, heading, color=colors[i], linewidth=1)
    ax.axhline(TARGET_HEADING_DEG, color="gray", linestyle="--", label="Target")
    ax.set_title("Heading over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # 4. Roll over time
    ax = axes[1, 0]
    for i in range(n_plot):
        _, _, roll, *_ = derive_fields(episodes[i]["obs"])
        t = np.arange(len(roll)) * DT_PER_STEP
        ax.plot(t, roll, color=colors[i], linewidth=1)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Roll (bank angle) over time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Roll (deg)")
    ax.grid(alpha=0.3)
 
    # 5. Actions for one example episode
    ax = axes[1, 1]
    actions = episodes[0]["actions"]
    t = np.arange(len(actions)) * DT_PER_STEP
    for j, label in enumerate(["aileron", "elevator", "rudder", "throttle"]):
        ax.plot(t, actions[:, j], linewidth=1, label=label)
    ax.set_title("Control inputs (episode 0)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Command (normalized)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # 6. Distribution of final tracking error across ALL episodes
    ax = axes[1, 2]
    final_errs = [tracking_error(ep["obs"][-1]) for ep in episodes]
    ep_lengths = [len(ep["obs"]) for ep in episodes]
    ax.hist(final_errs, bins=20, color="tab:blue", alpha=0.75)
    ax.set_title(f"Final tracking error across all {len(episodes)} episodes")
    ax.set_xlabel("Error (lower = better)")
    ax.set_ylabel("Episode count")
    ax.grid(alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
 
    max_len = max(ep_lengths)
    n_full_length = sum(1 for l in ep_lengths if l >= max_len)
    print(f"Saved plot to {args.out}")
    print(f"Mean episode length: {np.mean(ep_lengths):.0f} steps "
          f"({np.mean(ep_lengths) * DT_PER_STEP:.1f}s of simulated flight)")
    print(f"Episodes reaching full length (no loss-of-control): {n_full_length}/{len(episodes)}")
    print(f"Mean final tracking error: {np.mean(final_errs):.3f}")
 
 
if __name__ == "__main__":
    main()
 