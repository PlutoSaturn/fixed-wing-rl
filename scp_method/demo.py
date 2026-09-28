"""
demo.py
========
Standalone example of Method A solving a single benchmark instance:
a fixed-wing point-to-point trajectory through a small obstacle field.

Run with:
    pip install -r requirements.txt
    python demo.py

This mirrors the kind of instance your teammate's phase-01 benchmark
generator will eventually produce -- swap `build_demo_problem()` for
their generator's output once it exists; everything downstream
(the solver, the Solution schema) stays the same.
"""

from __future__ import annotations

import numpy as np

from rttg_method_a import (
    SCvxSolver,
    SCvxConfig,
    TrajectoryProblem,
    Obstacle,
    CostWeights,
    FixedWingPointMass3DOF,
    FixedWingParams,
)


def build_demo_problem(instance_id: str = "demo-0001", is_ood: bool = False) -> TrajectoryProblem:
    dynamics = FixedWingPointMass3DOF(FixedWingParams())
    path_constraints = dynamics.default_path_bounds()

    # Level start and end, 800 m apart, cruising speed.
    x0 = np.array([0.0, 0.0, 120.0, 20.0, 0.0, 0.0])
    xf = np.array([800.0, 150.0, 120.0, 20.0, 0.0, 0.0])

    obstacles = [
        Obstacle(center=np.array([250.0, 40.0, 120.0]), radius=60.0),
        Obstacle(center=np.array([500.0, -20.0, 120.0]), radius=50.0),
    ]

    if is_ood:
        # A deliberately out-of-distribution instance: tighter obstacle
        # field and a larger lateral offset relative to nominal spacing.
        obstacles = [
            Obstacle(center=np.array([220.0, 60.0, 120.0]), radius=70.0),
            Obstacle(center=np.array([420.0, -60.0, 120.0]), radius=70.0),
            Obstacle(center=np.array([620.0, 40.0, 120.0]), radius=60.0),
        ]
        xf = np.array([800.0, 250.0, 120.0, 20.0, 0.0, 0.0])

    return TrajectoryProblem(
        instance_id=instance_id,
        dynamics=dynamics,
        N=30,
        t0=0.0,
        tf=45.0,
        x0=x0,
        xf=xf,
        path_constraints=path_constraints,
        obstacles=obstacles,
        cost=CostWeights(control_effort=1.0),
        is_ood=is_ood,
    )


def main():
    problem = build_demo_problem()
    solver = SCvxSolver(SCvxConfig(
        verbose=True,
        max_iterations=25,
        eta_init=100.0,
        eta1=500.0
    ))

    print(f"Solving {problem.instance_id} (N={problem.N} nodes, "
          f"{len(problem.obstacles)} obstacles)...")
    solution = solver.timed_solve(problem)

    print()
    print(f"success={solution.success}  converged={solution.converged}  "
          f"feasible={solution.feasible}")
    print(f"iterations={solution.iterations}  wall_time_s={solution.wall_time_s:.4f}")
    print(f"cost={solution.cost:.3f}  virtual_control_norm={solution.virtual_control_norm:.3e}")
    print(f"max_constraint_violation={solution.max_constraint_violation:.3e}")

    if solution.x is not None:
        print()
        print("Final state:", np.round(solution.x[-1], 3))
        print("Target state:", problem.xf)

    # Optional: plot the ground track if matplotlib is available.
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(solution.x[:, 0], solution.x[:, 1], "o-", label="trajectory")
        for obs in problem.obstacles:
            circle = plt.Circle(obs.center[:2], obs.radius, color="r", alpha=0.3)
            ax.add_patch(circle)
        ax.plot(*problem.x0[:2], "g^", markersize=12, label="start")
        ax.plot(*problem.xf[:2], "b*", markersize=14, label="goal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{problem.instance_id} -- Method A (cold-start SCvx)")
        ax.legend()
        ax.axis("equal")
        fig.savefig("demo_trajectory.png", dpi=150)
        print("\nSaved plot to demo_trajectory.png")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()