Two Solvers: Birkhoff and Shooting

Birkhoff Solver is a form of Pseudospectral Solver. Pseudospectral solvers track rate of change of state across the path and then integrate that to get the state.
State and rate of change of state are tied to eachother via ODE's in this setup. At nodes, the ODE relating the state derivatives (rate of change), and rates must be satified.
This is how Pseudospectral method enforces dynamics. Birkhoff specificlly builds off a standard pseudospectral method by simplifying the integration step. It does this by
describing the state derivates at each node using a continuous polynomial that can be integrated with a matrix. This reduces error during the integration and maybe makes it faster.

Instead of the Pseudospectral method of tracking state derivative and integrating, the shooting method tracks state. Nodes positions are related to eachother using RK4 which estimates
state 2 from state 1 and the derivative of state 1. These RK4 equations are the method of enforcing dynamics, and are performed between nodes.

______________________

Some ideas for benchmarking:

How does the ability of the two systems to converge off an intiail guess differ? Is birkhoff more likely to come to a feasible solution if both are initialized off the exact same path?

The construction of matrices are very different for each. Birkhoff should produce a dense matrix and shooting a sparse matrix. How do the compute times vary with size and difficulty of environment?

Birkhoff integration should lead to less error accumulation over longer flights. The system uses a dynamic verifier (JSBsim_c1722.py) to look at how much error the path generated has 
over the course of the flight and whether the path is flyable. How much less often does Birkhoff fail or stray from the real dynamics of the aircraft? Both solvers have issues failing dynamics 
-- why is this in the first place?

A feasible solution must be close to the initial guess, within a certain region for the solver to find it. How is this region different between solvers (similar to first idea), but 
also what can this difference be attributed to.

______________________

Modal Setup -> run "pip install modal" and it should guide you
