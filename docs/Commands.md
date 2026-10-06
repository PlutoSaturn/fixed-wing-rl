cd %fixed-wing-rl file path%
python environment/envgen.py --size 12000 2500 600 --vehicle-radius 6 --margin 30 --occupancy 0.06 0.07 --endpoints ends --n 1 --seed 223 --out c172_envs.json --plot

generates --n random trajectories of size --size for a given vehicle radius. --occupancy defines randomized maximum and minimum infill rates. 

____

cd %fixed-wing-rl file path%
python -m solvers.scp_aircraft --envs c172_envs.json --verbose

Solves the set of generated trajectories using SCP methods and saves files.

____

cd %fixed-wing-rl file path%
python -m sim.jsbsim_c172 track --plot 1

flies the generated trajectories that are feasible using SCP, using JSBsim. --plot i to plot the i'th trajectory

____

cd %fixed-wing-rl file path%
python -m sim.jsbsim_c172 fgview 0 --speed 4

Generates a .bat file and scenery to initialize FlightGear simulation. Plays generated maneuvers from JSBsim.

____

[Demo Video](https://huggingface.co/datasets/PlutoJupiter/fixed-wing-rl/resolve/main/SCPdemo.mp4)
