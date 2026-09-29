python envgen.py --size 12000 2500 600 --vehicle-radius 6 --margin 30 --occupancy 0.05 0.08 --endpoints ends --n 3 --seed 1 --out c172_envs.json --plot

generates --n random trajectories of size --size for a given vehicle radius. --occupancy defines randomized maximum and minimum infill rates. 

____

python scp_aircraft.py --envs c172_envs.json --verbose

Solves the set of generated trajectories using SCP methods and saves files.

____

python jsbsim_c172.py track --plot 1

flies the generated trajectories that are feasible using SCP, using JSBsim. --plot i to plot the i'th trajectory

____

python jsbsim_c172.py fgview 1 --speed 4

Generates a .bat file and scenery to initialize FlightGear simulation. Plays generated maneuvers from JSBsim.

____

[Demo Video](https://huggingface.co/datasets/PlutoJupiter/fixed-wing-rl/resolve/main/SCPdemo.mp4)
