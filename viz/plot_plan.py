import json, sys
from types import SimpleNamespace
import numpy as np
from benchmarks.envgen import load_environments
from solvers.scp_aircraft import AircraftSCP, plot_result

env_file = "c172_envs.json"
results_file = "scp_aircraft_results.json"
index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

envs = load_environments(env_file)
r = next(r for r in json.load(open(results_file)) if r["env_index"] == index)
res = SimpleNamespace(x=np.array(r["x"]), u=np.array(r["u"]),
                      flight_time=r["flight_time"], status=r["status"])
plot_result(envs[index], res, AircraftSCP())