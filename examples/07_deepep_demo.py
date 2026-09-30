"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demo DeepEP communication: dispatch/combine for MoE."""
from ascendforge.comm.deepep import DeepEP
from ascendforge.comm.schedule import SuperNodeScheduler

ep = DeepEP(num_ranks=8, num_experts=8)
routed = ep.dispatch(list(range(100)))
print("dispatch routing:", {k: len(v) for k, v in routed.items()})
print("stats:", ep.report())

sched = SuperNodeScheduler(num_ranks=128, group_size=8)
print("super-node plan:", sched.plan())
