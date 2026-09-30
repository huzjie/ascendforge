"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demo MoE expert parallelism: routing + load balance."""
from ascendforge.moe.moe_layer import MoELayer
from ascendforge.moe.load_balance import aux_loss
from ascendforge.core.tensor import randn

layer = MoELayer(hidden=128, num_experts=8, top_k=2, expert_hidden=256)
x = randn((16, 128), "demo_moe")
out = layer.forward(x)
load = layer.load_counts()
print("output shape:", out.shape)
print("expert load:", load)
print("aux loss   :", aux_loss(load, sum(load)))
