"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Forward a token sequence through the MoE transformer."""
from ascendforge.model.transformer import AscendTransformer
from ascendforge.model.model_config import ModelConfig

cfg = ModelConfig(num_layers=2, hidden_size=128, num_heads=4,
                  num_experts=4, top_k=2, expert_hidden=256, vocab_size=1000)
model = AscendTransformer(cfg)
logits = model.forward([1, 2, 3, 4])
print("logits shape:", logits.shape)
