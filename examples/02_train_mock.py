"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Train the deterministic mock backend: skill rises 0.5 -> ~1.0."""
from ascendforge.backends import make_backend
from ascendforge.train.trainer import Trainer

backend = make_backend("mock")
trainer = Trainer(backend)
history = trainer.train(steps=200, log_every=50)
print("final_skill =", history["skill"][-1])
print("final_loss  =", history["loss"][-1])
print("eval        =", trainer.evaluate())
