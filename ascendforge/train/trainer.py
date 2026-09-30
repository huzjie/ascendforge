"""Training loop (deterministic mock-friendly)."""
from .optimizer import AdamW
from .scheduler import CosineScheduler
from ..utils.stable import stable_float


class Trainer:
    def __init__(self, backend, cfg=None, key="train"):
        self.backend = backend
        self.cfg = cfg
        self.key = key
        self.optimizer = AdamW(lr=3e-4)
        self.scheduler = CosineScheduler()
        self.history = {"loss": [], "skill": [], "lr": []}

    def train(self, steps=500, log_every=100):
        for step in range(steps):
            lr = self.scheduler.lr_at(step)
            loss = self.backend.train_step(step)
            self.history["loss"].append(loss)
            self.history["lr"].append(lr)
            skill = self.backend.current_skill()
            self.history["skill"].append(skill)
            if step % log_every == 0 or step == steps - 1:
                print(f"[train] step={step} loss={loss:.4f} skill={skill:.4f} lr={lr:.6f}", flush=True)
        return self.history

    def evaluate(self):
        return self.backend.evaluate()
