"""Learning-rate schedulers."""
import math


class WarmupScheduler:
    def __init__(self, warmup_steps=100, base_lr=3e-4):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr

    def lr_at(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        return self.base_lr


class CosineScheduler:
    def __init__(self, max_steps=2000, base_lr=3e-4, min_lr=3e-5, warmup_steps=100):
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup = WarmupScheduler(warmup_steps, base_lr)

    def lr_at(self, step):
        if step < self.warmup.warmup_steps:
            return self.warmup.lr_at(step)
        progress = (step - self.warmup.warmup_steps) / max(1, self.max_steps - self.warmup.warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return self.min_lr + (self.base_lr - self.min_lr) * coeff
