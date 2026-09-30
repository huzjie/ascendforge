"""Ascend NPU backend (stub + simulated performance model).

Real Ascend execution requires the CANN/Ascend C toolchain; this backend exposes
the same interface and reports a simulated performance model for planning.
"""
from ..core.device import Device
from .base import Backend
from . import register_backend


@register_backend("ascend")
class AscendBackend(Backend):
    name = "ascend"

    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.device = Device.NPU
        self.peak_tflops = 320.0  # simulated Ascend 950 per-card

    def generate(self, prompt, **kwargs):
        return {"device": self.device.value, "prompt": prompt}

    def train_step(self, step):
        return 0.0

    def current_skill(self):
        return 1.0

    def evaluate(self):
        return {"backend": "ascend", "peak_tflops": self.peak_tflops}

    def perf_model(self, flops, efficiency=0.5):
        """Estimate compute time in seconds."""
        return flops / (self.peak_tflops * 1e12 * efficiency)
