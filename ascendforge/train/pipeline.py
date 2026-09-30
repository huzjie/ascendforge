"""Pipeline parallelism (simulated micro-batching)."""


class PipelineParallel:
    def __init__(self, num_stages=8, num_microbatches=16):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

    def schedule(self):
        """Return a 1F1B schedule (forward/backward slots per stage)."""
        schedule = []
        for mb in range(self.num_microbatches):
            for stage in range(self.num_stages):
                schedule.append((mb, stage, "F"))
        for mb in range(self.num_microbatches):
            for stage in reversed(range(self.num_stages)):
                schedule.append((mb, stage, "B"))
        return schedule

    def bubble_ratio(self):
        n = self.num_stages
        m = self.num_microbatches
        return (n - 1) / (m + n - 1)
