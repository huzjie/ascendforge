"""Super-node scheduling (Ascend 950 128-card super-node, simulated)."""


class SuperNodeScheduler:
    """Simulate a 128-card super-node: data-parallel pods x expert-parallel within pod."""

    def __init__(self, num_ranks=128, group_size=8):
        self.num_ranks = num_ranks
        self.group_size = group_size
        self.num_groups = num_ranks // group_size

    def plan(self):
        plan = {
            "num_ranks": self.num_ranks,
            "num_groups": self.num_groups,
            "group_size": self.group_size,
            "strategy": "data-parallel x expert-parallel (hierarchical)",
        }
        return plan

    def placement(self, rank):
        return {"pod": rank // self.group_size, "local_rank": rank % self.group_size}
