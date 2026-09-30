"""DeepEP communication primitives (simulated, deterministic).

Implements MoE expert-parallel communication:
  - dispatch: route tokens to their assigned experts
  - combine:  gather expert outputs back to the origin ranks
  - all_to_all: full cross-rank exchange (non-MoE fallback)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepEP:
    num_ranks: int = 8
    num_experts: int = 8
    topology_kind: str = "hierarchical"
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        from .topology import build_topology
        self.topo = build_topology(self.num_ranks, self.topology_kind)
        self.stats = {"dispatch_tokens": 0, "combine_tokens": 0, "hops": 0}

    def dispatch(self, tokens: List[List[int]]):
        """Route tokens to experts: returns {expert_id: [token...]}."""
        routing = {}
        for t in tokens:
            expert = t % self.num_experts
            routing.setdefault(expert, []).append(t)
        self.stats["dispatch_tokens"] += sum(len(v) for v in routing.values())
        self.stats["hops"] += self._estimate_hops()
        return routing

    def combine(self, expert_outputs: dict):
        """Gather expert outputs back to origin ranks (identity reference)."""
        flat = []
        for _, outs in sorted(expert_outputs.items()):
            flat.extend(outs)
        self.stats["combine_tokens"] += len(flat)
        return flat

    def all_to_all(self, data: List):
        """Full cross-rank exchange."""
        return list(reversed(data))

    def _estimate_hops(self):
        return 1 if self.topology_kind == "hierarchical" else self.num_ranks // 2

    def report(self):
        return dict(self.stats)


def dispatch_combine(deepep: DeepEP, tokens, expert_outputs):
    routed = deepep.dispatch(tokens)
    combined = deepep.combine(expert_outputs)
    return routed, combined


def all_to_all(deepep: DeepEP, data):
    return deepep.all_to_all(data)
