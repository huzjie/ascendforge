"""Communication topologies: ring, mesh, hierarchical (128-card super-node)."""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Topology:
    num_ranks: int
    kind: str = "hierarchical"  # ring / mesh / hierarchical
    rank_groups: List[List[int]] = field(default_factory=list)
    links: Dict = field(default_factory=dict)

    def neighbors(self, rank):
        if self.kind == "ring":
            return [(rank - 1) % self.num_ranks, (rank + 1) % self.num_ranks]
        if self.kind == "mesh":
            side = int(self.num_ranks ** 0.5)
            r, c = divmod(rank, side)
            n = []
            if c > 0:
                n.append(rank - 1)
            if c < side - 1:
                n.append(rank + 1)
            if r > 0:
                n.append(rank - side)
            if r < side - 1:
                n.append(rank + side)
            return n
        # hierarchical: split into groups (simulated super-node pods)
        for grp in self.rank_groups:
            if rank in grp:
                return [g for g in grp if g != rank]
        return []


def build_topology(num_ranks, kind="hierarchical", group_size=8):
    topo = Topology(num_ranks=num_ranks, kind=kind)
    if kind == "hierarchical":
        topo.rank_groups = [list(range(i, min(i + group_size, num_ranks)))
                            for i in range(0, num_ranks, group_size)]
    return topo
