import unittest

from ascendforge.comm.deepep import DeepEP
from ascendforge.comm.topology import build_topology
from ascendforge.comm.schedule import SuperNodeScheduler


class TestComm(unittest.TestCase):
    def test_dispatch(self):
        ep = DeepEP(num_ranks=8, num_experts=8)
        routed = ep.dispatch(list(range(16)))
        self.assertEqual(sum(len(v) for v in routed.values()), 16)

    def test_topology_neighbors(self):
        topo = build_topology(16, "ring")
        self.assertEqual(len(topo.neighbors(0)), 2)

    def test_super_node(self):
        s = SuperNodeScheduler(num_ranks=128, group_size=8)
        self.assertEqual(s.num_groups, 16)


if __name__ == "__main__":
    unittest.main()
