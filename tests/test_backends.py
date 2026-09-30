import unittest

from ascendforge.backends import make_backend, list_backends


class TestBackends(unittest.TestCase):
    def test_list_backends(self):
        self.assertIn("mock", list_backends())
        self.assertIn("cpu", list_backends())
        self.assertIn("ascend", list_backends())
        self.assertIn("openai", list_backends())

    def test_mock_training_monotone(self):
        b = make_backend("mock")
        s0 = b.current_skill()
        for _ in range(20):
            b.train_step(_)
        self.assertGreater(b.current_skill(), s0)

    def test_mock_eval(self):
        b = make_backend("mock")
        res = b.evaluate()
        self.assertIn("accuracy", res)


if __name__ == "__main__":
    unittest.main()
