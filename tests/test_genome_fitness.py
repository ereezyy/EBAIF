import sys
import os
import unittest
from unittest.mock import MagicMock
import importlib.util

# --- Mocking Dependencies ---
# Since torch and numpy are not installed in this environment, we mock them.

class MockTensor:
    """Mocks torch.Tensor behavior needed for initialization."""
    def __init__(self, *args, **kwargs):
        pass

    def item(self):
        return 0.0

    def clone(self):
        return self

    def tolist(self):
        return []

    def __gt__(self, other):
        return MockTensor()

    def __lt__(self, other):
        return MockTensor()

    def __getitem__(self, key):
        return MockTensor()

    def __setitem__(self, key, value):
        pass

    def __repr__(self):
        return "MockTensor()"

mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.tensor = MagicMock(return_value=MockTensor())
mock_torch.zeros = MagicMock(return_value=MockTensor())
mock_torch.rand = MagicMock(return_value=MockTensor())
mock_torch.randint = MagicMock(return_value=MockTensor())
mock_torch.float32 = "float32"

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()

mock_numpy = MagicMock()
mock_numpy.random.randint.return_value = 12345
sys.modules["numpy"] = mock_numpy

# --- Import System Under Test ---

# We import the module directly by file path to avoid importing the top-level 'ebaif' package
# which has broken dependencies (missing ebaif.consensus.protocol).

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ebaif/behavior_genome/genome.py'))
spec = importlib.util.spec_from_file_location("genome", src_path)
genome_module = importlib.util.module_from_spec(spec)
sys.modules["genome"] = genome_module
spec.loader.exec_module(genome_module)

BehaviorGenome = genome_module.BehaviorGenome

class TestBehaviorGenomeFitness(unittest.TestCase):
    def test_fitness_history_limit(self):
        """Test that performance history is strictly limited to 1000 entries."""
        genome = BehaviorGenome()

        # Verify initial state
        self.assertEqual(len(genome.performance_history), 0)

        # Fill history up to the limit (0 to 999)
        for i in range(1000):
            genome.update_fitness(float(i))

        self.assertEqual(len(genome.performance_history), 1000)
        self.assertEqual(genome.performance_history[0], 0.0)
        self.assertEqual(genome.performance_history[-1], 999.0)

        # Add more entries to exceed the limit (1000 to 1499)
        for i in range(1000, 1500):
            genome.update_fitness(float(i))

        # Verify limit is maintained
        self.assertEqual(len(genome.performance_history), 1000)

        # Verify oldest entries are removed.
        # History should now contain 500.0 to 1499.0
        self.assertEqual(genome.performance_history[0], 500.0)
        self.assertEqual(genome.performance_history[-1], 1499.0)

if __name__ == '__main__':
    unittest.main()
