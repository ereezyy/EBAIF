import sys
import unittest
from unittest.mock import MagicMock
import os

# Set up mocks for torch and numpy before importing BehaviorGenome
class MockTensor:
    def __init__(self, *args, **kwargs):
        pass

    def __gt__(self, other):
        return MockTensor()

    def __lt__(self, other):
        return MockTensor()

    def __ge__(self, other):
        return MockTensor()

    def __le__(self, other):
        return MockTensor()

    def item(self):
        return 0.0

    def __setitem__(self, key, value):
        pass

    def __getattr__(self, name):
        return MockTensor()

    def __call__(self, *args, **kwargs):
        return MockTensor()

    def clone(self):
        return self

    def tolist(self):
        return []

    def __iter__(self):
        return iter([MockTensor()])

    def __len__(self):
        return 1

    def __bool__(self):
        return True

mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.tensor.side_effect = lambda *args, **kwargs: MockTensor()
mock_torch.zeros.side_effect = lambda *args, **kwargs: MockTensor()
mock_torch.rand.side_effect = lambda *args, **kwargs: MockTensor()
mock_torch.randint.side_effect = lambda *args, **kwargs: MockTensor()
mock_torch.float32 = "float32"
# Mock torch.bool for architecture genes mutation logic
mock_torch.bool = "bool"

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()

mock_numpy = MagicMock()
mock_numpy.random.randint.return_value = 12345
sys.modules["numpy"] = mock_numpy
sys.modules["ebaif.consensus.protocol"] = MagicMock()
sys.modules["ebaif.consensus.reputation"] = MagicMock()

# Add src to sys.path to allow importing the module
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from ebaif.behavior_genome.genome import BehaviorGenome

class TestBehaviorGenomeFitness(unittest.TestCase):
    def test_fitness_history_limit(self):
        """Test that performance history is limited to 1000 entries."""
        genome = BehaviorGenome()

        # Verify initial state
        self.assertEqual(len(genome.performance_history), 0)

        # Fill history up to the limit
        for i in range(1000):
            genome.update_fitness(float(i))

        self.assertEqual(len(genome.performance_history), 1000)
        self.assertEqual(genome.performance_history[0], 0.0)
        self.assertEqual(genome.performance_history[-1], 999.0)

        # Add more entries to exceed the limit
        for i in range(1000, 1500):
            genome.update_fitness(float(i))

        # Verify limit is maintained
        self.assertEqual(len(genome.performance_history), 1000)

        # Verify oldest entries are removed (should start from 500)
        self.assertEqual(genome.performance_history[0], 500.0)
        self.assertEqual(genome.performance_history[-1], 1499.0)

if __name__ == '__main__':
    unittest.main()
