import sys
import unittest
from unittest.mock import MagicMock
import os

# Create a MockTensor class to handle comparison operators and other tensor methods
class MockTensor:
    def __init__(self, *args, **kwargs):
        pass

    def __gt__(self, other):
        return MockTensor()

    def __lt__(self, other):
        return MockTensor()

    def item(self):
        return 0.0

    def __float__(self):
        return 0.0

    def __setitem__(self, key, value):
        pass

    def clone(self):
        return self

    def tolist(self):
        return []

    def __getattr__(self, name):
        return MagicMock()

# Mock torch
mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.tensor = MagicMock(return_value=MockTensor())
mock_torch.zeros = MagicMock(return_value=MockTensor())
mock_torch.rand = MagicMock(return_value=MockTensor())
mock_torch.randint = MagicMock(return_value=MockTensor())
mock_torch.float32 = "float32"

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()

# Mock numpy
mock_numpy = MagicMock()
mock_numpy.random.randint = MagicMock(return_value=12345)
sys.modules["numpy"] = mock_numpy

# Add src to path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '../src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Mock ebaif.consensus.protocol because it's missing in the filesystem
mock_consensus_protocol = MagicMock()
mock_consensus_protocol.ConsensusProtocol = MagicMock()
sys.modules["ebaif.consensus.protocol"] = mock_consensus_protocol

# Mock ebaif.consensus.reputation because it might be missing
mock_consensus_reputation = MagicMock()
sys.modules["ebaif.consensus.reputation"] = mock_consensus_reputation

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
        self.assertEqual(genome.fitness_score, 999.0)

        # Add more entries to exceed the limit
        for i in range(1000, 1500):
            genome.update_fitness(float(i))

        # Verify limit is maintained
        self.assertEqual(len(genome.performance_history), 1000)

        # Verify oldest entries are removed (should start from 500)
        self.assertEqual(genome.performance_history[0], 500.0)
        self.assertEqual(genome.performance_history[-1], 1499.0)
        self.assertEqual(genome.fitness_score, 1499.0)

if __name__ == '__main__':
    unittest.main()
