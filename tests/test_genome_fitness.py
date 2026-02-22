import sys
import unittest
from unittest.mock import MagicMock, patch

# Define MockTensor
class MockTensor:
    def __init__(self, value=0.0):
        self.value = value

    def __gt__(self, other):
        # Return a MockTensor that evaluates to False to simulate 'False' for initialization checks
        # or True if needed. For history limit test, it's mostly about appending floats.
        # But initialization does check config values.
        return MockTensor(0.0)

    def __lt__(self, other):
        return MockTensor(0.0)

    def item(self):
        return float(self.value)

    def __float__(self):
        return float(self.value)

    def __setitem__(self, key, value):
        pass

    def clone(self):
        return MockTensor(self.value)

    def tolist(self):
        return []

    def __getattr__(self, name):
        return MagicMock()

# Create a dictionary of mocks
mock_modules = {
    "torch": MagicMock(),
    "torch.nn": MagicMock(),
    "numpy": MagicMock(),
    "ebaif.consensus.protocol": MagicMock(),
    "ebaif.consensus.reputation": MagicMock(),
}

# Configure torch mock
mock_modules["torch"].Tensor = MockTensor
mock_modules["torch"].tensor = lambda x, **kwargs: MockTensor(x) if isinstance(x, (int, float)) else MockTensor()
mock_modules["torch"].zeros = lambda *args: MockTensor()
mock_modules["torch"].rand = lambda *args: MockTensor()
mock_modules["torch"].randint = lambda *args: MockTensor()
mock_modules["torch"].float32 = "float32"

# Configure numpy mock
mock_modules["numpy"].random.randint.return_value = 12345

# Patch sys.modules
patcher = patch.dict(sys.modules, mock_modules)
patcher.start()

from ebaif.behavior_genome.genome import BehaviorGenome

class TestBehaviorGenomeFitness(unittest.TestCase):
    def setUp(self):
        # Ensure mocks are active (redundant if patcher is global, but good practice)
        pass

    def tearDown(self):
        # Stop patcher if we wanted to isolate completely, but since we imported BehaviorGenome globally
        # with the mocks active, we should keep them active or reload the module.
        # For this simple test file, global patching is acceptable as it mimics the environment
        # for the imported module.
        pass

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
