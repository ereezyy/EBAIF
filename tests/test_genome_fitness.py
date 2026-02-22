import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import importlib

# Define strict MockTensor
class MockTensor:
    def __init__(self, *args, **kwargs):
        self.dtype = kwargs.get('dtype', MagicMock())
        self.shape = kwargs.get('shape', (1,))

    def __lt__(self, other):
        return MockTensor(dtype='bool')

    def __le__(self, other):
        return MockTensor(dtype='bool')

    def __gt__(self, other):
        return MockTensor(dtype='bool')

    def __ge__(self, other):
        return MockTensor(dtype='bool')

    def __invert__(self):
        return self

    def __setitem__(self, key, value):
        pass

    def item(self):
        return 0.0

    def clone(self):
        return self

    def tolist(self):
        return []

    def __repr__(self):
        return "MockTensor()"

class TestBehaviorGenomeFitness(unittest.TestCase):
    def setUp(self):
        # Setup path
        self.src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
        if self.src_path not in sys.path:
            sys.path.insert(0, self.src_path)

        # Setup mocks
        self.mock_torch = MagicMock()
        self.mock_torch.Tensor = MockTensor
        self.mock_torch.tensor.side_effect = lambda *args, **kwargs: MockTensor()
        self.mock_torch.zeros.side_effect = lambda *args, **kwargs: MockTensor()
        self.mock_torch.rand.side_effect = lambda *args, **kwargs: MockTensor()
        self.mock_torch.randint.side_effect = lambda *args, **kwargs: MockTensor()
        self.mock_torch.float32 = "float32"
        self.mock_torch.bool = "bool"

        self.mock_numpy = MagicMock()
        self.mock_numpy.random.randint.return_value = 12345

        # Patch sys.modules
        self.modules_patcher = patch.dict(sys.modules, {
            "torch": self.mock_torch,
            "torch.nn": MagicMock(),
            "numpy": self.mock_numpy,
            "ebaif.consensus.protocol": MagicMock(),
            "ebaif.consensus.reputation": MagicMock()
        })
        self.modules_patcher.start()

        # Import/Reload module under test
        try:
            import ebaif.behavior_genome.genome
            importlib.reload(ebaif.behavior_genome.genome)
            self.genome_module = ebaif.behavior_genome.genome
        except ImportError:
            self.fail("Could not import ebaif.behavior_genome.genome")

    def tearDown(self):
        self.modules_patcher.stop()
        if self.src_path in sys.path:
            sys.path.remove(self.src_path)

        # Clean up the module to prevent leakage of mocks to other tests
        if 'ebaif.behavior_genome.genome' in sys.modules:
            del sys.modules['ebaif.behavior_genome.genome']

    def test_fitness_history_limit(self):
        """Test that performance history is limited to 1000 entries."""
        # Use the class from the reloaded module
        BehaviorGenome = self.genome_module.BehaviorGenome
        genome = BehaviorGenome()

        # Verify initial state
        self.assertEqual(len(genome.performance_history), 0)
        self.assertEqual(genome.fitness_score, 0.0)

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
