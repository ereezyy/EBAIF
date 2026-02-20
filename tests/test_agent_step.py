import unittest
import sys
import types
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import os

# --- MOCK CLASSES ---

class MockTensor(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 'cpu'
        self.dtype = 'float32'

    def item(self):
        return 0.5

    def tolist(self):
        return [0.1, 0.2, 0.7]

    def dim(self):
        return 1

    def unsqueeze(self, dim):
        return self

    def clone(self):
        return self

    def detach(self):
        return self

    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __sub__(self, other): return self
    def __mul__(self, other): return self
    def __truediv__(self, other): return self

    # Comparison operators
    def __lt__(self, other): return True
    def __gt__(self, other): return False
    def __le__(self, other): return True
    def __ge__(self, other): return False
    def __eq__(self, other): return False
    def __ne__(self, other): return True

class MockTorch(types.ModuleType):
    def __init__(self):
        super().__init__('torch')
        self.Tensor = MockTensor
        self.float32 = 'float32'
        self.bool = 'bool'
        self.no_grad = MagicMock()

    def randn_like(self, tensor):
        return MockTensor()

    def zeros(self, *args, **kwargs):
        return MockTensor()

    def rand(self, *args):
        t = MockTensor()
        t.item = lambda: 0.5
        return t

    def tensor(self, data, **kwargs):
        t = MockTensor()
        if isinstance(data, (int, float)):
            t.item = lambda: data
        return t

    def softmax(self, tensor, dim=None):
        return tensor

    def zeros_like(self, tensor):
        return tensor

    def multinomial(self, input, num_samples):
        t = MockTensor()
        t.item = lambda: 0
        return t

    def argmax(self, input):
        t = MockTensor()
        t.item = lambda: 0
        return t

    def linspace(self, start, end, steps):
        return MockTensor()

    def clamp(self, input, min, max):
        return input

class MockNN(types.ModuleType):
    def __init__(self):
        super().__init__('torch.nn')
        self.Module = MagicMock
        self.Linear = MagicMock
        self.Conv1d = MagicMock
        self.LSTM = MagicMock
        self.Dropout = MagicMock
        self.ReLU = MagicMock
        self.Tanh = MagicMock
        self.Sigmoid = MagicMock
        self.LeakyReLU = MagicMock
        self.MultiheadAttention = MagicMock
        self.MSELoss = MagicMock

class MockOptim(types.ModuleType):
    def __init__(self):
        super().__init__('torch.optim')
        self.Adam = MagicMock

class MockNumpy(types.ModuleType):
    def __init__(self):
        super().__init__('numpy')

    def mean(self, a):
        return 0.5

    class random(object):
        @staticmethod
        def choice(a, size=None, replace=True, p=None):
            if isinstance(a, int):
                return 0
            if size is None:
                return a[0]
            if isinstance(a, list):
                return [a[0]] * (size if size else 1)
            return 0

        @staticmethod
        def randint(low, high=None, size=None):
            return low

# --- TEST CASE ---

class TestEmergentAgentStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mock modules
        mock_torch = MockTorch()
        mock_torch.nn = MockNN()
        mock_torch.optim = MockOptim()
        mock_numpy = MockNumpy()

        mock_consensus_protocol = types.ModuleType('ebaif.consensus.protocol')
        mock_consensus_protocol.ConsensusProtocol = MagicMock()

        mock_consensus_reputation = types.ModuleType('ebaif.consensus.reputation')
        mock_consensus_reputation.ReputationSystem = MagicMock()

        mock_neural_evolution = MagicMock()

        # Apply patch to sys.modules
        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_torch.nn,
            'torch.optim': mock_torch.optim,
            'numpy': mock_numpy,
            'ebaif.consensus.protocol': mock_consensus_protocol,
            'ebaif.consensus.reputation': mock_consensus_reputation,
            'ebaif.advanced.neural_evolution': mock_neural_evolution
        })
        cls.modules_patcher.start()

        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

        # Import the module under test
        # We need to import ebaif package structure or mock it?
        # Since we mocked submodules directly in sys.modules, import should work
        try:
            from ebaif.agents.agent import EmergentAgent, AgentConfig, AgentState, AgentRole
            from ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig

            cls.EmergentAgent = EmergentAgent
            cls.AgentConfig = AgentConfig
            cls.AgentState = AgentState
            cls.AgentRole = AgentRole
            cls.BehaviorGenome = BehaviorGenome
        except ImportError as e:
            print(f"Import failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    @classmethod
    def tearDownClass(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.config = self.AgentConfig(agent_id="test_agent")

        # Patch BehaviorGenome to avoid initialization logic that uses torch
        self.genome_patcher = patch('ebaif.agents.agent.BehaviorGenome')
        self.MockBehaviorGenome = self.genome_patcher.start()

        # Configure the mock genome instance
        self.mock_genome_instance = self.MockBehaviorGenome.return_value
        self.mock_genome_instance.genome_id = "test_genome"
        self.mock_genome_instance.get_behavior_parameters.return_value = {
            'exploration_rate': 0.1,
            'risk_tolerance': 0.5,
            'curiosity_factor': 0.5,
            'cooperation_tendency': 0.5,
            'communication_frequency': 0.5
        }
        self.mock_genome_instance.behavior_genes = {}
        # Make build_network return a mock
        self.mock_genome_instance.build_network.return_value = MagicMock()

        self.agent = self.EmergentAgent(self.config)

        # Mock the network and genome to be predictable
        self.agent.network = MagicMock()

        # Mock periodic activities
        self.agent._check_periodic_activities = AsyncMock()

        self.agent.peer_agents = {}

    def tearDown(self):
        self.genome_patcher.stop()

    def test_step_basic_flow(self):
        env_state = MockTensor()
        available_actions = [0, 1, 2]

        mock_logits = MockTensor()
        self.agent.network.return_value = mock_logits

        selected_action, step_info = asyncio.run(self.agent.step(env_state, available_actions))

        self.agent.network.assert_called_once()
        self.assertEqual(self.agent.state, self.AgentState.IDLE)
        self.assertEqual(self.agent.step_count, 1)
        self.agent._check_periodic_activities.assert_called_once()
        self.assertIsInstance(selected_action, int)
        self.assertIn('selected_action', step_info)

    def test_step_input_reshaping(self):
        env_state = MockTensor()
        env_state.dim = MagicMock(return_value=1)
        env_state.unsqueeze = MagicMock(return_value=env_state)

        available_actions = [0, 1]
        asyncio.run(self.agent.step(env_state, available_actions))
        env_state.unsqueeze.assert_called_with(0)

    def test_step_exploration(self):
        env_state = MockTensor()
        available_actions = [0, 1]

        self.mock_genome_instance.get_behavior_parameters.return_value = {
            'exploration_rate': 1.0
        }
        selected_action, _ = asyncio.run(self.agent.step(env_state, available_actions))
        self.assertIn(selected_action, available_actions)

    def test_step_periodic_activities(self):
        env_state = MockTensor()
        asyncio.run(self.agent.step(env_state, []))
        self.agent._check_periodic_activities.assert_called_once()

    def test_step_state_transition(self):
        env_state = MockTensor()
        asyncio.run(self.agent.step(env_state, []))
        self.assertEqual(self.agent.state, self.AgentState.IDLE)

    def test_step_with_no_available_actions(self):
        env_state = MockTensor()
        available_actions = []
        mock_logits = MockTensor()
        self.agent.network.return_value = mock_logits
        selected_action, step_info = asyncio.run(self.agent.step(env_state, available_actions))
        self.assertIsInstance(selected_action, int)

if __name__ == '__main__':
    unittest.main()
