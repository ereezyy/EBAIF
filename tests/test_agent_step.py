import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import os
import importlib

# --- Mocking Setup Start ---

# 1. Mock numpy
mock_numpy = MagicMock()
mock_numpy.random.choice = MagicMock(return_value=0)
mock_numpy.mean = MagicMock(return_value=0.5)

# 2. Mock torch
class MockTensor:
    def __init__(self, data=None):
        self.data = data
        self.shape = (1, 10) # Default shape

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, dim):
        # Return a new mock tensor with increased dimension
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        t = MockTensor()
        t.shape = tuple(new_shape)
        return t

    def tolist(self):
        return [0.1] * 10

    def item(self):
        # Return an integer to support indexing
        return 0

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def __add__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def sum(self):
        return MockTensor()

    def clone(self):
        return self

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return MockTensor()

class MockNNModule:
    def __init__(self, *args, **kwargs):
        pass
    def parameters(self):
        return []
    def __call__(self, *args, **kwargs):
        return MockTensor()

mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.nn = MagicMock()
mock_torch.nn.Module = MockNNModule
mock_torch.no_grad = MagicMock()
# Important: softmax needs to return a tensor that can be multiplied
mock_torch.softmax.return_value = MockTensor()
mock_torch.randn_like.return_value = MockTensor()
mock_torch.rand.return_value = MockTensor()
mock_torch.zeros_like.return_value = MockTensor()
mock_torch.multinomial.return_value = MockTensor()
mock_torch.argmax.return_value = MockTensor()
mock_torch.linspace.return_value = MockTensor()

# 3. Mock consensus modules
mock_consensus_engine = MagicMock()

# --- Mocking Setup End ---

class TestEmergentAgentStep(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Setup mocks dictionary
        self.mock_modules = {
            'numpy': mock_numpy,
            'torch': mock_torch,
            'torch.nn': mock_torch.nn,
            'ebaif.consensus.engine': mock_consensus_engine,
            'ebaif.consensus.protocol': MagicMock(),
            'ebaif.consensus.reputation': MagicMock(),
        }
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import EmergentAgent inside the patch
        sys.path.append(os.path.join(os.getcwd(), 'src'))

        # We need to import inside try/except block just in case
        try:
            import ebaif.agents.agent
            # Reload to ensure we get the version using our mocks
            importlib.reload(ebaif.agents.agent)
            from ebaif.agents.agent import EmergentAgent, AgentConfig, AgentState, AgentRole

            self.EmergentAgent = EmergentAgent
            self.AgentConfig = AgentConfig
            self.AgentState = AgentState
            self.AgentRole = AgentRole
        except ImportError as e:
            self.fail(f"Failed to import EmergentAgent: {e}")

        # Setup common test data
        self.config = self.AgentConfig(
            agent_id="test_agent",
            learning_rate=0.01,
            evolution_frequency=100
        )

        # Mock behavior genome
        self.mock_genome = MagicMock()
        self.mock_genome.genome_id = "genome_123"
        self.mock_genome.build_network.return_value = MockNNModule()
        self.mock_genome.get_behavior_parameters.return_value = {
            'exploration_rate': 0.1,
            'risk_tolerance': 0.5,
            'curiosity_factor': 0.5
        }

        # Initialize agent with mocked genome
        # We need to patch BehaviorGenome constructor since Agent creates one if not provided
        # Or better, we provide initial_genome
        self.agent = self.EmergentAgent(self.config, initial_genome=self.mock_genome)

        # We need to mock initialize logic if it uses heavy stuff
        # Agent.initialize calls self.genome.build_network() and creates optimizer
        # Optimizer needs parameters() which we mocked
        # It also calls optim.Adam which is in torch.optim
        mock_torch.optim = MagicMock()
        mock_torch.optim.Adam = MagicMock()

        await self.agent.initialize()

        # Mock internal methods to isolate step logic
        self.agent._check_periodic_activities = AsyncMock()

    async def asyncTearDown(self):
        self.patcher.stop()
        # Clean up sys.modules to remove ebaif modules that depend on mocks
        keys_to_remove = [k for k in sys.modules if k.startswith('ebaif')]
        for k in keys_to_remove:
            del sys.modules[k]

    async def test_step_basic_flow(self):
        """Test the basic flow of the step method."""
        # Setup inputs
        env_state = MockTensor()
        available_actions = [0, 1, 2]

        # Execute step
        action, info = await self.agent.step(env_state, available_actions)

        # Assertions
        self.assertEqual(self.agent.state, self.AgentState.IDLE)
        self.assertEqual(self.agent.step_count, 1)
        self.assertIn('selected_action', info)
        self.assertIn('action_probs', info)

        # Verify periodic activities check
        self.agent._check_periodic_activities.assert_called_once()

    async def test_step_1d_input(self):
        """Test step method with 1D input tensor (should be unsqueezed)."""
        env_state = MockTensor()
        env_state.shape = (10,) # 1D shape

        # Execute step
        await self.agent.step(env_state, [0, 1])

        # Verify step count increased
        self.assertEqual(self.agent.step_count, 1)

    async def test_step_no_available_actions(self):
        """Test step method when no actions are available."""
        env_state = MockTensor()
        available_actions = []

        # Execute step
        action, info = await self.agent.step(env_state, available_actions)

        # Verify behavior
        self.assertIsNotNone(action)
        self.agent._check_periodic_activities.assert_called_once()

    async def test_step_periodic_activities(self):
        """Test that periodic activities are checked."""
        env_state = MockTensor()

        await self.agent.step(env_state, [0])

        self.agent._check_periodic_activities.assert_called_once()

if __name__ == '__main__':
    unittest.main()
