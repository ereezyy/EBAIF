import sys
import os
import pytest
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock torch and numpy before importing ebaif
mock_torch = MagicMock()
mock_torch.tensor = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.Module = object  # Needs to be a class for inheritance
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn

mock_numpy = MagicMock()
mock_numpy.bool_ = bool  # Make it a type so isinstance works for pytest
mock_numpy.mean.return_value = 0.5 # Default for mean
sys.modules['numpy'] = mock_numpy

# Also mock ebaif.consensus if it has missing dependencies
mock_consensus = MagicMock()
sys.modules['ebaif.consensus'] = mock_consensus
sys.modules['ebaif.consensus.engine'] = mock_consensus
sys.modules['ebaif.consensus.validator'] = mock_consensus
sys.modules['ebaif.consensus.protocol'] = mock_consensus
sys.modules['ebaif.consensus.reputation'] = mock_consensus

from ebaif.agents.agent import EmergentAgent
from ebaif.behavior_genome.genome import BehaviorGenome

class TestAgentCompatibility:

    @pytest.fixture
    def mock_genome(self):
        genome = MagicMock(spec=BehaviorGenome)
        return genome

    @pytest.fixture
    def agent(self, mock_genome):
        # Create a mock agent but keep the method we want to test
        agent = MagicMock(spec=EmergentAgent)
        agent.genome = mock_genome

        # Bind the real method to our mock instance
        # We need to bind it as an instance method
        agent._calculate_compatibility = EmergentAgent._calculate_compatibility.__get__(agent, EmergentAgent)

        return agent

    @pytest.fixture
    def other_agent(self):
        other = MagicMock(spec=EmergentAgent)
        other.genome = MagicMock(spec=BehaviorGenome)
        return other

    def test_perfect_compatibility(self, agent, other_agent):
        """Test compatibility when parameters are identical."""
        params = {
            'cooperation_tendency': 0.5,
            'communication_frequency': 0.5,
            'exploration_rate': 0.5
        }

        agent.genome.get_behavior_parameters.return_value = params
        other_agent.genome.get_behavior_parameters.return_value = params

        # Calculation:
        # cooperation: (1 - 0) * 2 = 2
        # communication: (1 - 0) * 2 = 2
        # exploration: (1 - 0) * 1 = 1
        # total = 5
        # count = 3
        # result = 5 / 3 = 1.666...

        result = agent._calculate_compatibility(other_agent)

        expected = (2.0 + 2.0 + 1.0) / 3.0
        assert result == pytest.approx(expected)

    def test_zero_compatibility(self, agent, other_agent):
        """Test compatibility when parameters are completely opposite."""
        # Note: using 0.0 vs 1.0 for all parameters results in 0 similarity

        agent_params = {
            'cooperation_tendency': 0.0,
            'communication_frequency': 0.0,
            'exploration_rate': 0.0
        }
        other_params = {
            'cooperation_tendency': 1.0,
            'communication_frequency': 1.0,
            'exploration_rate': 1.0
        }

        agent.genome.get_behavior_parameters.return_value = agent_params
        other_agent.genome.get_behavior_parameters.return_value = other_params

        # Calculation:
        # cooperation: (1 - 1) * 2 = 0
        # communication: (1 - 1) * 2 = 0
        # exploration: (1 - 1) = 0
        # total = 0
        # count = 3

        result = agent._calculate_compatibility(other_agent)
        assert result == 0.0

    def test_weighted_keys(self, agent, other_agent):
        """Test that weighted keys have more impact."""
        # Case 1: Only weighted keys differ
        # cooperation_tendency is weighted * 2
        agent_params = {'cooperation_tendency': 0.5}
        other_params = {'cooperation_tendency': 0.6}

        agent.genome.get_behavior_parameters.return_value = agent_params
        other_agent.genome.get_behavior_parameters.return_value = other_params

        # similarity = 1.0 - 0.1 = 0.9
        # weighted = 0.9 * 2.0 = 1.8
        # result = 1.8 / 1 = 1.8

        result = agent._calculate_compatibility(other_agent)
        assert result == pytest.approx(1.8)

        # Case 2: Only non-weighted keys differ
        agent_params_unweighted = {'exploration_rate': 0.5}
        other_params_unweighted = {'exploration_rate': 0.6}

        agent.genome.get_behavior_parameters.return_value = agent_params_unweighted
        other_agent.genome.get_behavior_parameters.return_value = other_params_unweighted

        # similarity = 1.0 - 0.1 = 0.9
        # result = 0.9 / 1 = 0.9

        result_unweighted = agent._calculate_compatibility(other_agent)
        assert result_unweighted == pytest.approx(0.9)

    def test_partial_parameter_overlap(self, agent, other_agent):
        """Test when agents have different sets of parameters."""
        agent_params = {
            'common': 0.5,
            'agent_only': 0.5
        }
        other_params = {
            'common': 0.5,
            'other_only': 0.5
        }

        agent.genome.get_behavior_parameters.return_value = agent_params
        other_agent.genome.get_behavior_parameters.return_value = other_params

        # Only 'common' is compared (unweighted)
        # similarity = 1.0 - 0 = 1.0
        # count = 1
        # result = 1.0

        result = agent._calculate_compatibility(other_agent)
        assert result == 1.0

    def test_empty_parameters(self, agent, other_agent):
        """Test with empty parameter sets."""
        agent.genome.get_behavior_parameters.return_value = {}
        other_agent.genome.get_behavior_parameters.return_value = {}

        # param_count = 0
        # result = 0.0 / max(0, 1) = 0.0

        result = agent._calculate_compatibility(other_agent)
        assert result == 0.0

    def test_mixed_weighted_and_unweighted(self, agent, other_agent):
        """Test a mix of weighted and unweighted parameters with known values."""
        # Weighted keys: cooperation_tendency, communication_frequency

        agent_params = {
            'cooperation_tendency': 0.8,    # Diff 0.2 -> sim 0.8 -> *2 = 1.6
            'communication_frequency': 0.4, # Diff 0.1 -> sim 0.9 -> *2 = 1.8
            'risk_taking': 0.5              # Diff 0.4 -> sim 0.6 -> *1 = 0.6
        }
        other_params = {
            'cooperation_tendency': 0.6,
            'communication_frequency': 0.5,
            'risk_taking': 0.9
        }

        agent.genome.get_behavior_parameters.return_value = agent_params
        other_agent.genome.get_behavior_parameters.return_value = other_params

        # Total score = 1.6 + 1.8 + 0.6 = 4.0
        # Count = 3
        # Result = 4.0 / 3 = 1.333...

        result = agent._calculate_compatibility(other_agent)
        assert result == pytest.approx(4.0 / 3.0)
