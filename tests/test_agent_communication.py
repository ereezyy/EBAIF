import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock torch and numpy if they are not available
try:
    import torch
except ImportError:
    sys.modules['torch'] = MagicMock()
    sys.modules['torch.nn'] = MagicMock()
    sys.modules['torch.optim'] = MagicMock()

try:
    import numpy as np
except ImportError:
    sys.modules['numpy'] = MagicMock()

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock missing modules
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

# Now we can import the agent
from ebaif.agents.agent import EmergentAgent, AgentConfig

class TestAgentCommunication(unittest.TestCase):
    def setUp(self):
        # Create a mock agent
        self.config = AgentConfig(agent_id="agent_1")

        # Mock dependencies to avoid actual initialization
        with patch('ebaif.agents.agent.BehaviorGenome') as MockGenome:
            self.agent = EmergentAgent(self.config)

            # Setup mock genome behavior parameters
            self.agent.genome = MockGenome.return_value
            self.agent.genome.get_behavior_parameters.return_value = {
                'cooperation_tendency': 0.5,
                'communication_frequency': 0.5,
                'exploration_rate': 0.5
            }

            # Mock peer_agents dict
            self.agent.peer_agents = {}

    def test_select_partners_no_peers(self):
        """Test selection when no peers exist."""
        self.agent.peer_agents = {}
        partners = self.agent._select_communication_partners(num_partners=2)
        self.assertEqual(partners, [])

    def test_select_partners_sorting(self):
        """Test partners are sorted by compatibility score."""
        # Create mock peer agents
        peer1 = MagicMock()
        peer1.genome.get_behavior_parameters.return_value = {
            'cooperation_tendency': 0.5,  # Same as agent -> high compatibility
            'communication_frequency': 0.5,
            'exploration_rate': 0.5
        }

        peer2 = MagicMock()
        peer2.genome.get_behavior_parameters.return_value = {
            'cooperation_tendency': 0.9,  # Different -> lower compatibility
            'communication_frequency': 0.1,
            'exploration_rate': 0.9
        }

        peer3 = MagicMock()
        peer3.genome.get_behavior_parameters.return_value = {
            'cooperation_tendency': 0.4,  # Close -> medium compatibility
            'communication_frequency': 0.6,
            'exploration_rate': 0.4
        }

        # Set agent parameters explicitly for this test
        self.agent.genome.get_behavior_parameters.return_value = {
            'cooperation_tendency': 0.5,
            'communication_frequency': 0.5,
            'exploration_rate': 0.5
        }

        self.agent.peer_agents = {
            'peer1': peer1,
            'peer2': peer2,
            'peer3': peer3
        }

        partners = self.agent._select_communication_partners(num_partners=3)

        # Check order
        self.assertEqual(partners[0], 'peer1')
        self.assertEqual(partners[1], 'peer3')
        self.assertEqual(partners[2], 'peer2')

    def test_select_partners_limit(self):
        """Test selection respects the number of requested partners."""
        # Create mock peer agents
        peer1 = MagicMock()
        peer1.genome.get_behavior_parameters.return_value = {'cooperation_tendency': 0.5}

        peer2 = MagicMock()
        peer2.genome.get_behavior_parameters.return_value = {'cooperation_tendency': 0.5}

        peer3 = MagicMock()
        peer3.genome.get_behavior_parameters.return_value = {'cooperation_tendency': 0.5}

        self.agent.peer_agents = {
            'peer1': peer1,
            'peer2': peer2,
            'peer3': peer3
        }

        partners = self.agent._select_communication_partners(num_partners=2)
        self.assertEqual(len(partners), 2)

        partners = self.agent._select_communication_partners(num_partners=1)
        self.assertEqual(len(partners), 1)

        partners = self.agent._select_communication_partners(num_partners=5)
        self.assertEqual(len(partners), 3)

    def test_calculate_compatibility(self):
        """Test compatibility calculation logic."""
        # Setup agent's parameters
        self.agent.genome.get_behavior_parameters.return_value = {
            'param1': 0.5,
            'cooperation_tendency': 0.5,
            'communication_frequency': 0.5
        }

        # Perfect match
        other_agent = MagicMock()
        other_agent.genome.get_behavior_parameters.return_value = {
            'param1': 0.5,
            'cooperation_tendency': 0.5,
            'communication_frequency': 0.5
        }

        score = self.agent._calculate_compatibility(other_agent)
        # Expected: (1 + 2 + 2) / 3 = 1.666
        self.assertAlmostEqual(score, 5.0/3.0)

        # Worst match
        other_agent.genome.get_behavior_parameters.return_value = {
            'param1': 1.5,
            'cooperation_tendency': 1.5,
            'communication_frequency': 1.5
        }
        score = self.agent._calculate_compatibility(other_agent)
        self.assertAlmostEqual(score, 0.0)

if __name__ == '__main__':
    unittest.main()
