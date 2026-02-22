import sys
import os
import unittest
import time
from unittest.mock import MagicMock, patch

# Mock modules to avoid import errors
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

# Mock numpy to return float for mean
numpy_mock = MagicMock()
numpy_mock.mean.side_effect = lambda x: sum(x) / len(x) if x and len(x) > 0 else 0.0
sys.modules['numpy'] = numpy_mock

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Now import the target module
try:
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState
except ImportError:
    # Fallback if imports fail inside engine
    sys.modules['ebaif.behavior_genome'] = MagicMock()
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState

class TestConsensusEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = ConsensusEngine()
        self.engine.consensus_history = [] # Reset history
        self.engine.performance_metrics = {'average_consensus_time': 0.0}

    async def test_finalize_successful_consensus_adds_duration(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "test_proposal_success"
        proposal.timestamp = time.time() - 2.0  # Started 2 seconds ago
        proposal.consensus_score = 0.95
        proposal.votes = {'v1': 1.0, 'v2': 0.8}
        proposal.state = ConsensusState.COMPLETED

        # Call the method under test
        await self.engine._finalize_successful_consensus(proposal)

        # Verify history entry
        self.assertEqual(len(self.engine.consensus_history), 1)
        entry = self.engine.consensus_history[0]
        self.assertEqual(entry['proposal_id'], "test_proposal_success")
        self.assertTrue(entry['success'])

        # Verify duration exists and is reasonable
        self.assertIn('duration', entry, "Duration key is missing in consensus history")
        self.assertGreater(entry['duration'], 0.0)

    async def test_finalize_failed_consensus_adds_duration(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "test_proposal_fail"
        proposal.timestamp = time.time() - 3.0  # Started 3 seconds ago
        proposal.consensus_score = 0.4
        proposal.votes = {'v1': 0.2, 'v2': 0.3}
        proposal.state = ConsensusState.FAILED

        # Call the method under test
        await self.engine._finalize_failed_consensus(proposal)

        # Verify history entry
        self.assertEqual(len(self.engine.consensus_history), 1)
        entry = self.engine.consensus_history[0]
        self.assertEqual(entry['proposal_id'], "test_proposal_fail")
        self.assertFalse(entry['success'])

        # Verify duration exists and is reasonable
        self.assertIn('duration', entry, "Duration key is missing in consensus history")
        self.assertGreater(entry['duration'], 0.0)

if __name__ == '__main__':
    unittest.main()
