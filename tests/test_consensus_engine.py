import unittest
import sys
import os
import asyncio
import time
from unittest.mock import MagicMock, patch

# Mock dependencies before imports
# We need numpy.mean to return a float, not a Mock
numpy_mock = MagicMock()
numpy_mock.mean.side_effect = lambda x: sum(x) / len(x) if len(x) > 0 else 0.0
sys.modules['numpy'] = numpy_mock

sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

# Mock behavior_genome and its submodules
behavior_genome_mock = MagicMock()
sys.modules['ebaif.behavior_genome'] = behavior_genome_mock
sys.modules['ebaif.behavior_genome.genome'] = MagicMock()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState
except ImportError:
    # If import fails, we try to mock more aggressively
    sys.modules['ebaif.behavior_genome.genome'] = MagicMock()
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState

class TestConsensusEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ConsensusEngine()

    def test_duration_is_recorded_on_success(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "test_success"
        proposal.timestamp = time.time() - 2.0
        proposal.consensus_score = 0.9
        proposal.votes = {'v1': 1.0}
        proposal.state = ConsensusState.COMPLETED

        async def mock_async_method(*args, **kwargs):
            pass

        self.engine._emit_event = mock_async_method
        self.engine._propagate_behavior = mock_async_method

        # Run the method
        asyncio.run(self.engine._finalize_successful_consensus(proposal))

        # Verify history
        self.assertTrue(len(self.engine.consensus_history) > 0)
        last_entry = self.engine.consensus_history[-1]

        print(f"History entry (Success): {last_entry}")

        # This assertion should fail before the fix
        self.assertIn('duration', last_entry, "Duration should be recorded in consensus history")
        self.assertGreater(last_entry.get('duration', 0), 0, "Duration should be greater than 0")

    def test_duration_is_recorded_on_failure(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "test_failure"
        proposal.timestamp = time.time() - 2.0
        proposal.consensus_score = 0.4
        proposal.votes = {'v1': 0.0, 'v2': 0.1}
        proposal.state = ConsensusState.FAILED

        async def mock_async_method(*args, **kwargs):
            pass

        self.engine._emit_event = mock_async_method

        # Run the method
        asyncio.run(self.engine._finalize_failed_consensus(proposal))

        # Verify history
        self.assertTrue(len(self.engine.consensus_history) > 0)
        last_entry = self.engine.consensus_history[-1]

        print(f"History entry (Failure): {last_entry}")

        # This assertion should fail before the fix
        self.assertIn('duration', last_entry, "Duration should be recorded in consensus history for failure too")
        self.assertGreater(last_entry.get('duration', 0), 0, "Duration should be greater than 0")

if __name__ == '__main__':
    unittest.main()
