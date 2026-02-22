import sys
import os
import unittest
import time
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock missing dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

# Import the class to test
try:
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState
except ImportError:
    # If import fails, we mock behavior_genome too
    sys.modules['ebaif.behavior_genome'] = MagicMock()
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState

# Fix numpy mock to return float for mean
sys.modules['numpy'].mean.return_value = 0.5

class TestConsensusEngine(unittest.TestCase):
    def setUp(self):
        # Patch time in the module under test to control timestamps
        self.time_patcher = patch('ebaif.consensus.engine.time.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.return_value = 1000.0

        # Initialize engine with mocked dependencies
        self.engine = ConsensusEngine()

    def tearDown(self):
        self.time_patcher.stop()

    def test_finalize_successful_consensus_duration(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "prop_success"
        proposal.timestamp = 995.0 # Started 5 seconds ago
        proposal.consensus_score = 0.9
        proposal.votes = {'v1': 1.0}
        proposal.state = ConsensusState.PROPOSING

        # Mock _emit_event and _propagate_behavior to be awaitable
        self.engine._emit_event = AsyncMock()
        self.engine._propagate_behavior = AsyncMock()

        # Call the method (need to run async method synchronously for test)
        import asyncio
        asyncio.run(self.engine._finalize_successful_consensus(proposal))

        # Verify history entry
        self.assertEqual(len(self.engine.consensus_history), 1)
        history_entry = self.engine.consensus_history[0]

        # This assertion is expected to fail before the fix
        self.assertIn('duration', history_entry, "Duration key is missing in successful consensus history")
        self.assertAlmostEqual(history_entry['duration'], 5.0, places=5, msg="Duration calculation is incorrect")

    def test_finalize_failed_consensus_duration(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "prop_fail"
        proposal.timestamp = 990.0 # Started 10 seconds ago
        proposal.consensus_score = 0.4
        proposal.votes = {'v1': 0.2}
        proposal.state = ConsensusState.PROPOSING

        # Mock _emit_event to be awaitable
        self.engine._emit_event = AsyncMock()

        # Configure numpy mean return value for this test if needed
        # (It's already configured globally, but good to know)

        # Call the method
        import asyncio
        asyncio.run(self.engine._finalize_failed_consensus(proposal))

        # Verify history entry
        self.assertEqual(len(self.engine.consensus_history), 1)
        history_entry = self.engine.consensus_history[0]

        # This assertion is expected to fail before the fix
        self.assertIn('duration', history_entry, "Duration key is missing in failed consensus history")
        self.assertAlmostEqual(history_entry['duration'], 10.0, places=5, msg="Duration calculation is incorrect")

if __name__ == '__main__':
    unittest.main()
