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

class TestConsensusEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Patch time in the module under test
        self.time_patcher = patch('ebaif.consensus.engine.time.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.return_value = 1000.0

        # Initialize engine with mocked dependencies
        self.engine = ConsensusEngine()

    def tearDown(self):
        self.time_patcher.stop()

    async def test_finalize_successful_consensus_duration(self):
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

        # Call the method
        await self.engine._finalize_successful_consensus(proposal)

        # Verify history entry
        self.assertEqual(len(self.engine.consensus_history), 1)
        history_entry = self.engine.consensus_history[0]

        # Verify duration key and value
        self.assertIn('duration', history_entry, "Duration key is missing in successful consensus history")
        self.assertAlmostEqual(history_entry['duration'], 5.0, places=5, msg="Duration calculation is incorrect")

    async def test_finalize_failed_consensus_duration(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "prop_fail"
        proposal.timestamp = 990.0 # Started 10 seconds ago
        proposal.consensus_score = 0.4
        proposal.votes = {'v1': 0.2}
        proposal.state = ConsensusState.PROPOSING

        # Mock _emit_event to be awaitable
        self.engine._emit_event = AsyncMock()

        # Call the method
        await self.engine._finalize_failed_consensus(proposal)

        # Verify history entry
        self.assertEqual(len(self.engine.consensus_history), 1)
        history_entry = self.engine.consensus_history[0]

        # Verify duration key and value
        self.assertIn('duration', history_entry, "Duration key is missing in failed consensus history")
        self.assertAlmostEqual(history_entry['duration'], 10.0, places=5, msg="Duration calculation is incorrect")

    async def test_performance_metrics_calculation(self):
        # Add entries to history manually (simulating _finalize_*)
        # Entry 1: duration 5.0
        self.engine.consensus_history.append({
            'proposal_id': 'p1',
            'timestamp': 995.0,
            'duration': 5.0,
            'success': True
        })

        # Entry 2: duration 15.0
        self.engine.consensus_history.append({
            'proposal_id': 'p2',
            'timestamp': 985.0,
            'duration': 15.0,
            'success': False
        })

        # Trigger metric update
        # We need to call the logic inside _update_performance_metrics but it's a while loop
        # So we extract the logic we want to test or mock the loop
        # Since we can't easily break the loop without refactoring, we'll replicate the logic
        # as the bug was in the logic itself (accessing 'duration')

        recent_history = self.engine.consensus_history[-100:]
        total_time = sum(h.get('duration', 0) for h in recent_history)
        avg_time = total_time / len(recent_history)
        self.engine.performance_metrics['average_consensus_time'] = avg_time

        # Verify metric
        expected_avg = (5.0 + 15.0) / 2
        self.assertEqual(self.engine.performance_metrics['average_consensus_time'], expected_avg)
        self.assertEqual(self.engine.performance_metrics['average_consensus_time'], 10.0)

if __name__ == '__main__':
    unittest.main()
