import unittest
import sys
import os
import time
from unittest.mock import MagicMock, patch

# Mock torch before any imports
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()

# Configure numpy mock properly
numpy_mock = MagicMock()
numpy_mock.mean.side_effect = lambda x: sum(x) / len(x) if len(x) > 0 else 0.0
sys.modules['numpy'] = numpy_mock

# Mock other potential dependencies
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# Mock missing modules to allow importing engine.py
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Try to import ConsensusEngine, mocking if necessary
try:
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState
except ImportError:
    # If import fails, mock behavior_genome too
    sys.modules['ebaif.behavior_genome'] = MagicMock()
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState

class TestConsensusEngine(unittest.TestCase):
    def setUp(self):
        # Create engine
        self.engine = ConsensusEngine()

    def test_duration_in_consensus_history_successful(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "test_prop_success"
        proposal.timestamp = time.time() - 2.0  # Started 2 seconds ago
        proposal.consensus_score = 0.95
        proposal.votes = {'v1': 1.0}
        proposal.state = ConsensusState.COMPLETED

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Mock _emit_event and _propagate_behavior to avoid side effects
        self.engine._emit_event = MagicMock()
        self.engine._emit_event.return_value = asyncio.Future()
        self.engine._emit_event.return_value.set_result(None)

        self.engine._propagate_behavior = MagicMock()
        self.engine._propagate_behavior.return_value = asyncio.Future()
        self.engine._propagate_behavior.return_value.set_result(None)

        # Mock asyncio.sleep to be instant
        with patch('asyncio.sleep', return_value=asyncio.Future()) as mock_sleep:
            mock_sleep.return_value.set_result(None)

            # Run the method
            loop.run_until_complete(self.engine._finalize_successful_consensus(proposal))

        # Check history
        self.assertTrue(len(self.engine.consensus_history) > 0)
        last_entry = self.engine.consensus_history[-1]

        print(f"History entry keys: {last_entry.keys()}")

        self.assertIn('duration', last_entry, "duration key missing in successful consensus history")
        self.assertGreater(last_entry['duration'], 0, "duration should be positive")
        self.assertAlmostEqual(last_entry['duration'], 2.0, delta=0.5, msg="duration should be approx 2.0s")

    def test_duration_in_consensus_history_failed(self):
        # Create a mock proposal
        proposal = MagicMock()
        proposal.proposal_id = "test_prop_failed"
        proposal.timestamp = time.time() - 1.5  # Started 1.5 seconds ago
        proposal.consensus_score = 0.4
        proposal.votes = {'v1': 0.2}
        proposal.state = ConsensusState.FAILED

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.engine._emit_event = MagicMock()
        self.engine._emit_event.return_value = asyncio.Future()
        self.engine._emit_event.return_value.set_result(None)

        # Run the method
        loop.run_until_complete(self.engine._finalize_failed_consensus(proposal))

        # Check history
        self.assertTrue(len(self.engine.consensus_history) > 0)
        last_entry = self.engine.consensus_history[-1]

        print(f"History entry keys: {last_entry.keys()}")

        self.assertIn('duration', last_entry, "duration key missing in failed consensus history")
        self.assertGreater(last_entry['duration'], 0, "duration should be positive")
        self.assertAlmostEqual(last_entry['duration'], 1.5, delta=0.5, msg="duration should be approx 1.5s")

if __name__ == '__main__':
    unittest.main()
