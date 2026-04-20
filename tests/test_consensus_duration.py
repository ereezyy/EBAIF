import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import asyncio
import time
from collections import defaultdict

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['numpy'].mean = MagicMock(return_value=0.5) # Mock mean for failed consensus
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

# Import target
try:
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState
except ImportError as e:
    print(f"ImportError during setup: {e}")
    sys.modules['ebaif.behavior_genome'] = MagicMock()
    from ebaif.consensus.engine import ConsensusEngine, BehaviorProposal, ConsensusState

class TestConsensusDuration(unittest.TestCase):
    def setUp(self):
        self.engine = ConsensusEngine()
        # Use real dict for reputations
        self.engine.validator_reputations = defaultdict(lambda: 0.5)

        # Mock methods that would trigger external calls
        self.engine._emit_event = MagicMock()

        # We need to handle async properly. Since _emit_event is awaited,
        # it should return an awaitable.
        f = asyncio.Future()
        f.set_result(None)
        self.engine._emit_event.return_value = f

        self.engine._propagate_behavior = MagicMock()
        f2 = asyncio.Future()
        f2.set_result(None)
        self.engine._propagate_behavior.return_value = f2

        self.engine.logger = MagicMock()

        if not hasattr(self.engine, 'config') or self.engine.config is None:
             self.engine.config = MagicMock()
        self.engine.config.propagation_delay = 0.0

    def test_successful_consensus_duration(self):
        # Clear history
        self.engine.consensus_history = []

        # Setup proposal
        proposal = MagicMock()
        proposal.proposal_id = "p1"
        proposal.timestamp = time.time() - 5.0
        proposal.consensus_score = 0.9
        proposal.votes = {'v1': 1.0}
        proposal.state = ConsensusState.COMPLETED

        # Run async method
        # Since _emit_event returns the SAME future object, checking it twice is fine.
        # But we need to make sure the mocked method returns a NEW future each call if called multiple times?
        # Actually in this test it is called once per test method.

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.engine._finalize_successful_consensus(proposal))
        finally:
            loop.close()

        # Check history
        self.assertEqual(len(self.engine.consensus_history), 1)
        entry = self.engine.consensus_history[0]

        # This assertion is expected to fail before the fix
        self.assertIn('duration', entry, "Duration key missing in successful consensus")
        self.assertGreater(entry['duration'], 4.0, "Duration seems too short")

    def test_failed_consensus_duration(self):
        # Clear history
        self.engine.consensus_history = []

        proposal = MagicMock()
        proposal.proposal_id = "p2"
        proposal.timestamp = time.time() - 3.0
        proposal.consensus_score = 0.4
        proposal.votes = {'v1': 0.0}
        proposal.state = ConsensusState.FAILED

        # Reset mocks for new loop
        f = asyncio.Future()
        f.set_result(None)
        self.engine._emit_event.return_value = f

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.engine._finalize_failed_consensus(proposal))
        finally:
            loop.close()

        # Check history
        self.assertEqual(len(self.engine.consensus_history), 1)
        entry = self.engine.consensus_history[0]

        # This assertion is expected to fail before the fix
        self.assertIn('duration', entry, "Duration key missing in failed consensus")
        self.assertGreater(entry['duration'], 2.0, "Duration seems too short")

if __name__ == '__main__':
    unittest.main()
