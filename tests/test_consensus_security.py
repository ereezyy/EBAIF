import unittest
from unittest.mock import MagicMock
import sys
import os
import asyncio
import re

# Mock dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()
sys.modules['ebaif.behavior_genome.genome'] = MagicMock()

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ebaif.consensus.engine import ConsensusEngine

class TestConsensusSecurity(unittest.TestCase):
    def setUp(self):
        self.engine = ConsensusEngine()
        # Mock _is_behavior_duplicate to return a Future
        future = asyncio.Future()
        future.set_result(False)
        self.engine._is_behavior_duplicate = MagicMock(return_value=future)

    def test_proposal_id_is_uuid(self):
        """Test that proposal IDs are valid UUIDs."""
        genome = MagicMock()
        proposer_id = "test_user"

        # We need to run async code in sync test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            proposal_id = loop.run_until_complete(
                self.engine.propose_behavior(
                    genome,
                    proposer_id,
                    {'fitness': 1.0}
                )
            )

            print(f"Generated ID: {proposal_id}")

            # Check format: proposal_{proposer_id}_{uuid}
            parts = proposal_id.split('_')

            uuid_part = parts[-1]

            # Verify UUID format (simple regex for UUID v4)
            uuid_pattern = re.compile(
                r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
                re.IGNORECASE
            )

            self.assertTrue(uuid_pattern.match(uuid_part), f"ID part {uuid_part} is not a valid UUID v4")

        finally:
            loop.close()

    def test_engine_id_is_uuid(self):
        """Test that engine ID contains a valid UUID."""
        engine_id = self.engine.engine_id
        print(f"Engine ID: {engine_id}")

        parts = engine_id.split('_')
        uuid_part = parts[-1]

        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            re.IGNORECASE
        )

        self.assertTrue(uuid_pattern.match(uuid_part), f"Engine ID part {uuid_part} is not a valid UUID v4")

if __name__ == '__main__':
    unittest.main()
