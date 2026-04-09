import sys
from unittest.mock import MagicMock

# Mock dependencies that might be imported at top-level in ebaif
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()
sys.modules['src.ebaif.consensus.protocol'] = MagicMock()
sys.modules['src.ebaif.consensus.reputation'] = MagicMock()

import unittest
import asyncio
from src.ebaif.advanced.meta_learning import MetaLearningEngine, MetaModel, MetaLearningConfig

class TestMetaLearningLogic(unittest.TestCase):
    def setUp(self):
        self.config = MetaLearningConfig(meta_batch_size=2)
        self.engine = MetaLearningEngine(self.config)
        self.engine.initialize_meta_model(input_dim=4, output_dim=2, hidden_dims=[8])
        self.model = self.engine.meta_model

    def test_update_parameters_logic(self):
        params = {
            'W0': [[1.0, 2.0], [3.0, 4.0]],
            'b0': [0.5, 0.5]
        }
        grads = {
            'W0': [[0.1, 0.1], [0.1, 0.1]],
            'b0': [0.1, 0.1]
        }
        lr = 1.0
        updated = self.model.update_parameters(params, grads, lr)

        self.assertEqual(updated['W0'], [[0.9, 1.9], [2.9, 3.9]])
        self.assertEqual(updated['b0'], [0.4, 0.4])

    def test_accumulation_and_averaging_logic(self):
        # Use a real event loop if possible, otherwise mock it or use asyncio.run
        async def run_test():
            task = {
                'type': 'regression',
                'support_set': [([1.0, 1.0, 1.0, 1.0], 1.0), ([0.0, 0.0, 0.0, 0.0], 0.0)],
                'query_set': [([0.5, 0.5, 0.5, 0.5], 0.5)]
            }
            grads = await self.engine._compute_support_gradients(task, self.model.meta_parameters)
            return grads

        grads = asyncio.run(run_test())
        self.assertIsNotNone(grads)
        for name, val in grads.items():
            if isinstance(val[0], list):
                self.assertEqual(len(val), len(self.model.meta_parameters[name]))
            else:
                self.assertEqual(len(val), len(self.model.meta_parameters[name]))

    def test_meta_gradients_accumulation(self):
        async def run_test():
            meta_batch = [
                {'type': 'regression', 'support_set': [([0.1]*4, 0.1)], 'query_set': [([0.2]*4, 0.2)]},
                {'type': 'regression', 'support_set': [([0.3]*4, 0.3)], 'query_set': [([0.4]*4, 0.4)]}
            ]
            meta_grads = await self.engine._compute_meta_gradients(meta_batch)
            return meta_grads

        meta_grads = asyncio.run(run_test())
        self.assertIsNotNone(meta_grads)
        self.assertIn('W0', meta_grads)
        self.assertIn('b0', meta_grads)

if __name__ == '__main__':
    unittest.main()
