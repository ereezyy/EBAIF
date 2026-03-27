import unittest
import torch
import torch.nn as nn
from src.ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig, ArchitectureType

class TestGenomeArchitecture(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.input_dim = 16
        self.output_dim = 4
        self.batch_size = 2
        self.seq_len = 10

    def _create_dummy_input(self):
        """Create a dummy input tensor."""
        return torch.randn(self.batch_size, self.seq_len, self.input_dim)

    def test_transformer_architecture(self):
        """Test Transformer architecture generation."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.TRANSFORMER,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32, 16],
            attention_heads=4,
            num_layers=2
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIsInstance(network, nn.Module)

        # Verify components (accessing internal class name)
        self.assertTrue('TransformerGenome' in str(type(network)))

        # Verify forward pass
        dummy_input = self._create_dummy_input()
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_cnn_architecture(self):
        """Test CNN architecture generation."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.CNN,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32, 16]
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIsInstance(network, nn.Module)
        self.assertTrue('CNNGenome' in str(type(network)))

        # Verify forward pass
        dummy_input = self._create_dummy_input()
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_rnn_architecture(self):
        """Test RNN architecture generation."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.RNN,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32]
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIsInstance(network, nn.Module)
        self.assertTrue('RNNGenome' in str(type(network)))

        # Verify forward pass
        dummy_input = self._create_dummy_input()
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_hybrid_architecture(self):
        """Test Hybrid architecture generation."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.HYBRID,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32]
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIsInstance(network, nn.Module)
        self.assertTrue('HybridGenome' in str(type(network)))

        # Verify forward pass
        dummy_input = self._create_dummy_input()
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

if __name__ == '__main__':
    unittest.main()
