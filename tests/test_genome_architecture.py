import unittest
import torch
import torch.nn as nn
import sys
from unittest.mock import MagicMock

# Mock missing modules to allow imports
sys.modules['src.ebaif.consensus.protocol'] = MagicMock()
sys.modules['src.ebaif.consensus.reputation'] = MagicMock()

from src.ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig, ArchitectureType

class TestGenomeArchitecture(unittest.TestCase):
    def setUp(self):
        # Base config parameters
        self.input_dim = 16
        self.output_dim = 4
        self.batch_size = 2
        self.seq_len = 10

    def test_transformer_architecture(self):
        """Test Transformer architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.TRANSFORMER,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32, 16],
            num_layers=2,
            attention_heads=2
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type using class name as class is local
        self.assertIn('TransformerGenome', str(type(network)))
        self.assertIsInstance(network, nn.Module)

        # Verify forward pass
        # Transformer expects (batch_size, seq_len, input_dim)
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = network(dummy_input)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_cnn_architecture(self):
        """Test CNN architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.CNN,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32, 16]
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIn('CNNGenome', str(type(network)))
        self.assertIsInstance(network, nn.Module)

        # CNN implementation expects input that can be processed by Conv1d(1, ...).
        # The implementation unsqueezes 2D input (batch, dim) -> (batch, 1, dim).
        dummy_input = torch.randn(self.batch_size, self.input_dim)
        output = network(dummy_input)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_rnn_architecture(self):
        """Test RNN architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.RNN,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32],
            num_layers=1
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIn('RNNGenome', str(type(network)))
        self.assertIsInstance(network, nn.Module)

        # RNN expects (batch_size, seq_len, input_dim)
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = network(dummy_input)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_hybrid_architecture(self):
        """Test Hybrid architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.HYBRID,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[32],
            num_layers=1
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertIn('HybridGenome', str(type(network)))
        self.assertIsInstance(network, nn.Module)

        # Hybrid implementation expects 2D input (batch, dim) which it unsqueezes
        dummy_input = torch.randn(self.batch_size, self.input_dim)
        output = network(dummy_input)

        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_network_caching(self):
        """Test that the network is cached after being built."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.TRANSFORMER,
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        genome = BehaviorGenome(config)
        network1 = genome.build_network()
        network2 = genome.build_network()

        self.assertIs(network1, network2)

if __name__ == '__main__':
    unittest.main()
