import unittest
import torch
import torch.nn as nn
from src.ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig, ArchitectureType, ActivationFunction

class TestGenomeArchitecture(unittest.TestCase):
    """Test suite for BehaviorGenome network construction and variants."""

    def test_transformer_architecture(self):
        """Test building a Transformer-based network."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.TRANSFORMER,
            input_dim=32,
            output_dim=8,
            hidden_dims=[64, 32],
            num_layers=2,
            attention_heads=4
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertIsInstance(network, nn.Module)
        # Check if internal components exist (implementation detail, but good sanity check)
        self.assertTrue(hasattr(network, 'transformer'))
        self.assertTrue(hasattr(network, 'input_projection'))

        # Verify forward pass
        dummy_input = torch.randn(1, 10, 32)  # (batch, seq, dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (1, 8))

    def test_cnn_architecture(self):
        """Test building a CNN-based network."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.CNN,
            input_dim=32,
            output_dim=8,
            hidden_dims=[16, 32],  # Channel sizes
            dropout_rate=0.1
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertIsInstance(network, nn.Module)
        self.assertTrue(hasattr(network, 'layers'))
        self.assertTrue(hasattr(network, 'global_pool'))

        # Verify forward pass with 2D input (batch, dim) -> unsqueezed internally
        dummy_input = torch.randn(1, 32)
        output = network(dummy_input)
        self.assertEqual(output.shape, (1, 8))

        # Verify forward pass with 3D input (batch, channels, length) - standard for 1D conv
        # Note: CNN implementation expects (batch, channels, length) if 3D
        # Wait, let's check the implementation:
        # if x.dim() == 2: x = x.unsqueeze(1) -> (batch, 1, dim)
        # So it treats input_dim as length for the first layer?
        # "Assume input is 1D for simplicity... in_channels = 1"
        # So it expects (batch, 1, input_dim) or (batch, input_dim)

        dummy_input_3d = torch.randn(1, 1, 32)
        output_3d = network(dummy_input_3d)
        self.assertEqual(output_3d.shape, (1, 8))

    def test_rnn_architecture(self):
        """Test building an RNN-based network."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.RNN,
            input_dim=32,
            output_dim=8,
            hidden_dims=[64],
            num_layers=2
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertIsInstance(network, nn.Module)
        self.assertTrue(hasattr(network, 'rnn'))

        # Verify forward pass
        dummy_input = torch.randn(1, 10, 32)  # (batch, seq, dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (1, 8))

    def test_hybrid_architecture(self):
        """Test building a Hybrid network."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.HYBRID,
            input_dim=32,
            output_dim=8,
            hidden_dims=[64, 32]
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertIsInstance(network, nn.Module)
        self.assertTrue(hasattr(network, 'cnn'))
        self.assertTrue(hasattr(network, 'rnn'))
        self.assertTrue(hasattr(network, 'attention'))

        # Verify forward pass
        # Hybrid likely expects sequence data or flat data treated as sequence
        # Implementation:
        # CNN: (batch, 1, dim) -> (batch, 64, dim) -> transpose -> (batch, dim, 64)
        # RNN: (batch, dim, 64) -> (batch, dim, 128)
        # Attention: (batch, dim, 128)
        # Mean pooling -> (batch, 128) -> Linear -> (batch, output_dim)

        dummy_input = torch.randn(1, 32) # (batch, dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (1, 8))

    def test_network_caching(self):
        """Test that build_network returns the same instance if called multiple times."""
        config = GenomeConfig(architecture_type=ArchitectureType.CNN)
        genome = BehaviorGenome(config)

        net1 = genome.build_network()
        net2 = genome.build_network()

        self.assertIs(net1, net2)

if __name__ == '__main__':
    unittest.main()
