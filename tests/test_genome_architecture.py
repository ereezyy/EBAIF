import sys
from unittest.mock import MagicMock

# Mock missing modules before importing src
sys.modules['src.ebaif.consensus.protocol'] = MagicMock()
sys.modules['src.ebaif.consensus.reputation'] = MagicMock()

# Also need to mock the classes inside the mocked modules
sys.modules['src.ebaif.consensus.protocol'].ConsensusProtocol = MagicMock()
sys.modules['src.ebaif.consensus.reputation'].ReputationSystem = MagicMock()

import unittest
import torch
import torch.nn as nn
from src.ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig, ArchitectureType

class TestGenomeArchitecture(unittest.TestCase):
    def setUp(self):
        self.input_dim = 32
        self.output_dim = 8
        self.hidden_dims = [64, 32]
        self.batch_size = 4
        self.seq_len = 10

    def test_transformer_architecture(self):
        """Test Transformer architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.TRANSFORMER,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            num_layers=2,
            attention_heads=4
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        # Verify network type
        self.assertTrue("TransformerGenome" in str(type(network)),
                        f"Expected TransformerGenome, got {type(network)}")

        # Verify forward pass
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_cnn_architecture(self):
        """Test CNN architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.CNN,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertTrue("CNNGenome" in str(type(network)),
                        f"Expected CNNGenome, got {type(network)}")

        # CNN expects 1D input in this implementation (batch, length/features)
        dummy_input = torch.randn(self.batch_size, self.input_dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_rnn_architecture(self):
        """Test RNN architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.RNN,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertTrue("RNNGenome" in str(type(network)),
                        f"Expected RNNGenome, got {type(network)}")

        # RNN expects (batch, seq_len, input_dim)
        dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_hybrid_architecture(self):
        """Test Hybrid architecture creation and forward pass."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.HYBRID,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims
        )
        genome = BehaviorGenome(config)
        network = genome.build_network()

        self.assertTrue("HybridGenome" in str(type(network)),
                        f"Expected HybridGenome, got {type(network)}")

        # Hybrid implementation expects (batch, features)
        dummy_input = torch.randn(self.batch_size, self.input_dim)
        output = network(dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_network_caching(self):
        """Test that build_network returns the cached network on subsequent calls."""
        config = GenomeConfig(
            architecture_type=ArchitectureType.TRANSFORMER,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims
        )
        genome = BehaviorGenome(config)
        network1 = genome.build_network()
        network2 = genome.build_network()

        self.assertIs(network1, network2, "build_network should return the cached network instance")

if __name__ == '__main__':
    unittest.main()
