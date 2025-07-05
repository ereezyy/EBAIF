"""
Simple Test

Very basic test to verify the framework works.
"""

import torch
from src.ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig

def test_genome_creation():
    """Test basic genome creation."""
    print("Testing genome creation...")
    
    config = GenomeConfig(
        input_dim=32,
        output_dim=8,
        hidden_dims=[64, 32]
    )
    
    genome = BehaviorGenome(config)
    print(f"Created genome: {genome}")
    
    # Test building network
    network = genome.build_network()
    print(f"Built network: {network}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 10, 32)  # batch_size=1, seq_len=10, input_dim=32
    output = network(dummy_input)
    print(f"Network output shape: {output.shape}")
    
    # Test mutation
    mutated_genome = genome.mutate()
    print(f"Mutated genome: {mutated_genome}")
    
    print("Genome test passed!")

def test_behavior_parameters():
    """Test behavior parameter extraction."""
    print("Testing behavior parameters...")
    
    genome = BehaviorGenome()
    params = genome.get_behavior_parameters()
    print(f"Behavior parameters: {params}")
    
    print("Behavior parameters test passed!")

if __name__ == "__main__":
    test_genome_creation()
    test_behavior_parameters()
    print("All tests passed!")