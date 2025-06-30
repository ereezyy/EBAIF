"""
Behavior Genome Implementation

The BehaviorGenome class represents the core genetic structure that defines
an AI agent's neural architecture and behavioral parameters. It supports
dynamic evolution and adaptation based on performance feedback.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import hashlib
from enum import Enum

class ArchitectureType(Enum):
    """Supported neural architecture types."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class ActivationFunction(Enum):
    """Supported activation functions."""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"

@dataclass
class GenomeConfig:
    """Configuration for behavior genome."""
    architecture_type: ArchitectureType = ArchitectureType.TRANSFORMER
    input_dim: int = 512
    output_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: ActivationFunction = ActivationFunction.GELU
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 128
    attention_heads: int = 8
    num_layers: int = 6
    evolution_rate: float = 0.1
    mutation_probability: float = 0.05
    crossover_probability: float = 0.7

class BehaviorGenome:
    """
    Core behavior genome class that represents an AI agent's neural architecture
    and behavioral parameters. Supports evolution and adaptation.
    """
    
    def __init__(self, config: Optional[GenomeConfig] = None, genome_id: Optional[str] = None):
        """
        Initialize a behavior genome.
        
        Args:
            config: Genome configuration parameters
            genome_id: Unique identifier for this genome
        """
        self.config = config or GenomeConfig()
        self.genome_id = genome_id or self._generate_genome_id()
        self.generation = 0
        self.fitness_score = 0.0
        self.parent_ids: List[str] = []
        self.creation_time = torch.tensor(0.0)  # Will be set by framework
        
        # Neural architecture parameters
        self.architecture_genes = self._initialize_architecture_genes()
        self.behavior_genes = self._initialize_behavior_genes()
        
        # Performance tracking
        self.performance_history: List[float] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Network instance (lazy initialization)
        self._network: Optional[nn.Module] = None
        
    def _generate_genome_id(self) -> str:
        """Generate a unique genome ID based on configuration and timestamp."""
        config_str = json.dumps(self.config.__dict__, sort_keys=True, default=str)
        timestamp = str(torch.tensor(0.0).item())  # Placeholder for actual timestamp
        hash_input = f"{config_str}_{timestamp}_{np.random.randint(0, 1000000)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _initialize_architecture_genes(self) -> Dict[str, torch.Tensor]:
        """Initialize the neural architecture genes."""
        genes = {}
        
        # Layer configuration genes
        genes['layer_sizes'] = torch.tensor(self.config.hidden_dims, dtype=torch.float32)
        genes['num_layers'] = torch.tensor(self.config.num_layers, dtype=torch.float32)
        genes['attention_heads'] = torch.tensor(self.config.attention_heads, dtype=torch.float32)
        
        # Activation function genes (one-hot encoded)
        activation_map = {act: i for i, act in enumerate(ActivationFunction)}
        activation_vector = torch.zeros(len(ActivationFunction))
        activation_vector[activation_map[self.config.activation]] = 1.0
        genes['activation'] = activation_vector
        
        # Regularization genes
        genes['dropout_rate'] = torch.tensor(self.config.dropout_rate)
        genes['learning_rate'] = torch.tensor(self.config.learning_rate)
        
        # Architecture topology genes (for more complex evolution)
        genes['skip_connections'] = torch.rand(self.config.num_layers) > 0.5
        genes['layer_types'] = torch.randint(0, 3, (self.config.num_layers,))  # 0=dense, 1=conv, 2=attention
        
        return genes
    
    def _initialize_behavior_genes(self) -> Dict[str, torch.Tensor]:
        """Initialize behavioral parameter genes."""
        genes = {}
        
        # Exploration vs exploitation balance
        genes['exploration_rate'] = torch.tensor(0.1)
        genes['curiosity_factor'] = torch.tensor(0.5)
        genes['risk_tolerance'] = torch.tensor(0.3)
        
        # Social behavior parameters
        genes['cooperation_tendency'] = torch.tensor(0.6)
        genes['communication_frequency'] = torch.tensor(0.4)
        genes['leadership_inclination'] = torch.tensor(0.2)
        
        # Learning behavior parameters
        genes['adaptation_speed'] = torch.tensor(0.5)
        genes['memory_retention'] = torch.tensor(0.8)
        genes['pattern_recognition'] = torch.tensor(0.7)
        
        # Task-specific behavior genes
        genes['aggression_level'] = torch.tensor(0.3)
        genes['defensive_behavior'] = torch.tensor(0.6)
        genes['resource_sharing'] = torch.tensor(0.4)
        
        return genes
    
    def build_network(self) -> nn.Module:
        """Build the neural network based on current genome configuration."""
        if self._network is not None:
            return self._network
            
        # Extract architecture parameters from genes
        layer_sizes = self.architecture_genes['layer_sizes'].int().tolist()
        num_layers = int(self.architecture_genes['num_layers'].item())
        dropout_rate = self.architecture_genes['dropout_rate'].item()
        
        # Determine activation function
        activation_idx = torch.argmax(self.architecture_genes['activation']).item()
        activation_func = list(ActivationFunction)[activation_idx]
        
        # Build network based on architecture type
        if self.config.architecture_type == ArchitectureType.TRANSFORMER:
            self._network = self._build_transformer_network(layer_sizes, num_layers, dropout_rate, activation_func)
        elif self.config.architecture_type == ArchitectureType.CNN:
            self._network = self._build_cnn_network(layer_sizes, dropout_rate, activation_func)
        elif self.config.architecture_type == ArchitectureType.RNN:
            self._network = self._build_rnn_network(layer_sizes, dropout_rate, activation_func)
        else:
            self._network = self._build_hybrid_network(layer_sizes, num_layers, dropout_rate, activation_func)
            
        return self._network
    
    def _build_transformer_network(self, layer_sizes: List[int], num_layers: int, 
                                 dropout_rate: float, activation_func: ActivationFunction) -> nn.Module:
        """Build a transformer-based network."""
        class TransformerGenome(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout_rate):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout_rate,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                seq_len = x.size(1)
                x = self.input_projection(x)
                x = x + self.positional_encoding[:seq_len].unsqueeze(0)
                x = self.dropout(x)
                x = self.transformer(x)
                x = self.output_projection(x.mean(dim=1))  # Global average pooling
                return x
        
        hidden_dim = layer_sizes[0] if layer_sizes else 512
        num_heads = int(self.architecture_genes['attention_heads'].item())
        
        return TransformerGenome(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
    
    def _build_cnn_network(self, layer_sizes: List[int], dropout_rate: float, 
                          activation_func: ActivationFunction) -> nn.Module:
        """Build a CNN-based network."""
        class CNNGenome(nn.Module):
            def __init__(self, input_dim, output_dim, layer_sizes, dropout_rate, activation):
                super().__init__()
                self.layers = nn.ModuleList()
                
                # Assume input is 1D for simplicity, can be extended for 2D/3D
                in_channels = 1
                for size in layer_sizes:
                    self.layers.append(nn.Conv1d(in_channels, size, kernel_size=3, padding=1))
                    self.layers.append(self._get_activation(activation))
                    self.layers.append(nn.Dropout(dropout_rate))
                    in_channels = size
                
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.classifier = nn.Linear(layer_sizes[-1] if layer_sizes else input_dim, output_dim)
                
            def _get_activation(self, activation):
                activation_map = {
                    ActivationFunction.RELU: nn.ReLU(),
                    ActivationFunction.GELU: nn.GELU(),
                    ActivationFunction.SWISH: nn.SiLU(),
                    ActivationFunction.TANH: nn.Tanh(),
                    ActivationFunction.SIGMOID: nn.Sigmoid(),
                    ActivationFunction.LEAKY_RELU: nn.LeakyReLU(),
                }
                return activation_map.get(activation, nn.ReLU())
                
            def forward(self, x):
                # Reshape for 1D conv if needed
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.global_pool(x).squeeze(-1)
                x = self.classifier(x)
                return x
        
        return CNNGenome(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            activation=activation_func
        )
    
    def _build_rnn_network(self, layer_sizes: List[int], dropout_rate: float,
                          activation_func: ActivationFunction) -> nn.Module:
        """Build an RNN-based network."""
        class RNNGenome(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
                super().__init__()
                self.rnn = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout_rate if num_layers > 1 else 0,
                    batch_first=True
                )
                self.classifier = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                output, (hidden, cell) = self.rnn(x)
                # Use the last output
                x = output[:, -1, :]
                x = self.dropout(x)
                x = self.classifier(x)
                return x
        
        hidden_dim = layer_sizes[0] if layer_sizes else 256
        num_layers = len(layer_sizes) if layer_sizes else 2
        
        return RNNGenome(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
    
    def _build_hybrid_network(self, layer_sizes: List[int], num_layers: int,
                             dropout_rate: float, activation_func: ActivationFunction) -> nn.Module:
        """Build a hybrid network combining multiple architectures."""
        class HybridGenome(nn.Module):
            def __init__(self, input_dim, output_dim, layer_sizes, dropout_rate, activation):
                super().__init__()
                # Combine CNN and RNN components
                self.cnn = nn.Conv1d(1, 64, kernel_size=3, padding=1)
                self.rnn = nn.LSTM(64, 128, batch_first=True)
                self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
                self.classifier = nn.Linear(128, output_dim)
                self.dropout = nn.Dropout(dropout_rate)
                
            def forward(self, x):
                # CNN processing
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = self.cnn(x)
                x = x.transpose(1, 2)  # (batch, seq, features)
                
                # RNN processing
                x, _ = self.rnn(x)
                
                # Self-attention
                x, _ = self.attention(x, x, x)
                
                # Global pooling and classification
                x = x.mean(dim=1)
                x = self.dropout(x)
                x = self.classifier(x)
                return x
        
        return HybridGenome(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            layer_sizes=layer_sizes,
            dropout_rate=dropout_rate,
            activation=activation_func
        )
    
    def mutate(self, mutation_rate: Optional[float] = None) -> 'BehaviorGenome':
        """
        Create a mutated version of this genome.
        
        Args:
            mutation_rate: Override default mutation rate
            
        Returns:
            New mutated genome
        """
        mutation_rate = mutation_rate or self.config.mutation_probability
        
        # Create a copy of this genome
        new_genome = BehaviorGenome(self.config)
        new_genome.architecture_genes = {k: v.clone() for k, v in self.architecture_genes.items()}
        new_genome.behavior_genes = {k: v.clone() for k, v in self.behavior_genes.items()}
        new_genome.parent_ids = [self.genome_id]
        new_genome.generation = self.generation + 1
        
        # Mutate architecture genes
        for key, gene in new_genome.architecture_genes.items():
            if torch.rand(1).item() < mutation_rate:
                if gene.dtype == torch.bool:
                    # Flip boolean values
                    new_genome.architecture_genes[key] = ~gene
                elif key == 'activation':
                    # Randomly select new activation function
                    new_activation = torch.zeros_like(gene)
                    new_activation[torch.randint(0, len(gene), (1,))] = 1.0
                    new_genome.architecture_genes[key] = new_activation
                else:
                    # Add Gaussian noise to continuous values
                    noise = torch.randn_like(gene) * 0.1
                    new_genome.architecture_genes[key] = torch.clamp(gene + noise, 0.001, 10.0)
        
        # Mutate behavior genes
        for key, gene in new_genome.behavior_genes.items():
            if torch.rand(1).item() < mutation_rate:
                noise = torch.randn_like(gene) * 0.05
                new_genome.behavior_genes[key] = torch.clamp(gene + noise, 0.0, 1.0)
        
        return new_genome
    
    def crossover(self, other: 'BehaviorGenome') -> Tuple['BehaviorGenome', 'BehaviorGenome']:
        """
        Perform crossover with another genome to create offspring.
        
        Args:
            other: Another genome to crossover with
            
        Returns:
            Tuple of two offspring genomes
        """
        # Create two offspring
        offspring1 = BehaviorGenome(self.config)
        offspring2 = BehaviorGenome(self.config)
        
        # Set parent information
        offspring1.parent_ids = [self.genome_id, other.genome_id]
        offspring2.parent_ids = [self.genome_id, other.genome_id]
        offspring1.generation = max(self.generation, other.generation) + 1
        offspring2.generation = max(self.generation, other.generation) + 1
        
        # Crossover architecture genes
        for key in self.architecture_genes.keys():
            if torch.rand(1).item() < 0.5:
                offspring1.architecture_genes[key] = self.architecture_genes[key].clone()
                offspring2.architecture_genes[key] = other.architecture_genes[key].clone()
            else:
                offspring1.architecture_genes[key] = other.architecture_genes[key].clone()
                offspring2.architecture_genes[key] = self.architecture_genes[key].clone()
        
        # Crossover behavior genes
        for key in self.behavior_genes.keys():
            if torch.rand(1).item() < 0.5:
                offspring1.behavior_genes[key] = self.behavior_genes[key].clone()
                offspring2.behavior_genes[key] = other.behavior_genes[key].clone()
            else:
                offspring1.behavior_genes[key] = other.behavior_genes[key].clone()
                offspring2.behavior_genes[key] = self.behavior_genes[key].clone()
        
        return offspring1, offspring2
    
    def update_fitness(self, fitness_score: float):
        """Update the fitness score and performance history."""
        self.fitness_score = fitness_score
        self.performance_history.append(fitness_score)
        
        # Keep only recent history to manage memory
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_behavior_parameters(self) -> Dict[str, float]:
        """Get current behavior parameters as a dictionary."""
        return {key: gene.item() for key, gene in self.behavior_genes.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization."""
        return {
            'genome_id': self.genome_id,
            'generation': self.generation,
            'fitness_score': self.fitness_score,
            'parent_ids': self.parent_ids,
            'config': self.config.__dict__,
            'architecture_genes': {k: v.tolist() for k, v in self.architecture_genes.items()},
            'behavior_genes': {k: v.tolist() for k, v in self.behavior_genes.items()},
            'performance_history': self.performance_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehaviorGenome':
        """Create genome from dictionary."""
        config = GenomeConfig(**data['config'])
        genome = cls(config, data['genome_id'])
        genome.generation = data['generation']
        genome.fitness_score = data['fitness_score']
        genome.parent_ids = data['parent_ids']
        genome.performance_history = data['performance_history']
        
        # Restore gene tensors
        genome.architecture_genes = {k: torch.tensor(v) for k, v in data['architecture_genes'].items()}
        genome.behavior_genes = {k: torch.tensor(v) for k, v in data['behavior_genes'].items()}
        
        return genome
    
    def __repr__(self) -> str:
        return f"BehaviorGenome(id={self.genome_id[:8]}, gen={self.generation}, fitness={self.fitness_score:.3f})"

