"""
Configuration Management

Simple configuration management for EBAIF framework.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Config:
    """Configuration manager for EBAIF."""
    
    # Behavior Genome Settings
    evolution_rate: float = 0.1
    mutation_probability: float = 0.05
    selection_pressure: float = 0.8
    max_generations: int = 1000
    
    # Consensus Settings
    validation_threshold: float = 0.8
    propagation_delay: float = 1.0
    max_validators: int = 10
    consensus_timeout: float = 30.0
    
    # Agent Settings
    max_agents: int = 100
    update_frequency: float = 10.0
    learning_rate: float = 0.001
    memory_size: int = 10000
    
    # System Settings
    log_level: str = "INFO"
    metrics_enabled: bool = True
    distributed_mode: bool = False
    edge_optimization: bool = False
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        if not os.path.exists(filepath):
            return cls()
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)