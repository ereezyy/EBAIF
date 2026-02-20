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
        """Save configuration to JSON file.

        Security: This method validates the filepath to ensure it ends with .json
        and resides within the current working directory to prevent arbitrary file writes.
        """
        if not filepath.endswith('.json'):
            raise ValueError("Configuration file must have .json extension")

        # Resolve path and check against CWD to prevent directory traversal
        abs_path = os.path.abspath(filepath)
        cwd = os.getcwd()

        # Ensure we are writing within the current working directory
        try:
            if os.path.commonpath([abs_path, cwd]) != cwd:
                raise ValueError("Path must be within current working directory")
        except ValueError:
            # Handle case where paths are on different drives
            raise ValueError("Path must be within current working directory")

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
