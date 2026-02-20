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
        """Load configuration from JSON file.

        This method validates that the file is within the current working directory
        and has a .json extension to prevent path traversal attacks.
        """
        # Validate extension
        if not filepath.endswith('.json'):
            raise ValueError("Configuration file must have .json extension")

        # Normalize path and check against CWD
        cwd = os.getcwd()
        abs_path = os.path.abspath(filepath)

        # Use commonpath to check if abs_path is within cwd
        # Note: os.path.commonpath raises ValueError on different drives on Windows
        try:
            if os.path.commonpath([cwd, abs_path]) != cwd:
                raise ValueError(f"Access denied: Path '{filepath}' is outside the working directory.")
        except ValueError:
             raise ValueError(f"Access denied: Path '{filepath}' is invalid.")

        if not os.path.exists(abs_path):
            return cls()
            
        with open(abs_path, 'r') as f:
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
