"""
Emergent Behavior AI Framework (EBAIF)

A revolutionary framework for creating truly emergent AI behaviors in gaming
and interactive environments through neural architecture evolution and
distributed consensus mechanisms.

Author: Eddy Woods (ereezyy)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Eddy Woods"
__email__ = "contact@ereezyy.dev"
__license__ = "MIT"

# Core imports
from .behavior_genome import BehaviorGenome, GenomeEvolution
from .consensus import ConsensusEngine, BehaviorValidator
from .agents import EmergentAgent, AgentManager
from .utils import Config, Logger, Metrics

# Version info
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release": "alpha",
}

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO

# Package-level configuration
DEFAULT_CONFIG = {
    "behavior_genome": {
        "evolution_rate": 0.1,
        "mutation_probability": 0.05,
        "selection_pressure": 0.8,
        "max_generations": 1000,
    },
    "consensus": {
        "validation_threshold": 0.8,
        "propagation_delay": 1.0,
        "max_validators": 10,
        "consensus_timeout": 30.0,
    },
    "agents": {
        "max_agents": 100,
        "update_frequency": 10.0,
        "learning_rate": 0.001,
        "memory_size": 10000,
    },
    "system": {
        "log_level": "INFO",
        "metrics_enabled": True,
        "distributed_mode": False,
        "edge_optimization": False,
    },
}

# Export main classes and functions
__all__ = [
    # Core classes
    "BehaviorGenome",
    "GenomeEvolution", 
    "ConsensusEngine",
    "BehaviorValidator",
    "EmergentAgent",
    "AgentManager",
    
    # Utilities
    "Config",
    "Logger", 
    "Metrics",
    
    # Version functions
    "get_version",
    "get_version_info",
    
    # Configuration
    "DEFAULT_CONFIG",
]

