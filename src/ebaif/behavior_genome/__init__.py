"""
Behavior Genome Module

This module implements the core behavior genome system that enables AI agents
to evolve their neural architectures based on performance and interaction.

Key Components:
- BehaviorGenome: Core genome representation and evolution
- GenomeEvolution: Genetic algorithms for architecture evolution
- ArchitectureSearch: Neural architecture search implementation
- PerformanceEvaluator: Fitness evaluation for genome selection
"""

from .genome import BehaviorGenome
from .evolution import GenomeEvolution
from .architecture_search import ArchitectureSearch
from .evaluator import PerformanceEvaluator

__all__ = [
    "BehaviorGenome",
    "GenomeEvolution", 
    "ArchitectureSearch",
    "PerformanceEvaluator",
]

