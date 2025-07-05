"""
Performance Evaluator Implementation

Evaluates genome performance for EBAIF.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from .genome import BehaviorGenome

class PerformanceEvaluator:
    """Evaluates performance of behavior genomes."""
    
    def __init__(self):
        self.evaluation_history = []
        
    def evaluate_genome(self, genome: BehaviorGenome, 
                       task_results: Dict[str, float]) -> float:
        """Evaluate genome performance."""
        
        # Base performance from task results
        base_score = task_results.get('accuracy', 0.0)
        
        # Behavioral factors
        behavior_params = genome.get_behavior_parameters()
        
        # Cooperation bonus
        cooperation_bonus = behavior_params.get('cooperation_tendency', 0.5) * 0.1
        
        # Exploration bonus (balanced exploration)
        exploration = behavior_params.get('exploration_rate', 0.5)
        exploration_bonus = (1.0 - abs(exploration - 0.5)) * 0.1
        
        # Stability bonus
        if len(genome.performance_history) > 1:
            stability = 1.0 - np.std(genome.performance_history[-10:])
            stability_bonus = stability * 0.1
        else:
            stability_bonus = 0.0
            
        total_score = base_score + cooperation_bonus + exploration_bonus + stability_bonus
        
        # Update genome fitness
        genome.update_fitness(total_score)
        
        return total_score
        
    def compare_genomes(self, genome1: BehaviorGenome, 
                       genome2: BehaviorGenome) -> int:
        """Compare two genomes. Returns 1 if genome1 is better, -1 if genome2 is better, 0 if equal."""
        if genome1.fitness_score > genome2.fitness_score:
            return 1
        elif genome1.fitness_score < genome2.fitness_score:
            return -1
        else:
            return 0