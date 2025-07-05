"""
Genome Evolution Implementation

Simplified genome evolution system for EBAIF.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from .genome import BehaviorGenome

class GenomeEvolution:
    """Handles evolution of behavior genomes."""
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_ratio: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        
    def evolve_population(self, population: List[BehaviorGenome]) -> List[BehaviorGenome]:
        """Evolve a population of genomes."""
        # Sort by fitness
        population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        new_population = []
        
        # Keep elite individuals
        elite_count = int(len(population) * self.elite_ratio)
        new_population.extend(population[:elite_count])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if torch.rand(1).item() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
                
        # Mutation
        for genome in new_population[elite_count:]:
            if torch.rand(1).item() < self.mutation_rate:
                genome = genome.mutate()
                
        return new_population[:self.population_size]
        
    def _tournament_selection(self, population: List[BehaviorGenome], 
                            tournament_size: int = 3) -> BehaviorGenome:
        """Tournament selection."""
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda g: g.fitness_score)