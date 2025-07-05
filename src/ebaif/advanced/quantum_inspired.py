"""
Quantum-Inspired Optimization

Implements quantum-inspired algorithms for optimization and decision making,
including quantum annealing, superposition states, and entanglement effects.
"""

import asyncio
import numpy as np
import random
import math
import cmath
from typing import Dict, List, Tuple, Any, Optional, Complex
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class QuantumConfig:
    """Configuration for quantum-inspired optimization."""
    population_size: int = 50
    max_generations: int = 1000
    rotation_angle: float = 0.05
    quantum_crossover_rate: float = 0.8
    quantum_mutation_rate: float = 0.1
    measurement_collapse_rate: float = 0.3
    entanglement_strength: float = 0.2
    decoherence_rate: float = 0.01
    annealing_schedule: str = "exponential"  # linear, exponential, adaptive
    superposition_states: int = 8

class QuantumBit:
    """Quantum bit representation with superposition states."""
    
    def __init__(self, alpha: Complex = None, beta: Complex = None):
        """Initialize quantum bit with probability amplitudes."""
        if alpha is None and beta is None:
            # Initialize in superposition
            angle = random.uniform(0, 2 * math.pi)
            self.alpha = complex(math.cos(angle / 2), 0)
            self.beta = complex(math.sin(angle / 2), 0)
        else:
            self.alpha = alpha or complex(1, 0)
            self.beta = beta or complex(0, 0)
            
        # Normalize
        self._normalize()
        
    def _normalize(self):
        """Normalize probability amplitudes."""
        norm = (abs(self.alpha) ** 2 + abs(self.beta) ** 2) ** 0.5
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm
            
    def measure(self) -> int:
        """Measure quantum bit, collapsing to classical state."""
        prob_0 = abs(self.alpha) ** 2
        return 0 if random.random() < prob_0 else 1
        
    def rotate(self, theta: float):
        """Apply rotation gate to quantum bit."""
        cos_half = math.cos(theta / 2)
        sin_half = math.sin(theta / 2)
        
        new_alpha = cos_half * self.alpha - sin_half * self.beta
        new_beta = sin_half * self.alpha + cos_half * self.beta
        
        self.alpha = new_alpha
        self.beta = new_beta
        self._normalize()
        
    def probability_of_one(self) -> float:
        """Get probability of measuring |1⟩."""
        return abs(self.beta) ** 2
        
    def entangle_with(self, other: 'QuantumBit', strength: float):
        """Create entanglement with another quantum bit."""
        # Simplified entanglement: mix amplitudes
        factor = strength * 0.5
        
        alpha_mix = (1 - factor) * self.alpha + factor * other.alpha
        beta_mix = (1 - factor) * self.beta + factor * other.beta
        
        other_alpha_mix = factor * self.alpha + (1 - factor) * other.alpha
        other_beta_mix = factor * self.beta + (1 - factor) * other.beta
        
        self.alpha = alpha_mix
        self.beta = beta_mix
        other.alpha = other_alpha_mix
        other.beta = other_beta_mix
        
        self._normalize()
        other._normalize()
        
    def copy(self) -> 'QuantumBit':
        """Create a copy of this quantum bit."""
        return QuantumBit(self.alpha, self.beta)

class QuantumChromosome:
    """Quantum chromosome representing multiple quantum bits."""
    
    def __init__(self, length: int, qubits: List[QuantumBit] = None):
        self.length = length
        self.qubits = qubits or [QuantumBit() for _ in range(length)]
        self.fitness = 0.0
        self.measured_state: Optional[List[int]] = None
        self.entanglement_map = {}
        
    def measure_all(self) -> List[int]:
        """Measure all qubits and return classical bit string."""
        self.measured_state = [qubit.measure() for qubit in self.qubits]
        return self.measured_state
        
    def get_probability_vector(self) -> List[float]:
        """Get probability vector for all qubits being |1⟩."""
        return [qubit.probability_of_one() for qubit in self.qubits]
        
    def quantum_crossover(self, other: 'QuantumChromosome') -> Tuple['QuantumChromosome', 'QuantumChromosome']:
        """Perform quantum crossover creating entangled offspring."""
        offspring1 = QuantumChromosome(self.length)
        offspring2 = QuantumChromosome(self.length)
        
        for i in range(self.length):
            if random.random() < 0.5:
                # Take from parent 1
                offspring1.qubits[i] = self.qubits[i].copy()
                offspring2.qubits[i] = other.qubits[i].copy()
            else:
                # Take from parent 2  
                offspring1.qubits[i] = other.qubits[i].copy()
                offspring2.qubits[i] = self.qubits[i].copy()
                
            # Create quantum entanglement between offspring
            offspring1.qubits[i].entangle_with(offspring2.qubits[i], 0.1)
            
        return offspring1, offspring2
        
    def quantum_mutation(self, mutation_rate: float, rotation_angle: float):
        """Apply quantum mutation through rotations."""
        for qubit in self.qubits:
            if random.random() < mutation_rate:
                # Random rotation
                angle = random.gauss(0, rotation_angle)
                qubit.rotate(angle)
                
    def apply_decoherence(self, decoherence_rate: float):
        """Apply quantum decoherence to reduce quantum effects."""
        for qubit in self.qubits:
            if random.random() < decoherence_rate:
                # Partial decoherence - move toward classical state
                if qubit.probability_of_one() > 0.5:
                    qubit.beta = complex(abs(qubit.beta) * 0.9, 0)
                    qubit.alpha = complex((1 - abs(qubit.beta) ** 2) ** 0.5, 0)
                else:
                    qubit.alpha = complex(abs(qubit.alpha) * 0.9, 0)
                    qubit.beta = complex((1 - abs(qubit.alpha) ** 2) ** 0.5, 0)
                    
    def create_entanglements(self, entanglement_pairs: List[Tuple[int, int]], strength: float):
        """Create entanglements between specified qubit pairs."""
        for i, j in entanglement_pairs:
            if 0 <= i < self.length and 0 <= j < self.length:
                self.qubits[i].entangle_with(self.qubits[j], strength)
                self.entanglement_map[(i, j)] = strength
                
    def get_superposition_entropy(self) -> float:
        """Calculate entropy of superposition states."""
        total_entropy = 0.0
        
        for qubit in self.qubits:
            p0 = abs(qubit.alpha) ** 2
            p1 = abs(qubit.beta) ** 2
            
            if p0 > 0:
                total_entropy -= p0 * math.log2(p0)
            if p1 > 0:
                total_entropy -= p1 * math.log2(p1)
                
        return total_entropy
        
    def copy(self) -> 'QuantumChromosome':
        """Create a copy of this chromosome."""
        new_qubits = [qubit.copy() for qubit in self.qubits]
        new_chromosome = QuantumChromosome(self.length, new_qubits)
        new_chromosome.fitness = self.fitness
        new_chromosome.entanglement_map = self.entanglement_map.copy()
        return new_chromosome

class QuantumAnnealer:
    """Quantum annealing optimizer for complex optimization problems."""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.temperature = 1.0
        self.iteration = 0
        
    def anneal(self, energy_function: callable, initial_state: List[int], 
              max_iterations: int = 1000) -> Dict[str, Any]:
        """Perform quantum annealing optimization."""
        current_state = initial_state.copy()
        current_energy = energy_function(current_state)
        best_state = current_state.copy()
        best_energy = current_energy
        
        energy_history = []
        temperature_history = []
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # Update temperature based on annealing schedule
            self.temperature = self._update_temperature(iteration, max_iterations)
            
            # Generate neighbor state through quantum tunneling
            neighbor_state = self._quantum_tunneling_move(current_state)
            neighbor_energy = energy_function(neighbor_state)
            
            # Quantum acceptance probability
            accept_prob = self._quantum_acceptance_probability(
                current_energy, neighbor_energy, self.temperature
            )
            
            # Accept or reject move
            if random.random() < accept_prob:
                current_state = neighbor_state
                current_energy = neighbor_energy
                
                # Update best solution
                if neighbor_energy < best_energy:
                    best_state = neighbor_state.copy()
                    best_energy = neighbor_energy
                    
            # Record history
            energy_history.append(current_energy)
            temperature_history.append(self.temperature)
            
        return {
            'best_state': best_state,
            'best_energy': best_energy,
            'final_temperature': self.temperature,
            'energy_history': energy_history,
            'temperature_history': temperature_history,
            'iterations': max_iterations
        }
        
    def _update_temperature(self, iteration: int, max_iterations: int) -> float:
        """Update temperature based on annealing schedule."""
        progress = iteration / max_iterations
        
        if self.config.annealing_schedule == "linear":
            return 1.0 - progress
        elif self.config.annealing_schedule == "exponential":
            return math.exp(-5 * progress)
        elif self.config.annealing_schedule == "adaptive":
            # Adaptive schedule based on acceptance rate
            return max(0.01, 1.0 / (1.0 + iteration * 0.01))
        else:
            return 1.0 - progress
            
    def _quantum_tunneling_move(self, state: List[int]) -> List[int]:
        """Generate neighbor state using quantum tunneling effects."""
        new_state = state.copy()
        
        # Quantum tunneling allows multiple simultaneous bit flips
        tunnel_strength = self.temperature * 0.5
        num_flips = max(1, int(tunnel_strength * len(state) * 0.3))
        
        positions = random.sample(range(len(state)), min(num_flips, len(state)))
        
        for pos in positions:
            # Quantum superposition-influenced flip
            flip_prob = 0.5 + 0.3 * tunnel_strength * random.gauss(0, 1)
            flip_prob = max(0.1, min(0.9, flip_prob))
            
            if random.random() < flip_prob:
                new_state[pos] = 1 - new_state[pos]
                
        return new_state
        
    def _quantum_acceptance_probability(self, current_energy: float, 
                                      neighbor_energy: float, temperature: float) -> float:
        """Calculate quantum-enhanced acceptance probability."""
        if neighbor_energy < current_energy:
            return 1.0  # Always accept better solutions
            
        if temperature <= 0:
            return 0.0
            
        # Classical Metropolis criterion
        classical_prob = math.exp(-(neighbor_energy - current_energy) / temperature)
        
        # Quantum enhancement: tunneling probability
        energy_barrier = neighbor_energy - current_energy
        tunnel_prob = math.exp(-energy_barrier / (2 * temperature)) if energy_barrier > 0 else 1.0
        
        # Combine classical and quantum effects
        quantum_prob = 0.7 * classical_prob + 0.3 * tunnel_prob
        
        return min(1.0, quantum_prob)

class QuantumInspiredOptimization:
    """Main quantum-inspired optimization system."""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.population: List[QuantumChromosome] = []
        self.generation = 0
        self.best_chromosome: Optional[QuantumChromosome] = None
        
        # Quantum state tracking
        self.quantum_history = deque(maxlen=1000)
        self.entanglement_history = deque(maxlen=100)
        self.superposition_entropy_history = deque(maxlen=100)
        
        # Annealer for local optimization
        self.annealer = QuantumAnnealer(config)
        
    def initialize_population(self, chromosome_length: int):
        """Initialize quantum population."""
        self.population = []
        
        for i in range(self.config.population_size):
            chromosome = QuantumChromosome(chromosome_length)
            
            # Create some initial entanglements
            if chromosome_length > 2:
                num_entanglements = min(chromosome_length // 4, 5)
                entanglement_pairs = []
                
                for _ in range(num_entanglements):
                    i, j = random.sample(range(chromosome_length), 2)
                    entanglement_pairs.append((i, j))
                    
                chromosome.create_entanglements(entanglement_pairs, self.config.entanglement_strength)
                
            self.population.append(chromosome)
            
    async def evolve(self, fitness_function: callable, max_generations: int = None) -> Dict[str, Any]:
        """Evolve the quantum population."""
        max_generations = max_generations or self.config.max_generations
        evolution_history = []
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate fitness for all chromosomes
            await self._evaluate_population(fitness_function)
            
            # Update best chromosome
            self._update_best_chromosome()
            
            # Record quantum statistics
            quantum_stats = self._record_quantum_statistics()
            evolution_history.append(quantum_stats)
            
            # Quantum evolution operations
            new_population = await self._quantum_evolution_step()
            
            # Apply decoherence
            self._apply_population_decoherence()
            
            # Replace population
            self.population = new_population
            
            # Adaptive parameter adjustment
            self._adapt_quantum_parameters()
            
            # Check convergence
            if self._check_quantum_convergence():
                break
                
        return {
            'best_chromosome': self.best_chromosome,
            'best_fitness': self.best_chromosome.fitness if self.best_chromosome else 0.0,
            'generations_completed': self.generation,
            'evolution_history': evolution_history,
            'final_quantum_state': self._get_population_quantum_state()
        }
        
    async def _evaluate_population(self, fitness_function: callable):
        """Evaluate fitness for all chromosomes in population."""
        evaluation_tasks = []
        
        for chromosome in self.population:
            task = asyncio.create_task(self._evaluate_chromosome(chromosome, fitness_function))
            evaluation_tasks.append(task)
            
        await asyncio.gather(*evaluation_tasks)
        
    async def _evaluate_chromosome(self, chromosome: QuantumChromosome, fitness_function: callable):
        """Evaluate fitness for a single chromosome."""
        try:
            # Multiple measurements for quantum averaging
            fitness_measurements = []
            
            for _ in range(3):  # 3 measurements per chromosome
                measured_state = chromosome.measure_all()
                fitness = await fitness_function(measured_state)
                fitness_measurements.append(fitness)
                
            # Average fitness across measurements
            chromosome.fitness = sum(fitness_measurements) / len(fitness_measurements)
            
        except Exception as e:
            chromosome.fitness = float('-inf')
            
    def _update_best_chromosome(self):
        """Update the best chromosome found so far."""
        if not self.population:
            return
            
        current_best = max(self.population, key=lambda c: c.fitness)
        
        if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = current_best.copy()
            
    async def _quantum_evolution_step(self) -> List[QuantumChromosome]:
        """Perform one step of quantum evolution."""
        new_population = []
        
        # Elite preservation
        elite_count = max(1, int(self.config.population_size * 0.1))
        elites = sorted(self.population, key=lambda c: c.fitness, reverse=True)[:elite_count]
        new_population.extend([elite.copy() for elite in elites])
        
        # Quantum reproduction
        while len(new_population) < self.config.population_size:
            # Quantum selection
            parent1 = self._quantum_selection()
            parent2 = self._quantum_selection()
            
            # Quantum crossover
            if random.random() < self.config.quantum_crossover_rate:
                offspring1, offspring2 = parent1.quantum_crossover(parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
                
            # Quantum mutation
            offspring1.quantum_mutation(self.config.quantum_mutation_rate, self.config.rotation_angle)
            offspring2.quantum_mutation(self.config.quantum_mutation_rate, self.config.rotation_angle)
            
            new_population.extend([offspring1, offspring2])
            
        return new_population[:self.config.population_size]
        
    def _quantum_selection(self) -> QuantumChromosome:
        """Quantum-inspired selection based on superposition of fitness."""
        # Calculate selection probabilities with quantum enhancement
        fitness_values = [c.fitness for c in self.population]
        min_fitness = min(fitness_values)
        
        # Shift to positive values
        shifted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        
        # Quantum superposition weights
        quantum_weights = []
        for i, chromosome in enumerate(self.population):
            base_weight = shifted_fitness[i]
            quantum_entropy = chromosome.get_superposition_entropy()
            quantum_weight = base_weight * (1.0 + 0.2 * quantum_entropy)
            quantum_weights.append(quantum_weight)
            
        # Weighted selection
        total_weight = sum(quantum_weights)
        if total_weight <= 0:
            return random.choice(self.population)
            
        selection_point = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(quantum_weights):
            cumulative_weight += weight
            if cumulative_weight >= selection_point:
                return self.population[i]
                
        return self.population[-1]  # Fallback
        
    def _apply_population_decoherence(self):
        """Apply decoherence to the entire population."""
        for chromosome in self.population:
            chromosome.apply_decoherence(self.config.decoherence_rate)
            
    def _adapt_quantum_parameters(self):
        """Adapt quantum parameters based on evolution progress."""
        if self.generation > 50:
            # Increase decoherence in later generations
            self.config.decoherence_rate = min(0.05, self.config.decoherence_rate * 1.01)
            
            # Decrease rotation angle for fine-tuning
            self.config.rotation_angle = max(0.01, self.config.rotation_angle * 0.99)
            
    def _record_quantum_statistics(self) -> Dict[str, Any]:
        """Record quantum statistics for current generation."""
        if not self.population:
            return {}
            
        fitness_values = [c.fitness for c in self.population]
        entropy_values = [c.get_superposition_entropy() for c in self.population]
        
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitness_values),
            'average_fitness': sum(fitness_values) / len(fitness_values),
            'fitness_variance': np.var(fitness_values),
            'average_quantum_entropy': sum(entropy_values) / len(entropy_values),
            'total_entanglements': sum(len(c.entanglement_map) for c in self.population),
            'decoherence_rate': self.config.decoherence_rate,
            'rotation_angle': self.config.rotation_angle
        }
        
        self.quantum_history.append(stats)
        return stats
        
    def _check_quantum_convergence(self) -> bool:
        """Check if quantum evolution has converged."""
        if len(self.quantum_history) < 20:
            return False
            
        recent_fitness = [stats['best_fitness'] for stats in list(self.quantum_history)[-10:]]
        fitness_variance = np.var(recent_fitness)
        
        # Also check quantum entropy convergence
        recent_entropy = [stats['average_quantum_entropy'] for stats in list(self.quantum_history)[-10:]]
        entropy_variance = np.var(recent_entropy)
        
        return fitness_variance < 1e-8 and entropy_variance < 0.01
        
    def _get_population_quantum_state(self) -> Dict[str, Any]:
        """Get comprehensive quantum state of population."""
        if not self.population:
            return {}
            
        return {
            'population_size': len(self.population),
            'average_quantum_entropy': sum(c.get_superposition_entropy() for c in self.population) / len(self.population),
            'total_entanglements': sum(len(c.entanglement_map) for c in self.population),
            'best_chromosome_state': {
                'fitness': self.best_chromosome.fitness if self.best_chromosome else 0.0,
                'quantum_entropy': self.best_chromosome.get_superposition_entropy() if self.best_chromosome else 0.0,
                'probability_vector': self.best_chromosome.get_probability_vector() if self.best_chromosome else []
            },
            'quantum_parameters': {
                'decoherence_rate': self.config.decoherence_rate,
                'rotation_angle': self.config.rotation_angle,
                'entanglement_strength': self.config.entanglement_strength
            }
        }
        
    async def hybrid_optimize(self, fitness_function: callable, 
                            initial_state: List[int] = None) -> Dict[str, Any]:
        """Hybrid quantum-classical optimization combining evolution and annealing."""
        chromosome_length = len(initial_state) if initial_state else 20
        
        # Phase 1: Quantum evolutionary search
        self.initialize_population(chromosome_length)
        evolution_result = await self.evolve(fitness_function, max_generations=500)
        
        # Phase 2: Quantum annealing refinement
        if self.best_chromosome:
            best_classical_state = self.best_chromosome.measure_all()
        else:
            best_classical_state = initial_state or [random.randint(0, 1) for _ in range(chromosome_length)]
            
        annealing_result = self.annealer.anneal(
            lambda state: -fitness_function(state),  # Minimize negative fitness
            best_classical_state,
            max_iterations=1000
        )
        
        return {
            'evolution_phase': evolution_result,
            'annealing_phase': annealing_result,
            'final_best_state': annealing_result['best_state'],
            'final_best_fitness': -annealing_result['best_energy'],  # Convert back to fitness
            'hybrid_optimization_complete': True
        }
        
    def get_quantum_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of quantum optimization."""
        return {
            'generations_completed': self.generation,
            'population_size': len(self.population),
            'best_fitness_achieved': self.best_chromosome.fitness if self.best_chromosome else 0.0,
            'quantum_state_summary': self._get_population_quantum_state(),
            'evolution_history': list(self.quantum_history)[-20:],  # Last 20 generations
            'convergence_status': self._check_quantum_convergence()
        }

class QuantumInspiredOptimization:
    """Main quantum-inspired optimization interface."""
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.optimizer = QuantumInspiredOptimization(config)
        
    async def optimize(self, fitness_function: callable, problem_size: int = 20) -> Dict[str, Any]:
        """Run quantum-inspired optimization."""
        return await self.optimizer.hybrid_optimize(fitness_function)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return self.optimizer.get_quantum_summary()