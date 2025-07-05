"""
Swarm Intelligence Implementation

Implements collective intelligence algorithms including particle swarm optimization,
ant colony optimization, and emergent swarm behaviors for distributed AI systems.
"""

import asyncio
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
import time

@dataclass
class SwarmConfig:
    """Configuration for swarm intelligence system."""
    swarm_size: int = 100
    max_iterations: int = 1000
    communication_radius: float = 0.3
    information_decay: float = 0.95
    exploration_factor: float = 0.4
    exploitation_factor: float = 0.6
    social_learning_rate: float = 0.02
    individual_learning_rate: float = 0.01
    collective_memory_size: int = 1000
    emergence_threshold: float = 0.7
    adaptation_rate: float = 0.1

class SwarmAgent:
    """Individual agent in the swarm with collective intelligence capabilities."""
    
    def __init__(self, agent_id: str, position: List[float] = None, dimension: int = 10):
        self.agent_id = agent_id
        self.dimension = dimension
        self.position = position or [random.uniform(-5, 5) for _ in range(dimension)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimension)]
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('-inf')
        self.current_fitness = float('-inf')
        
        # Social intelligence attributes
        self.neighbors: List[str] = []
        self.social_information = {}
        self.communication_history = deque(maxlen=100)
        self.trust_levels: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Learning and adaptation
        self.learning_rate = random.uniform(0.005, 0.015)
        self.exploration_tendency = random.uniform(0.2, 0.8)
        self.cooperation_level = random.uniform(0.3, 0.9)
        self.specialization_score = 0.0
        
        # Memory and experience
        self.local_memory = deque(maxlen=50)
        self.experience_buffer = deque(maxlen=200)
        self.skill_repertoire = {}
        
        # State tracking
        self.age = 0
        self.energy_level = 1.0
        self.role = "explorer"  # explorer, exploiter, communicator, leader
        
    def update_position(self, global_best_position: List[float], swarm_config: SwarmConfig):
        """Update agent position using PSO dynamics with enhancements."""
        w = 0.729  # Inertia weight
        c1 = 1.49445  # Cognitive coefficient
        c2 = 1.49445  # Social coefficient
        
        for i in range(self.dimension):
            # Standard PSO velocity update
            cognitive_component = c1 * random.random() * (self.personal_best_position[i] - self.position[i])
            social_component = c2 * random.random() * (global_best_position[i] - self.position[i])
            
            # Add swarm intelligence enhancements
            exploration_component = self._calculate_exploration_component(i, swarm_config)
            social_learning_component = self._calculate_social_learning_component(i, swarm_config)
            
            # Update velocity
            self.velocity[i] = (w * self.velocity[i] + 
                              cognitive_component + 
                              social_component +
                              exploration_component +
                              social_learning_component)
            
            # Apply velocity limits
            max_velocity = 2.0
            self.velocity[i] = max(-max_velocity, min(max_velocity, self.velocity[i]))
            
            # Update position
            self.position[i] += self.velocity[i]
            
            # Apply boundary constraints
            self.position[i] = max(-10, min(10, self.position[i]))
            
    def _calculate_exploration_component(self, dimension: int, config: SwarmConfig) -> float:
        """Calculate exploration component based on agent's exploration tendency."""
        if self.role == "explorer":
            exploration_magnitude = config.exploration_factor * self.exploration_tendency
            return exploration_magnitude * random.gauss(0, 0.5)
        return 0.0
        
    def _calculate_social_learning_component(self, dimension: int, config: SwarmConfig) -> float:
        """Calculate social learning component from neighboring agents."""
        if not self.neighbors or not self.social_information:
            return 0.0
            
        # Weighted average of neighbor positions
        weighted_position = 0.0
        total_weight = 0.0
        
        for neighbor_id in self.neighbors:
            if neighbor_id in self.social_information:
                neighbor_info = self.social_information[neighbor_id]
                trust = self.trust_levels[neighbor_id]
                fitness_weight = max(0.1, neighbor_info.get('fitness', 0.1))
                
                weight = trust * fitness_weight
                weighted_position += weight * neighbor_info.get('position', [0] * self.dimension)[dimension]
                total_weight += weight
                
        if total_weight > 0:
            avg_neighbor_position = weighted_position / total_weight
            return config.social_learning_rate * (avg_neighbor_position - self.position[dimension])
            
        return 0.0
        
    def communicate_with_neighbor(self, neighbor: 'SwarmAgent', information: Dict[str, Any]):
        """Share information with a neighboring agent."""
        # Send information
        message = {
            'sender_id': self.agent_id,
            'timestamp': time.time(),
            'position': self.position.copy(),
            'fitness': self.current_fitness,
            'role': self.role,
            'specialization': self.specialization_score,
            'custom_info': information
        }
        
        neighbor.receive_communication(message)
        
        # Update trust based on information quality
        self._update_trust(neighbor.agent_id, information)
        
    def receive_communication(self, message: Dict[str, Any]):
        """Receive and process communication from another agent."""
        sender_id = message['sender_id']
        
        # Store social information
        self.social_information[sender_id] = {
            'position': message['position'],
            'fitness': message['fitness'],
            'role': message['role'],
            'specialization': message['specialization'],
            'timestamp': message['timestamp']
        }
        
        # Record communication
        self.communication_history.append(message)
        
        # Learn from high-performing agents
        if message['fitness'] > self.current_fitness:
            self._learn_from_superior_agent(message)
            
    def _update_trust(self, neighbor_id: str, information: Dict[str, Any]):
        """Update trust level for a neighbor based on information quality."""
        # Simple trust update based on information usefulness
        info_quality = information.get('quality_score', 0.5)
        current_trust = self.trust_levels[neighbor_id]
        
        # Gradual trust update
        alpha = 0.1
        new_trust = alpha * info_quality + (1 - alpha) * current_trust
        self.trust_levels[neighbor_id] = max(0.0, min(1.0, new_trust))
        
    def _learn_from_superior_agent(self, message: Dict[str, Any]):
        """Learn from agents with superior performance."""
        sender_fitness = message['fitness']
        fitness_advantage = sender_fitness - self.current_fitness
        
        if fitness_advantage > 0.1:  # Significant advantage
            # Adjust position slightly toward superior agent
            for i in range(self.dimension):
                adjustment = self.learning_rate * fitness_advantage * (message['position'][i] - self.position[i])
                self.position[i] += adjustment * 0.1  # Small adjustment
                
            # Learn behavioral traits
            if message['role'] != self.role and random.random() < 0.1:
                self.role = message['role']
                
    def adapt_role(self, swarm_context: Dict[str, Any]):
        """Adapt agent role based on swarm context and personal performance."""
        performance_ratio = self.current_fitness / max(swarm_context.get('avg_fitness', 1.0), 0.1)
        neighbor_count = len(self.neighbors)
        
        # Role adaptation logic
        if performance_ratio > 1.5 and self.cooperation_level > 0.7:
            self.role = "leader"
        elif neighbor_count > 5 and self.cooperation_level > 0.6:
            self.role = "communicator"
        elif performance_ratio > 1.2:
            self.role = "exploiter"
        else:
            self.role = "explorer"
            
        # Update specialization score
        role_consistency = sum(1 for msg in list(self.communication_history)[-10:] 
                             if msg.get('role') == self.role) / 10.0
        self.specialization_score = 0.9 * self.specialization_score + 0.1 * role_consistency
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent."""
        return {
            'agent_id': self.agent_id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'fitness': self.current_fitness,
            'role': self.role,
            'neighbors': len(self.neighbors),
            'trust_network_size': len(self.trust_levels),
            'specialization_score': self.specialization_score,
            'cooperation_level': self.cooperation_level,
            'age': self.age
        }

class CollectiveIntelligence:
    """Manages collective intelligence emergence in the swarm."""
    
    def __init__(self):
        self.collective_memory = deque(maxlen=1000)
        self.emergent_patterns = {}
        self.group_decisions = deque(maxlen=100)
        self.consensus_history = deque(maxlen=50)
        
    def detect_emergent_behavior(self, swarm_agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Detect emergent behaviors in the swarm."""
        emergent_behaviors = {}
        
        # Clustering behavior
        clustering_score = self._detect_clustering(swarm_agents)
        emergent_behaviors['clustering'] = clustering_score
        
        # Leadership emergence
        leadership_structure = self._detect_leadership(swarm_agents)
        emergent_behaviors['leadership'] = leadership_structure
        
        # Specialization emergence
        specialization_patterns = self._detect_specialization(swarm_agents)
        emergent_behaviors['specialization'] = specialization_patterns
        
        # Communication network structure
        network_structure = self._analyze_communication_network(swarm_agents)
        emergent_behaviors['network_structure'] = network_structure
        
        return emergent_behaviors
        
    def _detect_clustering(self, agents: List[SwarmAgent]) -> float:
        """Detect clustering behavior in agent positions."""
        if len(agents) < 3:
            return 0.0
            
        positions = [agent.position for agent in agents]
        
        # Calculate average pairwise distances
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])) ** 0.5
                total_distance += distance
                pair_count += 1
                
        avg_distance = total_distance / pair_count
        
        # Calculate clustering metric (lower average distance = higher clustering)
        max_possible_distance = 20.0 * (len(positions[0]) ** 0.5)  # Approximate max distance
        clustering_score = 1.0 - (avg_distance / max_possible_distance)
        
        return max(0.0, min(1.0, clustering_score))
        
    def _detect_leadership(self, agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Detect leadership structure in the swarm."""
        leaders = [agent for agent in agents if agent.role == "leader"]
        communicators = [agent for agent in agents if agent.role == "communicator"]
        
        leadership_ratio = len(leaders) / len(agents)
        communication_ratio = len(communicators) / len(agents)
        
        # Analyze influence networks
        influence_scores = {}
        for agent in agents:
            # Count how many agents trust this agent highly
            high_trust_count = sum(1 for other in agents 
                                 if other.trust_levels.get(agent.agent_id, 0.5) > 0.7)
            influence_scores[agent.agent_id] = high_trust_count
            
        return {
            'leadership_ratio': leadership_ratio,
            'communication_ratio': communication_ratio,
            'influence_distribution': influence_scores,
            'hierarchy_strength': max(influence_scores.values()) / len(agents) if influence_scores else 0.0
        }
        
    def _detect_specialization(self, agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Detect role specialization patterns."""
        role_distribution = defaultdict(int)
        specialization_scores = []
        
        for agent in agents:
            role_distribution[agent.role] += 1
            specialization_scores.append(agent.specialization_score)
            
        avg_specialization = sum(specialization_scores) / len(specialization_scores)
        role_diversity = len(role_distribution) / 4.0  # 4 possible roles
        
        return {
            'role_distribution': dict(role_distribution),
            'average_specialization': avg_specialization,
            'role_diversity': role_diversity,
            'specialization_variance': np.var(specialization_scores)
        }
        
    def _analyze_communication_network(self, agents: List[SwarmAgent]) -> Dict[str, Any]:
        """Analyze the structure of the communication network."""
        total_connections = sum(len(agent.neighbors) for agent in agents)
        avg_connections = total_connections / len(agents) if agents else 0.0
        
        # Calculate network density
        max_possible_connections = len(agents) * (len(agents) - 1)
        network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0.0
        
        # Identify highly connected nodes (hubs)
        connection_counts = [len(agent.neighbors) for agent in agents]
        avg_connections_per_agent = sum(connection_counts) / len(connection_counts) if connection_counts else 0.0
        hub_threshold = avg_connections_per_agent * 1.5
        hub_count = sum(1 for count in connection_counts if count > hub_threshold)
        
        return {
            'average_connections': avg_connections,
            'network_density': network_density,
            'hub_count': hub_count,
            'connectivity_distribution': connection_counts
        }
        
    def make_collective_decision(self, agents: List[SwarmAgent], 
                               decision_options: List[Any]) -> Dict[str, Any]:
        """Make a collective decision using swarm intelligence."""
        if not agents or not decision_options:
            return {'decision': None, 'confidence': 0.0}
            
        # Weight votes by agent trust and performance
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        
        for agent in agents:
            # Agent's vote weight based on fitness and trust network
            base_weight = max(0.1, agent.current_fitness)
            trust_bonus = sum(agent.trust_levels.values()) / len(agent.trust_levels) if agent.trust_levels else 0.5
            final_weight = base_weight * (1.0 + trust_bonus)
            
            # Agent votes for an option (simplified random choice for demo)
            vote = random.choice(decision_options)
            weighted_votes[vote] += final_weight
            total_weight += final_weight
            
        # Normalize votes
        if total_weight > 0:
            for option in weighted_votes:
                weighted_votes[option] /= total_weight
                
        # Select option with highest weighted vote
        best_option = max(weighted_votes.items(), key=lambda x: x[1])
        decision = best_option[0]
        confidence = best_option[1]
        
        # Record decision
        decision_record = {
            'decision': decision,
            'confidence': confidence,
            'vote_distribution': dict(weighted_votes),
            'participating_agents': len(agents),
            'timestamp': time.time()
        }
        
        self.group_decisions.append(decision_record)
        
        return decision_record

class SwarmIntelligence:
    """Main swarm intelligence system coordinating collective behavior."""
    
    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.agents: List[SwarmAgent] = []
        self.collective_intelligence = CollectiveIntelligence()
        
        # Optimization state
        self.global_best_position: Optional[List[float]] = None
        self.global_best_fitness = float('-inf')
        self.iteration = 0
        
        # Performance tracking
        self.fitness_history = deque(maxlen=1000)
        self.diversity_history = deque(maxlen=1000)
        self.emergence_history = deque(maxlen=1000)
        
        # Communication network
        self.communication_network = {}
        
    def initialize_swarm(self, dimension: int = 10):
        """Initialize the swarm with agents."""
        self.agents = []
        
        for i in range(self.config.swarm_size):
            agent = SwarmAgent(f"agent_{i}", dimension=dimension)
            self.agents.append(agent)
            
        # Initialize communication network
        self._initialize_communication_network()
        
    def _initialize_communication_network(self):
        """Initialize communication network between agents."""
        for i, agent in enumerate(self.agents):
            # Each agent connects to nearby agents
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    distance = self._calculate_distance(agent.position, other_agent.position)
                    if distance < self.config.communication_radius * 10:  # Scale factor
                        agent.neighbors.append(other_agent.agent_id)
                        
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5
        
    async def optimize(self, fitness_function: Callable, max_iterations: int = None) -> Dict[str, Any]:
        """Run swarm optimization algorithm."""
        max_iterations = max_iterations or self.config.max_iterations
        
        optimization_history = []
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # Evaluate fitness for all agents
            await self._evaluate_fitness(fitness_function)
            
            # Update global best
            self._update_global_best()
            
            # Update agent positions
            self._update_agent_positions()
            
            # Facilitate communication
            await self._facilitate_communication()
            
            # Adapt agent roles
            self._adapt_agent_roles()
            
            # Update communication network
            if iteration % 10 == 0:
                self._update_communication_network()
                
            # Analyze emergent behaviors
            if iteration % 20 == 0:
                emergent_behaviors = self.collective_intelligence.detect_emergent_behavior(self.agents)
                self.emergence_history.append(emergent_behaviors)
                
            # Record iteration statistics
            iteration_stats = self._record_iteration_stats()
            optimization_history.append(iteration_stats)
            
            # Check convergence
            if self._check_convergence():
                break
                
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'iterations_completed': self.iteration,
            'optimization_history': optimization_history,
            'emergent_behaviors': list(self.emergence_history)[-5:],  # Last 5 emergence analyses
            'final_swarm_state': [agent.get_state() for agent in self.agents]
        }
        
    async def _evaluate_fitness(self, fitness_function: Callable):
        """Evaluate fitness for all agents."""
        evaluation_tasks = []
        
        for agent in self.agents:
            task = asyncio.create_task(self._evaluate_agent_fitness(agent, fitness_function))
            evaluation_tasks.append(task)
            
        await asyncio.gather(*evaluation_tasks)
        
    async def _evaluate_agent_fitness(self, agent: SwarmAgent, fitness_function: Callable):
        """Evaluate fitness for a single agent."""
        try:
            fitness = await fitness_function(agent.position)
            agent.current_fitness = fitness
            
            # Update personal best
            if fitness > agent.personal_best_fitness:
                agent.personal_best_fitness = fitness
                agent.personal_best_position = agent.position.copy()
                
            # Store in experience
            experience = {
                'position': agent.position.copy(),
                'fitness': fitness,
                'timestamp': time.time(),
                'iteration': self.iteration
            }
            agent.experience_buffer.append(experience)
            
        except Exception as e:
            agent.current_fitness = float('-inf')
            
    def _update_global_best(self):
        """Update global best position and fitness."""
        for agent in self.agents:
            if agent.current_fitness > self.global_best_fitness:
                self.global_best_fitness = agent.current_fitness
                self.global_best_position = agent.position.copy()
                
    def _update_agent_positions(self):
        """Update positions for all agents."""
        if self.global_best_position is None:
            return
            
        for agent in self.agents:
            agent.update_position(self.global_best_position, self.config)
            agent.age += 1
            
    async def _facilitate_communication(self):
        """Facilitate communication between neighboring agents."""
        communication_tasks = []
        
        for agent in self.agents:
            if agent.neighbors and random.random() < 0.3:  # 30% chance to communicate
                task = asyncio.create_task(self._agent_communication_round(agent))
                communication_tasks.append(task)
                
        await asyncio.gather(*communication_tasks)
        
    async def _agent_communication_round(self, agent: SwarmAgent):
        """Perform one round of communication for an agent."""
        # Select neighbors to communicate with
        active_neighbors = random.sample(
            agent.neighbors, 
            min(3, len(agent.neighbors))  # Communicate with up to 3 neighbors
        )
        
        for neighbor_id in active_neighbors:
            neighbor = next((a for a in self.agents if a.agent_id == neighbor_id), None)
            if neighbor:
                # Share information
                info_to_share = {
                    'quality_score': random.uniform(0.3, 1.0),  # Simplified quality metric
                    'discovery': {
                        'position': agent.personal_best_position.copy(),
                        'fitness': agent.personal_best_fitness
                    }
                }
                
                agent.communicate_with_neighbor(neighbor, info_to_share)
                
    def _adapt_agent_roles(self):
        """Adapt roles for all agents based on swarm context."""
        # Calculate swarm statistics
        fitness_values = [agent.current_fitness for agent in self.agents if agent.current_fitness != float('-inf')]
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
        
        swarm_context = {
            'avg_fitness': avg_fitness,
            'best_fitness': self.global_best_fitness,
            'iteration': self.iteration,
            'swarm_size': len(self.agents)
        }
        
        for agent in self.agents:
            agent.adapt_role(swarm_context)
            
    def _update_communication_network(self):
        """Update the communication network based on current positions."""
        for agent in self.agents:
            # Clear current neighbors
            agent.neighbors = []
            
            # Find new neighbors within communication radius
            for other_agent in self.agents:
                if agent.agent_id != other_agent.agent_id:
                    distance = self._calculate_distance(agent.position, other_agent.position)
                    if distance < self.config.communication_radius * 10:
                        agent.neighbors.append(other_agent.agent_id)
                        
    def _record_iteration_stats(self) -> Dict[str, Any]:
        """Record statistics for current iteration."""
        fitness_values = [agent.current_fitness for agent in self.agents if agent.current_fitness != float('-inf')]
        
        if not fitness_values:
            return {'iteration': self.iteration}
            
        stats = {
            'iteration': self.iteration,
            'best_fitness': self.global_best_fitness,
            'average_fitness': sum(fitness_values) / len(fitness_values),
            'fitness_std': np.std(fitness_values),
            'diversity': self._calculate_population_diversity(),
            'role_distribution': self._get_role_distribution(),
            'network_connectivity': self._calculate_network_connectivity()
        }
        
        self.fitness_history.append(stats['best_fitness'])
        self.diversity_history.append(stats['diversity'])
        
        return stats
        
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of agent positions."""
        if len(self.agents) < 2:
            return 0.0
            
        positions = [agent.position for agent in self.agents]
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = self._calculate_distance(positions[i], positions[j])
                total_distance += distance
                pair_count += 1
                
        avg_distance = total_distance / pair_count
        return avg_distance
        
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of agent roles."""
        role_counts = defaultdict(int)
        for agent in self.agents:
            role_counts[agent.role] += 1
        return dict(role_counts)
        
    def _calculate_network_connectivity(self) -> float:
        """Calculate connectivity of communication network."""
        total_connections = sum(len(agent.neighbors) for agent in self.agents)
        max_connections = len(self.agents) * (len(self.agents) - 1)
        return total_connections / max_connections if max_connections > 0 else 0.0
        
    def _check_convergence(self) -> bool:
        """Check if the swarm has converged."""
        if len(self.fitness_history) < 50:
            return False
            
        recent_fitness = list(self.fitness_history)[-20:]
        fitness_variance = np.var(recent_fitness)
        
        return fitness_variance < 1e-6
        
    def get_swarm_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of swarm state."""
        return {
            'swarm_size': len(self.agents),
            'iterations_completed': self.iteration,
            'global_best_fitness': self.global_best_fitness,
            'current_diversity': self._calculate_population_diversity(),
            'role_distribution': self._get_role_distribution(),
            'network_connectivity': self._calculate_network_connectivity(),
            'convergence_status': self._check_convergence(),
            'recent_fitness_history': list(self.fitness_history)[-20:],
            'emergent_behavior_count': len(self.emergence_history)
        }