"""
Emergent Agent Implementation

The EmergentAgent class represents an individual AI agent with evolving
behavior capabilities. It integrates with the behavior genome system
and consensus engine to continuously adapt and improve.
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging
from collections import deque, defaultdict

from ..behavior_genome.genome import BehaviorGenome, GenomeConfig
from ..consensus.engine import ConsensusEngine

class AgentState(Enum):
    """States of an emergent agent."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    LEARNING = "learning"
    ACTING = "acting"
    EVOLVING = "evolving"
    COMMUNICATING = "communicating"
    SHUTDOWN = "shutdown"

class AgentRole(Enum):
    """Roles that an agent can take in the system."""
    EXPLORER = "explorer"
    OPTIMIZER = "optimizer"
    COMMUNICATOR = "communicator"
    LEADER = "leader"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"

@dataclass
class AgentConfig:
    """Configuration for emergent agents."""
    agent_id: str = ""
    initial_genome_config: Optional[GenomeConfig] = None
    learning_rate: float = 0.001
    evolution_frequency: int = 100  # Steps between evolution attempts
    communication_frequency: int = 50  # Steps between communications
    memory_size: int = 10000
    experience_buffer_size: int = 1000
    fitness_window_size: int = 100
    adaptation_threshold: float = 0.1
    social_learning_enabled: bool = True
    self_evolution_enabled: bool = True
    max_genome_age: int = 1000  # Max steps before forced evolution
    role_adaptation_enabled: bool = True

class Experience:
    """Represents an agent's experience for learning."""
    
    def __init__(self, state: torch.Tensor, action: torch.Tensor, 
                 reward: float, next_state: torch.Tensor, done: bool,
                 context: Dict[str, Any] = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.context = context or {}
        self.timestamp = time.time()

class EmergentAgent:
    """
    An individual AI agent with evolving behavior capabilities.
    Integrates genome evolution, consensus participation, and social learning.
    """
    
    def __init__(self, 
                 config: AgentConfig,
                 consensus_engine: Optional[ConsensusEngine] = None,
                 initial_genome: Optional[BehaviorGenome] = None):
        """
        Initialize an emergent agent.
        
        Args:
            config: Agent configuration
            consensus_engine: Consensus engine for behavior validation
            initial_genome: Starting behavior genome (optional)
        """
        self.config = config
        self.agent_id = config.agent_id or f"agent_{int(time.time() * 1000)}"
        self.consensus_engine = consensus_engine
        
        # Core agent state
        self.state = AgentState.INITIALIZING
        self.role = AgentRole.GENERALIST
        self.step_count = 0
        self.last_evolution_step = 0
        self.last_communication_step = 0
        
        # Behavior genome and network
        self.genome = initial_genome or BehaviorGenome(
            config.initial_genome_config or GenomeConfig()
        )
        self.network = None
        self.optimizer = None
        
        # Learning and memory
        self.experience_buffer = deque(maxlen=config.experience_buffer_size)
        self.memory = {}
        self.fitness_history = deque(maxlen=config.fitness_window_size)
        self.current_fitness = 0.0
        
        # Social learning
        self.peer_agents: Dict[str, 'EmergentAgent'] = {}
        self.communication_history = deque(maxlen=100)
        self.learned_behaviors: List[BehaviorGenome] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_reward': 0.0,
            'average_reward': 0.0,
            'learning_progress': 0.0,
            'evolution_count': 0,
            'successful_communications': 0,
            'role_changes': 0,
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        self.logger = logging.getLogger(f"EmergentAgent.{self.agent_id}")
        
    async def initialize(self):
        """Initialize the agent and its neural network."""
        self.state = AgentState.INITIALIZING
        
        # Build neural network from genome
        self.network = self.genome.build_network()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Determine initial role based on genome
        self.role = self._determine_role_from_genome()
        
        self.state = AgentState.IDLE
        self.logger.info(f"Agent {self.agent_id} initialized with role {self.role.value}")
        
        await self._emit_event('agent_initialized', {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'genome_id': self.genome.genome_id
        })
        
    async def step(self, environment_state: torch.Tensor, 
                   available_actions: List[int]) -> Tuple[int, Dict[str, Any]]:
        """
        Execute one step of agent behavior.
        
        Args:
            environment_state: Current state of the environment
            available_actions: List of available action indices
            
        Returns:
            Tuple of (selected_action, step_info)
        """
        self.step_count += 1
        step_info = {}
        
        # Update state
        self.state = AgentState.ACTING
        
        # Get action from network
        with torch.no_grad():
            if environment_state.dim() == 1:
                environment_state = environment_state.unsqueeze(0)
            
            action_logits = self.network(environment_state)
            
            # Apply behavior genome parameters
            action_probs = self._apply_behavioral_modulation(action_logits, available_actions)
            
            # Select action based on exploration/exploitation balance
            selected_action = self._select_action(action_probs, available_actions)
            
        step_info.update({
            'action_probs': action_probs.tolist(),
            'selected_action': selected_action,
            'genome_id': self.genome.genome_id,
            'role': self.role.value,
            'step_count': self.step_count
        })
        
        # Check for periodic activities
        await self._check_periodic_activities()
        
        self.state = AgentState.IDLE
        return selected_action, step_info
        
    def _apply_behavioral_modulation(self, action_logits: torch.Tensor, 
                                   available_actions: List[int]) -> torch.Tensor:
        """Apply behavior genome parameters to action selection."""
        # Get behavior parameters
        behavior_params = self.genome.get_behavior_parameters()
        
        # Apply exploration modulation
        exploration_rate = behavior_params.get('exploration_rate', 0.1)
        if exploration_rate > 0:
            noise = torch.randn_like(action_logits) * exploration_rate
            action_logits = action_logits + noise
            
        # Apply risk tolerance modulation
        risk_tolerance = behavior_params.get('risk_tolerance', 0.5)
        if risk_tolerance < 0.5:
            # Conservative behavior - prefer safer actions (assume lower indices are safer)
            safety_bias = torch.linspace(0.1, -0.1, action_logits.size(-1))
            action_logits = action_logits + safety_bias * (1.0 - risk_tolerance)
            
        # Apply curiosity factor
        curiosity = behavior_params.get('curiosity_factor', 0.5)
        if curiosity > 0.5 and hasattr(self, '_action_counts'):
            # Boost less-tried actions
            for i, count in enumerate(self._action_counts):
                if count < np.mean(self._action_counts):
                    action_logits[0, i] += curiosity * 0.1
                    
        # Convert to probabilities
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Mask unavailable actions
        if available_actions:
            mask = torch.zeros_like(action_probs)
            mask[0, available_actions] = 1.0
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            
        return action_probs
        
    def _select_action(self, action_probs: torch.Tensor, available_actions: List[int]) -> int:
        """Select an action based on probabilities and behavior parameters."""
        behavior_params = self.genome.get_behavior_parameters()
        exploration_rate = behavior_params.get('exploration_rate', 0.1)
        
        if torch.rand(1).item() < exploration_rate:
            # Random exploration
            return np.random.choice(available_actions) if available_actions else 0
        else:
            # Exploitation based on probabilities
            if available_actions:
                available_probs = action_probs[0, available_actions]
                selected_idx = torch.multinomial(available_probs, 1).item()
                return available_actions[selected_idx]
            else:
                return torch.argmax(action_probs).item()
                
    async def learn_from_experience(self, experience: Experience):
        """Learn from a single experience."""
        self.state = AgentState.LEARNING
        
        # Add to experience buffer
        self.experience_buffer.append(experience)
        
        # Update performance metrics
        self.performance_metrics['total_reward'] += experience.reward
        self.current_fitness += experience.reward
        
        # Perform learning if we have enough experiences
        if len(self.experience_buffer) >= 32:  # Minimum batch size
            await self._perform_batch_learning()
            
        # Update fitness history
        if self.step_count % 10 == 0:  # Update every 10 steps
            avg_fitness = self.current_fitness / 10
            self.fitness_history.append(avg_fitness)
            self.genome.update_fitness(avg_fitness)
            self.current_fitness = 0.0
            
        self.state = AgentState.IDLE
        
    async def _perform_batch_learning(self):
        """Perform batch learning from experience buffer."""
        if not self.experience_buffer:
            return
            
        # Sample batch from experience buffer
        batch_size = min(32, len(self.experience_buffer))
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch_experiences = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch tensors
        states = torch.stack([exp.state for exp in batch_experiences])
        actions = torch.stack([exp.action for exp in batch_experiences])
        rewards = torch.tensor([exp.reward for exp in batch_experiences], dtype=torch.float32)
        next_states = torch.stack([exp.next_state for exp in batch_experiences])
        dones = torch.tensor([exp.done for exp in batch_experiences], dtype=torch.bool)
        
        # Compute loss and update network
        self.optimizer.zero_grad()
        
        # Forward pass
        current_q_values = self.network(states)
        next_q_values = self.network(next_states)
        
        # Compute target values (simplified Q-learning)
        target_q_values = rewards + 0.99 * torch.max(next_q_values, dim=1)[0] * (~dones)
        
        # Compute loss (MSE between current and target Q-values)
        # For simplicity, we'll use the action with highest probability
        current_action_values = torch.max(current_q_values, dim=1)[0]
        loss = nn.MSELoss()(current_action_values, target_q_values.detach())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update learning progress
        self.performance_metrics['learning_progress'] = float(loss.item())
        
    async def _check_periodic_activities(self):
        """Check and perform periodic activities like evolution and communication."""
        # Check for evolution
        if (self.config.self_evolution_enabled and 
            self.step_count - self.last_evolution_step >= self.config.evolution_frequency):
            await self._attempt_evolution()
            
        # Check for communication
        if (self.config.social_learning_enabled and
            self.step_count - self.last_communication_step >= self.config.communication_frequency):
            await self._attempt_communication()
            
        # Check for role adaptation
        if self.config.role_adaptation_enabled and self.step_count % 500 == 0:
            await self._adapt_role()
            
    async def _attempt_evolution(self):
        """Attempt to evolve the agent's behavior genome."""
        self.state = AgentState.EVOLVING
        self.last_evolution_step = self.step_count
        
        # Check if evolution is warranted
        if not self._should_evolve():
            self.state = AgentState.IDLE
            return
            
        # Create mutated genome
        mutated_genome = self.genome.mutate()
        
        # Evaluate mutation (simplified - in practice, would test performance)
        mutation_fitness = await self._evaluate_genome_fitness(mutated_genome)
        current_fitness = np.mean(list(self.fitness_history)) if self.fitness_history else 0.0
        
        # Decide whether to adopt mutation
        if mutation_fitness > current_fitness + self.config.adaptation_threshold:
            # Propose to consensus engine if available
            if self.consensus_engine:
                try:
                    proposal_id = await self.consensus_engine.propose_behavior(
                        genome=mutated_genome,
                        proposer_id=self.agent_id,
                        performance_metrics={
                            'fitness_improvement': mutation_fitness - current_fitness,
                            'current_fitness': current_fitness,
                            'mutation_fitness': mutation_fitness,
                        },
                        context={'evolution_step': self.step_count}
                    )
                    self.logger.info(f"Proposed genome evolution: {proposal_id}")
                except Exception as e:
                    self.logger.error(f"Failed to propose evolution: {e}")
            else:
                # Direct adoption without consensus
                await self._adopt_genome(mutated_genome)
                
        self.performance_metrics['evolution_count'] += 1
        self.state = AgentState.IDLE
        
    def _should_evolve(self) -> bool:
        """Determine if the agent should attempt evolution."""
        # Evolve if performance is stagnating
        if len(self.fitness_history) >= 20:
            recent_fitness = list(self.fitness_history)[-10:]
            older_fitness = list(self.fitness_history)[-20:-10]
            
            if np.mean(recent_fitness) <= np.mean(older_fitness) + 0.01:
                return True
                
        # Evolve if genome is too old
        genome_age = self.step_count - self.last_evolution_step
        if genome_age >= self.config.max_genome_age:
            return True
            
        # Evolve based on curiosity factor
        curiosity = self.genome.behavior_genes.get('curiosity_factor', torch.tensor(0.5)).item()
        if torch.rand(1).item() < curiosity * 0.1:  # Curiosity-driven evolution
            return True
            
        return False
        
    async def _evaluate_genome_fitness(self, genome: BehaviorGenome) -> float:
        """Evaluate the fitness of a genome (simplified simulation)."""
        # In a real implementation, this would run the genome in a test environment
        # For now, we'll use a simplified heuristic based on genome parameters
        
        behavior_params = genome.get_behavior_parameters()
        
        # Calculate fitness based on balance of parameters
        exploration = behavior_params.get('exploration_rate', 0.5)
        cooperation = behavior_params.get('cooperation_tendency', 0.5)
        adaptation = behavior_params.get('adaptation_speed', 0.5)
        
        # Reward balanced behaviors
        balance_score = 1.0 - np.std([exploration, cooperation, adaptation])
        
        # Add some randomness to simulate environment interaction
        noise = np.random.normal(0, 0.1)
        
        return balance_score + noise
        
    async def _adopt_genome(self, new_genome: BehaviorGenome):
        """Adopt a new behavior genome."""
        old_genome_id = self.genome.genome_id
        self.genome = new_genome
        
        # Rebuild network with new genome
        self.network = self.genome.build_network()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Update role based on new genome
        new_role = self._determine_role_from_genome()
        if new_role != self.role:
            self.role = new_role
            self.performance_metrics['role_changes'] += 1
            
        self.logger.info(f"Adopted new genome {new_genome.genome_id} (was {old_genome_id})")
        
        await self._emit_event('genome_adopted', {
            'agent_id': self.agent_id,
            'old_genome_id': old_genome_id,
            'new_genome_id': new_genome.genome_id,
            'new_role': self.role.value
        })
        
    async def _attempt_communication(self):
        """Attempt to communicate with peer agents."""
        self.state = AgentState.COMMUNICATING
        self.last_communication_step = self.step_count
        
        if not self.peer_agents:
            self.state = AgentState.IDLE
            return
            
        # Select communication partners based on behavior parameters
        communication_freq = self.genome.behavior_genes.get('communication_frequency', torch.tensor(0.4)).item()
        cooperation = self.genome.behavior_genes.get('cooperation_tendency', torch.tensor(0.6)).item()
        
        # Determine number of agents to communicate with
        num_partners = max(1, int(len(self.peer_agents) * communication_freq))
        
        # Select partners (prefer cooperative agents)
        partners = self._select_communication_partners(num_partners)
        
        # Communicate with selected partners
        for partner_id in partners:
            if partner_id in self.peer_agents:
                await self._communicate_with_agent(self.peer_agents[partner_id])
                
        self.state = AgentState.IDLE
        
    def _select_communication_partners(self, num_partners: int) -> List[str]:
        """Select communication partners based on compatibility."""
        if not self.peer_agents:
            return []
            
        # Score partners based on compatibility
        partner_scores = {}
        for partner_id, partner in self.peer_agents.items():
            # Calculate compatibility based on behavior similarity
            compatibility = self._calculate_compatibility(partner)
            partner_scores[partner_id] = compatibility
            
        # Select top partners
        sorted_partners = sorted(partner_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [partner_id for partner_id, _ in sorted_partners[:num_partners]]
        
        return selected
        
    def _calculate_compatibility(self, other_agent: 'EmergentAgent') -> float:
        """Calculate compatibility with another agent."""
        # Compare behavior parameters
        my_params = self.genome.get_behavior_parameters()
        other_params = other_agent.genome.get_behavior_parameters()
        
        compatibility = 0.0
        param_count = 0
        
        for key in my_params.keys():
            if key in other_params:
                # Higher compatibility for similar cooperation and communication values
                if key in ['cooperation_tendency', 'communication_frequency']:
                    similarity = 1.0 - abs(my_params[key] - other_params[key])
                    compatibility += similarity * 2.0  # Weight these more heavily
                else:
                    similarity = 1.0 - abs(my_params[key] - other_params[key])
                    compatibility += similarity
                param_count += 1
                
        return compatibility / max(param_count, 1)
        
    async def _communicate_with_agent(self, other_agent: 'EmergentAgent'):
        """Communicate with another agent."""
        try:
            # Share behavior information
            message = {
                'sender_id': self.agent_id,
                'message_type': 'behavior_sharing',
                'genome_summary': {
                    'genome_id': self.genome.genome_id,
                    'fitness': np.mean(list(self.fitness_history)) if self.fitness_history else 0.0,
                    'behavior_params': self.genome.get_behavior_parameters(),
                },
                'performance_metrics': self.performance_metrics.copy(),
                'timestamp': time.time()
            }
            
            # Send message to other agent
            await other_agent.receive_message(message)
            
            self.performance_metrics['successful_communications'] += 1
            
        except Exception as e:
            self.logger.error(f"Communication failed with {other_agent.agent_id}: {e}")
            
    async def receive_message(self, message: Dict[str, Any]):
        """Receive and process a message from another agent."""
        self.communication_history.append(message)
        
        # Process different message types
        if message['message_type'] == 'behavior_sharing':
            await self._process_behavior_sharing(message)
        elif message['message_type'] == 'collaboration_request':
            await self._process_collaboration_request(message)
            
        await self._emit_event('message_received', message)
        
    async def _process_behavior_sharing(self, message: Dict[str, Any]):
        """Process behavior sharing message from another agent."""
        sender_fitness = message['genome_summary']['fitness']
        my_fitness = np.mean(list(self.fitness_history)) if self.fitness_history else 0.0
        
        # Learn from higher-performing agents
        if sender_fitness > my_fitness * 1.1:  # 10% improvement threshold
            # Consider adopting aspects of the shared behavior
            shared_params = message['genome_summary']['behavior_params']
            await self._learn_from_peer_behavior(shared_params, sender_fitness)
            
    async def _learn_from_peer_behavior(self, peer_params: Dict[str, float], peer_fitness: float):
        """Learn from peer agent's behavior parameters."""
        if not self.config.social_learning_enabled:
            return
            
        # Gradually adjust our parameters toward successful peer's parameters
        learning_rate = 0.1  # How much to adjust toward peer
        my_params = self.genome.get_behavior_parameters()
        
        # Create adjusted genome
        adjusted_genome = BehaviorGenome(self.genome.config)
        adjusted_genome.architecture_genes = {k: v.clone() for k, v in self.genome.architecture_genes.items()}
        adjusted_genome.behavior_genes = {k: v.clone() for k, v in self.genome.behavior_genes.items()}
        
        # Adjust behavior parameters
        for key, peer_value in peer_params.items():
            if key in my_params:
                my_value = my_params[key]
                # Move toward peer's value
                new_value = my_value + learning_rate * (peer_value - my_value)
                adjusted_genome.behavior_genes[key] = torch.tensor(new_value)
                
        # Test the adjusted genome
        adjusted_fitness = await self._evaluate_genome_fitness(adjusted_genome)
        current_fitness = np.mean(list(self.fitness_history)) if self.fitness_history else 0.0
        
        # Adopt if improvement is significant
        if adjusted_fitness > current_fitness + 0.05:
            await self._adopt_genome(adjusted_genome)
            self.logger.info(f"Learned from peer behavior, fitness improved to {adjusted_fitness:.3f}")
            
    async def _adapt_role(self):
        """Adapt agent role based on current behavior and performance."""
        new_role = self._determine_role_from_genome()
        
        if new_role != self.role:
            old_role = self.role
            self.role = new_role
            self.performance_metrics['role_changes'] += 1
            
            self.logger.info(f"Role changed from {old_role.value} to {new_role.value}")
            
            await self._emit_event('role_changed', {
                'agent_id': self.agent_id,
                'old_role': old_role.value,
                'new_role': new_role.value,
                'step_count': self.step_count
            })
            
    def _determine_role_from_genome(self) -> AgentRole:
        """Determine agent role based on behavior genome."""
        params = self.genome.get_behavior_parameters()
        
        exploration = params.get('exploration_rate', 0.5)
        cooperation = params.get('cooperation_tendency', 0.5)
        leadership = params.get('leadership_inclination', 0.2)
        communication = params.get('communication_frequency', 0.4)
        
        # Determine role based on parameter combinations
        if exploration > 0.7:
            return AgentRole.EXPLORER
        elif leadership > 0.6 and communication > 0.6:
            return AgentRole.LEADER
        elif communication > 0.7:
            return AgentRole.COMMUNICATOR
        elif cooperation > 0.8:
            return AgentRole.SPECIALIST
        else:
            return AgentRole.GENERALIST
            
    def add_peer_agent(self, agent: 'EmergentAgent'):
        """Add a peer agent for communication and learning."""
        self.peer_agents[agent.agent_id] = agent
        
    def remove_peer_agent(self, agent_id: str):
        """Remove a peer agent."""
        if agent_id in self.peer_agents:
            del self.peer_agents[agent_id]
            
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler."""
        self.event_handlers[event_type].append(handler)
        
    async def _emit_event(self, event_type: str, data: Any):
        """Emit an event to all registered handlers."""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                self.logger.error(f"Event handler error for {event_type}: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'role': self.role.value,
            'step_count': self.step_count,
            'genome_id': self.genome.genome_id,
            'current_fitness': np.mean(list(self.fitness_history)) if self.fitness_history else 0.0,
            'performance_metrics': self.performance_metrics.copy(),
            'peer_count': len(self.peer_agents),
            'experience_count': len(self.experience_buffer),
        }
        
    async def shutdown(self):
        """Shutdown the agent gracefully."""
        self.state = AgentState.SHUTDOWN
        
        # Save important data
        final_status = self.get_status()
        
        await self._emit_event('agent_shutdown', final_status)
        
        self.logger.info(f"Agent {self.agent_id} shutdown complete")

