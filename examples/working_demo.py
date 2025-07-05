"""
Working EBAIF Demo

Complete working demonstration of the EBAIF framework.
"""

import asyncio
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.ebaif import BehaviorGenome, ConsensusEngine, EmergentAgent
from src.ebaif.behavior_genome.genome import GenomeConfig
from src.ebaif.agents.agent import AgentConfig, Experience
from src.ebaif.agents.manager import AgentManager
from src.ebaif.consensus.engine import ConsensusConfig
from src.ebaif.consensus.validator import BehaviorValidator
from src.ebaif.utils import Logger, Config

class SimpleEnvironment:
    """Simple environment for testing agents."""
    
    def __init__(self, size: int = 10):
        self.size = size
        self.agent_positions = {}
        self.resources = torch.rand(size, size)
        self.step_count = 0
        
    def reset(self):
        """Reset the environment."""
        self.resources = torch.rand(self.size, self.size)
        self.agent_positions.clear()
        self.step_count = 0
        
    def add_agent(self, agent_id: str):
        """Add an agent to the environment."""
        x, y = np.random.randint(0, self.size, 2)
        self.agent_positions[agent_id] = (x, y)
        
    def get_state(self, agent_id: str) -> torch.Tensor:
        """Get environment state for an agent."""
        if agent_id not in self.agent_positions:
            return torch.zeros(self.size * self.size + 2)
            
        x, y = self.agent_positions[agent_id]
        
        # Flatten environment and add agent position
        state = torch.cat([
            self.resources.flatten(),
            torch.tensor([x / self.size, y / self.size])
        ])
        
        return state
        
    def step(self, agent_id: str, action: int) -> float:
        """Execute action and return reward."""
        if agent_id not in self.agent_positions:
            return 0.0
            
        x, y = self.agent_positions[agent_id]
        
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=collect
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.size - 1:
            x += 1
        elif action == 4:
            # Collect resource
            reward = self.resources[x, y].item()
            self.resources[x, y] = 0.0
            self.agent_positions[agent_id] = (x, y)
            return reward
            
        self.agent_positions[agent_id] = (x, y)
        return -0.01  # Small penalty for movement

async def run_demo():
    """Run the complete EBAIF demo."""
    
    # Setup
    logger = Logger()
    log = logger.get_logger("Demo")
    log.info("Starting EBAIF Demo...")
    
    # Create environment
    env = SimpleEnvironment(size=8)
    
    # Create consensus system
    consensus_config = ConsensusConfig(
        validation_threshold=0.6,
        min_validators=1,
        max_validators=2
    )
    consensus_engine = ConsensusEngine(consensus_config)
    
    # Add validators
    validator = BehaviorValidator("validator_1")
    consensus_engine.register_validator("validator_1", validator)
    
    await consensus_engine.start()
    
    # Create agent manager
    agent_manager = AgentManager(max_agents=3)
    await agent_manager.start()
    
    # Create agents
    agents = []
    for i in range(3):
        genome_config = GenomeConfig(
            input_dim=env.size * env.size + 2,  # Environment + position
            output_dim=5,  # 5 possible actions
            hidden_dims=[64, 32],
            num_layers=2
        )
        
        agent_config = AgentConfig(
            agent_id=f"agent_{i}",
            initial_genome_config=genome_config,
            learning_rate=0.01,
            evolution_frequency=20
        )
        
        agent = EmergentAgent(
            config=agent_config,
            consensus_engine=consensus_engine
        )
        
        await agent.initialize()
        agent_manager.add_agent(agent)
        env.add_agent(agent.agent_id)
        agents.append(agent)
        
    log.info(f"Created {len(agents)} agents")
    
    # Track performance
    performance_history = {agent.agent_id: [] for agent in agents}
    evolution_events = []
    
    # Run simulation
    num_episodes = 10
    steps_per_episode = 50
    
    for episode in range(num_episodes):
        log.info(f"Episode {episode + 1}/{num_episodes}")
        env.reset()
        
        # Re-add agents to environment
        for agent in agents:
            env.add_agent(agent.agent_id)
            
        episode_rewards = {agent.agent_id: 0.0 for agent in agents}
        
        for step in range(steps_per_episode):
            for agent in agents:
                # Get current state
                state = env.get_state(agent.agent_id)
                available_actions = list(range(5))  # 0-4
                
                # Agent decides action
                action, step_info = await agent.step(state, available_actions)
                
                # Execute action in environment
                reward = env.step(agent.agent_id, action)
                episode_rewards[agent.agent_id] += reward
                
                # Get next state for learning
                next_state = env.get_state(agent.agent_id)
                
                # Create experience
                experience = Experience(
                    state=state,
                    action=torch.tensor([action]),
                    reward=reward,
                    next_state=next_state,
                    done=(step == steps_per_episode - 1)
                )
                
                # Agent learns
                await agent.learn_from_experience(experience)
                
        # Record episode performance
        for agent in agents:
            performance_history[agent.agent_id].append(episode_rewards[agent.agent_id])
            
        # Print episode summary
        avg_rewards = {aid: np.mean(performance_history[aid][-5:]) 
                      for aid in performance_history.keys()}
        log.info(f"Episode rewards: {episode_rewards}")
        log.info(f"Recent avg rewards: {avg_rewards}")
        
        # Brief pause between episodes
        await asyncio.sleep(0.1)
    
    # Final statistics
    log.info("\n" + "="*50)
    log.info("FINAL RESULTS")
    log.info("="*50)
    
    for agent in agents:
        status = agent.get_status()
        avg_reward = np.mean(performance_history[agent.agent_id])
        log.info(f"{agent.agent_id}:")
        log.info(f"  Average Reward: {avg_reward:.3f}")
        log.info(f"  Evolution Count: {status['performance_metrics']['evolution_count']}")
        log.info(f"  Role: {status['role']}")
        
        # Show behavior parameters
        behavior_params = agent.genome.get_behavior_parameters()
        log.info(f"  Behavior: {behavior_params}")
    
    # Consensus engine stats
    consensus_status = consensus_engine.get_status()
    log.info(f"\nConsensus Engine:")
    log.info(f"  Successful Consensus: {consensus_status['performance_metrics']['successful_consensus']}")
    log.info(f"  Failed Consensus: {consensus_status['performance_metrics']['failed_consensus']}")
    
    # Cleanup
    await agent_manager.stop()
    await consensus_engine.stop()
    
    log.info("Demo completed successfully!")
    
    return performance_history

if __name__ == "__main__":
    # Run the demo
    performance_data = asyncio.run(run_demo())
    
    # Create simple performance plot if matplotlib is available
    try:
        plt.figure(figsize=(10, 6))
        for agent_id, rewards in performance_data.items():
            plt.plot(rewards, label=agent_id, marker='o')
        
        plt.title('Agent Performance Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('agent_performance.png')
        print("Performance plot saved as 'agent_performance.png'")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")