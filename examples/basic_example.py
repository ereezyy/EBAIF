"""
Basic EBAIF Example

This example shows how to create a simple emergent agent and run it.
"""

import asyncio
import torch
import numpy as np
from src.ebaif import BehaviorGenome, ConsensusEngine, EmergentAgent
from src.ebaif.behavior_genome.genome import GenomeConfig
from src.ebaif.agents.agent import AgentConfig
from src.ebaif.consensus.engine import ConsensusConfig
from src.ebaif.consensus.validator import BehaviorValidator
from src.ebaif.utils import Logger, Config

async def main():
    """Run basic EBAIF example."""
    
    # Setup logging
    logger = Logger()
    log = logger.get_logger("BasicExample")
    log.info("Starting basic EBAIF example...")
    
    # Create configuration
    config = Config()
    
    # Create a simple behavior genome
    genome_config = GenomeConfig(
        input_dim=64,
        output_dim=10,
        hidden_dims=[128, 64],
        num_layers=3
    )
    
    genome = BehaviorGenome(genome_config)
    log.info(f"Created genome: {genome}")
    
    # Create consensus engine
    consensus_config = ConsensusConfig(
        validation_threshold=0.7,
        min_validators=1,  # Reduced for simple example
        max_validators=3
    )
    consensus_engine = ConsensusEngine(consensus_config)
    
    # Create a simple validator
    validator = BehaviorValidator("validator_1")
    consensus_engine.register_validator("validator_1", validator)
    
    # Start consensus engine
    await consensus_engine.start()
    log.info("Consensus engine started")
    
    # Create an emergent agent
    agent_config = AgentConfig(
        agent_id="agent_1",
        initial_genome_config=genome_config,
        learning_rate=0.001,
        evolution_frequency=50
    )
    
    agent = EmergentAgent(
        config=agent_config,
        consensus_engine=consensus_engine,
        initial_genome=genome
    )
    
    # Initialize the agent
    await agent.initialize()
    log.info(f"Agent initialized: {agent.agent_id}")
    
    # Simulate some agent steps
    log.info("Running agent simulation...")
    
    for step in range(10):
        # Create dummy environment state
        env_state = torch.randn(64)  # Random environment state
        available_actions = list(range(10))  # All actions available
        
        # Agent takes a step
        action, step_info = await agent.step(env_state, available_actions)
        
        # Simulate reward (random for this example)
        reward = np.random.random()
        
        # Create experience for learning
        from src.ebaif.agents.agent import Experience
        experience = Experience(
            state=env_state,
            action=torch.tensor([action]),
            reward=reward,
            next_state=torch.randn(64),  # Random next state
            done=False
        )
        
        # Agent learns from experience
        await agent.learn_from_experience(experience)
        
        log.info(f"Step {step}: Action={action}, Reward={reward:.3f}")
        
        # Brief pause between steps
        await asyncio.sleep(0.1)
    
    # Get agent status
    status = agent.get_status()
    log.info(f"Final agent status: {status}")
    
    # Test evolution
    log.info("Testing behavior evolution...")
    await agent._attempt_evolution()
    
    # Get consensus engine status
    consensus_status = consensus_engine.get_status()
    log.info(f"Consensus engine status: {consensus_status}")
    
    # Cleanup
    await consensus_engine.stop()
    await agent.shutdown()
    
    log.info("Example completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())