"""
Learning Coordinator Implementation

Coordinates learning between agents in EBAIF.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from ..behavior_genome.genome import BehaviorGenome
from ..utils import Logger

class LearningCoordinator:
    """Coordinates collective learning between agents."""
    
    def __init__(self):
        self.shared_knowledge: Dict[str, Any] = {}
        self.learning_sessions: List[Dict[str, Any]] = []
        self.logger = Logger.get_logger("LearningCoordinator")
        
    async def coordinate_learning(self, agents: List[Any]):
        """Coordinate learning between multiple agents."""
        if len(agents) < 2:
            return
            
        # Collect performance data from all agents
        performance_data = []
        for agent in agents:
            if hasattr(agent, 'fitness_history') and agent.fitness_history:
                avg_fitness = np.mean(list(agent.fitness_history))
                performance_data.append({
                    'agent_id': agent.agent_id,
                    'fitness': avg_fitness,
                    'genome': agent.genome
                })
                
        if not performance_data:
            return
            
        # Sort by performance
        performance_data.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Share knowledge from top performers
        top_performers = performance_data[:min(3, len(performance_data))]
        
        for performer in top_performers:
            await self._share_knowledge(performer, agents)
            
    async def _share_knowledge(self, performer: Dict[str, Any], 
                              agents: List[Any]):
        """Share knowledge from a high-performing agent."""
        performer_id = performer['agent_id']
        performer_genome = performer['genome']
        
        # Create knowledge package
        knowledge = {
            'source_agent': performer_id,
            'behavior_params': performer_genome.get_behavior_parameters(),
            'fitness': performer['fitness'],
            'knowledge_type': 'behavioral_pattern'
        }
        
        # Share with other agents
        for agent in agents:
            if agent.agent_id != performer_id:
                await self._transfer_knowledge(knowledge, agent)
                
    async def _transfer_knowledge(self, knowledge: Dict[str, Any], 
                                 target_agent: Any):
        """Transfer knowledge to a target agent."""
        try:
            # Simple knowledge transfer - adjust behavior parameters slightly
            source_params = knowledge['behavior_params']
            
            if hasattr(target_agent, 'genome') and target_agent.genome:
                current_params = target_agent.genome.get_behavior_parameters()
                
                # Blend parameters (10% influence from source)
                blend_factor = 0.1
                
                for key in current_params.keys():
                    if key in source_params:
                        current_value = current_params[key]
                        source_value = source_params[key]
                        new_value = current_value + blend_factor * (source_value - current_value)
                        
                        # Update the behavior gene
                        target_agent.genome.behavior_genes[key] = torch.tensor(new_value)
                        
        except Exception as e:
            self.logger.error(f"Knowledge transfer failed: {e}")
            
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning coordination statistics."""
        return {
            'total_sessions': len(self.learning_sessions),
            'shared_knowledge_items': len(self.shared_knowledge),
        }