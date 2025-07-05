"""
Agent Manager Implementation

Manages multiple emergent agents in EBAIF.
"""

import asyncio
from typing import Dict, List, Optional, Any
from .agent import EmergentAgent, AgentConfig
from ..utils import Logger

class AgentManager:
    """Manages multiple emergent agents."""
    
    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.agents: Dict[str, EmergentAgent] = {}
        self.logger = Logger.get_logger("AgentManager")
        self._running = False
        
    async def start(self):
        """Start the agent manager."""
        self._running = True
        self.logger.info("Agent manager started")
        
    async def stop(self):
        """Stop the agent manager."""
        self._running = False
        
        # Shutdown all agents
        for agent in self.agents.values():
            await agent.shutdown()
            
        self.logger.info("Agent manager stopped")
        
    def add_agent(self, agent: EmergentAgent):
        """Add an agent to the manager."""
        if len(self.agents) >= self.max_agents:
            raise ValueError("Maximum number of agents reached")
            
        self.agents[agent.agent_id] = agent
        
        # Connect agents for peer learning
        for other_agent in self.agents.values():
            if other_agent.agent_id != agent.agent_id:
                agent.add_peer_agent(other_agent)
                other_agent.add_peer_agent(agent)
                
        self.logger.info(f"Added agent {agent.agent_id}")
        
    def remove_agent(self, agent_id: str):
        """Remove an agent from the manager."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            
            # Remove from peer networks
            for other_agent in self.agents.values():
                other_agent.remove_peer_agent(agent_id)
                
            self.logger.info(f"Removed agent {agent_id}")
            
    def get_agent_statuses(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            agent_id: agent.get_status() 
            for agent_id, agent in self.agents.items()
        }