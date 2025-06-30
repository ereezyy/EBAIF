"""
Agents Module

This module implements the emergent AI agent system that coordinates
individual agents, manages their evolution, and facilitates communication
and learning between agents.

Key Components:
- EmergentAgent: Individual AI agent with evolving behavior
- AgentManager: Coordinates multiple agents and their interactions
- AgentCommunication: Handles inter-agent communication protocols
- LearningCoordinator: Manages collective learning and knowledge sharing
"""

from .agent import EmergentAgent
from .manager import AgentManager
from .communication import AgentCommunication
from .learning import LearningCoordinator

__all__ = [
    "EmergentAgent",
    "AgentManager",
    "AgentCommunication", 
    "LearningCoordinator",
]

