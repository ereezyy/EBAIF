"""
Consensus Module

This module implements the distributed consensus mechanism for validating
and propagating emergent behaviors across the agent network. It ensures
that only beneficial behaviors are shared while maintaining system stability.

Key Components:
- ConsensusEngine: Core consensus coordination and decision making
- BehaviorValidator: Individual behavior validation and scoring
- ConsensusProtocol: Communication protocol for distributed consensus
- ReputationSystem: Trust and reputation management for validators
"""

from .engine import ConsensusEngine
from .validator import BehaviorValidator
from .protocol import ConsensusProtocol
from .reputation import ReputationSystem

__all__ = [
    "ConsensusEngine",
    "BehaviorValidator", 
    "ConsensusProtocol",
    "ReputationSystem",
]

