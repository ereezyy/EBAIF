"""
Consensus Engine Implementation

The ConsensusEngine coordinates distributed validation of emergent behaviors
across the agent network. It implements a novel consensus protocol that
balances behavior innovation with system stability.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging
from collections import defaultdict, deque

from ..behavior_genome.genome import BehaviorGenome

class ConsensusState(Enum):
    """States of the consensus process."""
    IDLE = "idle"
    PROPOSING = "proposing"
    VALIDATING = "validating"
    VOTING = "voting"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"

class BehaviorProposal:
    """Represents a behavior proposal for consensus validation."""
    
    def __init__(self, 
                 proposal_id: str,
                 genome: BehaviorGenome,
                 proposer_id: str,
                 performance_metrics: Dict[str, float],
                 context: Dict[str, Any]):
        self.proposal_id = proposal_id
        self.genome = genome
        self.proposer_id = proposer_id
        self.performance_metrics = performance_metrics
        self.context = context
        self.timestamp = time.time()
        self.votes: Dict[str, float] = {}  # validator_id -> vote_score
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.consensus_score = 0.0
        self.state = ConsensusState.PROPOSING

@dataclass
class ConsensusConfig:
    """Configuration for the consensus engine."""
    validation_threshold: float = 0.8
    min_validators: int = 3
    max_validators: int = 10
    consensus_timeout: float = 30.0
    vote_weight_decay: float = 0.95
    reputation_weight: float = 0.3
    performance_weight: float = 0.4
    novelty_weight: float = 0.3
    max_concurrent_proposals: int = 5
    behavior_similarity_threshold: float = 0.85
    propagation_delay: float = 1.0

class ConsensusEngine:
    """
    Core consensus engine that coordinates distributed validation of
    emergent behaviors across the agent network.
    """
    
    def __init__(self, 
                 config: Optional[ConsensusConfig] = None,
                 validator_factory: Optional[Callable] = None):
        """
        Initialize the consensus engine.
        
        Args:
            config: Consensus configuration parameters
            validator_factory: Factory function for creating validators
        """
        self.config = config or ConsensusConfig()
        self.validator_factory = validator_factory
        
        # Core state
        self.engine_id = f"consensus_{int(time.time() * 1000)}"
        self.active_proposals: Dict[str, BehaviorProposal] = {}
        self.completed_proposals: deque = deque(maxlen=1000)
        self.validators: Dict[str, Any] = {}  # validator_id -> validator_instance
        self.validator_reputations: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Network state
        self.connected_nodes: Set[str] = set()
        self.network_topology: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.consensus_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_proposals': 0,
            'successful_consensus': 0,
            'failed_consensus': 0,
            'average_consensus_time': 0.0,
            'behavior_adoption_rate': 0.0,
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Async components
        self._running = False
        self._consensus_tasks: Dict[str, asyncio.Task] = {}
        
        self.logger = logging.getLogger(f"ConsensusEngine.{self.engine_id}")
        
    async def start(self):
        """Start the consensus engine."""
        self._running = True
        self.logger.info(f"Consensus engine {self.engine_id} started")
        
        # Start background tasks
        asyncio.create_task(self._cleanup_expired_proposals())
        asyncio.create_task(self._update_performance_metrics())
        
    async def stop(self):
        """Stop the consensus engine."""
        self._running = False
        
        # Cancel all active consensus tasks
        for task in self._consensus_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._consensus_tasks:
            await asyncio.gather(*self._consensus_tasks.values(), return_exceptions=True)
        
        self.logger.info(f"Consensus engine {self.engine_id} stopped")
        
    def register_validator(self, validator_id: str, validator_instance: Any):
        """Register a behavior validator with the consensus engine."""
        self.validators[validator_id] = validator_instance
        if validator_id not in self.validator_reputations:
            self.validator_reputations[validator_id] = 0.5  # Initial neutral reputation
        self.logger.info(f"Registered validator {validator_id}")
        
    def unregister_validator(self, validator_id: str):
        """Unregister a behavior validator."""
        if validator_id in self.validators:
            del self.validators[validator_id]
            self.logger.info(f"Unregistered validator {validator_id}")
            
    async def propose_behavior(self, 
                             genome: BehaviorGenome,
                             proposer_id: str,
                             performance_metrics: Dict[str, float],
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Propose a new behavior for consensus validation.
        
        Args:
            genome: The behavior genome to validate
            proposer_id: ID of the agent proposing the behavior
            performance_metrics: Performance data supporting the proposal
            context: Additional context information
            
        Returns:
            Proposal ID for tracking the consensus process
        """
        # Check if we can accept new proposals
        if len(self.active_proposals) >= self.config.max_concurrent_proposals:
            raise RuntimeError("Maximum concurrent proposals reached")
            
        # Generate proposal ID
        proposal_id = f"proposal_{proposer_id}_{int(time.time() * 1000)}"
        
        # Check for similar existing behaviors
        if await self._is_behavior_duplicate(genome):
            self.logger.info(f"Behavior proposal {proposal_id} rejected: too similar to existing behavior")
            return proposal_id
            
        # Create proposal
        proposal = BehaviorProposal(
            proposal_id=proposal_id,
            genome=genome,
            proposer_id=proposer_id,
            performance_metrics=performance_metrics,
            context=context or {}
        )
        
        self.active_proposals[proposal_id] = proposal
        self.performance_metrics['total_proposals'] += 1
        
        # Start consensus process
        consensus_task = asyncio.create_task(self._run_consensus(proposal))
        self._consensus_tasks[proposal_id] = consensus_task
        
        self.logger.info(f"Started consensus for proposal {proposal_id}")
        await self._emit_event('proposal_created', proposal)
        
        return proposal_id
        
    async def _run_consensus(self, proposal: BehaviorProposal):
        """Run the complete consensus process for a proposal."""
        try:
            proposal.state = ConsensusState.VALIDATING
            await self._emit_event('consensus_started', proposal)
            
            # Phase 1: Distributed validation
            validation_results = await self._conduct_validation(proposal)
            proposal.validation_results = validation_results
            
            # Phase 2: Voting based on validation results
            proposal.state = ConsensusState.VOTING
            votes = await self._conduct_voting(proposal)
            proposal.votes = votes
            
            # Phase 3: Calculate consensus score
            proposal.state = ConsensusState.FINALIZING
            consensus_score = self._calculate_consensus_score(proposal)
            proposal.consensus_score = consensus_score
            
            # Phase 4: Finalize decision
            if consensus_score >= self.config.validation_threshold:
                proposal.state = ConsensusState.COMPLETED
                await self._finalize_successful_consensus(proposal)
                self.performance_metrics['successful_consensus'] += 1
            else:
                proposal.state = ConsensusState.FAILED
                await self._finalize_failed_consensus(proposal)
                self.performance_metrics['failed_consensus'] += 1
                
        except asyncio.CancelledError:
            proposal.state = ConsensusState.FAILED
            self.logger.info(f"Consensus for proposal {proposal.proposal_id} was cancelled")
        except Exception as e:
            proposal.state = ConsensusState.FAILED
            self.logger.error(f"Consensus failed for proposal {proposal.proposal_id}: {e}")
        finally:
            # Clean up
            if proposal.proposal_id in self.active_proposals:
                self.completed_proposals.append(self.active_proposals.pop(proposal.proposal_id))
            if proposal.proposal_id in self._consensus_tasks:
                del self._consensus_tasks[proposal.proposal_id]
                
            await self._emit_event('consensus_completed', proposal)
            
    async def _conduct_validation(self, proposal: BehaviorProposal) -> Dict[str, Dict[str, Any]]:
        """Conduct distributed validation of the proposed behavior."""
        # Select validators for this proposal
        selected_validators = self._select_validators(proposal)
        
        # Run validation in parallel
        validation_tasks = []
        for validator_id in selected_validators:
            validator = self.validators[validator_id]
            task = asyncio.create_task(
                self._run_validator(validator_id, validator, proposal)
            )
            validation_tasks.append(task)
            
        # Wait for validation results with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*validation_tasks, return_exceptions=True),
                timeout=self.config.consensus_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Validation timeout for proposal {proposal.proposal_id}")
            results = [None] * len(validation_tasks)
            
        # Process validation results
        validation_results = {}
        for i, (validator_id, result) in enumerate(zip(selected_validators, results)):
            if isinstance(result, Exception):
                self.logger.error(f"Validator {validator_id} failed: {result}")
                validation_results[validator_id] = {'error': str(result), 'score': 0.0}
            elif result is not None:
                validation_results[validator_id] = result
            else:
                validation_results[validator_id] = {'timeout': True, 'score': 0.0}
                
        return validation_results
        
    async def _run_validator(self, validator_id: str, validator: Any, proposal: BehaviorProposal) -> Dict[str, Any]:
        """Run a single validator on the proposal."""
        try:
            # Call the validator's validate method
            if hasattr(validator, 'validate_behavior'):
                result = await validator.validate_behavior(
                    proposal.genome,
                    proposal.performance_metrics,
                    proposal.context
                )
            else:
                # Fallback for simple validators
                result = {'score': 0.5, 'details': 'Basic validation'}
                
            return result
        except Exception as e:
            self.logger.error(f"Validator {validator_id} error: {e}")
            raise
            
    def _select_validators(self, proposal: BehaviorProposal) -> List[str]:
        """Select validators for a proposal based on reputation and availability."""
        available_validators = list(self.validators.keys())
        
        if len(available_validators) <= self.config.min_validators:
            return available_validators
            
        # Sort by reputation (higher is better)
        validators_by_reputation = sorted(
            available_validators,
            key=lambda v_id: self.validator_reputations[v_id],
            reverse=True
        )
        
        # Select top validators up to max_validators
        num_validators = min(len(validators_by_reputation), self.config.max_validators)
        num_validators = max(num_validators, self.config.min_validators)
        
        return validators_by_reputation[:num_validators]
        
    async def _conduct_voting(self, proposal: BehaviorProposal) -> Dict[str, float]:
        """Conduct voting based on validation results."""
        votes = {}
        
        for validator_id, validation_result in proposal.validation_results.items():
            if 'error' in validation_result or 'timeout' in validation_result:
                votes[validator_id] = 0.0
                continue
                
            # Extract validation score
            base_score = validation_result.get('score', 0.0)
            
            # Weight by validator reputation
            reputation = self.validator_reputations[validator_id]
            weighted_score = base_score * (1.0 + self.config.reputation_weight * (reputation - 0.5))
            
            # Clamp to valid range
            votes[validator_id] = max(0.0, min(1.0, weighted_score))
            
        return votes
        
    def _calculate_consensus_score(self, proposal: BehaviorProposal) -> float:
        """Calculate the final consensus score for a proposal."""
        if not proposal.votes:
            return 0.0
            
        # Weighted average of votes
        total_weight = 0.0
        weighted_sum = 0.0
        
        for validator_id, vote in proposal.votes.items():
            reputation = self.validator_reputations[validator_id]
            weight = 1.0 + self.config.reputation_weight * (reputation - 0.5)
            
            weighted_sum += vote * weight
            total_weight += weight
            
        if total_weight == 0.0:
            return 0.0
            
        consensus_score = weighted_sum / total_weight
        
        # Apply performance bonus
        performance_bonus = self._calculate_performance_bonus(proposal)
        consensus_score += performance_bonus * self.config.performance_weight
        
        # Apply novelty bonus
        novelty_bonus = self._calculate_novelty_bonus(proposal)
        consensus_score += novelty_bonus * self.config.novelty_weight
        
        return max(0.0, min(1.0, consensus_score))
        
    def _calculate_performance_bonus(self, proposal: BehaviorProposal) -> float:
        """Calculate performance bonus based on metrics."""
        metrics = proposal.performance_metrics
        
        # Define performance criteria
        performance_score = 0.0
        criteria_count = 0
        
        # Fitness improvement
        if 'fitness_improvement' in metrics:
            performance_score += max(0.0, min(1.0, metrics['fitness_improvement']))
            criteria_count += 1
            
        # Efficiency metrics
        if 'efficiency' in metrics:
            performance_score += max(0.0, min(1.0, metrics['efficiency']))
            criteria_count += 1
            
        # Stability metrics
        if 'stability' in metrics:
            performance_score += max(0.0, min(1.0, metrics['stability']))
            criteria_count += 1
            
        return performance_score / max(1, criteria_count)
        
    def _calculate_novelty_bonus(self, proposal: BehaviorProposal) -> float:
        """Calculate novelty bonus for unique behaviors."""
        # Compare with recent successful behaviors
        novelty_score = 1.0  # Start with maximum novelty
        
        for completed_proposal in list(self.completed_proposals)[-50:]:  # Check last 50
            if completed_proposal.state == ConsensusState.COMPLETED:
                similarity = self._calculate_behavior_similarity(
                    proposal.genome, 
                    completed_proposal.genome
                )
                novelty_score = min(novelty_score, 1.0 - similarity)
                
        return max(0.0, novelty_score)
        
    def _calculate_behavior_similarity(self, genome1: BehaviorGenome, genome2: BehaviorGenome) -> float:
        """Calculate similarity between two behavior genomes."""
        # Compare behavior genes
        behavior_similarity = 0.0
        gene_count = 0
        
        for key in genome1.behavior_genes.keys():
            if key in genome2.behavior_genes:
                gene1 = genome1.behavior_genes[key]
                gene2 = genome2.behavior_genes[key]
                
                # Calculate normalized difference
                diff = torch.abs(gene1 - gene2).mean().item()
                similarity = 1.0 - diff
                behavior_similarity += similarity
                gene_count += 1
                
        if gene_count == 0:
            return 0.0
            
        return behavior_similarity / gene_count
        
    async def _is_behavior_duplicate(self, genome: BehaviorGenome) -> bool:
        """Check if a behavior is too similar to existing ones."""
        # Check against active proposals
        for proposal in self.active_proposals.values():
            similarity = self._calculate_behavior_similarity(genome, proposal.genome)
            if similarity > self.config.behavior_similarity_threshold:
                return True
                
        # Check against recent successful behaviors
        for completed_proposal in list(self.completed_proposals)[-20:]:
            if completed_proposal.state == ConsensusState.COMPLETED:
                similarity = self._calculate_behavior_similarity(genome, completed_proposal.genome)
                if similarity > self.config.behavior_similarity_threshold:
                    return True
                    
        return False
        
    async def _finalize_successful_consensus(self, proposal: BehaviorProposal):
        """Finalize a successful consensus and propagate the behavior."""
        self.logger.info(f"Consensus successful for proposal {proposal.proposal_id} "
                        f"(score: {proposal.consensus_score:.3f})")
        
        # Update validator reputations (positive feedback)
        for validator_id, vote in proposal.votes.items():
            if vote > 0.5:  # Validators who voted positively
                self.validator_reputations[validator_id] = min(
                    1.0, 
                    self.validator_reputations[validator_id] + 0.01
                )
                
        # Record consensus history
        self.consensus_history.append({
            'proposal_id': proposal.proposal_id,
            'timestamp': proposal.timestamp,
            'consensus_score': proposal.consensus_score,
            'success': True,
            'validator_count': len(proposal.votes),
        })
        
        # Emit success event
        await self._emit_event('behavior_accepted', proposal)
        
        # Propagate behavior to network (with delay)
        await asyncio.sleep(self.config.propagation_delay)
        await self._propagate_behavior(proposal)
        
    async def _finalize_failed_consensus(self, proposal: BehaviorProposal):
        """Finalize a failed consensus."""
        self.logger.info(f"Consensus failed for proposal {proposal.proposal_id} "
                        f"(score: {proposal.consensus_score:.3f})")
        
        # Update validator reputations (negative feedback for outliers)
        avg_vote = np.mean(list(proposal.votes.values())) if proposal.votes else 0.0
        for validator_id, vote in proposal.votes.items():
            # Penalize validators who voted very differently from consensus
            if abs(vote - avg_vote) > 0.3:
                self.validator_reputations[validator_id] = max(
                    0.0,
                    self.validator_reputations[validator_id] - 0.005
                )
                
        # Record consensus history
        self.consensus_history.append({
            'proposal_id': proposal.proposal_id,
            'timestamp': proposal.timestamp,
            'consensus_score': proposal.consensus_score,
            'success': False,
            'validator_count': len(proposal.votes),
        })
        
        # Emit failure event
        await self._emit_event('behavior_rejected', proposal)
        
    async def _propagate_behavior(self, proposal: BehaviorProposal):
        """Propagate accepted behavior to the network."""
        # This would integrate with the network layer to share the behavior
        # For now, just emit an event
        await self._emit_event('behavior_propagated', proposal)
        self.logger.info(f"Behavior {proposal.proposal_id} propagated to network")
        
    async def _cleanup_expired_proposals(self):
        """Background task to clean up expired proposals."""
        while self._running:
            current_time = time.time()
            expired_proposals = []
            
            for proposal_id, proposal in self.active_proposals.items():
                if current_time - proposal.timestamp > self.config.consensus_timeout * 2:
                    expired_proposals.append(proposal_id)
                    
            for proposal_id in expired_proposals:
                proposal = self.active_proposals.pop(proposal_id, None)
                if proposal:
                    proposal.state = ConsensusState.FAILED
                    self.completed_proposals.append(proposal)
                    self.logger.warning(f"Proposal {proposal_id} expired")
                    
            await asyncio.sleep(10)  # Check every 10 seconds
            
    async def _update_performance_metrics(self):
        """Background task to update performance metrics."""
        while self._running:
            if self.consensus_history:
                # Calculate average consensus time
                recent_history = self.consensus_history[-100:]  # Last 100 consensus
                total_time = sum(
                    h.get('duration', 0) for h in recent_history
                )
                self.performance_metrics['average_consensus_time'] = total_time / len(recent_history)
                
                # Calculate success rate
                successful = sum(1 for h in recent_history if h['success'])
                self.performance_metrics['behavior_adoption_rate'] = successful / len(recent_history)
                
            await asyncio.sleep(60)  # Update every minute
            
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler for consensus events."""
        self.event_handlers[event_type].append(handler)
        
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove an event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            
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
        """Get current consensus engine status."""
        return {
            'engine_id': self.engine_id,
            'running': self._running,
            'active_proposals': len(self.active_proposals),
            'registered_validators': len(self.validators),
            'performance_metrics': self.performance_metrics.copy(),
            'validator_reputations': dict(self.validator_reputations),
        }
        
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific proposal."""
        # Check active proposals
        if proposal_id in self.active_proposals:
            proposal = self.active_proposals[proposal_id]
            return {
                'proposal_id': proposal.proposal_id,
                'state': proposal.state.value,
                'consensus_score': proposal.consensus_score,
                'votes': proposal.votes,
                'timestamp': proposal.timestamp,
            }
            
        # Check completed proposals
        for proposal in self.completed_proposals:
            if proposal.proposal_id == proposal_id:
                return {
                    'proposal_id': proposal.proposal_id,
                    'state': proposal.state.value,
                    'consensus_score': proposal.consensus_score,
                    'votes': proposal.votes,
                    'timestamp': proposal.timestamp,
                }
                
        return None

