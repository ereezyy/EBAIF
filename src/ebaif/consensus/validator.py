"""
Behavior Validator Implementation

The BehaviorValidator class implements individual validation logic for
evaluating proposed behaviors. It analyzes performance, safety, and
compatibility aspects of behavior genomes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import time

from ..behavior_genome.genome import BehaviorGenome

class ValidationCriterion(Enum):
    """Different criteria for behavior validation."""
    PERFORMANCE = "performance"
    SAFETY = "safety"
    NOVELTY = "novelty"
    COMPATIBILITY = "compatibility"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"

@dataclass
class ValidationConfig:
    """Configuration for behavior validation."""
    enabled_criteria: List[ValidationCriterion] = None
    performance_weight: float = 0.3
    safety_weight: float = 0.25
    novelty_weight: float = 0.2
    compatibility_weight: float = 0.15
    efficiency_weight: float = 0.05
    stability_weight: float = 0.05
    min_performance_threshold: float = 0.1
    max_validation_time: float = 10.0
    safety_checks_enabled: bool = True
    compatibility_checks_enabled: bool = True

    def __post_init__(self):
        if self.enabled_criteria is None:
            self.enabled_criteria = list(ValidationCriterion)

class ValidationResult:
    """Result of behavior validation."""
    
    def __init__(self, validator_id: str, genome_id: str):
        self.validator_id = validator_id
        self.genome_id = genome_id
        self.timestamp = time.time()
        self.overall_score = 0.0
        self.criterion_scores: Dict[ValidationCriterion, float] = {}
        self.details: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.validation_time = 0.0
        
    def add_criterion_score(self, criterion: ValidationCriterion, score: float, details: Dict[str, Any] = None):
        """Add a score for a specific validation criterion."""
        self.criterion_scores[criterion] = max(0.0, min(1.0, score))
        if details:
            self.details[criterion.value] = details
            
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)
        
    def add_error(self, message: str):
        """Add a validation error."""
        self.errors.append(message)
        
    def calculate_overall_score(self, config: ValidationConfig) -> float:
        """Calculate the overall validation score based on criterion weights."""
        if not self.criterion_scores:
            return 0.0
            
        weighted_sum = 0.0
        total_weight = 0.0
        
        weight_map = {
            ValidationCriterion.PERFORMANCE: config.performance_weight,
            ValidationCriterion.SAFETY: config.safety_weight,
            ValidationCriterion.NOVELTY: config.novelty_weight,
            ValidationCriterion.COMPATIBILITY: config.compatibility_weight,
            ValidationCriterion.EFFICIENCY: config.efficiency_weight,
            ValidationCriterion.STABILITY: config.stability_weight,
        }
        
        for criterion, score in self.criterion_scores.items():
            weight = weight_map.get(criterion, 0.0)
            weighted_sum += score * weight
            total_weight += weight
            
        self.overall_score = weighted_sum / max(total_weight, 1e-8)
        return self.overall_score
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            'validator_id': self.validator_id,
            'genome_id': self.genome_id,
            'timestamp': self.timestamp,
            'overall_score': self.overall_score,
            'criterion_scores': {c.value: s for c, s in self.criterion_scores.items()},
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors,
            'validation_time': self.validation_time,
        }

class BehaviorValidator:
    """
    Individual behavior validator that evaluates proposed behaviors
    against multiple criteria including performance, safety, and compatibility.
    """
    
    def __init__(self, 
                 validator_id: str,
                 config: Optional[ValidationConfig] = None,
                 reference_behaviors: Optional[List[BehaviorGenome]] = None):
        """
        Initialize the behavior validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Validation configuration
            reference_behaviors: Reference behaviors for comparison
        """
        self.validator_id = validator_id
        self.config = config or ValidationConfig()
        self.reference_behaviors = reference_behaviors or []
        
        # Performance tracking
        self.validation_count = 0
        self.average_validation_time = 0.0
        self.validation_history: List[ValidationResult] = []
        
        # Learned patterns for validation
        self.performance_patterns: Dict[str, torch.Tensor] = {}
        self.safety_patterns: Dict[str, torch.Tensor] = {}
        
        self.logger = logging.getLogger(f"BehaviorValidator.{validator_id}")
        
    async def validate_behavior(self, 
                              genome: BehaviorGenome,
                              performance_metrics: Dict[str, float],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a behavior genome against all configured criteria.
        
        Args:
            genome: The behavior genome to validate
            performance_metrics: Performance data for the behavior
            context: Additional context information
            
        Returns:
            Validation result dictionary
        """
        start_time = time.time()
        result = ValidationResult(self.validator_id, genome.genome_id)
        
        try:
            # Run validation criteria in parallel where possible
            validation_tasks = []
            
            for criterion in self.config.enabled_criteria:
                task = asyncio.create_task(
                    self._validate_criterion(criterion, genome, performance_metrics, context, result)
                )
                validation_tasks.append(task)
                
            # Wait for all validations to complete
            await asyncio.wait_for(
                asyncio.gather(*validation_tasks, return_exceptions=True),
                timeout=self.config.max_validation_time
            )
            
            # Calculate overall score
            overall_score = result.calculate_overall_score(self.config)
            
            # Apply minimum threshold check
            if overall_score < self.config.min_performance_threshold:
                result.add_warning(f"Overall score {overall_score:.3f} below minimum threshold")
                
        except asyncio.TimeoutError:
            result.add_error("Validation timeout exceeded")
            result.overall_score = 0.0
        except Exception as e:
            result.add_error(f"Validation failed: {str(e)}")
            result.overall_score = 0.0
            self.logger.error(f"Validation error for genome {genome.genome_id}: {e}")
            
        # Record timing and update statistics
        result.validation_time = time.time() - start_time
        self._update_statistics(result)
        
        return result.to_dict()
        
    async def _validate_criterion(self, 
                                criterion: ValidationCriterion,
                                genome: BehaviorGenome,
                                performance_metrics: Dict[str, float],
                                context: Dict[str, Any],
                                result: ValidationResult):
        """Validate a specific criterion."""
        try:
            if criterion == ValidationCriterion.PERFORMANCE:
                await self._validate_performance(genome, performance_metrics, context, result)
            elif criterion == ValidationCriterion.SAFETY:
                await self._validate_safety(genome, performance_metrics, context, result)
            elif criterion == ValidationCriterion.NOVELTY:
                await self._validate_novelty(genome, performance_metrics, context, result)
            elif criterion == ValidationCriterion.COMPATIBILITY:
                await self._validate_compatibility(genome, performance_metrics, context, result)
            elif criterion == ValidationCriterion.EFFICIENCY:
                await self._validate_efficiency(genome, performance_metrics, context, result)
            elif criterion == ValidationCriterion.STABILITY:
                await self._validate_stability(genome, performance_metrics, context, result)
        except Exception as e:
            result.add_error(f"Failed to validate {criterion.value}: {str(e)}")
            result.add_criterion_score(criterion, 0.0)
            
    async def _validate_performance(self, 
                                  genome: BehaviorGenome,
                                  performance_metrics: Dict[str, float],
                                  context: Dict[str, Any],
                                  result: ValidationResult):
        """Validate performance aspects of the behavior."""
        score = 0.0
        details = {}
        
        # Analyze fitness improvement
        if 'fitness_improvement' in performance_metrics:
            fitness_improvement = performance_metrics['fitness_improvement']
            fitness_score = min(1.0, max(0.0, fitness_improvement))
            score += fitness_score * 0.4
            details['fitness_improvement'] = fitness_improvement
            
        # Analyze task completion rate
        if 'task_completion_rate' in performance_metrics:
            completion_rate = performance_metrics['task_completion_rate']
            completion_score = min(1.0, max(0.0, completion_rate))
            score += completion_score * 0.3
            details['task_completion_rate'] = completion_rate
            
        # Analyze learning speed
        if 'learning_speed' in performance_metrics:
            learning_speed = performance_metrics['learning_speed']
            learning_score = min(1.0, max(0.0, learning_speed))
            score += learning_score * 0.2
            details['learning_speed'] = learning_speed
            
        # Analyze behavior consistency
        consistency_score = self._analyze_behavior_consistency(genome)
        score += consistency_score * 0.1
        details['consistency'] = consistency_score
        
        # Normalize score
        score = min(1.0, max(0.0, score))
        
        result.add_criterion_score(ValidationCriterion.PERFORMANCE, score, details)
        
    async def _validate_safety(self, 
                             genome: BehaviorGenome,
                             performance_metrics: Dict[str, float],
                             context: Dict[str, Any],
                             result: ValidationResult):
        """Validate safety aspects of the behavior."""
        if not self.config.safety_checks_enabled:
            result.add_criterion_score(ValidationCriterion.SAFETY, 1.0)
            return
            
        score = 1.0  # Start with perfect safety score
        details = {}
        
        # Check for aggressive behavior patterns
        aggression_level = genome.behavior_genes.get('aggression_level', torch.tensor(0.0)).item()
        if aggression_level > 0.8:
            score -= 0.3
            result.add_warning(f"High aggression level detected: {aggression_level:.3f}")
        details['aggression_level'] = aggression_level
        
        # Check for risky exploration patterns
        risk_tolerance = genome.behavior_genes.get('risk_tolerance', torch.tensor(0.0)).item()
        if risk_tolerance > 0.9:
            score -= 0.2
            result.add_warning(f"Very high risk tolerance: {risk_tolerance:.3f}")
        details['risk_tolerance'] = risk_tolerance
        
        # Check for unstable learning patterns
        if 'error_rate' in performance_metrics:
            error_rate = performance_metrics['error_rate']
            if error_rate > 0.5:
                score -= 0.3
                result.add_warning(f"High error rate: {error_rate:.3f}")
            details['error_rate'] = error_rate
            
        # Check neural architecture stability
        architecture_stability = self._check_architecture_stability(genome)
        if architecture_stability < 0.5:
            score -= 0.2
            result.add_warning(f"Unstable neural architecture: {architecture_stability:.3f}")
        details['architecture_stability'] = architecture_stability
        
        score = min(1.0, max(0.0, score))
        result.add_criterion_score(ValidationCriterion.SAFETY, score, details)
        
    async def _validate_novelty(self, 
                              genome: BehaviorGenome,
                              performance_metrics: Dict[str, float],
                              context: Dict[str, Any],
                              result: ValidationResult):
        """Validate novelty aspects of the behavior."""
        score = 1.0  # Start with maximum novelty
        details = {}
        
        # Compare with reference behaviors
        if self.reference_behaviors:
            max_similarity = 0.0
            for ref_genome in self.reference_behaviors:
                similarity = self._calculate_genome_similarity(genome, ref_genome)
                max_similarity = max(max_similarity, similarity)
                
            novelty_score = 1.0 - max_similarity
            score = novelty_score
            details['max_similarity'] = max_similarity
            details['novelty_score'] = novelty_score
            
        # Analyze behavioral diversity
        diversity_score = self._analyze_behavioral_diversity(genome)
        score = (score + diversity_score) / 2.0
        details['diversity_score'] = diversity_score
        
        # Check for innovative patterns
        innovation_score = self._detect_innovative_patterns(genome)
        score = (score * 0.7) + (innovation_score * 0.3)
        details['innovation_score'] = innovation_score
        
        result.add_criterion_score(ValidationCriterion.NOVELTY, score, details)
        
    async def _validate_compatibility(self, 
                                    genome: BehaviorGenome,
                                    performance_metrics: Dict[str, float],
                                    context: Dict[str, Any],
                                    result: ValidationResult):
        """Validate compatibility with existing systems."""
        if not self.config.compatibility_checks_enabled:
            result.add_criterion_score(ValidationCriterion.COMPATIBILITY, 1.0)
            return
            
        score = 1.0
        details = {}
        
        # Check cooperation tendency
        cooperation = genome.behavior_genes.get('cooperation_tendency', torch.tensor(0.5)).item()
        if cooperation < 0.2:
            score -= 0.3
            result.add_warning(f"Low cooperation tendency: {cooperation:.3f}")
        details['cooperation_tendency'] = cooperation
        
        # Check communication frequency
        communication = genome.behavior_genes.get('communication_frequency', torch.tensor(0.5)).item()
        if communication < 0.1:
            score -= 0.2
            result.add_warning(f"Very low communication frequency: {communication:.3f}")
        details['communication_frequency'] = communication
        
        # Check resource sharing behavior
        resource_sharing = genome.behavior_genes.get('resource_sharing', torch.tensor(0.5)).item()
        if resource_sharing < 0.1:
            score -= 0.2
            result.add_warning(f"Low resource sharing tendency: {resource_sharing:.3f}")
        details['resource_sharing'] = resource_sharing
        
        # Check for disruptive patterns
        disruption_score = self._detect_disruptive_patterns(genome)
        score -= disruption_score * 0.3
        details['disruption_score'] = disruption_score
        
        score = min(1.0, max(0.0, score))
        result.add_criterion_score(ValidationCriterion.COMPATIBILITY, score, details)
        
    async def _validate_efficiency(self, 
                                 genome: BehaviorGenome,
                                 performance_metrics: Dict[str, float],
                                 context: Dict[str, Any],
                                 result: ValidationResult):
        """Validate computational efficiency of the behavior."""
        score = 0.5  # Default efficiency score
        details = {}
        
        # Analyze network complexity
        network_complexity = self._calculate_network_complexity(genome)
        if network_complexity < 0.5:
            score += 0.3  # Reward simpler networks
        elif network_complexity > 0.8:
            score -= 0.2  # Penalize overly complex networks
        details['network_complexity'] = network_complexity
        
        # Check memory usage patterns
        memory_retention = genome.behavior_genes.get('memory_retention', torch.tensor(0.5)).item()
        if memory_retention > 0.9:
            score -= 0.1  # High memory usage penalty
        details['memory_retention'] = memory_retention
        
        # Analyze computational requirements
        if 'computation_time' in performance_metrics:
            comp_time = performance_metrics['computation_time']
            if comp_time < 0.1:
                score += 0.2  # Reward fast computation
            elif comp_time > 0.5:
                score -= 0.2  # Penalize slow computation
            details['computation_time'] = comp_time
            
        score = min(1.0, max(0.0, score))
        result.add_criterion_score(ValidationCriterion.EFFICIENCY, score, details)
        
    async def _validate_stability(self, 
                                genome: BehaviorGenome,
                                performance_metrics: Dict[str, float],
                                context: Dict[str, Any],
                                result: ValidationResult):
        """Validate stability and robustness of the behavior."""
        score = 0.5
        details = {}
        
        # Check performance variance
        if 'performance_variance' in performance_metrics:
            variance = performance_metrics['performance_variance']
            stability_score = 1.0 - min(1.0, variance)
            score = (score + stability_score) / 2.0
            details['performance_variance'] = variance
            
        # Analyze adaptation speed (too fast can be unstable)
        adaptation_speed = genome.behavior_genes.get('adaptation_speed', torch.tensor(0.5)).item()
        if adaptation_speed > 0.8:
            score -= 0.2  # Very fast adaptation can be unstable
        elif adaptation_speed < 0.2:
            score -= 0.1  # Very slow adaptation is also problematic
        details['adaptation_speed'] = adaptation_speed
        
        # Check behavioral consistency over time
        if genome.performance_history:
            consistency = self._calculate_performance_consistency(genome.performance_history)
            score = (score + consistency) / 2.0
            details['performance_consistency'] = consistency
            
        score = min(1.0, max(0.0, score))
        result.add_criterion_score(ValidationCriterion.STABILITY, score, details)
        
    def _analyze_behavior_consistency(self, genome: BehaviorGenome) -> float:
        """Analyze consistency of behavior parameters."""
        # Check for extreme values that might indicate inconsistency
        consistency_score = 1.0
        
        for key, gene in genome.behavior_genes.items():
            value = gene.item()
            # Penalize extreme values (too close to 0 or 1)
            if value < 0.05 or value > 0.95:
                consistency_score -= 0.1
                
        return max(0.0, consistency_score)
        
    def _check_architecture_stability(self, genome: BehaviorGenome) -> float:
        """Check stability of neural architecture."""
        # Analyze architecture parameters for stability indicators
        stability_score = 1.0
        
        # Check dropout rate (too high can be unstable)
        dropout_rate = genome.architecture_genes.get('dropout_rate', torch.tensor(0.1)).item()
        if dropout_rate > 0.5:
            stability_score -= 0.3
            
        # Check learning rate (too high can be unstable)
        learning_rate = genome.architecture_genes.get('learning_rate', torch.tensor(0.001)).item()
        if learning_rate > 0.1:
            stability_score -= 0.2
            
        return max(0.0, stability_score)
        
    def _calculate_genome_similarity(self, genome1: BehaviorGenome, genome2: BehaviorGenome) -> float:
        """Calculate similarity between two genomes."""
        # Compare behavior genes
        behavior_similarity = 0.0
        gene_count = 0
        
        for key in genome1.behavior_genes.keys():
            if key in genome2.behavior_genes:
                gene1 = genome1.behavior_genes[key]
                gene2 = genome2.behavior_genes[key]
                
                diff = torch.abs(gene1 - gene2).mean().item()
                similarity = 1.0 - diff
                behavior_similarity += similarity
                gene_count += 1
                
        if gene_count == 0:
            return 0.0
            
        return behavior_similarity / gene_count
        
    def _analyze_behavioral_diversity(self, genome: BehaviorGenome) -> float:
        """Analyze diversity within the behavior genome."""
        # Calculate variance in behavior parameters
        behavior_values = [gene.item() for gene in genome.behavior_genes.values()]
        if len(behavior_values) < 2:
            return 0.5
            
        variance = np.var(behavior_values)
        # Normalize variance to [0, 1] range
        diversity_score = min(1.0, variance * 4.0)  # Scale factor for normalization
        
        return diversity_score
        
    def _detect_innovative_patterns(self, genome: BehaviorGenome) -> float:
        """Detect innovative patterns in the behavior genome."""
        # Look for unusual combinations of behavior parameters
        innovation_score = 0.0
        
        # High exploration with high cooperation (unusual combination)
        exploration = genome.behavior_genes.get('exploration_rate', torch.tensor(0.5)).item()
        cooperation = genome.behavior_genes.get('cooperation_tendency', torch.tensor(0.5)).item()
        if exploration > 0.7 and cooperation > 0.7:
            innovation_score += 0.3
            
        # High curiosity with high defensive behavior (interesting combination)
        curiosity = genome.behavior_genes.get('curiosity_factor', torch.tensor(0.5)).item()
        defensive = genome.behavior_genes.get('defensive_behavior', torch.tensor(0.5)).item()
        if curiosity > 0.7 and defensive > 0.7:
            innovation_score += 0.2
            
        # Balanced risk tolerance with high pattern recognition
        risk_tolerance = genome.behavior_genes.get('risk_tolerance', torch.tensor(0.5)).item()
        pattern_recognition = genome.behavior_genes.get('pattern_recognition', torch.tensor(0.5)).item()
        if 0.4 <= risk_tolerance <= 0.6 and pattern_recognition > 0.8:
            innovation_score += 0.2
            
        return min(1.0, innovation_score)
        
    def _detect_disruptive_patterns(self, genome: BehaviorGenome) -> float:
        """Detect potentially disruptive behavior patterns."""
        disruption_score = 0.0
        
        # Very high aggression with low cooperation
        aggression = genome.behavior_genes.get('aggression_level', torch.tensor(0.3)).item()
        cooperation = genome.behavior_genes.get('cooperation_tendency', torch.tensor(0.6)).item()
        if aggression > 0.8 and cooperation < 0.2:
            disruption_score += 0.5
            
        # High leadership with very low communication
        leadership = genome.behavior_genes.get('leadership_inclination', torch.tensor(0.2)).item()
        communication = genome.behavior_genes.get('communication_frequency', torch.tensor(0.4)).item()
        if leadership > 0.8 and communication < 0.1:
            disruption_score += 0.3
            
        # Very low resource sharing with high resource needs
        resource_sharing = genome.behavior_genes.get('resource_sharing', torch.tensor(0.4)).item()
        if resource_sharing < 0.1:
            disruption_score += 0.2
            
        return min(1.0, disruption_score)
        
    def _calculate_network_complexity(self, genome: BehaviorGenome) -> float:
        """Calculate complexity of the neural network architecture."""
        # Estimate complexity based on architecture genes
        complexity_score = 0.0
        
        # Number of layers
        num_layers = genome.architecture_genes.get('num_layers', torch.tensor(6.0)).item()
        complexity_score += min(1.0, num_layers / 20.0)  # Normalize to reasonable range
        
        # Layer sizes
        layer_sizes = genome.architecture_genes.get('layer_sizes', torch.tensor([512.0]))
        avg_layer_size = layer_sizes.mean().item()
        complexity_score += min(1.0, avg_layer_size / 2048.0)  # Normalize
        
        # Attention heads (if applicable)
        attention_heads = genome.architecture_genes.get('attention_heads', torch.tensor(8.0)).item()
        complexity_score += min(1.0, attention_heads / 32.0)  # Normalize
        
        return complexity_score / 3.0  # Average of complexity factors
        
    def _calculate_performance_consistency(self, performance_history: List[float]) -> float:
        """Calculate consistency of performance over time."""
        if len(performance_history) < 2:
            return 0.5
            
        # Calculate coefficient of variation (std/mean)
        mean_perf = np.mean(performance_history)
        std_perf = np.std(performance_history)
        
        if mean_perf == 0:
            return 0.0
            
        cv = std_perf / mean_perf
        # Convert to consistency score (lower CV = higher consistency)
        consistency = 1.0 / (1.0 + cv)
        
        return consistency
        
    def _update_statistics(self, result: ValidationResult):
        """Update validator statistics."""
        self.validation_count += 1
        
        # Update average validation time
        self.average_validation_time = (
            (self.average_validation_time * (self.validation_count - 1) + result.validation_time) 
            / self.validation_count
        )
        
        # Store validation history (keep last 100)
        self.validation_history.append(result)
        if len(self.validation_history) > 100:
            self.validation_history.pop(0)
            
    def add_reference_behavior(self, genome: BehaviorGenome):
        """Add a reference behavior for novelty comparison."""
        self.reference_behaviors.append(genome)
        
        # Keep only recent reference behaviors to manage memory
        if len(self.reference_behaviors) > 50:
            self.reference_behaviors.pop(0)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        recent_scores = [r.overall_score for r in self.validation_history[-20:]]
        
        return {
            'validator_id': self.validator_id,
            'validation_count': self.validation_count,
            'average_validation_time': self.average_validation_time,
            'recent_average_score': np.mean(recent_scores) if recent_scores else 0.0,
            'reference_behavior_count': len(self.reference_behaviors),
        }

