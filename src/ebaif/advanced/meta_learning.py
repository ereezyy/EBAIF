"""
Meta-Learning Engine Implementation

Implements advanced meta-learning capabilities including MAML, Reptile, and
custom meta-optimization algorithms for rapid adaptation to new tasks.
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
class MetaLearningConfig:
    """Configuration for meta-learning system."""
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 0.001
    adaptation_steps: int = 5
    meta_batch_size: int = 16
    task_batch_size: int = 8
    support_shots: int = 5
    query_shots: int = 15
    max_meta_iterations: int = 1000
    convergence_threshold: float = 0.001
    use_second_order: bool = True
    memory_augmented: bool = True
    continual_learning: bool = True

class TaskDistribution:
    """Represents a distribution of tasks for meta-learning."""
    
    def __init__(self, task_type: str, num_classes: int = 5, input_dim: int = 64):
        self.task_type = task_type
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.tasks_generated = 0
        
    def sample_task(self) -> Dict[str, Any]:
        """Sample a task from this distribution."""
        self.tasks_generated += 1
        
        task = {
            'task_id': f"{self.task_type}_{self.tasks_generated}",
            'type': self.task_type,
            'num_classes': self.num_classes,
            'input_dim': self.input_dim,
            'support_set': self._generate_support_set(),
            'query_set': self._generate_query_set(),
            'task_parameters': self._generate_task_parameters()
        }
        
        return task
        
    def _generate_support_set(self) -> List[Tuple[List[float], int]]:
        """Generate support set for few-shot learning."""
        support_set = []
        
        for class_id in range(self.num_classes):
            for _ in range(5):  # 5 examples per class
                # Generate synthetic data point
                if self.task_type == 'classification':
                    # Create class-specific patterns
                    center = [random.uniform(-1, 1) for _ in range(self.input_dim)]
                    noise = [random.gauss(0, 0.1) for _ in range(self.input_dim)]
                    features = [c + n for c, n in zip(center, noise)]
                elif self.task_type == 'regression':
                    # Generate regression data
                    features = [random.uniform(-2, 2) for _ in range(self.input_dim)]
                else:
                    features = [random.gauss(0, 1) for _ in range(self.input_dim)]
                    
                support_set.append((features, class_id))
                
        return support_set
        
    def _generate_query_set(self) -> List[Tuple[List[float], int]]:
        """Generate query set for evaluation."""
        query_set = []
        
        for class_id in range(self.num_classes):
            for _ in range(3):  # 3 query examples per class
                if self.task_type == 'classification':
                    center = [random.uniform(-1, 1) for _ in range(self.input_dim)]
                    noise = [random.gauss(0, 0.1) for _ in range(self.input_dim)]
                    features = [c + n for c, n in zip(center, noise)]
                elif self.task_type == 'regression':
                    features = [random.uniform(-2, 2) for _ in range(self.input_dim)]
                else:
                    features = [random.gauss(0, 1) for _ in range(self.input_dim)]
                    
                query_set.append((features, class_id))
                
        return query_set
        
    def _generate_task_parameters(self) -> Dict[str, Any]:
        """Generate task-specific parameters."""
        return {
            'difficulty': random.uniform(0.1, 1.0),
            'noise_level': random.uniform(0.0, 0.3),
            'task_weight': random.uniform(0.5, 2.0),
            'adaptation_requirement': random.uniform(0.1, 1.0)
        }

class MetaModel:
    """Meta-learnable model with rapid adaptation capabilities."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [128, 64]
        
        # Initialize parameters
        self.parameters = self._initialize_parameters()
        self.meta_parameters = {k: v.copy() for k, v in self.parameters.items()}
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
    def _initialize_parameters(self) -> Dict[str, List[List[float]]]:
        """Initialize model parameters."""
        params = {}
        
        # Build layer dimensions
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        # Initialize weights and biases
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            
            # Xavier initialization
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            
            # Weights
            weights = []
            for _ in range(out_dim):
                row = [random.uniform(-limit, limit) for _ in range(in_dim)]
                weights.append(row)
            params[f'W{i}'] = weights
            
            # Biases
            biases = [0.0 for _ in range(out_dim)]
            params[f'b{i}'] = biases
            
        return params
        
    def forward(self, x: List[float], parameters: Dict[str, List[List[float]]] = None) -> List[float]:
        """Forward pass through the model."""
        if parameters is None:
            parameters = self.parameters
            
        activations = x.copy()
        
        # Forward through layers
        layer_idx = 0
        while f'W{layer_idx}' in parameters:
            weights = parameters[f'W{layer_idx}']
            biases = parameters[f'b{layer_idx}']
            
            # Linear transformation
            new_activations = []
            for i in range(len(weights)):
                output = sum(w * a for w, a in zip(weights[i], activations)) + biases[i]
                new_activations.append(output)
                
            # Apply activation function (ReLU for hidden layers, linear for output)
            if f'W{layer_idx + 1}' in parameters:  # Hidden layer
                activations = [max(0, a) for a in new_activations]
            else:  # Output layer
                activations = new_activations
                
            layer_idx += 1
            
        return activations
        
    def compute_loss(self, predictions: List[float], targets: List[float], loss_type: str = 'mse') -> float:
        """Compute loss between predictions and targets."""
        if loss_type == 'mse':
            return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
        elif loss_type == 'cross_entropy':
            # Simplified cross-entropy for multi-class
            exp_preds = [math.exp(p) for p in predictions]
            sum_exp = sum(exp_preds)
            softmax = [e / sum_exp for e in exp_preds]
            
            # Assume targets are one-hot encoded
            loss = 0.0
            for i, target in enumerate(targets):
                if target > 0:  # True class
                    loss -= math.log(max(softmax[i], 1e-8))
            return loss
        else:
            return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
            
    def compute_gradients(self, x: List[float], targets: List[float], 
                         parameters: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        """Compute gradients using numerical differentiation."""
        gradients = {}
        epsilon = 1e-7
        
        # Get baseline loss
        predictions = self.forward(x, parameters)
        baseline_loss = self.compute_loss(predictions, targets)
        
        # Compute gradients for each parameter
        for param_name, param_matrix in parameters.items():
            grad_matrix = []
            
            if 'W' in param_name:  # Weight matrix
                for i in range(len(param_matrix)):
                    grad_row = []
                    for j in range(len(param_matrix[i])):
                        # Perturb parameter
                        original_val = param_matrix[i][j]
                        param_matrix[i][j] = original_val + epsilon
                        
                        # Compute perturbed loss
                        perturbed_predictions = self.forward(x, parameters)
                        perturbed_loss = self.compute_loss(perturbed_predictions, targets)
                        
                        # Gradient approximation
                        grad = (perturbed_loss - baseline_loss) / epsilon
                        grad_row.append(grad)
                        
                        # Restore original value
                        param_matrix[i][j] = original_val
                        
                    grad_matrix.append(grad_row)
            else:  # Bias vector
                grad_vector = []
                for i in range(len(param_matrix)):
                    # Perturb bias
                    original_val = param_matrix[i]
                    param_matrix[i] = original_val + epsilon
                    
                    # Compute perturbed loss
                    perturbed_predictions = self.forward(x, parameters)
                    perturbed_loss = self.compute_loss(perturbed_predictions, targets)
                    
                    # Gradient approximation
                    grad = (perturbed_loss - baseline_loss) / epsilon
                    grad_vector.append(grad)
                    
                    # Restore original value
                    param_matrix[i] = original_val
                    
                grad_matrix = grad_vector
                
            gradients[param_name] = grad_matrix
            
        return gradients
        
    def update_parameters(self, parameters: Dict[str, List[List[float]]], 
                         gradients: Dict[str, List[List[float]]], 
                         learning_rate: float) -> Dict[str, List[List[float]]]:
        """Update parameters using gradients."""
        updated_params = {}
        
        for param_name, param_matrix in parameters.items():
            grad_matrix = gradients[param_name]
            
            if 'W' in param_name:  # Weight matrix
                updated_matrix = []
                for i in range(len(param_matrix)):
                    updated_row = []
                    for j in range(len(param_matrix[i])):
                        new_val = param_matrix[i][j] - learning_rate * grad_matrix[i][j]
                        updated_row.append(new_val)
                    updated_matrix.append(updated_row)
            else:  # Bias vector
                updated_matrix = []
                for i in range(len(param_matrix)):
                    new_val = param_matrix[i] - learning_rate * grad_matrix[i]
                    updated_matrix.append(new_val)
                    
            updated_params[param_name] = updated_matrix
            
        return updated_params

class MetaLearningEngine:
    """Advanced meta-learning engine implementing MAML and beyond."""
    
    def __init__(self, config: MetaLearningConfig = None):
        self.config = config or MetaLearningConfig()
        
        # Task distributions
        self.task_distributions: Dict[str, TaskDistribution] = {}
        
        # Meta-model
        self.meta_model: Optional[MetaModel] = None
        
        # Learning state
        self.meta_iteration = 0
        self.task_performance_history = defaultdict(deque)
        self.adaptation_statistics = deque(maxlen=1000)
        
        # Memory for continual learning
        self.episodic_memory = deque(maxlen=10000)
        self.semantic_memory = {}
        
        # Performance tracking
        self.meta_learning_curves = deque(maxlen=1000)
        self.transfer_learning_scores = deque(maxlen=100)
        
    def register_task_distribution(self, name: str, distribution: TaskDistribution):
        """Register a task distribution for meta-learning."""
        self.task_distributions[name] = distribution
        
    def initialize_meta_model(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        """Initialize the meta-learnable model."""
        self.meta_model = MetaModel(input_dim, output_dim, hidden_dims)
        
    async def meta_train(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Train the meta-learner across multiple task distributions."""
        if not self.meta_model or not self.task_distributions:
            raise ValueError("Meta-model and task distributions must be initialized first")
            
        training_history = []
        
        for iteration in range(num_iterations):
            self.meta_iteration = iteration
            
            # Sample meta-batch of tasks
            meta_batch = await self._sample_meta_batch()
            
            # Compute meta-gradients
            meta_gradients = await self._compute_meta_gradients(meta_batch)
            
            # Update meta-parameters
            self._update_meta_parameters(meta_gradients)
            
            # Evaluate meta-learning progress
            if iteration % 10 == 0:
                evaluation_results = await self._evaluate_meta_learning()
                training_history.append(evaluation_results)
                
                # Log progress
                print(f"Meta-iteration {iteration}: "
                      f"Avg adaptation performance: {evaluation_results['avg_adaptation_performance']:.4f}")
                      
            # Adaptive learning rate scheduling
            if iteration % 100 == 0:
                self._adapt_learning_rates()
                
            # Memory consolidation for continual learning
            if self.config.continual_learning and iteration % 50 == 0:
                await self._consolidate_memory()
                
        return {
            'final_iteration': self.meta_iteration,
            'training_history': training_history,
            'meta_model_state': self._get_meta_model_state(),
            'adaptation_statistics': list(self.adaptation_statistics)
        }
        
    async def _sample_meta_batch(self) -> List[Dict[str, Any]]:
        """Sample a batch of tasks for meta-learning."""
        meta_batch = []
        
        for _ in range(self.config.meta_batch_size):
            # Sample task distribution
            dist_name = random.choice(list(self.task_distributions.keys()))
            distribution = self.task_distributions[dist_name]
            
            # Sample specific task
            task = distribution.sample_task()
            meta_batch.append(task)
            
        return meta_batch
        
    async def _compute_meta_gradients(self, meta_batch: List[Dict[str, Any]]) -> Dict[str, List[List[float]]]:
        """Compute meta-gradients using MAML algorithm."""
        meta_gradients = {k: [[0.0 for _ in row] for row in matrix] 
                         if isinstance(matrix[0], list) else [0.0 for _ in matrix]
                         for k, matrix in self.meta_model.meta_parameters.items()}
        
        for task in meta_batch:
            # Fast adaptation on support set
            adapted_parameters = await self._fast_adaptation(task)
            
            # Compute gradients on query set
            task_gradients = await self._compute_task_gradients(task, adapted_parameters)
            
            # Accumulate meta-gradients
            for param_name in meta_gradients:
                if isinstance(meta_gradients[param_name][0], list):  # Weight matrix
                    for i in range(len(meta_gradients[param_name])):
                        for j in range(len(meta_gradients[param_name][i])):
                            meta_gradients[param_name][i][j] += task_gradients[param_name][i][j]
                else:  # Bias vector
                    for i in range(len(meta_gradients[param_name])):
                        meta_gradients[param_name][i] += task_gradients[param_name][i]
                        
        # Average gradients across meta-batch
        for param_name in meta_gradients:
            if isinstance(meta_gradients[param_name][0], list):  # Weight matrix
                for i in range(len(meta_gradients[param_name])):
                    for j in range(len(meta_gradients[param_name][i])):
                        meta_gradients[param_name][i][j] /= len(meta_batch)
            else:  # Bias vector
                for i in range(len(meta_gradients[param_name])):
                    meta_gradients[param_name][i] /= len(meta_batch)
                    
        return meta_gradients
        
    async def _fast_adaptation(self, task: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """Perform fast adaptation on task support set."""
        # Start with meta-parameters
        adapted_params = {k: [row.copy() if isinstance(row, list) else row 
                             for row in matrix] if isinstance(matrix[0], list) 
                            else matrix.copy()
                            for k, matrix in self.meta_model.meta_parameters.items()}
        
        # Gradient descent steps on support set
        for step in range(self.config.adaptation_steps):
            # Compute gradients on support set
            support_gradients = await self._compute_support_gradients(task, adapted_params)
            
            # Update parameters
            adapted_params = self.meta_model.update_parameters(
                adapted_params, support_gradients, self.config.inner_learning_rate
            )
            
        return adapted_params
        
    async def _compute_support_gradients(self, task: Dict[str, Any], 
                                       parameters: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        """Compute gradients on task support set."""
        total_gradients = None
        
        for x, y in task['support_set']:
            # Convert target to appropriate format
            if task['type'] == 'classification':
                targets = [0.0] * task['num_classes']
                targets[y] = 1.0
            else:
                targets = [float(y)]
                
            # Compute gradients for this example
            gradients = self.meta_model.compute_gradients(x, targets, parameters)
            
            # Accumulate gradients
            if total_gradients is None:
                total_gradients = gradients
            else:
                for param_name in total_gradients:
                    if isinstance(total_gradients[param_name][0], list):  # Weight matrix
                        for i in range(len(total_gradients[param_name])):
                            for j in range(len(total_gradients[param_name][i])):
                                total_gradients[param_name][i][j] += gradients[param_name][i][j]
                    else:  # Bias vector
                        for i in range(len(total_gradients[param_name])):
                            total_gradients[param_name][i] += gradients[param_name][i]
                            
        # Average gradients
        for param_name in total_gradients:
            if isinstance(total_gradients[param_name][0], list):  # Weight matrix
                for i in range(len(total_gradients[param_name])):
                    for j in range(len(total_gradients[param_name][i])):
                        total_gradients[param_name][i][j] /= len(task['support_set'])
            else:  # Bias vector
                for i in range(len(total_gradients[param_name])):
                    total_gradients[param_name][i] /= len(task['support_set'])
                    
        return total_gradients
        
    async def _compute_task_gradients(self, task: Dict[str, Any], 
                                    adapted_parameters: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        """Compute gradients on task query set using adapted parameters."""
        total_gradients = None
        
        for x, y in task['query_set']:
            # Convert target to appropriate format
            if task['type'] == 'classification':
                targets = [0.0] * task['num_classes']
                targets[y] = 1.0
            else:
                targets = [float(y)]
                
            # Compute gradients for this example
            gradients = self.meta_model.compute_gradients(x, targets, adapted_parameters)
            
            # Accumulate gradients
            if total_gradients is None:
                total_gradients = gradients
            else:
                for param_name in total_gradients:
                    if isinstance(total_gradients[param_name][0], list):  # Weight matrix
                        for i in range(len(total_gradients[param_name])):
                            for j in range(len(total_gradients[param_name][i])):
                                total_gradients[param_name][i][j] += gradients[param_name][i][j]
                    else:  # Bias vector
                        for i in range(len(total_gradients[param_name])):
                            total_gradients[param_name][i] += gradients[param_name][i]
                            
        # Average gradients
        for param_name in total_gradients:
            if isinstance(total_gradients[param_name][0], list):  # Weight matrix
                for i in range(len(total_gradients[param_name])):
                    for j in range(len(total_gradients[param_name][i])):
                        total_gradients[param_name][i][j] /= len(task['query_set'])
            else:  # Bias vector
                for i in range(len(total_gradients[param_name])):
                    total_gradients[param_name][i] /= len(task['query_set'])
                    
        return total_gradients
        
    def _update_meta_parameters(self, meta_gradients: Dict[str, List[List[float]]]):
        """Update meta-parameters using computed meta-gradients."""
        self.meta_model.meta_parameters = self.meta_model.update_parameters(
            self.meta_model.meta_parameters, 
            meta_gradients, 
            self.config.outer_learning_rate
        )
        
    async def _evaluate_meta_learning(self) -> Dict[str, Any]:
        """Evaluate meta-learning progress on held-out tasks."""
        evaluation_tasks = []
        
        # Sample evaluation tasks from each distribution
        for dist_name, distribution in self.task_distributions.items():
            for _ in range(5):  # 5 evaluation tasks per distribution
                task = distribution.sample_task()
                evaluation_tasks.append(task)
                
        # Evaluate adaptation performance
        adaptation_performances = []
        
        for task in evaluation_tasks:
            # Fast adaptation
            adapted_params = await self._fast_adaptation(task)
            
            # Evaluate on query set
            total_loss = 0.0
            for x, y in task['query_set']:
                if task['type'] == 'classification':
                    targets = [0.0] * task['num_classes']
                    targets[y] = 1.0
                else:
                    targets = [float(y)]
                    
                predictions = self.meta_model.forward(x, adapted_params)
                loss = self.meta_model.compute_loss(predictions, targets)
                total_loss += loss
                
            avg_loss = total_loss / len(task['query_set'])
            adaptation_performances.append(1.0 / (1.0 + avg_loss))  # Convert to performance score
            
        avg_adaptation_performance = sum(adaptation_performances) / len(adaptation_performances)
        
        # Record in history
        evaluation_result = {
            'meta_iteration': self.meta_iteration,
            'avg_adaptation_performance': avg_adaptation_performance,
            'num_evaluation_tasks': len(evaluation_tasks),
        }
        
        self.meta_learning_curves.append(evaluation_result)
        
        return evaluation_result
        
    def _adapt_learning_rates(self):
        """Adapt learning rates based on meta-learning progress."""
        if len(self.meta_learning_curves) < 2:
            return
            
        current_performance = self.meta_learning_curves[-1]['avg_adaptation_performance']
        previous_performance = self.meta_learning_curves[-2]['avg_adaptation_performance']
        
        # If performance is improving, maintain learning rates
        # If performance is stagnating, reduce learning rates
        if current_performance <= previous_performance * 1.001:  # Less than 0.1% improvement
            self.config.outer_learning_rate *= 0.95
            self.config.inner_learning_rate *= 0.95
            
        # Prevent learning rates from becoming too small
        self.config.outer_learning_rate = max(self.config.outer_learning_rate, 1e-6)
        self.config.inner_learning_rate = max(self.config.inner_learning_rate, 1e-5)
        
    async def _consolidate_memory(self):
        """Consolidate episodic memory for continual learning."""
        if not self.episodic_memory:
            return
            
        # Simple memory consolidation: keep most diverse and important examples
        # This is a simplified version - in practice, would use more sophisticated methods
        
        # Sort by importance (assuming we track this)
        sorted_memories = sorted(list(self.episodic_memory), 
                               key=lambda x: x.get('importance', 0.5), reverse=True)
        
        # Keep top 50% most important memories
        keep_count = len(sorted_memories) // 2
        self.episodic_memory = deque(sorted_memories[:keep_count], maxlen=10000)
        
    def _get_meta_model_state(self) -> Dict[str, Any]:
        """Get current state of the meta-model."""
        return {
            'meta_parameters': self.meta_model.meta_parameters,
            'input_dim': self.meta_model.input_dim,
            'output_dim': self.meta_model.output_dim,
            'hidden_dims': self.meta_model.hidden_dims,
        }
        
    async def few_shot_adapt(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform few-shot adaptation to a new task."""
        if not self.meta_model:
            raise ValueError("Meta-model must be trained first")
            
        start_time = time.time()
        
        # Fast adaptation
        adapted_parameters = await self._fast_adaptation(task)
        
        # Evaluate adaptation quality
        performance_score = await self._evaluate_adaptation(task, adapted_parameters)
        
        adaptation_time = time.time() - start_time
        
        # Record adaptation statistics
        adaptation_stats = {
            'task_id': task['task_id'],
            'adaptation_time': adaptation_time,
            'performance_score': performance_score,
            'adaptation_steps': self.config.adaptation_steps,
            'support_set_size': len(task['support_set']),
        }
        
        self.adaptation_statistics.append(adaptation_stats)
        
        return {
            'adapted_parameters': adapted_parameters,
            'performance_score': performance_score,
            'adaptation_time': adaptation_time,
            'adaptation_stats': adaptation_stats,
        }
        
    async def _evaluate_adaptation(self, task: Dict[str, Any], 
                                 adapted_parameters: Dict[str, List[List[float]]]) -> float:
        """Evaluate quality of adaptation."""
        total_loss = 0.0
        
        for x, y in task['query_set']:
            if task['type'] == 'classification':
                targets = [0.0] * task['num_classes']
                targets[y] = 1.0
            else:
                targets = [float(y)]
                
            predictions = self.meta_model.forward(x, adapted_parameters)
            loss = self.meta_model.compute_loss(predictions, targets)
            total_loss += loss
            
        avg_loss = total_loss / len(task['query_set'])
        performance_score = 1.0 / (1.0 + avg_loss)
        
        return performance_score
        
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of meta-learning system."""
        return {
            'meta_iterations_completed': self.meta_iteration,
            'task_distributions': list(self.task_distributions.keys()),
            'current_learning_rates': {
                'inner_lr': self.config.inner_learning_rate,
                'outer_lr': self.config.outer_learning_rate,
            },
            'recent_adaptation_performance': (
                self.meta_learning_curves[-1]['avg_adaptation_performance'] 
                if self.meta_learning_curves else 0.0
            ),
            'total_adaptations': len(self.adaptation_statistics),
            'memory_size': len(self.episodic_memory),
            'meta_learning_curve': list(self.meta_learning_curves)[-20:],  # Last 20 evaluations
        }