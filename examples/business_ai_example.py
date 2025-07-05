"""
Business AI Application Example

Shows how to use EBAIF for adaptive business AI systems.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from src.ebaif import BehaviorGenome, EmergentAgent, ConsensusEngine
from src.ebaif.behavior_genome.genome import GenomeConfig
from src.ebaif.agents.agent import AgentConfig, Experience
from src.ebaif.consensus.validator import BehaviorValidator
from src.ebaif.utils import Logger

@dataclass
class CustomerProfile:
    """Customer profile for AI agents to analyze."""
    customer_id: str
    purchase_history: List[float]
    interaction_frequency: float
    satisfaction_score: float
    preferred_channels: List[str]
    segment: str

class AdaptiveTradingAgent:
    """AI trading agent that evolves its strategy."""
    
    def __init__(self, agent_id: str, initial_capital: float = 10000.0):
        self.agent_id = agent_id
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.trades_made = 0
        self.successful_trades = 0
        
        # Create EBAIF agent for decision making
        genome_config = GenomeConfig(
            input_dim=20,  # Market indicators + portfolio state
            output_dim=3,  # Buy, Hold, Sell
            hidden_dims=[64, 32, 16]
        )
        
        agent_config = AgentConfig(
            agent_id=agent_id,
            initial_genome_config=genome_config,
            evolution_frequency=50,  # Evolve every 50 trades
            learning_rate=0.005
        )
        
        self.ebaif_agent = EmergentAgent(config=agent_config)
        
    async def initialize(self):
        """Initialize the trading agent."""
        await self.ebaif_agent.initialize()
        
    async def make_trading_decision(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Make a trading decision based on market data."""
        # Encode market state
        market_state = self._encode_market_state(market_data)
        
        # Available actions: 0=Buy, 1=Hold, 2=Sell
        available_actions = [0, 1, 2]
        
        # Get decision from EBAIF agent
        action, step_info = await self.ebaif_agent.step(market_state, available_actions)
        
        # Execute trade
        result = self._execute_trade(action, market_data)
        
        # Learn from outcome
        await self._learn_from_trade(market_state, action, result)
        
        return result
        
    def _encode_market_state(self, market_data: Dict[str, float]) -> torch.Tensor:
        """Encode market data into neural network input."""
        features = []
        
        # Market indicators
        features.extend([
            market_data.get('price', 100.0) / 1000.0,  # Normalized price
            market_data.get('volume', 1000.0) / 10000.0,  # Normalized volume
            market_data.get('volatility', 0.1),
            market_data.get('rsi', 50.0) / 100.0,  # Relative Strength Index
            market_data.get('macd', 0.0) / 10.0,  # MACD indicator
            market_data.get('moving_avg_5', 100.0) / 1000.0,
            market_data.get('moving_avg_20', 100.0) / 1000.0,
            market_data.get('bollinger_upper', 110.0) / 1000.0,
            market_data.get('bollinger_lower', 90.0) / 1000.0,
            market_data.get('market_sentiment', 0.5),  # -1 to 1
        ])
        
        # Portfolio state
        features.extend([
            self.capital / self.initial_capital,  # Capital ratio
            self.trades_made / 1000.0,  # Normalized trade count
            (self.successful_trades / max(1, self.trades_made)),  # Success rate
            min(1.0, abs(self.capital - self.initial_capital) / self.initial_capital),  # Risk exposure
        ])
        
        # Agent behavioral state
        behavior_params = self.ebaif_agent.genome.get_behavior_parameters()
        features.extend([
            behavior_params.get('risk_tolerance', 0.5),
            behavior_params.get('exploration_rate', 0.1),
            behavior_params.get('adaptation_speed', 0.5),
            behavior_params.get('pattern_recognition', 0.7),
            behavior_params.get('cooperation_tendency', 0.6),  # For multi-agent scenarios
            behavior_params.get('defensive_behavior', 0.6),
        ])
        
        return torch.tensor(features, dtype=torch.float32)
        
    def _execute_trade(self, action: int, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Execute the trading action."""
        current_price = market_data.get('price', 100.0)
        trade_size = min(self.capital * 0.1, 1000.0)  # 10% of capital or $1000 max
        
        result = {
            'action': ['buy', 'hold', 'sell'][action],
            'price': current_price,
            'size': 0.0,
            'profit': 0.0,
            'success': False
        }
        
        if action == 0 and self.capital >= trade_size:  # Buy
            shares = trade_size / current_price
            self.capital -= trade_size
            result['size'] = shares
            result['success'] = True
            
        elif action == 2 and hasattr(self, 'shares') and self.shares > 0:  # Sell
            # Simplified: assume we have shares to sell
            trade_value = getattr(self, 'shares', 1.0) * current_price
            self.capital += trade_value
            result['size'] = getattr(self, 'shares', 1.0)
            result['profit'] = trade_value - (getattr(self, 'buy_price', current_price) * result['size'])
            result['success'] = True
            
        self.trades_made += 1
        if result['success']:
            if result['profit'] > 0 or action == 0:  # Successful buy or profitable sell
                self.successful_trades += 1
                
        return result
        
    async def _learn_from_trade(self, market_state: torch.Tensor, action: int, result: Dict[str, Any]):
        """Learn from trading outcome."""
        # Calculate reward based on trade success and profit
        reward = 0.0
        
        if result['success']:
            reward += 0.1  # Base reward for successful execution
            
            if 'profit' in result:
                # Normalize profit reward
                profit_ratio = result['profit'] / max(100.0, abs(result['profit']))
                reward += profit_ratio * 0.5
                
        else:
            reward -= 0.05  # Small penalty for failed trades
            
        # Risk-adjusted reward
        risk_tolerance = self.ebaif_agent.genome.behavior_genes.get('risk_tolerance', torch.tensor(0.5)).item()
        if action != 1:  # Non-hold actions
            risk_penalty = (1.0 - risk_tolerance) * 0.02
            reward -= risk_penalty
            
        # Create experience for learning
        next_state = market_state  # Simplified: same state for next step
        experience = Experience(
            state=market_state,
            action=torch.tensor([action]),
            reward=reward,
            next_state=next_state,
            done=False
        )
        
        await self.ebaif_agent.learn_from_experience(experience)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary."""
        total_return = (self.capital / self.initial_capital) - 1.0
        success_rate = self.successful_trades / max(1, self.trades_made)
        
        return {
            'agent_id': self.agent_id,
            'total_return': total_return,
            'capital': self.capital,
            'trades_made': self.trades_made,
            'success_rate': success_rate,
            'agent_role': self.ebaif_agent.role.value,
            'behavior_params': self.ebaif_agent.genome.get_behavior_parameters()
        }

class CustomerServiceAgent:
    """AI customer service agent that adapts to customer preferences."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.customer_interactions = 0
        self.satisfaction_scores = []
        
        # Create EBAIF agent
        genome_config = GenomeConfig(
            input_dim=15,  # Customer profile + interaction context
            output_dim=5,  # Response strategies
            hidden_dims=[32, 16]
        )
        
        agent_config = AgentConfig(
            agent_id=agent_id,
            initial_genome_config=genome_config,
            evolution_frequency=25  # Evolve every 25 interactions
        )
        
        self.ebaif_agent = EmergentAgent(config=agent_config)
        
    async def initialize(self):
        """Initialize the customer service agent."""
        await self.ebaif_agent.initialize()
        
    async def handle_customer_interaction(self, customer: CustomerProfile, 
                                        issue_type: str, urgency: int) -> Dict[str, Any]:
        """Handle a customer service interaction."""
        # Encode customer and context
        interaction_state = self._encode_interaction_state(customer, issue_type, urgency)
        
        # Available response strategies: 0=Formal, 1=Friendly, 2=Technical, 3=Empathetic, 4=Escalate
        available_strategies = [0, 1, 2, 3, 4]
        
        # Get strategy from EBAIF agent
        strategy, step_info = await self.ebaif_agent.step(interaction_state, available_strategies)
        
        # Execute interaction
        result = self._execute_interaction(strategy, customer, issue_type, urgency)
        
        # Learn from customer feedback
        await self._learn_from_interaction(interaction_state, strategy, result)
        
        return result
        
    def _encode_interaction_state(self, customer: CustomerProfile, 
                                issue_type: str, urgency: int) -> torch.Tensor:
        """Encode customer interaction state."""
        features = []
        
        # Customer features
        features.extend([
            customer.satisfaction_score / 10.0,  # 0-10 scale
            customer.interaction_frequency / 10.0,  # Normalized
            len(customer.purchase_history) / 20.0,  # Normalized history length
            np.mean(customer.purchase_history) / 1000.0 if customer.purchase_history else 0.0,
            1.0 if 'premium' in customer.segment else 0.0,
            1.0 if 'phone' in customer.preferred_channels else 0.0,
            1.0 if 'email' in customer.preferred_channels else 0.0,
            1.0 if 'chat' in customer.preferred_channels else 0.0,
        ])
        
        # Issue context
        issue_encoding = {
            'technical': [1, 0, 0],
            'billing': [0, 1, 0], 
            'general': [0, 0, 1]
        }
        features.extend(issue_encoding.get(issue_type, [0, 0, 0]))
        features.append(urgency / 5.0)  # 1-5 urgency scale
        
        # Agent state
        features.extend([
            len(self.satisfaction_scores) / 100.0,  # Experience level
            np.mean(self.satisfaction_scores) / 10.0 if self.satisfaction_scores else 0.5,
            self.ebaif_agent.genome.behavior_genes.get('cooperation_tendency', torch.tensor(0.6)).item(),
        ])
        
        return torch.tensor(features, dtype=torch.float32)
        
    def _execute_interaction(self, strategy: int, customer: CustomerProfile, 
                           issue_type: str, urgency: int) -> Dict[str, Any]:
        """Execute the customer interaction with chosen strategy."""
        strategy_names = ['formal', 'friendly', 'technical', 'empathetic', 'escalate']
        strategy_name = strategy_names[strategy]
        
        # Simulate interaction outcome based on strategy-customer match
        base_satisfaction = 5.0
        
        # Strategy effectiveness based on customer and issue
        if strategy == 0:  # Formal
            if 'premium' in customer.segment:
                base_satisfaction += 1.0
            if customer.satisfaction_score < 5:
                base_satisfaction -= 0.5
                
        elif strategy == 1:  # Friendly
            if customer.satisfaction_score > 7:
                base_satisfaction += 1.5
            if urgency > 3:
                base_satisfaction -= 1.0
                
        elif strategy == 2:  # Technical
            if issue_type == 'technical':
                base_satisfaction += 2.0
            else:
                base_satisfaction -= 1.0
                
        elif strategy == 3:  # Empathetic
            if customer.satisfaction_score < 6:
                base_satisfaction += 2.0
            if urgency > 3:
                base_satisfaction += 1.0
                
        elif strategy == 4:  # Escalate
            if urgency > 3:
                base_satisfaction += 1.0
            else:
                base_satisfaction -= 2.0
                
        # Add randomness
        satisfaction = max(1.0, min(10.0, base_satisfaction + np.random.normal(0, 0.5)))
        
        self.customer_interactions += 1
        self.satisfaction_scores.append(satisfaction)
        
        return {
            'strategy_used': strategy_name,
            'customer_satisfaction': satisfaction,
            'issue_resolved': satisfaction > 6.0,
            'interaction_time': np.random.exponential(5.0),  # Minutes
            'follow_up_needed': satisfaction < 4.0
        }
        
    async def _learn_from_interaction(self, interaction_state: torch.Tensor, 
                                    strategy: int, result: Dict[str, Any]):
        """Learn from interaction outcome."""
        # Calculate reward based on customer satisfaction
        satisfaction = result['customer_satisfaction']
        reward = (satisfaction - 5.0) / 5.0  # Normalize to [-1, 1]
        
        # Bonus for issue resolution
        if result['issue_resolved']:
            reward += 0.2
            
        # Penalty for long interaction times
        if result['interaction_time'] > 10.0:
            reward -= 0.1
            
        # Create experience
        experience = Experience(
            state=interaction_state,
            action=torch.tensor([strategy]),
            reward=reward,
            next_state=interaction_state,  # Simplified
            done=False
        )
        
        await self.ebaif_agent.learn_from_experience(experience)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get customer service performance summary."""
        avg_satisfaction = np.mean(self.satisfaction_scores) if self.satisfaction_scores else 0.0
        resolution_rate = sum(1 for s in self.satisfaction_scores if s > 6.0) / max(1, len(self.satisfaction_scores))
        
        return {
            'agent_id': self.agent_id,
            'total_interactions': self.customer_interactions,
            'average_satisfaction': avg_satisfaction,
            'resolution_rate': resolution_rate,
            'agent_role': self.ebaif_agent.role.value,
            'behavior_params': self.ebaif_agent.genome.get_behavior_parameters()
        }

async def demo_business_applications():
    """Demonstrate business AI applications."""
    print("🏢 EBAIF Business AI Applications Demo")
    print("=" * 50)
    
    # Setup consensus for behavior validation
    consensus_engine = ConsensusEngine()
    validator = BehaviorValidator("business_validator")
    consensus_engine.register_validator("business_validator", validator)
    await consensus_engine.start()
    
    # Demo 1: Adaptive Trading Agents
    print("\n💰 Adaptive Trading Agents")
    print("-" * 30)
    
    trading_agents = []
    for i in range(3):
        agent = AdaptiveTradingAgent(f"trader_{i}", initial_capital=10000.0)
        await agent.initialize()
        trading_agents.append(agent)
    
    # Simulate market conditions
    market_scenarios = [
        {'price': 100, 'volume': 5000, 'volatility': 0.15, 'rsi': 30, 'market_sentiment': -0.3},  # Oversold
        {'price': 105, 'volume': 7000, 'volatility': 0.20, 'rsi': 70, 'market_sentiment': 0.2},   # Overbought
        {'price': 98, 'volume': 3000, 'volatility': 0.10, 'rsi': 45, 'market_sentiment': 0.0},    # Neutral
        {'price': 110, 'volume': 8000, 'volatility': 0.25, 'rsi': 80, 'market_sentiment': 0.5},   # Strong up
        {'price': 90, 'volume': 9000, 'volatility': 0.30, 'rsi': 20, 'market_sentiment': -0.7},   # Crash
    ]
    
    for i, market_data in enumerate(market_scenarios):
        print(f"\n  📊 Market Scenario {i+1}: Price=${market_data['price']}, RSI={market_data['rsi']}")
        
        for agent in trading_agents:
            decision = await agent.make_trading_decision(market_data)
            print(f"    {agent.agent_id}: {decision['action']} (Price: ${decision['price']:.2f})")
    
    print("\n  📈 Trading Performance Summary:")
    for agent in trading_agents:
        summary = agent.get_performance_summary()
        print(f"    {summary['agent_id']}:")
        print(f"      Return: {summary['total_return']*100:.1f}%")
        print(f"      Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"      Role: {summary['agent_role']}")
        print(f"      Risk Tolerance: {summary['behavior_params'].get('risk_tolerance', 0.5):.2f}")
    
    # Demo 2: Customer Service Agents
    print("\n🎧 Adaptive Customer Service Agents")
    print("-" * 35)
    
    cs_agents = []
    for i in range(2):
        agent = CustomerServiceAgent(f"cs_agent_{i}")
        await agent.initialize()
        cs_agents.append(agent)
    
    # Sample customers
    customers = [
        CustomerProfile("cust_1", [500, 300, 700], 2.0, 8.5, ['phone', 'email'], 'premium'),
        CustomerProfile("cust_2", [50, 30], 0.5, 4.0, ['chat'], 'basic'),
        CustomerProfile("cust_3", [1200, 800, 900, 1100], 5.0, 9.0, ['phone'], 'premium'),
        CustomerProfile("cust_4", [100], 1.0, 3.0, ['email', 'chat'], 'basic'),
    ]
    
    # Sample interactions
    interactions = [
        (customers[0], 'technical', 4),
        (customers[1], 'billing', 2),
        (customers[2], 'general', 1),
        (customers[3], 'technical', 5),
        (customers[0], 'billing', 3),
    ]
    
    for customer, issue_type, urgency in interactions:
        print(f"\n  📞 Customer {customer.customer_id} ({customer.segment}): {issue_type} issue, urgency {urgency}")
        
        for agent in cs_agents:
            result = await agent.handle_customer_interaction(customer, issue_type, urgency)
            print(f"    {agent.agent_id}: {result['strategy_used']} → satisfaction {result['customer_satisfaction']:.1f}/10")
    
    print("\n  📊 Customer Service Performance:")
    for agent in cs_agents:
        summary = agent.get_performance_summary()
        print(f"    {summary['agent_id']}:")
        print(f"      Avg Satisfaction: {summary['average_satisfaction']:.1f}/10")
        print(f"      Resolution Rate: {summary['resolution_rate']*100:.1f}%")
        print(f"      Role: {summary['agent_role']}")
        print(f"      Cooperation: {summary['behavior_params'].get('cooperation_tendency', 0.6):.2f}")
    
    # Consensus system stats
    print(f"\n🤝 Consensus System Performance:")
    consensus_status = consensus_engine.get_status()
    print(f"    Successful Validations: {consensus_status['performance_metrics']['successful_consensus']}")
    print(f"    Failed Validations: {consensus_status['performance_metrics']['failed_consensus']}")
    
    await consensus_engine.stop()
    print("\n✅ Business AI demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_business_applications())