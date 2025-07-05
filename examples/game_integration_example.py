"""
Game Integration Example

Shows how to integrate EBAIF with a game engine.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Tuple
from src.ebaif import BehaviorGenome, EmergentAgent
from src.ebaif.behavior_genome.genome import GenomeConfig
from src.ebaif.agents.agent import AgentConfig, Experience
from src.ebaif.utils import Logger

class GameNPC:
    """Game NPC powered by EBAIF."""
    
    def __init__(self, npc_id: str, npc_type: str = "guard"):
        self.npc_id = npc_id
        self.npc_type = npc_type
        
        # Create EBAIF agent
        genome_config = GenomeConfig(
            input_dim=32,  # Game state representation
            output_dim=8,  # Possible actions
            hidden_dims=[64, 32]
        )
        
        agent_config = AgentConfig(
            agent_id=npc_id,
            initial_genome_config=genome_config,
            evolution_frequency=100  # Evolve every 100 interactions
        )
        
        self.agent = EmergentAgent(config=agent_config)
        
        # Game-specific state
        self.position = (0, 0)
        self.health = 100
        self.player_interactions = 0
        self.personality_traits = {}
        
    async def initialize(self):
        """Initialize the NPC."""
        await self.agent.initialize()
        
        # Set initial personality based on behavior genome
        behavior_params = self.agent.genome.get_behavior_parameters()
        self.personality_traits = {
            'aggressiveness': behavior_params.get('aggression_level', 0.3),
            'friendliness': behavior_params.get('cooperation_tendency', 0.6),
            'curiosity': behavior_params.get('curiosity_factor', 0.5),
            'cautiousness': behavior_params.get('defensive_behavior', 0.6)
        }
        
    async def interact_with_player(self, player_state: Dict, context: Dict) -> Dict:
        """Handle player interaction."""
        self.player_interactions += 1
        
        # Convert game state to neural network input
        game_state = self._encode_game_state(player_state, context)
        
        # Define available actions
        available_actions = self._get_available_actions(context)
        
        # Agent decides action
        action_idx, step_info = await self.agent.step(game_state, available_actions)
        action = self._decode_action(action_idx)
        
        # Execute action in game
        result = self._execute_action(action, player_state, context)
        
        # Learn from interaction outcome
        reward = self._calculate_reward(result, context)
        next_state = self._encode_game_state(result.get('new_player_state', player_state), context)
        
        experience = Experience(
            state=game_state,
            action=torch.tensor([action_idx]),
            reward=reward,
            next_state=next_state,
            done=False
        )
        
        await self.agent.learn_from_experience(experience)
        
        return result
        
    def _encode_game_state(self, player_state: Dict, context: Dict) -> torch.Tensor:
        """Convert game state to neural network input."""
        # Example encoding
        features = []
        
        # Player features
        features.extend([
            player_state.get('health', 100) / 100.0,
            player_state.get('level', 1) / 50.0,
            player_state.get('reputation', 0) / 100.0,
        ])
        
        # Context features
        features.extend([
            context.get('time_of_day', 12) / 24.0,
            context.get('location_danger', 0) / 10.0,
            context.get('recent_aggression', 0) / 10.0,
        ])
        
        # NPC state
        features.extend([
            self.health / 100.0,
            self.personality_traits.get('aggressiveness', 0.3),
            self.personality_traits.get('friendliness', 0.6),
            len(self.agent.peer_agents) / 10.0,  # Social connections
        ])
        
        # Pad to input_dim
        while len(features) < 32:
            features.append(0.0)
            
        return torch.tensor(features[:32], dtype=torch.float32)
        
    def _get_available_actions(self, context: Dict) -> List[int]:
        """Get available actions based on context."""
        # Example actions: 0=greet, 1=ignore, 2=attack, 3=trade, 4=help, 5=flee, 6=guard, 7=patrol
        base_actions = [0, 1, 6, 7]  # Always available
        
        if context.get('player_has_items', False):
            base_actions.append(3)  # Trade
            
        if context.get('player_needs_help', False):
            base_actions.append(4)  # Help
            
        if context.get('hostile_situation', False):
            base_actions.extend([2, 5])  # Attack or flee
            
        return base_actions
        
    def _decode_action(self, action_idx: int) -> str:
        """Convert action index to action name."""
        action_map = {
            0: 'greet',
            1: 'ignore', 
            2: 'attack',
            3: 'trade',
            4: 'help',
            5: 'flee',
            6: 'guard',
            7: 'patrol'
        }
        return action_map.get(action_idx, 'ignore')
        
    def _execute_action(self, action: str, player_state: Dict, context: Dict) -> Dict:
        """Execute the chosen action."""
        result = {
            'action_taken': action,
            'npc_id': self.npc_id,
            'dialogue': '',
            'reputation_change': 0,
            'combat_initiated': False,
            'trade_offered': False,
            'new_player_state': player_state.copy()
        }
        
        if action == 'greet':
            friendliness = self.personality_traits.get('friendliness', 0.6)
            if friendliness > 0.7:
                result['dialogue'] = "Hello there, friend! How can I help you?"
                result['reputation_change'] = 1
            else:
                result['dialogue'] = "Hello."
                
        elif action == 'ignore':
            result['dialogue'] = ""
            
        elif action == 'attack':
            aggressiveness = self.personality_traits.get('aggressiveness', 0.3)
            if aggressiveness > 0.8 and context.get('hostile_situation', False):
                result['dialogue'] = "You've made a mistake coming here!"
                result['combat_initiated'] = True
                result['reputation_change'] = -5
                
        elif action == 'trade':
            result['dialogue'] = "I have some items that might interest you."
            result['trade_offered'] = True
            result['reputation_change'] = 1
            
        elif action == 'help':
            result['dialogue'] = "Let me assist you with that."
            result['reputation_change'] = 3
            
        elif action == 'flee':
            result['dialogue'] = "I... I need to go!"
            result['reputation_change'] = -1
            
        elif action == 'guard':
            result['dialogue'] = "I'm keeping watch here."
            
        elif action == 'patrol':
            result['dialogue'] = "Just making my rounds."
            
        return result
        
    def _calculate_reward(self, result: Dict, context: Dict) -> float:
        """Calculate reward for the action taken."""
        reward = 0.0
        
        # Positive rewards
        if result['reputation_change'] > 0:
            reward += result['reputation_change'] * 0.1
            
        if result['trade_offered']:
            reward += 0.2
            
        # Negative rewards
        if result['combat_initiated'] and not context.get('hostile_situation', False):
            reward -= 0.5  # Unprovoked aggression
            
        if result['reputation_change'] < 0:
            reward += result['reputation_change'] * 0.1
            
        # Role-based rewards
        agent_role = self.agent.role.value
        if agent_role == 'leader' and result['action_taken'] in ['help', 'greet']:
            reward += 0.1
        elif agent_role == 'explorer' and result['action_taken'] == 'patrol':
            reward += 0.1
            
        return reward
        
    def get_personality_description(self) -> str:
        """Get a text description of the NPC's personality."""
        traits = self.personality_traits
        
        desc_parts = []
        
        if traits.get('aggressiveness', 0) > 0.7:
            desc_parts.append("aggressive")
        elif traits.get('aggressiveness', 0) < 0.3:
            desc_parts.append("peaceful")
            
        if traits.get('friendliness', 0) > 0.7:
            desc_parts.append("friendly")
        elif traits.get('friendliness', 0) < 0.3:
            desc_parts.append("aloof")
            
        if traits.get('curiosity', 0) > 0.7:
            desc_parts.append("curious")
        elif traits.get('curiosity', 0) < 0.3:
            desc_parts.append("disinterested")
            
        if traits.get('cautiousness', 0) > 0.7:
            desc_parts.append("cautious")
        elif traits.get('cautiousness', 0) < 0.3:
            desc_parts.append("reckless")
            
        return f"This {self.npc_type} appears {', '.join(desc_parts)}."

class GameWorld:
    """Example game world with EBAIF NPCs."""
    
    def __init__(self):
        self.npcs: Dict[str, GameNPC] = {}
        self.player_state = {
            'health': 100,
            'level': 5,
            'reputation': 50,
            'location': 'town_square'
        }
        self.logger = Logger.get_logger("GameWorld")
        
    async def initialize(self):
        """Initialize the game world."""
        # Create some NPCs
        npc_types = ['guard', 'merchant', 'villager', 'captain']
        
        for i, npc_type in enumerate(npc_types):
            npc = GameNPC(f"{npc_type}_{i}", npc_type)
            await npc.initialize()
            self.npcs[npc.npc_id] = npc
            
        self.logger.info(f"Initialized game world with {len(self.npcs)} NPCs")
        
    async def player_interacts_with_npc(self, npc_id: str, 
                                       interaction_context: Dict = None) -> Dict:
        """Handle player interaction with an NPC."""
        if npc_id not in self.npcs:
            return {'error': 'NPC not found'}
            
        context = interaction_context or {}
        context.update({
            'time_of_day': 14,  # 2 PM
            'location_danger': 2,
            'recent_aggression': 0
        })
        
        npc = self.npcs[npc_id]
        result = await npc.interact_with_player(self.player_state, context)
        
        # Update player state based on interaction
        if 'new_player_state' in result:
            self.player_state.update(result['new_player_state'])
            
        if result.get('reputation_change', 0) != 0:
            self.player_state['reputation'] += result['reputation_change']
            self.player_state['reputation'] = max(0, min(100, self.player_state['reputation']))
            
        return result
        
    def get_npc_descriptions(self) -> Dict[str, str]:
        """Get personality descriptions for all NPCs."""
        return {
            npc_id: npc.get_personality_description() 
            for npc_id, npc in self.npcs.items()
        }

async def demo_game_integration():
    """Demonstrate game integration."""
    print("🎮 EBAIF Game Integration Demo")
    print("=" * 40)
    
    # Initialize game world
    world = GameWorld()
    await world.initialize()
    
    # Show initial NPC personalities
    print("\n📝 Initial NPC Personalities:")
    descriptions = world.get_npc_descriptions()
    for npc_id, desc in descriptions.items():
        print(f"  {npc_id}: {desc}")
    
    # Simulate player interactions
    print("\n🎭 Player Interactions:")
    
    interactions = [
        ('guard_0', {'hostile_situation': False, 'player_needs_help': False}),
        ('merchant_1', {'player_has_items': True, 'hostile_situation': False}),
        ('villager_2', {'player_needs_help': True, 'hostile_situation': False}),
        ('captain_3', {'hostile_situation': False, 'player_has_items': False}),
        ('guard_0', {'hostile_situation': True, 'recent_aggression': 5}),  # Second interaction with guard
    ]
    
    for npc_id, context in interactions:
        result = await world.player_interacts_with_npc(npc_id, context)
        
        print(f"\n  Player -> {npc_id}:")
        print(f"    Action: {result.get('action_taken', 'unknown')}")
        print(f"    Dialogue: \"{result.get('dialogue', 'silence')}\"")
        print(f"    Reputation change: {result.get('reputation_change', 0)}")
        
        if result.get('combat_initiated'):
            print(f"    ⚔️ Combat initiated!")
        if result.get('trade_offered'):
            print(f"    💰 Trade offered!")
    
    print(f"\n📊 Final player reputation: {world.player_state['reputation']}")
    
    # Show how personalities might have evolved
    print("\n🧬 NPC Evolution Status:")
    for npc_id, npc in world.npcs.items():
        status = npc.agent.get_status()
        print(f"  {npc_id}:")
        print(f"    Interactions: {npc.player_interactions}")
        print(f"    Evolution count: {status['performance_metrics']['evolution_count']}")
        print(f"    Role: {status['role']}")
    
    print("\n✅ Game integration demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_game_integration())