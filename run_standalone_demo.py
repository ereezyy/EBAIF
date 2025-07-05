"""
Standalone WebContainer Demo

Runs a simplified version of the EBAIF demo that works with standard library only.
"""

import os
import sys
import random
import time

def run_standalone_demo():
    """Run a simplified demo of the EBAIF system using only standard library."""
    print("🚀 EBAIF Standalone Demo")
    print("=" * 50)
    print("Demonstrating emergent AI behavior with standard library Python only")
    
    # Create environment (6x6 grid with resources)
    print("\n🌍 Creating resource collection environment")
    grid_size = 6
    environment = [[random.random() for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Print environment
    print("\nInitial resource distribution:")
    for row in environment:
        print("  " + " ".join([f"{cell:.2f}" for cell in row]))
    
    # Create agents
    print(f"\n🤖 Creating 3 learning agents")
    
    class Agent:
        """Simple agent that can learn and evolve."""
        
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            self.resources_collected = 0
            self.moves = 0
            
            # Learning parameters
            self.exploration_rate = random.uniform(0.1, 0.5)
            self.learning_rate = random.uniform(0.01, 0.1)
            
            # Memory of good positions
            self.memory = {}
            
            print(f"  Agent {agent_id} created at position {self.position}")
            print(f"    Exploration rate: {self.exploration_rate:.2f}")
            print(f"    Learning rate: {self.learning_rate:.2f}")
        
        def decide_move(self):
            """Decide where to move."""
            # Exploration vs exploitation
            if random.random() < self.exploration_rate:
                # Random move (exploration)
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                dx, dy = random.choice(directions)
                
                x, y = self.position
                new_x = max(0, min(grid_size-1, x + dx))
                new_y = max(0, min(grid_size-1, y + dy))
                
                return (new_x, new_y)
            else:
                # Use memory (exploitation)
                best_pos = None
                best_value = -1
                
                # Check memory and adjacent cells
                for y in range(grid_size):
                    for x in range(grid_size):
                        pos = (x, y)
                        # Prefer closer positions
                        distance = abs(x - self.position[0]) + abs(y - self.position[1])
                        if distance <= 2:  # Only consider nearby cells
                            if pos in self.memory:
                                value = self.memory[pos] / (distance + 1)
                                if value > best_value:
                                    best_value = value
                                    best_pos = pos
                
                if best_pos and best_value > 0:
                    return best_pos
                
                # Default to random nearby move
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                dx, dy = random.choice(directions)
                x, y = self.position
                new_x = max(0, min(grid_size-1, x + dx))
                new_y = max(0, min(grid_size-1, y + dy))
                return (new_x, new_y)
        
        def collect_resources(self, environment):
            """Collect resources at current position."""
            x, y = self.position
            resources = environment[y][x]
            
            if resources > 0:
                self.resources_collected += resources
                environment[y][x] = 0  # Resource is collected
                
                # Remember this position
                self.memory[self.position] = resources
                
                return resources
            return 0
        
        def adapt(self, reward):
            """Adapt behavior based on rewards."""
            if reward > 0.5:
                # Good reward, reduce exploration
                self.exploration_rate = max(0.1, self.exploration_rate - self.learning_rate)
            else:
                # Poor reward, increase exploration
                self.exploration_rate = min(0.9, self.exploration_rate + self.learning_rate)
    
    agents = [Agent(f"agent_{i}") for i in range(3)]
    
    # Run simulation
    episodes = 5
    steps_per_episode = 10
    
    for episode in range(episodes):
        print(f"\n📈 Episode {episode+1}/{episodes}")
        
        # Reset environment
        environment = [[random.random() for _ in range(grid_size)] for _ in range(grid_size)]
        
        for agent in agents:
            agent.position = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
        
        episode_rewards = {agent.agent_id: 0 for agent in agents}
        
        for step in range(steps_per_episode):
            for agent in agents:
                # Decide where to move
                new_position = agent.decide_move()
                
                # Move
                agent.position = new_position
                agent.moves += 1
                
                # Collect resources
                reward = agent.collect_resources(environment)
                episode_rewards[agent.agent_id] += reward
                
                # Learn from outcome
                agent.adapt(reward)
                
                print(f"  {agent.agent_id} moved to {agent.position}, collected {reward:.2f}")
        
        print(f"  Episode rewards: {episode_rewards}")
        
        for agent in agents:
            print(f"  {agent.agent_id}: exploration rate now {agent.exploration_rate:.2f}")
    
    print("\n📊 Final Results:")
    print("-" * 30)
    
    for agent in agents:
        print(f"🤖 {agent.agent_id}:")
        print(f"   Total Resources: {agent.resources_collected:.2f}")
        print(f"   Final Exploration Rate: {agent.exploration_rate:.2f}")
        print(f"   Moves Made: {agent.moves}")
        print(f"   Memory Size: {len(agent.memory)}")
        print()
    
    print("✅ Demo Completed!")
    print()
    print("🎯 What This Demonstrated:")
    print("  ✅ Agents learning from environment")
    print("  ✅ Behavioral parameter evolution")
    print("  ✅ Memory-based decision making")
    print("  ✅ Adaptation based on performance")

if __name__ == "__main__":
    try:
        run_standalone_demo()
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()