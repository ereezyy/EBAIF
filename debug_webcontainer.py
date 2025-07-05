"""
Simplified WebContainer Demo

This script is a simplified version of the EBAIF demo that works
with standard library Python only in WebContainer.
"""

import os
import sys
import random
import time

print("🚀 Starting Simplified EBAIF Demo")
print("=" * 50)

# Check Python version
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

class SimpleAgent:
    """Very simple agent implementation using only standard library."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.position = (random.randint(0, 5), random.randint(0, 5))
        self.fitness = 0.0
        print(f"Agent {agent_id} initialized at position {self.position}")
    
    def move(self):
        """Move in a random direction."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = random.choice(directions)
        
        x, y = self.position
        new_x = max(0, min(5, x + dx))
        new_y = max(0, min(5, y + dy))
        
        self.position = (new_x, new_y)
        return self.position
    
    def collect_resource(self, resource_map):
        """Collect resource at current position."""
        x, y = self.position
        resource = resource_map[y][x]
        if resource > 0:
            self.fitness += resource
            resource_map[y][x] = 0
            return resource
        return 0

def run_demo():
    """Run a simplified demo of agents collecting resources."""
    # Create a simple 6x6 environment with resources
    environment = [[random.random() for _ in range(6)] for _ in range(6)]
    
    # Create agents
    agents = [SimpleAgent(f"agent_{i}") for i in range(3)]
    
    # Run simulation
    print("\n🌍 Running simulation...")
    for step in range(10):
        print(f"\n📊 Step {step+1}/10")
        
        # Each agent takes a turn
        for agent in agents:
            # Move
            position = agent.move()
            
            # Collect resource
            resource = agent.collect_resource(environment)
            
            print(f"  {agent.agent_id}: moved to {position}, collected {resource:.2f}")
        
        # Display environment (simple ASCII representation)
        print("\n  Environment state:")
        for row in environment:
            print("  " + " ".join([f"{cell:.1f}" for cell in row]))
    
    # Show final results
    print("\n🏆 Final Results:")
    for agent in agents:
        print(f"  {agent.agent_id}: Total fitness = {agent.fitness:.2f}")
    
    print("\n✅ Demo completed successfully!")
    print("This demonstrates the core concept of emergent behavior in a simple way.")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()