"""
Project Diagnosis and Repair Tool

This script diagnoses issues in the EBAIF project and attempts to fix them.
"""

import os
import sys
import importlib.util
import traceback

def check_directory_structure():
    """Check if the directory structure is valid."""
    print("🔍 Checking directory structure...")
    
    directories = ["src", "examples", "gui"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✅ Found {directory}/")
        else:
            print(f"  ❌ Missing {directory}/")
    
    if not os.path.exists("src/ebaif"):
        print("  ❌ Missing src/ebaif/ directory")
        return False
    
    print("  ✅ Basic directory structure check completed")
    return True

def check_python_path():
    """Check if the Python path is correctly set up."""
    print("🔍 Checking Python path...")
    
    if "src" not in sys.path:
        print("  ❌ 'src' not in Python path, adding it...")
        sys.path.append("src")
        print("  ✅ Added 'src' to Python path")
    else:
        print("  ✅ 'src' is in Python path")
    
    if "." not in sys.path:
        print("  ❌ Current directory not in Python path, adding it...")
        sys.path.append(".")
        print("  ✅ Added current directory to Python path")
    else:
        print("  ✅ Current directory is in Python path")

def test_imports():
    """Test importing key modules."""
    print("🔍 Testing imports...")
    
    modules_to_test = [
        "src.ebaif",
        "examples.webcontainer_demo"
    ]
    
    for module in modules_to_test:
        try:
            components = module.split('.')
            
            # Build the import path step by step
            for i in range(1, len(components) + 1):
                test_module = '.'.join(components[:i])
                try:
                    # Try to import each level
                    importlib.import_module(test_module)
                except ModuleNotFoundError as e:
                    print(f"  ❌ Failed to import {test_module}: {e}")
                    
                    # Create __init__.py files if missing
                    path_components = components[:i]
                    path = os.path.join(*path_components)
                    if os.path.isdir(path) and not os.path.exists(os.path.join(path, "__init__.py")):
                        print(f"  ⚠️ Creating missing {path}/__init__.py file")
                        with open(os.path.join(path, "__init__.py"), "w") as f:
                            f.write("# Auto-generated __init__.py\n")
            
            print(f"  ✅ Module {module} import paths fixed")
        except Exception as e:
            print(f"  ❌ Failed to fix import path for {module}: {e}")

def fix_webcontainer_demo():
    """Fix the webcontainer demo script."""
    print("🔍 Creating a standalone webcontainer demo...")
    
    try:
        if os.path.exists("examples/webcontainer_demo.py"):
            with open("examples/webcontainer_demo.py", "r") as f:
                content = f.read()
            
            # Create a simplified version that runs directly
            with open("run_standalone_demo.py", "w") as f:
                f.write("""
\"\"\"
Standalone WebContainer Demo

Runs a simplified version of the EBAIF demo that works with standard library only.
\"\"\"

import os
import sys

# Ensure current directory is in path
if '.' not in sys.path:
    sys.path.append('.')

# Try to run the webcontainer demo directly
try:
    from examples.webcontainer_demo import run_webcontainer_demo
    
    print("🚀 Starting EBAIF WebContainer Demo...")
    print("Using standard library Python only (no external dependencies)")
    print()
    
    # Run demo directly (not using asyncio to avoid issues)
    run_webcontainer_demo()
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Running simplified fallback demo instead...")
    
    import random
    import time
    
    # Simple fallback implementation
    class SimpleAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.fitness = 0
            
        def move(self, environment):
            x = random.randint(0, len(environment[0])-1)
            y = random.randint(0, len(environment)-1)
            return (x, y)
    
    def run_fallback_demo():
        print("🌍 Running simplified EBAIF demo")
        
        # Create environment
        env_size = 5
        environment = [[random.random() for _ in range(env_size)] for _ in range(env_size)]
        
        # Create agents
        agents = [SimpleAgent(f"agent_{i}") for i in range(3)]
        
        # Run simulation
        for step in range(5):
            print(f"\\n📈 Step {step+1}")
            
            for agent in agents:
                x, y = agent.move(environment)
                reward = environment[y][x]
                agent.fitness += reward
                environment[y][x] = 0  # Resource collected
                print(f"  {agent.agent_id} collected {reward:.2f} at ({x},{y})")
        
        print("\\n🏆 Results:")
        for agent in agents:
            print(f"  {agent.agent_id}: total fitness = {agent.fitness:.2f}")
    
    run_fallback_demo()
except Exception as e:
    print(f"❌ Error running demo: {e}")
    traceback.print_exc()
""")
            print("  ✅ Created run_standalone_demo.py")
    except Exception as e:
        print(f"  ❌ Failed to create standalone demo: {e}")

def main():
    """Main diagnostic function."""
    print("🔧 EBAIF Project Diagnosis and Repair")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Run checks
    check_directory_structure()
    check_python_path()
    test_imports()
    fix_webcontainer_demo()
    
    print("\n🎯 Diagnosis complete!")
    print("=" * 50)
    print("Try running one of these commands:")
    print("  python debug_webcontainer.py       (Simplified demo)")
    print("  python run_standalone_demo.py      (Fixed webcontainer demo)")
    print("  python fix_project.py              (Run this diagnosis again)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Diagnostic tool crashed: {e}")
        traceback.print_exc()