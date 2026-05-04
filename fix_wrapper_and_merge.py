import sys
import re

filepath = 'src/ebaif/advanced/quantum_inspired.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Rename QuantumOptimizer back to QuantumInspiredOptimization (first occurrence)
# We assume implementation is QuantumOptimizer (from previous change)
if 'class QuantumOptimizer:' in content:
    content = content.replace('class QuantumOptimizer:', 'class QuantumInspiredOptimization:', 1)
    print("Renamed QuantumOptimizer back to QuantumInspiredOptimization.")

# 2. Add wrapper methods to the class
# We need to add `optimize` and `get_summary` to QuantumInspiredOptimization
# And modify `optimize` to call `hybrid_optimize` instead of `self.optimizer.hybrid_optimize`
# And modify `get_summary` to call `get_quantum_summary` instead of `self.optimizer.get_quantum_summary`

wrapper_methods = """
    async def optimize(self, fitness_function: callable, problem_size: int = 20) -> Dict[str, Any]:
        \"\"\"Run quantum-inspired optimization.\"\"\"
        return await self.hybrid_optimize(fitness_function)

    def get_summary(self) -> Dict[str, Any]:
        \"\"\"Get optimization summary.\"\"\"
        return self.get_quantum_summary()
"""

# Find end of class QuantumInspiredOptimization (before the wrapper class definition)
# We look for the last method `get_quantum_summary`
match = re.search(r'    def get_quantum_summary\(self\) -> Dict\[str, Any\]:', content)
if match:
    # Find end of method
    start_idx = match.start()
    lines = content[start_idx:].split('\n')
    method_lines = []
    method_lines.append(lines[0])

    end_idx = start_idx + len(lines[0]) + 1

    # Iterate to find end of method (next class or unindented)
    idx = 1
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == '':
            method_lines.append(line)
            idx += 1
            continue
        if line.startswith('class ') or (not line.startswith('        ') and not line.startswith('    ') and line.strip() != ''):
            break
        method_lines.append(line)
        idx += 1

    original_method = '\n'.join(method_lines)

    # Append new methods after get_quantum_summary
    new_code = original_method + wrapper_methods
    content = content.replace(original_method, new_code, 1)
    print("Added wrapper methods to implementation class.")

# 3. Remove the wrapper class definition at the end
# The wrapper class starts with `class QuantumInspiredOptimization:` (second occurrence now?)
# Or since we renamed the first one back, both are named QuantumInspiredOptimization.
# But regex replacement only replaced the first occurrence.
# So the wrapper is still named QuantumInspiredOptimization (or QuantumOptimizer if I messed up previously, but verify_fix.py used QuantumOptimizer).
# Wait, verify_fix.py used QuantumInspiredOptimization as wrapper and QuantumOptimizer as implementation.
# If I rename implementation back to QuantumInspiredOptimization, I have TWO classes with same name.
# Python will use the second one.
# So I must remove the second one.

# Find the second occurrence of `class QuantumInspiredOptimization:`
matches = list(re.finditer(r'class QuantumInspiredOptimization:', content))
if len(matches) >= 2:
    second_match = matches[1]
    # Remove everything from second match to end of file (or next class, but it's at the end)
    content = content[:second_match.start()]
    print("Removed duplicate wrapper class definition.")
else:
    # Maybe wrapper was named differently? Check if `class QuantumInspiredOptimization` exists only once now?
    # If I renamed implementation back, I should have two.
    # If I didn't rename yet (step 1 above), I have QuantumOptimizer and QuantumInspiredOptimization.
    # But step 1 runs first.
    pass

# Clean up trailing newlines
content = content.rstrip() + '\n'

with open(filepath, 'w') as f:
    f.write(content)

print("File updated successfully.")
