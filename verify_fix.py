import sys
import random
from unittest.mock import MagicMock

# Mock torch properly
torch_mock = MagicMock()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()

# Mock numpy
numpy_mock = MagicMock()
numpy_mock.var = MagicMock(return_value=0.0)
sys.modules['numpy'] = numpy_mock

# Mock consensus
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

try:
    sys.path.append('src')
    from ebaif.advanced.quantum_inspired import QuantumInspiredOptimization, QuantumConfig
    print("Import successful!")

    # Test instantiation
    config = QuantumConfig(population_size=100)
    optimizer = QuantumInspiredOptimization(config)
    print("Instantiation successful!")

    # Check internal optimizer (no longer exists)
    if hasattr(optimizer, 'optimizer'):
        print("Internal optimizer type: {type(optimizer.optimizer)}")
    else:
        print("No internal optimizer (wrapper merged)")

    # Check if initialize_population exists
    if hasattr(optimizer, 'initialize_population'):
        print("initialize_population exists")
        optimizer.initialize_population(100)
        print(f"Population initialized: {len(optimizer.population)}")
    else:
        print("Error: initialize_population missing!")

    # Check if evolve exists
    if hasattr(optimizer, 'evolve'):
        print("evolve exists")
    else:
        print("Error: evolve missing!")

    # Check optimize exists
    if hasattr(optimizer, 'optimize'):
        print("optimize exists")
    else:
        print("Error: optimize missing!")

    # Run precompute
    import time
    start = time.time()
    optimizer._precompute_selection_probabilities()
    end = time.time()
    print(f"Precompute took {end - start:.6f}s")
    print(f"Cum weights calculated: {len(optimizer._selection_cum_weights)}")

    # Run selection benchmark
    start = time.time()
    for _ in range(10000):
        optimizer._quantum_selection()
    end = time.time()
    duration = end - start
    print(f"10000 selections took {duration:.6f}s ({10000/duration:.2f} ops/s)")

    # Verify evolution step
    import asyncio
    async def run_evolution():
        new_pop = await optimizer._quantum_evolution_step()
        return new_pop

    start = time.time()
    new_pop = asyncio.run(run_evolution())
    end = time.time()
    print(f"Evolution step took {end - start:.6f}s for population 100")
    print(f"New population size: {len(new_pop)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
