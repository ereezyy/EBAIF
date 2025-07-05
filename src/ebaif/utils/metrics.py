"""
Performance Metrics

Simple metrics collection for EBAIF framework.
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

class Metrics:
    """Simple metrics collector for EBAIF."""
    
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.gauges: Dict[str, float] = {}
        self.histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] += value
        
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self.gauges[name] = value
        self.histories[name].append(value)
        
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return TimerContext(self, name)
        
    def add_timer(self, name: str, duration: float):
        """Add a timer measurement."""
        self.timers[name].append(duration)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timers': {}
        }
        
        # Calculate timer statistics
        for name, times in self.timers.items():
            if times:
                stats['timers'][name] = {
                    'count': len(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                }
                
        return stats
        
    def reset(self):
        """Reset all metrics."""
        self.counters.clear()
        self.timers.clear()
        self.gauges.clear()
        self.histories.clear()

class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: Metrics, name: str):
        self.metrics = metrics
        self.name = name
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.add_timer(self.name, duration)