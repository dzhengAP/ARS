"""
Energy measurement utilities
"""

import time


class EnergyMeter:
    """
    Simple energy meter based on time and average power consumption.
    
    This is a simplified approach. For production, consider using:
    - NVIDIA SMI for GPU power
    - Intel RAPL for CPU power
    - Specialized power measurement hardware
    """
    
    def __init__(self, avg_power_w: float = 6.0):
        """
        Initialize energy meter.
        
        Args:
            avg_power_w: Average power consumption in watts
        """
        self.avg_power_w = avg_power_w
        self._last_energy = 0.0
        self._start_time = 0.0
    
    def __enter__(self):
        """Start timing on context entry"""
        self._start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Calculate energy on context exit"""
        elapsed_time = time.time() - self._start_time
        self._last_energy = elapsed_time * self.avg_power_w
    
    def read_last(self) -> float:
        """
        Get energy consumption from last measurement.
        
        Returns:
            Energy in joules
        """
        return self._last_energy
