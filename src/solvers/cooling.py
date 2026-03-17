from abc import ABC, abstractmethod
import math

class SACoolingBase(ABC):
    # abstract base class for Simulated Annealing cooling schedules. ensures a consistent interface for different cooling strategies.
    
    @abstractmethod
    def get_temperature(self, current_temp: float, step: int, initial_temp: float) -> float:
        """
        calculates the temperature for the next iteration
        
        Args:
            current_temp: temperature from the previous step
            step: current iteration count
            initial_temp: starting temperature of the SA algorithm
        """
        pass

class GeometricCooling(SACoolingBase):

    # geometric cooling: T_next = T_curr * alpha, this is the most common strategy where temperature decreases by a fixed percentage.

    def __init__(self, cooling_rate: float = 0.995):
        self.alpha = cooling_rate

    def get_temperature(self, current_temp: float, step: int, initial_temp: float) -> float:
        return current_temp * self.alpha

class LinearCooling(SACoolingBase):
    # linear cooling: T_next = T_initial * (1 - step / max_steps),the temperature decreases linearly toward zero over a fixed number of steps.
  
    def __init__(self, max_steps: int = 10000):
        self.max_steps = max_steps

    def get_temperature(self, current_temp: float, step: int, initial_temp: float) -> float:
        # prevents temperature from becoming zero or negative
        progress = step / self.max_steps
        return max(1e-10, initial_temp * (1 - progress))

class LogarithmicCooling(SACoolingBase):
    # logarithmic cooling: T_next = T_initial / log(step + 2), very slow cooling schedule that is theoretically guaranteed to find the global optimum but often too slow for practical use.
    
    def get_temperature(self, current_temp: float, step: int, initial_temp: float) -> float:
        # step + 2 is used to avoid division by log(1)=0 and log(0)
        return initial_temp / math.log(step + 2)