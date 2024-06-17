from abc import ABC, abstractmethod

class Heuristic(ABC):
    """
    This class
    """
    @abstractmethod
    def estimate(self, state):
        pass