from abc import ABC, abstractmethod


class Heuristic(ABC):
    """ This class is  an abstract base
     for heuristics functions"""
    @abstractmethod
    def estimate(self, state):
        pass