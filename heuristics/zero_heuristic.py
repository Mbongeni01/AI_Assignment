from heuristics.heuristic import Heuristic


class ZeroHeuristic(Heuristic):
    """ This class returns a zero heurtistic """
    def estimate(self, state):
        return 0