from heuristics.heuristic import Heuristic


class ZeroHeuristic(Heuristic):
    def estimate(self, state):
        return 0