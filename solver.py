import torch
import numpy as np
from heuristics.learned_heuristic import LearnedHeuristic
from heuristics.zero_heuristic import ZeroHeuristic
from problems.puzzle import Puzzle
from problems.pancake import Pancake
from problems.blocksworld import Blocksworld
from search_algorithms.search_algorithms import SearchAlgorithms


class Solver:
    def __init__(self, problem, heuristic, search_algorithm='astar',
                 weight=1.0):
        self.problem = problem
        self.heuristic = heuristic
        self.search_algorithm = search_algorithm
        self.weight = weight

    def solve(self, initial_state):
        search_algorithms = SearchAlgorithms(self.problem, self.heuristic)
        if self.search_algorithm == 'astar':
            return search_algorithms.astar(initial_state)
        elif self.search_algorithm == 'gbfs':
            return search_algorithms.gbfs(initial_state)
        elif self.search_algorithm == 'weighted_astar':
            return search_algorithms.weighted_astar(initial_state, self.weight)
        else:
            raise ValueError(
                f"Unknown search algorithm: {self.search_algorithm}")


def load_data(data_path, problem_size):
    x_data = np.load(f"{data_path}_x.npy")
    y_data = np.load(f"{data_path}_y.npy")
    return torch.tensor(x_data, dtype=torch.float32), \
        torch.tensor(y_data,dtype=torch.float32)


def main():
    # Example usage
    problem = Puzzle(size=4)
    heuristic = LearnedHeuristic('data/15_puzzle_model.pt')
    solver = Solver(problem, heuristic, search_algorithm='astar')

    initial_state_str = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0'
    initial_state = problem.parse_state(initial_state_str)

    solution = solver.solve(initial_state)
    print("Solution:", solution)


if __name__ == "__main__":
    main()
