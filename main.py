import argparse
from solver import Solver
from heuristics.learned_heuristic import LearnedHeuristic
from heuristics.zero_heuristic import ZeroHeuristic
from problems.puzzle import Puzzle
from problems.pancake import Pancake
from problems.blocksworld import Blocksworld


def run_experiment(problem_type, heuristic_type,
                   initial_state_str, search_algorithm, weight):
    if problem_type == 'puzzle':
        problem = Puzzle(size=4)
    elif problem_type == 'pancake':
        problem = Pancake(size=10)
    elif problem_type == 'blocksworld':
        problem = Blocksworld(size=15)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    if heuristic_type == 'learned':
        heuristic = LearnedHeuristic(f'data/{problem_type}_model.pt')
    elif heuristic_type == 'zero':
        heuristic = ZeroHeuristic()
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")

    solver = Solver(problem, heuristic, search_algorithm=search_algorithm,
                    weight=weight)
    initial_state = problem.parse_state(initial_state_str)
    solution = solver.solve(initial_state)
    print("Solution:", solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run heuristic search '
                                                 'experiments.')
    parser.add_argument('--problem_type', type=str, required=True,
                        choices=['puzzle', 'pancake', 'blocksworld'],
                        help='Type of problem to solve.')
    parser.add_argument('--heuristic_type', type=str, required=True,
                        choices=['learned', 'zero'],
                        help='Type of heuristic to use.')
    parser.add_argument('--initial_state', type=str, required=True,
                        help='Initial state of the problem '
                             'as a comma-separated string.')
    parser.add_argument('--search_algorithm', type=str, default='astar',
                        choices=['astar', 'gbfs', 'weighted_astar'],
                        help='Search algorithm to use.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='Weight for weighted A* search.')

    args = parser.parse_args()
    run_experiment(args.problem_type, args.heuristic_type,
                   args.initial_state, args.search_algorithm, args.weight)
