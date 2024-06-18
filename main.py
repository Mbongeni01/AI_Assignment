import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from solver import Solver
from heuristics.learned_heuristic import LearnedHeuristic
from heuristics.zero_heuristic import ZeroHeuristic
from problems.puzzle import Puzzle
from problems.pancake import Pancake
from problems.blocksworld import Blocksworld


def run_experiment(problem_type, heuristic_type,
                   initial_state_str, search_algorithm, weight):
    """
    This function runs experiments for the
    specified problem and heuristic type
    :param problem_type:
    :param heuristic_type:
    :param initial_state_str:
    :param search_algorithm:
    :param weight:
    :return:
    """
    if problem_type == 'puzzle':
        problem = Puzzle(size=4)
    elif problem_type == 'pancake':
        problem = Pancake(size=10)
    elif problem_type == 'blocksworld':
        problem = Blocksworld(size=15)
    else:
        raise ValueError(f"Unknown problem type: "
                         f"{problem_type}")

    if heuristic_type == 'learned':
        heuristic = LearnedHeuristic(
            f'data/{problem_type}_model.pt')
    elif heuristic_type == 'zero':
        heuristic = ZeroHeuristic()
    else:
        raise ValueError(f"Unknown heuristic type: "
                         f"{heuristic_type}")

    solver = Solver(problem, heuristic,
                    search_algorithm=search_algorithm,
                    weight=weight)

    initial_state = problem.parse_state(initial_state_str)

    start_time = time.time()
    solution = solver.solve(initial_state)
    end_time = time.time()

    elapsed_time = end_time - start_time
    expanded_nodes = solver.expanded_nodes

    return solution, expanded_nodes, elapsed_time

def main():
    parser = argparse.ArgumentParser(description=
                                     'Run heuristic search experiments.')
    parser.add_argument('--problem_type', type=str,
                        required=True,
                        choices=['puzzle', 'pancake', 'blocksworld'],
                        help='Type of problem to solve.')
    parser.add_argument('--heuristic_type', type=str,
                        required=True, choices=['learned', 'zero'],
                        help='Type of heuristic to use.')
    parser.add_argument('--initial_state', type=str, required=True,
                        help='Initial state of the problem as'
                             ' a comma-separated string.')
    parser.add_argument('--search_algorithm', type=str,
                        default='astar',
                        choices=['astar', 'gbfs', 'weighted_astar'],
                        help='Search algorithm to use.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='Weight for weighted A* search.')

    args = parser.parse_args()
    experiments = [{
        "problem_type": args.problem_type,
        "heuristic_type": args.heuristic_type,
        "initial_state": args.initial_state,
        "search_algorithm": args.search_algorithm,
        "weight": args.weight
    }]

    results = []
    for experiment in experiments:
        solution, expanded_nodes, elapsed_time = run_experiment(**experiment)
        result = {
            "experiment": experiment,
            "solution": solution,
            "expanded_nodes": expanded_nodes,
            "elapsed_time": elapsed_time
        }
        results.append(result)
        print(json.dumps(result, indent=4))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
