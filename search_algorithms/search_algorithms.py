import heapq
import numpy as np


class SearchAlgorithms:
    """
    This class implements the various search algorithms
    """
    def __init__(self, problem, heuristic):
        self.problem = problem
        self.heuristic = heuristic
        self.expanded_nodes = 0

    def astar(self, initial_state):
        """
        This function performs the A-star search
        :param initial_state:
        :return:
        """
        open_list = []
        closed_set = set()
        backtrack = {}

        initial_cost = 0
        initial_heuristic = self.heuristic.estimate(initial_state)
        initial_f_score = initial_cost + initial_heuristic

        heapq.heappush(open_list,
                       (initial_f_score, initial_cost, tuple(initial_state)))
        backtrack[tuple(initial_state)] = (None, initial_cost)

        while open_list:
            current_f_score, current_cost, current_state = heapq.heappop(
                open_list)

            if self.problem.is_goal_state(np.array(current_state)):
                return self.reconstruct_path(backtrack, current_state)

            closed_set.add(current_state)
            self.expanded_nodes += 1

            for action in self.problem.get_actions(np.array(current_state)):
                successor_state = self.problem.get_successor(
                    np.array(current_state), action)
                successor_tuple = tuple(successor_state)

                if successor_tuple in closed_set:
                    continue

                successor_cost = current_cost + 1
                successor_heuristic = self.heuristic.estimate(successor_state)
                successor_f_score = successor_cost + successor_heuristic

                if successor_tuple not in backtrack or successor_cost < \
                        backtrack[successor_tuple][1]:
                    backtrack[successor_tuple] = (
                    current_state, successor_cost)
                    heapq.heappush(open_list, (
                    successor_f_score, successor_cost, successor_tuple))

        return None

    def gbfs(self, initial_state):
        """
        This function performs the Greedy Best-First Search aka GBFS
        :param initial_state:
        :return:
        """
        open_list = []
        closed_set = set()
        backtrack = {}

        initial_heuristic = self.heuristic.estimate(initial_state)

        heapq.heappush(open_list, (initial_heuristic, tuple(initial_state)))
        backtrack[tuple(initial_state)] = (None, 0)

        while open_list:
            current_heuristic, current_state = heapq.heappop(open_list)

            if self.problem.is_goal_state(np.array(current_state)):
                return self.reconstruct_path(backtrack, current_state)

            closed_set.add(current_state)
            self.expanded_nodes += 1

            for action in self.problem.get_actions(np.array(current_state)):
                successor_state = self.problem.get_successor(
                    np.array(current_state), action)
                successor_tuple = tuple(successor_state)

                if successor_tuple in closed_set:
                    continue

                successor_heuristic = self.heuristic.estimate(successor_state)

                if successor_tuple not in backtrack:
                    backtrack[successor_tuple] = (current_state, 0)
                    heapq.heappush(open_list,
                                   (successor_heuristic, successor_tuple))

        return None

    def weighted_astar(self, initial_state, weight):
        """
        Performs the weighted A-star search
        :param initial_state:
        :param weight:
        :return:
        """
        open_list = []
        closed_set = set()
        backtrack = {}

        initial_cost = 0
        initial_heuristic = self.heuristic.estimate(initial_state)
        initial_f_score = initial_cost + weight * initial_heuristic

        heapq.heappush(open_list,
                       (initial_f_score, initial_cost, tuple(initial_state)))
        backtrack[tuple(initial_state)] = (None, initial_cost)

        while open_list:
            current_f_score, current_cost, current_state = heapq.heappop(
                open_list)

            if self.problem.is_goal_state(np.array(current_state)):
                return self.reconstruct_path(backtrack, current_state)

            closed_set.add(current_state)
            self.expanded_nodes += 1

            for action in self.problem.get_actions(np.array(current_state)):
                successor_state = self.problem.get_successor(
                    np.array(current_state), action)
                successor_tuple = tuple(successor_state)

                if successor_tuple in closed_set:
                    continue

                successor_cost = current_cost + 1
                successor_heuristic = self.heuristic.estimate(successor_state)
                successor_f_score = successor_cost + weight * \
                                    successor_heuristic

                if successor_tuple not in backtrack or successor_cost < \
                        backtrack[successor_tuple][1]:
                    backtrack[successor_tuple] = (
                    current_state, successor_cost)
                    heapq.heappush(open_list, (
                    successor_f_score, successor_cost, successor_tuple))

        return None

    def reconstruct_path(self, backtrack, goal_state):
        """
        Reconstruct the path from the initial state to the goal state
        :param backtrack:
        :param goal_state:
        :return:
        """

        path = []
        current_state = goal_state

        while current_state is not None:
            path.append(current_state)
            current_state = backtrack[current_state][0]

        path.reverse()
        return path
