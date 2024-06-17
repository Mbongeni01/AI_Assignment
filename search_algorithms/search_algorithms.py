import heapq


class SearchAlgorithms:
    def __init__(self, problem, heuristic):
        self.problem = problem
        self.heuristic = heuristic
        self.expanded_nodes = 0

    def astar(self, initial_state):
        open_list = []
        closed_set = set()
        backtrack = {}

        initial_cost = 0
        initial_heuristic = self.heuristic.estimate(initial_state)
        initial_f_score = initial_cost + initial_heuristic

        heapq.heappush(open_list, (initial_f_score, initial_cost, initial_state))
        backtrack[tuple(initial_state)] = None

        while open_list:
            current_f_score, current_cost, current_state = heapq.heappop(open_list)

            if self.problem.is_goal_state(current_state):
                return self.reconstruct_path(backtrack, current_state)

            closed_set.add(tuple(current_state))
            self.expanded_nodes += 1

            for action in self.problem.get_actions(current_state):
                successor_state = self.problem.get_successor(current_state, action)
                successor_tuple = tuple(successor_state)

                if successor_tuple in closed_set:
                    continue

                successor_cost = current_cost + 1
                successor_heuristic = self.heuristic.estimate(successor_state)
                successor_f_score = successor_cost + successor_heuristic

                if successor_tuple not in backtrack or successor_f_score < backtrack[successor_tuple][0]:
                    backtrack[successor_tuple] = (current_state, action)
                    heapq.heappush(open_list, (successor_f_score, successor_cost, successor_state))

        return None

    def gbfs(self, initial_state):
        open_list = []
        closed_set = set()
        backtrack = {}

        initial_heuristic = self.heuristic.estimate(initial_state)

        heapq.heappush(open_list, (initial_heuristic, initial_state))
        backtrack[tuple(initial_state)] = None

        while open_list:
            current_heuristic, current_state = heapq.heappop(open_list)

            if self.problem.is_goal_state(current_state):
                return self.reconstruct_path(backtrack, current_state)

            closed_set.add(tuple(current_state))
            self.expanded_nodes += 1

            for action in self.problem.get_actions(current_state):
                successor_state = self.problem.get_successor(current_state, action)
                successor_tuple = tuple(successor_state)

                if successor_tuple in closed_set:
                    continue

                successor_heuristic = self.heuristic.estimate(successor_state)

                if successor_tuple not in backtrack:
                    backtrack[successor_tuple] = (current_state, action)
                    heapq.heappush(open_list, (successor_heuristic, successor_state))

        return None

    def weighted_astar(self, initial_state, weight):
        open_list = []
        closed_set = set()
        backtrack = {}

        initial_cost = 0
        initial_heuristic = self.heuristic.estimate(initial_state)
        initial_f_score = initial_cost + weight * initial_heuristic

        heapq.heappush(open_list, (initial_f_score, initial_cost, initial_state))
        backtrack[tuple(initial_state)] = None

        while open_list:
            current_f_score, current_cost, current_state = heapq.heappop(open_list)

            if self.problem.is_goal_state(current_state):
                return self.reconstruct_path(backtrack, current_state)

            closed_set.add(tuple(current_state))
            self.expanded_nodes += 1

            for action in self.problem.get_actions(current_state):
                successor_state = self.problem.get_successor(current_state, action)
                successor_tuple = tuple(successor_state)

                if successor_tuple in closed_set:
                    continue

                successor_cost = current_cost + 1
                successor_heuristic = self.heuristic.estimate(successor_state)
                successor_f_score = successor_cost + weight * successor_heuristic

                if successor_tuple not in backtrack or successor_f_score < backtrack[successor_tuple][0]:
                    backtrack[successor_tuple] = (current_state, action)
                    heapq.heappush(open_list, (successor_f_score, successor_cost, successor_state))

        return None

    def reconstruct_path(self, backtrack, goal_state):
        path = []
        current_state = goal_state

        while current_state is not None:
            path.append(current_state)
            current_state = backtrack[tuple(current_state)][0]

        path.reverse()
        return path