import numpy as np
from .problem import Problem


class Blocksworld(Problem):
    def __init__(self, size=8):
        self.size = size
        self.goal_state = np.arange(size)

    def get_actions(self, state):
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    actions.append((i, j))
        return actions

    def get_successor(self, state, action):
        source, target = action
        successor = state.copy()
        successor[source], successor[target] = successor[target], successor[source]
        return successor

    def is_goal_state(self, state):
        return np.array_equal(state, self.goal_state)

    def get_goal_state(self):
        return self.goal_state

    def parse_state(self, state_str):
        return np.array([int(x) for x in state_str.split(',')])