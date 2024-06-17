import numpy as np
from .problem import Problem


class Pancake(Problem):
    def __init__(self, size=10):
        self.size = size
        self.goal_state = np.arange(size)

    def get_actions(self, state):
        return [i for i in range(2, self.size + 1)]

    def get_successor(self, state, action):
        successor = state.copy()
        successor[:action] = np.flip(successor[:action])
        return successor

    def is_goal_state(self, state):
        return np.array_equal(state, self.goal_state)

    def get_goal_state(self):
        return self.goal_state

    def parse_state(self, state_str):
        return np.array([int(x) for x in state_str.split(',')])
