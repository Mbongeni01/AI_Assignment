import numpy as np
from .problem import Problem

class Puzzle(Problem):
    def __init__(self, size=4):
        # Initialize the goal state
        self.size = size
        self.goal_state = np.arange(size * size)

    def get_actions(self, state):
        """

        :param state:
        :return:
        """
        zero_index = np.where(state == 0)[0][0]
        actions = []

        if zero_index >= self.size:
            actions.append('up')
        if zero_index < self.size * (self.size - 1):
            actions.append('down')
        if zero_index % self.size > 0:
            actions.append('left')
        if zero_index % self.size < self.size - 1:
            actions.append('right')

        return actions

    def get_successor(self, state, action):
        """
        Returns the successor given teh state and action
        :param state:
        :param action:
        :return:
        """
        zero_index = np.where(state == 0)[0][0]
        successor = state.copy()

        if action == 'up':
            swap_index = zero_index - self.size
        elif action == 'down':
            swap_index = zero_index + self.size
        elif action == 'left':
            swap_index = zero_index - 1
        elif action == 'right':
            swap_index = zero_index + 1
        else:
            raise ValueError(f'Unknown action: {action}')

        successor[zero_index], successor[swap_index] = successor[swap_index], successor[zero_index]
        return successor

    def is_goal_state(self, state):
        """
        Checks if the given state is the goal state
        :param state:
        :return:
        """
        return np.array_equal(state, self.goal_state)

    def get_goal_state(self):
        return self.goal_state

    def parse_state(self, state_str):
        return np.array([int(x) for x in state_str.split(',')])
