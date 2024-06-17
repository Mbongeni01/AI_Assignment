from abc import ABC, abstractmethod


class Problem(ABC):
    """
    This class is an abstract base for problem domains
    """
    @abstractmethod
    def get_actions(self, state):
        """
        This method gets the list of possible actions
        for a state
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def get_successor(self, state, action):
        """
        Given a state and action, this function returns the next state
        :param state:
        :param action:
        :return:
        """
        pass

    @abstractmethod
    def is_goal_state(self, state):
        """
        This function is to check if the
        given state is a goal state
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def parse_state(self, state_str):
        """
        Parses a state from its string representation
        :param state_str:
        :return:
        """
        pass