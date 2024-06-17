import torch
from .heuristic import Heuristic


class LearnedHeuristic(Heuristic):
    """
    This class is the learned hueristic based on
    a learned model.
    """
    def __init__(self, data_path):
        # Initialize the learned heuristic with a trained model
        self.model = self.load_model(data_path)

    def load_model(self, data_path):
        # Load the model from the specified path
        model = torch.load(data_path)
        model.eval()
        return model

    def preprocess_state(self, state):
        # Process the state for the model
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def estimate(self, state):
        # Using the trained model estimate the
        # heuristic value for a given state
        input_data = self.preprocess_state(state)
        with torch.no_grad():
            heuristic_value = self.model(input_data)
        return heuristic_value.item()
