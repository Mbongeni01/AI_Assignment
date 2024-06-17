from heuristics.heuristic import Heuristic
import torch


class LearnedHeuristic(Heuristic):
    def __init__(self, data_path):
        self.model = self.load_model(data_path)

    def load_model(self, data_path):
        model = torch.load(data_path)
        model.eval()
        return model

    def estimate(self, state):
        input_data = self.preprocess_state(state)
        with torch.no_grad():
            heuristic_value = self.model(input_data)
        return heuristic_value.item()