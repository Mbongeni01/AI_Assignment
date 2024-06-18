import numpy as np
import torch
import os


class DataGenerator:
    """
    This class generates the data required
    """
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def is_solvable(self, puzzle, problem_type):
        """
        Check if the given puzzle configuration is solvable.
        """
        inversions = 0
        if problem_type == "puzzle":
            # Remove the blank tile for inversion count
            puzzle = puzzle[puzzle != 0]
        for i in range(len(puzzle)):
            for j in range(i + 1, len(puzzle)):
                if puzzle[i] > puzzle[j]:
                    inversions += 1
        return inversions % 2 == 0

    def generate_problem_data(self, problem_type, problem_size, num_samples):
        """
        Generate problem data for the given problem type and size.
        """
        x_data = []
        y_data = []
        for _ in range(num_samples):
            if problem_type in ["puzzle", "pancake"]:
                while True:
                    # Generate a random puzzle state
                    puzzle = np.random.permutation(problem_size)
                    if self.is_solvable(puzzle, problem_type):
                        break
            else:
                # Generate a random 15-blocksworld state
                puzzle = np.random.permutation(problem_size)
            # Placeholder heuristic value
            heuristic_value = np.random.randint(0, 100)
            x_data.append(puzzle)
            y_data.append(heuristic_value)
        return np.array(x_data), np.array(y_data)

    def save_data(self, x_data, y_data, filename):
        """
        Save the generated data to a file.
        """
        x_path = os.path.join(self.output_dir, f"{filename}_x.npy")
        y_path = os.path.join(self.output_dir, f"{filename}_y.npy")
        np.save(x_path, x_data)
        np.save(y_path, y_data)
        print(f"Data saved: {x_path}, {y_path}")

    def generate_and_save_all(self, num_samples=1000):
        """
        Generate and save data for all problem types and sizes.
        """
        problems = [
            ("puzzle", 16, "15_puzzle"),
            ("puzzle", 25, "24_puzzle"),
            ("pancake", 24, "24_pancake"),
            ("blocksworld", 15, "15_blocksworld")
        ]

        for problem_type, problem_size, filename in problems:
            x_data, y_data = self.generate_problem_data(problem_type,
                                                        problem_size,
                                                        num_samples)
            self.save_data(x_data, y_data, filename)


if __name__ == "__main__":
    data_generator = DataGenerator()
    data_generator.generate_and_save_all(1000)