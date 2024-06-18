import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BayesianNN(nn.Module):
    """
    This class is a Bayesian Neural Network (BNN) for heuristic estimation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_bnn_model(data_path, input_dim, hidden_dim,
                    output_dim, epochs=100, use_gpu=True):
    """
    This function trains the BNN model
    :param data_path:
    :param input_dim:
    :param hidden_dim:
    :param output_dim:
    :param epochs:
    :param use_gpu:
    :return:
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model = BayesianNN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Load training data
    x_train = torch.tensor(np.load(f"{data_path}_x.npy"),
                           dtype=torch.float32).to(device)
    y_train = torch.tensor(np.load(f"{data_path}_y.npy"),
                           dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

    # Save the trained model
    model_save_path = f"{data_path}_model.pt"
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    # Train BNN models for all domains
    domains = [
        ('data/15_puzzle', 16, 50, 1),
        ('data/24_puzzle', 25, 50, 1),
        ('data/24_pancake', 24, 50, 1),
        ('data/15_blocksworld', 15, 50, 1)
    ]

    for data_path, input_dim, hidden_dim, output_dim in domains:
        train_bnn_model(data_path, input_dim, hidden_dim, output_dim, use_gpu=True)
