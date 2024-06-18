import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BayesianNN(nn.Module):
    """
    This class
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_bnn_model(data_path, input_dim, hidden_dim, output_dim, epochs=100):
    model = BayesianNN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Load training data
    x_train = torch.tensor(np.load(f"{data_path}_x.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load(f"{data_path}_y.npy"), dtype=torch.float32)

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
    torch.save(model, f"{data_path}_model.pt")
    print(f"Model saved to {data_path}_model.pt")

