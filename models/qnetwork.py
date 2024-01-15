import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the Q-Network.
        :param input_size: The number of input features (size of the state space)
        :param output_size: The number of output features (size of the action space)
        """
        super(QNetwork, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor
        """
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function after second layer
        x = self.fc3(x)  # Output layer
        return x
