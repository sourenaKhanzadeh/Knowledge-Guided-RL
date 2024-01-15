import torch.optim as optim
from Envs.maze_env import MazeEnvironment
from models.qnetwork import QNetwork


WITH_KNOWLEDGE = False
VISUALIZE = True

width, height = 6, 6
start = (0, 0)
goal = (5, 5)
walls = [(1, 1), (1, 2), (2, 1), (2, 2)]  # Add more walls as needed

# Example of a more complex maze setup
# width, height = 15, 15  # Increased size of the maze
# start = (0, 0)
# goal = (14, 14)
# # Define walls in more complex patterns
# walls = [(1, i) for i in range(1, 14)] + [(i, 13) for i in range(1, 14)]
# walls += [(3, i) for i in range(3, 10)] + [(i, 3) for i in range(4, 8)]
# walls += [(7, i) for i in range(5, 14)] + [(i, 7) for i in range(8, 14)]
# walls += [(10, i) for i in range(1, 7)] + [(i, 10) for i in range(11, 14)]


input_size = 2  # (x, y) position
output_size = 4  # Number of actions

if WITH_KNOWLEDGE:
    input_size += 1  # Add distance to goal as an input feature

# Initialize environment, Q-network, and optimizer
env = MazeEnvironment(width, height, start, goal, walls)
q_network = QNetwork(input_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# Training hyperparameters
num_episodes = 1000
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
number_episodes_to_watch = 500
penalty_for_hitting_walls = -1

