from setting import *
import torch
import numpy as np
import torch.nn as nn
import time
from knowledge.maze_domain import modified_reward, augment_state_with_distance_to_goal
import pygame.time


clock = pygame.time.Clock()
start_time = time.time()
for episode in range(num_episodes):
    state = env.reset()
    if WITH_KNOWLEDGE:
        state = augment_state_with_distance_to_goal(state, env.goal)
    done = False
    total_reward = 0

    while not done:
        if VISUALIZE:
            # and episode == num_episodes - number_episodes_to_watch:
            env.render(episode, start_time)
            clock.tick(10)

        # Exploration-exploitation trade-off
        if np.random.rand() < epsilon:
            action = env.sample_action()  # Random action
        else:
            state_tensor = torch.FloatTensor(state)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        # Take action and observe the outcome
        next_state, reward, done = env.step(action)
        if WITH_KNOWLEDGE:
            next_state = augment_state_with_distance_to_goal(next_state, env.goal)
            # reward = modified_reward(state, next_state, env.goal, reward, penalty_for_hitting_walls)
        total_reward += reward

        # Update Q-values using the Bellman Equation
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state)
            future_q_values = q_network(next_state_tensor)
            max_future_q = torch.max(future_q_values)
            target_q = reward + gamma * max_future_q

        # Calculate loss and backpropagate
        optimizer.zero_grad()
        current_q = q_network(torch.FloatTensor(state))[action]
        loss = nn.functional.mse_loss(current_q, target_q)
        loss.backward()
        optimizer.step()

        # Move to the next state
        state = next_state

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Optional: log the total reward of this episode
    print(f"Episode {episode}, Total Reward: {total_reward}")

print(f"Training took {(time.time() - start_time):.2f}s")
