import numpy as np


def modified_reward(state: np.ndarray, next_state: np.ndarray, goal: np.ndarray, original_reward, penalty_for_hitting_walls):
    if np.array_equal(next_state, state):  # The agent hit a wall
        return original_reward - penalty_for_hitting_walls
    elif np.array_equal(next_state, goal):  # The agent reached the goal
        return original_reward + 10  # Large reward for reaching the goal
    else:
        return original_reward  # Standard reward for normal moves


def augment_state_with_distance_to_goal(state, goal):
    """
    Augment the state with the Manhattan distance to the goal.
    :param state: Current state (x, y).
    :param goal: Goal position (x, y).
    :return: Augmented state.
    """
    x, y = state
    goal_x, goal_y = goal
    manhattan_distance = abs(x - goal_x) + abs(y - goal_y)
    augmented_state = np.array([x, y, manhattan_distance])
    return augmented_state
