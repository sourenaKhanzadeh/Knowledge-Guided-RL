import numpy as np
import pygame
import time


class MazeEnvironment:
    def __init__(self, width, height, start, goal, walls):
        """
        Initialize the maze environment.
        :param width: Width of the maze
        :param height: Height of the maze
        :param start: Starting position (x, y)
        :param goal: Goal position (x, y)
        :param walls: List of wall positions [(x1, y1), (x2, y2), ...]
        """
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.walls = walls
        self.state = start

        self.cell_size = 40  # Size of each cell in pixels
        self.window_size = (width * self.cell_size, height * self.cell_size)
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Maze Solver")

        # Initialize Pygame font
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 18)  # You can choose any available font and size
        self.text_color = (255, 0, 0)  # Red color

        # Define colors
        self.bg_color = (255, 255, 255)  # White
        self.wall_color = (0, 0, 0)  # Black
        self.agent_color = (0, 0, 255)  # Blue
        self.goal_color = (0, 255, 0)  # Green

    def reset(self):
        """
        Reset the environment to the starting state.
        """
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Take an action and return the next state, reward, and done flag.
        :param action: Action to take (0: up, 1: right, 2: down, 3: left)
        """
        x, y = self.state
        next_state = self.state
        if action == 0:   # Up
            next_state = (x, max(y - 1, 0))
        elif action == 1: # Right
            next_state = (min(x + 1, self.width - 1), y)
        elif action == 2: # Down
            next_state = (x, min(y + 1, self.height - 1))
        elif action == 3: # Left
            next_state = (max(x - 1, 0), y)

        # Check for walls
        if next_state in self.walls:
            next_state = self.state

        # Update state
        self.state = next_state

        # Check for goal
        reward = 1 if self.state == self.goal else 0
        done = self.state == self.goal
        return next_state, reward, done

    @staticmethod
    def sample_action():
        """
        Sample a random action.
        """
        return np.random.choice([0, 1, 2, 3])

    def render(self, episode_number, start_time):
        self.screen.fill(self.bg_color)
        # Render episode number
        episode_text = f"Episode: {episode_number}"
        text_surface = self.font.render(episode_text, True, self.text_color)
        self.screen.blit(text_surface, (10, 10))  # Position the text at the top-left corner

        # Render elapsed time
        elapsed_time = time.time() - start_time
        time_text = f"Time: {elapsed_time:.2f}s"
        time_surface = self.font.render(time_text, True, self.text_color)
        time_rect = time_surface.get_rect(topright=(self.window_size[0] - 10, 10))
        self.screen.blit(time_surface, time_rect)  # Position the text at the top-right corner

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.wall_color,
                             (wall[0] * self.cell_size, wall[1] * self.cell_size,
                              self.cell_size, self.cell_size))

        # Draw agent
        pygame.draw.rect(self.screen, self.agent_color,
                         (self.state[0] * self.cell_size, self.state[1] * self.cell_size,
                          self.cell_size, self.cell_size))

        # Draw goal
        pygame.draw.rect(self.screen, self.goal_color,
                         (self.goal[0] * self.cell_size, self.goal[1] * self.cell_size,
                          self.cell_size, self.cell_size))

        pygame.display.flip()


    @staticmethod
    def close():
        pygame.quit()
