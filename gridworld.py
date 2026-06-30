"""
Chapter 1: GridWorld MDP

A simple 4x4 grid environment implementing the MDP tuple (S, A, P, R, γ).
The agent starts at (0,0) and must reach the goal at (3,3), avoiding a wall at (1,1).

This is a deterministic environment — P(s'|s,a) is always 0 or 1.
"""

import numpy as np


class GridWorld:
    """
    4x4 grid MDP.

    States: (row, col) tuples, 0-indexed from top-left
    Actions: 0=up, 1=down, 2=left, 3=right
    """

    ACTIONS = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }
    ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

    def __init__(self, rows=4, cols=4, start=(0, 0), goal=(3, 3),
                 walls=None, step_reward=-0.01, goal_reward=1.0, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = walls or {(1, 1)}
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.gamma = gamma

        # Build the state set S — all cells minus walls
        self.states = [
            (r, c) for r in range(rows) for c in range(cols)
            if (r, c) not in self.walls
        ]
        self.n_states = len(self.states)
        self.n_actions = len(self.ACTIONS)

        # Current state
        self.state = start

    def reset(self):
        """Reset agent to start. Returns initial state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Take an action. Returns (next_state, reward, done).

        This is P(s'|s,a) and R(s,a,s') combined — the standard
        gym-style interface.
        """
        dr, dc = self.ACTIONS[action]
        new_r = self.state[0] + dr
        new_c = self.state[1] + dc

        # Boundary check and wall check — stay put if invalid
        if (0 <= new_r < self.rows and 0 <= new_c < self.cols
                and (new_r, new_c) not in self.walls):
            self.state = (new_r, new_c)

        # Check if we reached the goal
        done = self.state == self.goal
        reward = self.goal_reward if done else self.step_reward

        return self.state, reward, done

    def get_transition(self, state, action):
        """
        Pure transition function P(s'|s,a) — doesn't modify the environment.
        Returns (next_state, reward, done).

        Useful for planning algorithms (value iteration) that need to
        look ahead without actually taking steps.
        """
        dr, dc = self.ACTIONS[action]
        new_r = state[0] + dr
        new_c = state[1] + dc

        if (0 <= new_r < self.rows and 0 <= new_c < self.cols
                and (new_r, new_c) not in self.walls):
            next_state = (new_r, new_c)
        else:
            next_state = state

        done = next_state == self.goal
        reward = self.goal_reward if done else self.step_reward

        return next_state, reward, done

    def render(self, policy=None, values=None):
        """
        Print the grid. Optionally overlay a policy (arrows) or values.
        """
        arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

        print()
        for r in range(self.rows):
            row_top = ""
            row_bot = ""
            for c in range(self.cols):
                if (r, c) in self.walls:
                    row_top += "│ ██ "
                    row_bot += "│    "
                elif (r, c) == self.goal:
                    row_top += "│ G  "
                    row_bot += "│    "
                elif (r, c) == self.state:
                    row_top += "│ @  "
                    row_bot += "│    "
                elif policy is not None and values is not None:
                    row_top += f"│{values.get((r, c), 0):5.2f}"
                    row_bot += f"│  {arrows[policy[(r, c)]]}  "
                elif policy is not None:
                    row_top += f"│  {arrows[policy[(r, c)]]}  "
                    row_bot += "│    "
                elif values is not None:
                    row_top += f"│{values.get((r, c), 0):5.2f}"
                    row_bot += "│    "
                else:
                    row_top += "│    "
                    row_bot += "│    "

            print("┌────" * self.cols + "┐" if r == 0 else "├────" * self.cols + "┤")
            print(row_top + "│")
        print("└────" * self.cols + "┘")
        print()


def random_agent(env, max_steps=100):
    """A random agent — takes random actions. Our baseline."""
    state = env.reset()
    total_reward = 0
    discount = 1.0

    print(f"Start: {state}")
    for step in range(max_steps):
        action = np.random.randint(env.n_actions)
        next_state, reward, done = env.step(action)
        total_reward += discount * reward
        discount *= env.gamma

        print(f"  Step {step+1}: {env.ACTION_NAMES[action]:>5} → {next_state}  "
              f"r={reward:.2f}")

        if done:
            print(f"\nReached goal in {step+1} steps!")
            print(f"Discounted return: {total_reward:.4f}")
            return total_reward

    print(f"\nFailed to reach goal in {max_steps} steps.")
    print(f"Discounted return: {total_reward:.4f}")
    return total_reward


if __name__ == "__main__":
    env = GridWorld()

    print("=" * 50)
    print("GridWorld MDP")
    print("=" * 50)
    print(f"States:      {env.n_states}")
    print(f"Actions:     {env.n_actions}")
    print(f"Start:       {env.start}")
    print(f"Goal:        {env.goal}")
    print(f"Walls:       {env.walls}")
    print(f"γ (gamma):   {env.gamma}")
    print(f"Step reward:  {env.step_reward}")
    print(f"Goal reward:  {env.goal_reward}")

    print("\nThe grid:")
    env.render()

    print("=" * 50)
    print("Random agent episode")
    print("=" * 50)
    np.random.seed(42)
    random_agent(env)
