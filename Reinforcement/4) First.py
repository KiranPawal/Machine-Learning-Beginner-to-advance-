"""
Module 1 — Foundations of RL
Practical: Value Iteration on a 4x4 Gridworld

This solves the Bellman optimality equation directly:
    V*(s) = max_a  sum_s' P(s'|s,a) [ R(s,a,s') + gamma * V*(s') ]
"""

import numpy as np

# ---------------------------------------------------------------------
# 1. Define the Gridworld MDP
# ---------------------------------------------------------------------
# 4x4 grid. States are (row, col). Two terminal states: top-left (0,0)
# and bottom-right (3,3), both give reward 0. Every other move costs -1.
# This is the classic "get to the goal in as few steps as possible" task.

GRID_SIZE = 4
GAMMA = 1.0          # no discounting -- we just want shortest path
TERMINAL_STATES = {(0, 0), (3, 3)}
ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

                                                  # This creates all possible states.
def all_states():
    return [(r, c) for r in range(GRID_SIZE) 
            for c in range(GRID_SIZE)]
'''                                             [
                                                (0,0), (0,1), (0,2), (0,3),
                                                (1,0), (1,1), (1,2), (1,3),
                                                (2,0), (2,1), (2,2), (2,3),
                                                (3,0), (3,1), (3,2), (3,3)
                                                ]'''

def step(state, action):
    '''it determines: If the agent is in a state and performs an action, 
                                    where will it go and what reward will it receive?'''
    if state in TERMINAL_STATES:
        return state, 0              # Terminal State Check: If the agent is already in a terminal state, it stays there and receives a reward of 0.

    dr, dc = ACTION_DELTAS[action]       # Get Movement         movine raw and column
    r, c = state                         # Current Position            state = raw and column of the agent 
    nr, nc = r + dr, c + dc              # Calculate Next Position nr= next raw, nc = next column

    # stay in 3*3 bounds
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        nr, nc = r, c                                                  # This prevents the agent from leaving the grid.

    reward = -1                               # Every movement costs -1.
    return (nr, nc), reward


# ---------------------------------------------------------------------
# 2. Value Iteration: Repeatedly applies the Bellman optimality backup to every state
#    until the value function stops changing by more than `theta`.
# ---------------------------------------------------------------------
def value_iteration(theta=1e-4, max_iterations=1000):

    V = {s: 0.0 for s in all_states()}                     # Initially, every state has value: 0

    for iteration in range(max_iterations):
        delta = 0.0                           # delta stores the largest change in value during one iteration.
        new_V = V.copy()                  # This creates a copy.

        for s in all_states():
            if s in TERMINAL_STATES:              # Terminal states don't need updating.
                continue

            # Bellman optimality backup: try every action, keep the best.
            action_values = []                               # This will store the value of each possible action.
            for a in ACTIONS:
                s_next, reward = step(s, a)
                action_values.append(reward + GAMMA * V[s_next])

            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations (delta={delta:.6f})")
            break

    return V


def extract_policy(V):
    """A policy means: What action should the agent take in each state?"""
    policy = {}
    for s in all_states():
        if s in TERMINAL_STATES:
            policy[s] = "-"
            continue
        best_action, best_value = None, -float("inf")
        for a in ACTIONS:
            s_next, reward = step(s, a)
            value = reward + GAMMA * V[s_next]
            if value > best_value:
                best_value, best_action = value, a
        policy[s] = best_action
    return policy


# ---------------------------------------------------------------------
# 3. Run it and visualize the result
# ---------------------------------------------------------------------
def print_grid(values_or_policy, is_policy=False):
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">", "-": "*"}
    for r in range(GRID_SIZE):
        row_cells = []
        for c in range(GRID_SIZE):
            val = values_or_policy[(r, c)]
            if is_policy:
                row_cells.append(f" {arrows[val]} ")
            else:
                row_cells.append(f"{val:6.1f}")
        print(" | ".join(row_cells))
    print()


if __name__ == "__main__":
    print("Running Value Iteration on 4x4 Gridworld...\n")
    V_star = value_iteration()

    print("\nOptimal value function V*(s):")
    print_grid(V_star)

    policy = extract_policy(V_star)
    print("Optimal policy (greedy w.r.t. V*):")
    print_grid(policy, is_policy=True)

    print("Reading: every non-terminal cell's arrow shows the fastest")
    print("direction to a terminal state (top-left or bottom-right).")




