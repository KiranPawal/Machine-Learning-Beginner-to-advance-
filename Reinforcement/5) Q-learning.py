'''
SARSA in Reinforcement Learning
    SARSA is a model-free, on-policy Reinforcement Learning algorithm used to learn the optimal action-value function \(Q(s,a)\).
    SARSA stands for:
        S → State
        A → Action
        R → Reward
        S → Next State
        A → Next Action

1. What does SARSA learn?
    SARSA learns the Q-value:
        Q(s,a) 
    This represents:
        How good is it to take action \(a\) when we are in state \(s\)?
        The agent learns through interaction with the environment.

2. SARSA Flow
    The sequence is:
        (S, A, R, S', A') 
    Step-by-step:
        Agent observes the current State (S)
        Agent selects an Action (A)
        Environment gives a Reward (R)
        Agent moves to Next State (S')
        Agent selects Next Action (A')
        Update the Q-value
    Visual representation
                    Current State (S)
                        ↓
                    Choose Action (A)
                        ↓
                    Take Action
                        ↓
                    Receive Reward (R)
                        ↓
                    Next State (S')
                        ↓
                    Choose Next Action (A')
                        ↓
                    Update Q-value

3. SARSA Update Formula
The SARSA update equation is:
    Q(S,A) <----- Q(S,A) + alpha [R + gamma Q(S',A') - Q(S,A)] 
    Where:
        Q(S,A) = Current Q-value
        alpha = Learning rate
        R = Reward
        gamma = Discount factor
        Q(S',A') = Q-value of the next state and next action

5. Simple Example
    Suppose:
    Current State = A
    Current Action = Right

    Reward = 10

    Next State = B
    Next Action = Up

    Assume:
        Q(A,Right)=5
        Q(B, Up) = 8 

    Learning rate:
        alpha = 0.1 

    Discount factor:
        gamma = 0.9 

    Calculate:
        Q(A,Right) = 5 + 0.1[10 + 0.9(8) - 5]  
        = 5 + 0.1[10 + 7.2 - 5] 
        = 5 + 0.1(12.2) 
        = 6.22 

    So:
        Q(A,Right)=6.22
'''


import numpy as np
import random


# ==========================
# Environment Configuration
# ==========================

ROWS = 4
COLS = 4

START = (0, 0)
GOAL = (3, 3)

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]


# ==========================
# Hyperparameters
# ==========================

alpha = 0.1        # Learning rate
gamma = 0.9        # Discount factor
epsilon = 0.1      # Exploration rate

episodes = 1000


# ==========================
# Q-Table
# ==========================

Q = np.zeros((ROWS, COLS, 4))


# ==========================
# Choose Action
# ε-Greedy Policy
# ==========================

def choose_action(state):

    row, col = state

    # Exploration
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)

    # Exploitation
    else:
        return np.argmax(Q[row, col])


# ==========================
# Take Action
# ==========================

def take_action(state, action):

    row, col = state

    if action == UP:
        row = max(row - 1, 0)

    elif action == DOWN:
        row = min(row + 1, ROWS - 1)

    elif action == LEFT:
        col = max(col - 1, 0)

    elif action == RIGHT:
        col = min(col + 1, COLS - 1)

    next_state = (row, col)

    # Check if goal is reached
    if next_state == GOAL:
        reward = 10
        done = True

    else:
        reward = -1
        done = False

    return next_state, reward, done


# ==========================
# SARSA Training
# ==========================

for episode in range(episodes):

    # Start state
    state = START

    # Choose initial action
    action = choose_action(state)

    total_reward = 0


    while True:

        # Take action
        next_state, reward, done = take_action(state, action)

        total_reward += reward


        # If goal reached
        if done:

            row, col = state

            # Update Q-value
            Q[row, col, action] = Q[row, col, action] + \
                alpha * (
                    reward -
                    Q[row, col, action]
                )

            break


        # Choose next action
        next_action = choose_action(next_state)


        # Current state
        row, col = state

        # Next state
        next_row, next_col = next_state


        # SARSA Update
        Q[row, col, action] = Q[row, col, action] + \
            alpha * (
                reward +
                gamma * Q[next_row, next_col, next_action]
                -
                Q[row, col, action]
            )


        # Move to next state and action
        state = next_state
        action = next_action


    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode + 1}, "
            f"Total Reward = {total_reward}"
        )


# ==========================
# Display Best Policy
# ==========================

print("\nBest Policy:")

symbols = {
    UP: "↑",
    DOWN: "↓",
    LEFT: "←",
    RIGHT: "→"
}


for row in range(ROWS):

    for col in range(COLS):

        if (row, col) == GOAL:
            print("G", end="  ")

        else:

            best_action = np.argmax(Q[row, col])

            print(symbols[best_action], end="  ")

    print()


# ==========================
# Display Q-table
# ==========================

print("\nQ-Table:")

print(Q)