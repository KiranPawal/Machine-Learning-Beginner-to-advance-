import gymnasium as gym
import numpy as np

cliffEnv=gym.make("CliffWalking-v1")
q_table=np.zeros((48,4))

#create the policy for the action selection
def policy(state,explore=0.0):
    action=int(np.argmax(q_table[state]))
    if np.random.random()<=explore:
        action=int(np.random.randint(0,4))
    return action

#parameters
EPSILON=0.1
ALPHA=0.1
GAMMA=0.9
NUM_EPISODES=500

state,info=cliffEnv.reset()
action=policy(state,EPSILON)
done=False


for episode in range(NUM_EPISODES):
    state, info = cliffEnv.reset()
    state = int(state)
    action = policy(state, EPSILON)
    done = False

    while not done:

        next_state, reward, terminated, truncated, info = cliffEnv.step(action)

        next_state = int(next_state)
        reward = float(reward)

        done = terminated or truncated

        if done:
            target = reward
        else:
            next_action = policy(next_state, EPSILON)
            target = reward + GAMMA * q_table[next_state, next_action]

        # SARSA update
        q_table[state, action] += ALPHA * (
            target - q_table[state, action]
        )

        state = next_state
        if not done:
            action = next_action

cliffEnv.close()

print("Training completed!")
print(q_table)
