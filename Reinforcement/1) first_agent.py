import gymnasium as gym                   #gymnasium provides reinforcement learning environments.
import numpy as np

#create environment using make
cliffEnv=gym.make("CliffWalking-v1", render_mode="ansi")
done=False                #done indicates whether the episode has finished.
state, info=cliffEnv.reset()              #initialization: Resets the environment to the starting state.


while not done:
    print(cliffEnv.render())                #The agent's current position is shown in the text output.

    action=int(np.random.randint(0,4, size=1)[0])           #choose the random action
    print(state,"-->",["Up","right","down","left"][action])             #Current state number.         #Action selected.

    next_state, reward, terminated, truncated, info=cliffEnv.step(action)                 #receive rewars,next state, done, execute action
    done = terminated or truncated               #goal is reached (terminated)      #episode ends because of a time limit (truncated).
cliffEnv.close()

