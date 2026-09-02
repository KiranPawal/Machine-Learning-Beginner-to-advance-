import gymnasium as gym                   #gymnasium provides reinforcement learning environments.
import numpy as np
import cv2

#create environment using make
cliffEnv=gym.make("CliffWalking-v1", render_mode="ansi")



def initialize_frame():
    width, height = 600, 200
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    margin_horizontal = 6
    margin_vertical = 2

    # Vertical lines
    for i in range(13):
        cv2.line(
            img,
            (49 * i + margin_horizontal, margin_vertical),
            (49 * i + margin_horizontal, 200 - margin_vertical),
            (0, 0, 0),
            1,
        )

    # Horizontal lines
    for i in range(5):
        cv2.line(
            img,
            (margin_horizontal, 49 * i + margin_vertical),
            (600 - margin_horizontal, 49 * i + margin_vertical),           #cv2.line(image, start_point, end_point, color, thickness)
            (0, 0, 0),
            1,
        )

    # Cliff
    cv2.rectangle(
        img,
        (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2),
        (255, 0, 255),
        -1,
    )

    cv2.putText(           # #cv2.line(image, start_point, end_point, color, thickness)
        img,
        "Cliff", 
        (49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),          
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        img,
        "G",
        (49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    return img


def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img



done=False                
frame=initialize_frame()

state, info=cliffEnv.reset()             

while not done:
    frame2=put_agent(frame.copy(),state)        
    cv2.imshow("Cliff Walking",frame2)
    cv2.waitKey(250)

    action=int(np.random.randint(0,4, size=1)[0])               

    next_state, reward, terminated, truncated, info=cliffEnv.step(action)
    state = next_state                  
    done = terminated or truncated           
cliffEnv.close()

