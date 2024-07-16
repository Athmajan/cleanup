from cleanup import CleanupEnv
import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_image(img: np.ndarray, pause_time: float = 1):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")
    
    plt.figure()

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time) 


def resize_and_text(img, scale_factor, titleText):
    height, width, _ = img.shape
    size = (int(width * scale_factor), int(height * scale_factor))
    upscaled_frame = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    # Create a blank image with space for the title above the frame
    title_height = 50  # Adjust as needed
    new_height = size[1] + title_height
    combined_frame = np.zeros((new_height, size[0], 3), dtype=np.uint8)

    # Place the upscaled frame in the lower part of the combined frame
    combined_frame[title_height:new_height, 0:size[0]] = upscaled_frame

    # Add white background for the title
    cv2.rectangle(combined_frame, (0, 0), (size[0], title_height), (255, 255, 255), -1)

    # Add titleText with black font
    cv2.putText(combined_frame, titleText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    return combined_frame



env = CleanupEnv(num_agents=2, render=True)
env.reset()
grid = env.get_map_with_agents()
print(grid)
visualize_image(env.renderIMG())
# for i in range(1000):
#     rand_action = np.random.randint(8, size=2)
#     obs, rew, dones, info, = env.step(
#                         {('agent-' + str(j)): rand_action[j] for j in range(0, 2)})
#     grid = env.get_map_with_agents()
#     if i % 100 ==0:
#         np.save('customGrid_'+str(i)+'_.npy', grid)



   
fileName = 'customGrid_'+str(3*100)+'_.npy'
loaded_array = np.load(fileName)
print(loaded_array)
observations = env.reset_from_observation(loaded_array)
for simu_i in range(10):
    rand_action = np.random.randint(8, size=2)
    print(rand_action)
    obs, rew, dones, info, = env.step(
                    {('agent-' + str(j)): rand_action[j] for j in range(0, 2)})
    grid = env.get_map_with_agents()
    visualize_image(env.renderIMG())
    
    input("press enter for next")