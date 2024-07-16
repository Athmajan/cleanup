import argparse
import numpy as np
import os
import sys
import shutil

from cleanup import CleanupOnlyAgent, CleanupEnv
import matplotlib.pyplot as plt
from utility_funcs import make_video_from_rgb_imgs
import cv2
import io
from PIL import Image
import time
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

#####################################
parser = argparse.ArgumentParser(description='')

parser.add_argument(
    '--vid_path',
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
    help='Path to directory where videos are saved.')

parser.add_argument(
    '--env',
    type=str,
    default='cleanup',
    help='Name of the environment to rollout.')

parser.add_argument(
    '--render_type',
    type=str,
    default='pretty',
    help='Can be pretty or fast. Implications obvious.')

parser.add_argument(
    '--num_agents',
    type=int,
    default=2,
    help='Number of agents.')

parser.add_argument(
    '--fps',
    type=int,
    default=8,
    help='Number of frames per second.')

args = parser.parse_args()

#######################################

def visualize_image1(img: np.ndarray, pause_time: float = 3):
    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")
    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 

def visualize_image(img: np.ndarray, pause_time: float = 3, top_border: int = 14, right_border: int = 15, bottom_border: int = 15, left_border: int = 15):
    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")
    # Crop the image to remove the specified borders
    cropped_img = img[top_border:-bottom_border, left_border:-right_border]
    plt.imshow(cropped_img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close()

def rgb_to_grayscale(rgb_img):
    """
    Convert an RGB image to grayscale using the luminosity method.
    
    Parameters:
    rgb_img (np.ndarray): Input image of shape (28, 18, 3)
    
    Returns:
    np.ndarray: Grayscale image of shape (28, 18)
    """
    if rgb_img.shape != (26, 18, 3):
        raise ValueError("Input image must be of shape (28, 18, 3)")

    # Convert to grayscale using the luminosity method
    grayscale_img = 0.2989 * rgb_img[:, :, 0] + 0.5870 * rgb_img[:, :, 1] + 0.1140 * rgb_img[:, :, 2]
    
    return grayscale_img

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        # Convolutional layers
        # The input image is 26x18 with 1 channel (black and white)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)  # Output: 13x9
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)  # Output: 7x5
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # Output: 7x5
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: 7x5
        
        # Calculate the size after the convolutions
        self.size = 32 * 7 * 5  # Flatten the output from conv layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)  # Output 16 probabilities (8 for each agent)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        x = F.softmax(x.view(-1, 2, 8), dim=args.num_agents)  # Apply softmax along the last dimension for each agent
        return x



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Example usage
# policy = Policy().to(device)

# env = CleanupEnv(num_agents=args.num_agents, render=True)

# env.reset()

# rgb_arr = env.render_preprocess()
# BWImg = rgb_to_grayscale(rgb_arr)

# # Convert BWImg to a PyTorch tensor
# BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


# probs = policy(BWImg)



# actions = torch.argmax(probs, dim=2).detach().numpy()
# categorical_dist = D.Categorical(probs.view(-1, 8))  # Flatten the probabilities for sampling
# actions = categorical_dist.sample().view(args.num_agents, -1).detach().cpu().numpy()



# # actions = actions[0]

               


# print(actions)

