import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import cv2
import json


from cleanup import CleanupEnv
import argparse
import os
import sys
import shutil

import matplotlib.pyplot as plt
from utility_funcs import make_video_from_rgb_imgs
import io
from PIL import Image
import time
from tqdm import tqdm

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



# Define constants and configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        actions = self.output_layer(x)
        action_probs = F.softmax(actions, dim=1)
        return action_probs

# Function to preprocess observations
def preprocess_observations(obs, convert_to_gray=True):
    obs = np.array(obs, dtype=np.uint8)
    if len(obs.shape) != 3 or obs.shape[2] != 3:
        raise ValueError("Expected observation with shape (H, W, 3), but got shape {}".format(obs.shape))
    if convert_to_gray:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = obs.flatten()
    return obs / 255.0


def OHE_action(action):
    if len(action) != 2 or not all(0 <= x <= 8 for x in action):
        raise ValueError("Action must be a list of two elements, each between 1 and 9 (inclusive).")
    
    # Initialize the one-hot encoded vector with zeros
    oheAction = np.zeros(18, dtype=int)
    
    # Set the corresponding positions to 1
    oheAction[action[0] - 1] = 1
    oheAction[9 + action[1] - 1] = 1
    
    return oheAction

def set_action_positions(inputVector):
    pos1 = inputVector[0]
    pos2 = inputVector[1]
    if not (0 <= pos1 < 18) or not (0 <= pos2 < 18):
        raise ValueError("Positions must be in the range [0, 17]")

    vector = [0] * 18  # Initialize the 18-long vector with zeros
    vector[pos1] = 1   # Set the value at pos1 to 1
    vector[pos2] = 1   # Set the value at pos2 to 1
    
    return vector
    
def decode_OHE_action(ohe_action):
    '''Decodes a one-hot encoded action back into the original action elements.
    
    Args:
    - ohe_action (Array): A one-hot encoded array of length 18
    
    Returns:
    - (list): A list of two elements representing the original action
    '''
    if len(ohe_action) != 18 or not all(x in [0, 1] for x in ohe_action):
        raise ValueError("One-hot encoded action must be an array of length 18 with binary values (0 or 1).")
    
    # Find the positions of the 1s in the one-hot encoded vector
    pos1 = np.argmax(ohe_action[:9])
    pos2 = np.argmax(ohe_action[9:])
    
    return [pos1, pos2]
    


def create_actions_dict(action_list):
    if len(action_list) != 2:
        raise ValueError("Action list must contain exactly two elements.")

    actions_dict = {}
    for i, action_value in enumerate(action_list):
        agent_id = f"agent-{i}"
        actions_dict[agent_id] = int(action_value)

    return actions_dict


# Function to select actions given current state
def select_action(network, state):
    ''' Selects two actions given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment
    
    Return:
    - (tuple): two actions that are selected
    - (tuple): log probabilities of selecting those actions given state and network
    '''
    
    # Convert state to float tensor, add 1 dimension, allocate tensor on device
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    
    # Use network to predict action probabilities
    action_probs = network(state)
    state = state.detach()
    
    # Sample two actions using the probability distribution
    m = Categorical(action_probs)
    action1 = m.sample()
    action2 = m.sample()
    
    # Return actions and their log probabilities
    return (action1.item(), action2.item()), (m.log_prob(action1), m.log_prob(action2))

# Function to load saved models
def load_models(model_file):
    checkpoint = torch.load(model_file, map_location=DEVICE)
    policy_network = PolicyNetwork(450, 18).to(DEVICE)
    policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
    policy_network.eval()  # Set to evaluation mode for inference
    return policy_network

def create_movie_clip(frames: list, output_file: str, fps: int = 10, scale_factor: int = 1):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width * scale_factor, height * scale_factor)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        # Upscale the frame
        upscaled_frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
        out.write(cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR))
    
    out.release()


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

LENGTH_EPISODE = 2000
SCALEFACTOR = 20

if __name__ == "__main__":
    frames = []

    # Example usage
    model_file = "final_model.pth"  # Replace with your saved model file

    # Load the trained policy network
    policy_network = load_models(model_file)

    # Example observation (replace with actual observation from your environment)

    env = CleanupEnv(num_agents=args.num_agents, render=True)
    agents = list(env.agents.values())
    obs = env.map_to_colors()

    for i in tqdm(range(LENGTH_EPISODE)):
        state = preprocess_observations(obs)
        # Select action based on the observation
        actions, log_probs = select_action(policy_network, state)
        result_vector = set_action_positions(actions)
        actionList = decode_OHE_action(result_vector)
        action = create_actions_dict(actionList)
        
        rgb_arrNew = resize_and_text(obs,SCALEFACTOR,"Full Frame View")
        frames.append(rgb_arrNew) 

        try:
            env.step(action)
            obs = env.map_to_colors()
        except:
            pass

    create_movie_clip(frames,"basePolicy_2_agents.mp4")
    
        
        