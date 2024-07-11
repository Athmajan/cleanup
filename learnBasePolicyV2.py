import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tqdm import tqdm_notebook
import numpy as np
from collections import deque
import cv2
import os
import glob
import json

# discount factor for future utilities
DISCOUNT_FACTOR = 0.99

# number of episodes to run
NUM_EPISODES = 1000

# max steps per episode
MAX_STEPS = 10000

# score agent needs for environment to be solved
SOLVED_SCORE = 195

# device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
    
    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)
    
    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)
        
        # relu activation
        x = F.relu(x)
        
        # actions
        actions = self.output_layer(x)
        
        # get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)
        
        return action_probs
    

# Using a neural network to learn state value
class StateValueNetwork(nn.Module):
    
    # Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)
        
    def forward(self, x):
        # input layer
        x = self.input_layer(x)
        
        # activiation relu
        x = F.relu(x)
        
        # get state value
        state_value = self.output_layer(x)
        
        return state_value
    

def preprocess_observations(obs, convert_to_gray=True):
    # Ensure the input is a NumPy array
    obs = np.array(obs, dtype=np.uint8)

    # Check the shape of the observation to ensure it's (H, W, C)
    if len(obs.shape) != 3 or obs.shape[2] != 3:
        raise ValueError("Expected observation with shape (H, W, 3), but got shape {}".format(obs.shape))

    if convert_to_gray:
        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    obs = obs.flatten()
    
    return obs/255.0


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

def save_models(policy_network, stateval_network, policy_optimizer, stateval_optimizer, filename):
    '''Saves the models and optimizers state dictionaries to a file'''
    torch.save({
        'policy_network_state_dict': policy_network.state_dict(),
        'stateval_network_state_dict': stateval_network.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        'stateval_optimizer_state_dict': stateval_optimizer.state_dict()
    }, filename)


if __name__ == "__main__":
    policy_network = PolicyNetwork(450, 18).to(DEVICE)
    stateval_network = StateValueNetwork(450).to(DEVICE)

    # Init optimizer
    policy_optimizer = optim.SGD(policy_network.parameters(), lr=0.001)
    stateval_optimizer = optim.SGD(stateval_network.parameters(), lr=0.001)

    scores = []
    recent_scores = deque(maxlen=100)

    observations = []
    actions = []
    rewards = []
    next_observations = []

    file_pattern = "Episode_*_Rewards_Observations.json"
    files = glob.glob(file_pattern)
    filtered_files = [f for f in files if any(f"Episode_{i}_Rewards_Observations.json" in f for i in range(5))]

    for file in filtered_files:
        score = 0
        I = 1
        done = False
        
        print(file)
        with open(file, 'r') as f:
            EpisodeData = json.load(f)
    
        for i, frame in enumerate(EpisodeData):
            obs = frame["Observation"]
            rew = 0  # Initialize reward to zero
            if frame["Reward"]["agent-0"] > 0 and frame["Reward"]["agent-1"] > 0:
                rew = frame["Reward"]["agent-0"] + frame["Reward"]["agent-1"]

            state = preprocess_observations(obs)

            action = frame["Action"]  # Assuming actions are also stored

            actions, log_probs = select_action(policy_network, state)
            result_vector = set_action_positions(actions)
            action = decode_OHE_action(result_vector)

            if i < len(EpisodeData) - 1:
                next_obs = EpisodeData[i + 1]["Observation"]
                nextState = preprocess_observations(next_obs)
                new_state_tensor = torch.from_numpy(nextState).float().unsqueeze(0).to(DEVICE)  
                new_state_val = stateval_network(new_state_tensor)
            else:
                # Terminal State
                nextState = np.zeros_like(state)
                done = True
                new_state_val = torch.tensor([0.0]).float().unsqueeze(0).to(DEVICE)

            score += rew
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            state_val = stateval_network(state_tensor)

            val_loss = F.mse_loss(rew + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I

            advantage = (rew + DISCOUNT_FACTOR * new_state_val.item()) - state_val.item()
            advantage = torch.tensor([advantage]).to(DEVICE)
            policy_loss = -log_probs[0] * advantage - log_probs[1] * advantage
            policy_loss *= I

            # Backpropagate policy
            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            # Backpropagate value
            stateval_optimizer.zero_grad()
            val_loss.backward()
            stateval_optimizer.step()
            
            if done:
                break

            I *= DISCOUNT_FACTOR

        # append episode score 
        scores.append(score)
        recent_scores.append(score)

        save_models(policy_network, stateval_network, policy_optimizer, stateval_optimizer, "model_checkpoint.pth")

        # early stopping if we meet solved score goal
        if np.array(recent_scores).mean() >= SOLVED_SCORE:
            break
    save_models(policy_network, stateval_network, policy_optimizer, stateval_optimizer, "final_model.pth")
