import numpy as np
import cv2
import json
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, action_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    


# def preprocess_observations(obs, convert_to_gray=True):
#     # Ensure the input is a NumPy array
#     obs = np.array(obs, dtype=np.uint8)

#     # Check the shape of the observation to ensure it's (H, W, C)
#     if len(obs.shape) != 3 or obs.shape[2] != 3:
#         raise ValueError("Expected observation with shape (H, W, 3), but got shape {}".format(obs.shape))

#     if convert_to_gray:
#         # Convert to grayscale
#         obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         # Add an extra dimension to match the expected output shape
#         obs = np.expand_dims(obs, axis=-1)
    
#     return obs/255.0

def preprocess_observations(obs):
    obs = np.array(obs, dtype=np.uint8)
    obs = cv2.resize(obs, (8, 8))  # Resize image to 8x8
    obs = obs / 255.0  # Normalize pixel values to range [0, 1]
    obs = np.transpose(obs, (2, 0, 1))  # Transpose to (C, H, W) format
    #obs = np.expand_dims(obs, axis=0)  # Add batch dimension
    return obs




if __name__ == "__main__":
    if os.path.exists('observations.npy') and os.path.exists('actions.npy') and \
    os.path.exists('rewards.npy') and os.path.exists('next_observations.npy'):
        # Load from .npy files
        observations = np.load('observations.npy')
        actions = np.load('actions.npy', allow_pickle=True)  # Load actions with allow_pickle=True
        rewards = np.load('rewards.npy')
        next_observations = np.load('next_observations.npy')

    else:
        observations = []
        actions = []
        rewards = []
        next_observations = []

        file_pattern = "Episode_*_Rewards_Observations.json"
        files = glob.glob(file_pattern)
        filtered_files = [f for f in files if any(f"Episode_{i}_Rewards_Observations.json" in f for i in range(5))]

        for file in filtered_files:
            print(file)
            with open(file, 'r') as f:
                EpisodeData = json.load(f)
        
            for i, frame in enumerate(EpisodeData):
                obs = frame["Observation"]
                rew = 0  # Initialize reward to zero
                if frame["Reward"]["agent-0"] > 0 and frame["Reward"]["agent-1"] > 0:
                    rew = frame["Reward"]["agent-0"] + frame["Reward"]["agent-1"]

                action = frame["Action"]  # Assuming actions are also stored

                processed_obs = preprocess_observations(obs)

                observations.append(processed_obs)
                actions.append(action)
                rewards.append(rew)

            if i < len(EpisodeData) - 1:
                next_obs = EpisodeData[i + 1]["Observation"]
                processed_next_obs = preprocess_observations(next_obs)
                next_observations.append(processed_next_obs)
            else:
                next_observations.append(np.zeros_like(processed_obs))  # Terminal state

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)

        np.save('observations.npy', observations)
        np.save('actions.npy', actions)
        np.save('rewards.npy', rewards)
        np.save('next_observations.npy', next_observations)

    # Convert to PyTorch tensors
    observations = torch.tensor(observations, dtype=torch.float32).unsqueeze(1)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_observations = torch.tensor(next_observations, dtype=torch.float32).unsqueeze(1)

    # Define Actor and Critic networks
    action_size = len(np.unique(actions))  # Number of unique actions
    actor = Actor(action_size)
    critic = Critic()

    # Define optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.0001)

    def compute_advantage(rewards, values, next_values, dones, gamma=0.99):
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantage = deltas.clone()
        return advantage

    def train_actor_critic(observations, actions, rewards, next_observations, actor, critic, actor_optimizer, critic_optimizer, gamma=0.99):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # Compute critic predictions
        values = critic(observations)
        next_values = critic(next_observations)

        # Compute advantage
        target_values = rewards + gamma * next_values
        advantage = target_values - values

        # Actor loss
        log_probs = torch.log(actor(observations))
        selected_log_probs = advantage.detach() * log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -selected_log_probs.mean()

        # Critic loss
        critic_loss = F.mse_loss(values, target_values.detach())

        # Backpropagation
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
    

    # Training loop
    epochs = 1000
    batch_size = 32
    for epoch in range(epochs):
        permutation = np.random.permutation(len(observations))
        for i in range(0, len(observations), batch_size):
            indices = np.random.randint(0, 5, batch_size)
            batch_observations = observations[indices]
            batch_actions = actions[indices]
            batch_rewards = rewards[indices]
            batch_next_observations = next_observations[indices]

            actor_loss, critic_loss = train_actor_critic(
                batch_observations, batch_actions, batch_rewards,
                batch_next_observations,
                actor, critic, actor_optimizer, critic_optimizer
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")