import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from reinforce import Policy
from cleanup import CleanupOnlyAgent, CleanupEnv


import argparse
import numpy as np
import os
import sys
import shutil

import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import time
from tqdm import tqdm
import json

import torch.optim as optim



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





class REINFORCEController():
    def __init__(self):
        self.env = CleanupEnv(num_agents=args.num_agents, render=True)
        self.env.reset()
        self.agents = list(self.env.agents.values())
        self.actionDim= self.agents[0].action_space.n
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = Policy().to(self.device)
        

    def resize_and_text(self,img, scale_factor, titleText):
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
    

    def makeBWTensor(
            self,
            rgbImage
    ):
        BWImg = 0.2989 * rgbImage[:, :, 0] + 0.5870 * rgbImage[:, :, 1] + 0.1140 * rgbImage[:, :, 2]
        BWImage = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        return BWImage



    def getActionDict(
            self,
            probs,
                      ):
        
        categorical_dist = D.Categorical(probs.view(-1, 8))  # Flatten the probabilities for sampling
        actions = categorical_dist.sample().view(args.num_agents, -1).detach().cpu().numpy().flatten()
        actiInput = {('agent-' + str(j)): actions[j] for j in range(0, args.num_agents)}
        return actions, actiInput
    
    def collect_trajectories(
            self,
            terminateReward,
            ):
        
        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        


        # Take one step
        randomAction = np.random.randint(self.actionDim,size=args.num_agents)
        obs, rew, dones, info, = self.env.step(
                        {('agent-' + str(j)): randomAction[j] for j in range(0, args.num_agents)})
        
        
        cumulativeReward = 0
        while cumulativeReward<=terminateReward:
       
            # Update observation
            rgb_arr = self.env.render_preprocess()
            BWImg  = self.makeBWTensor(rgb_arr)
            
            probs = self.policy(BWImg)

            actions, actiInput = self.getActionDict(probs)
            
            obs, rew, dones, info, = self.env.step(actiInput)

            reward = 0
            for agent in self.agents:
                if rew[agent.agent_id] >0 :
                    reward = reward + rew[agent.agent_id]
                    cumulativeReward = cumulativeReward + rew[agent.agent_id]

            state_list.append(BWImg)
            action_list.append(actions) # collection of 1x2 
            prob_list.append(probs.squeeze().cpu().detach().numpy()) # collection os 2x8 matrices
            reward_list.append(reward)

        return prob_list, state_list, action_list, reward_list
    


def create_movie_clip(frames: list, output_file: str, fps: int = 30, scale_factor: int = 1):
    # Assuming all frames have the same shape
    height, width,layers = frames[0].shape
    size = (width * scale_factor, height * scale_factor)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        # Upscale the frame
        upscaled_frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
        out.write(cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR))
    
    out.release()


# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1,*states.shape[-3:])
    policyOut = policy(policy_input)
    return policyOut




# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards,device,
              discount = 0.995, beta=0.01):
    sizeofEpisode = len(states)
    rewardsTensor = torch.tensor(rewards, dtype=torch.float, device=device).view(sizeofEpisode, 1, 1)

    discount = discount**np.arange(len(rewards))

    rewards = np.asarray(rewards)*discount[:,np.newaxis]

    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(np.array(actions), dtype=torch.int64, device=device) # torch.Size([200, 2])
    old_probs = torch.tensor(np.array(old_probs), dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)


    
    

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)



    entropyOld = -torch.sum(old_probs * torch.log(old_probs + 1e-10), dim=2)
    entropyNew = -torch.sum(new_probs * torch.log(new_probs + 1e-10), dim=2)
    
    entropy  = (entropyOld + entropyNew)/2
    att1 = beta*entropy
    att1Copy = att1.view(sizeofEpisode, 2, 1)
    ratio = new_probs/old_probs
    att2 = ratio * rewardsTensor
    outputtt = torch.mean(att1Copy + att2)
    return outputtt



SCALEFACTOR = 20
TERMINATION_REWARD = 10
NUM_EPISODE = 9000
beta = 0.01
discountRate = 0.99
mean_rewards = []



if __name__ == "__main__":
    control = REINFORCEController()
    policy = Policy().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    pbar = tqdm(total=NUM_EPISODE, desc='Episodes')

    for epi in range(NUM_EPISODE):
        prob_list, state_list, action_list, reward_list= control.collect_trajectories(terminateReward=TERMINATION_REWARD) # TODO Fix True condition
        
        Lsur = -surrogate(Policy().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), 
                  prob_list, 
                  state_list, 
                  action_list, 
                  reward_list,
                  torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                  )
        totalRewards = np.sum(reward_list,axis=0)
        


        optimizer.zero_grad()
        Lsur.backward()
        optimizer.step()
        del Lsur

        beta*=0.995
        mean_rewards.append(np.mean(totalRewards))

        pbar.update(1)
        # display some progress every 20 iterations
        if (epi+1)%20 ==0 :
            torch.save(policy, 'REINFORCE.policy')

    pbar.close()

        
