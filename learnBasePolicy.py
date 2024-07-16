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

    def collect_trajectories(
            self,
            horizon,
            ):
        
        # initialize returning lists and start the game!
        state_list = []
        reward_list = []
        prob_list = []
        action_list = []
        obsList = []
        obsList2 = []

        randomAction = np.random.randint(self.actionDim,size=args.num_agents)

        obs, rew, dones, info, = self.env.step(
                        {('agent-' + str(j)): randomAction[j] for j in range(0, args.num_agents)})
        
        
        cumulativeReward = 0
        for t in range(horizon):
            # Update observation
            
            rgb_arr = self.env.render_preprocess()
            rgb_arrScaled = self.resize_and_text(self.env.renderIMG(),SCALEFACTOR,"Full Frame")
            obsList.append(rgb_arrScaled)
            obsList2.append(self.resize_and_text(rgb_arr,SCALEFACTOR,"Full Frame"))

            BWImg = 0.2989 * rgb_arr[:, :, 0] + 0.5870 * rgb_arr[:, :, 1] + 0.1140 * rgb_arr[:, :, 2]

            
            BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            state_list.append(BWImg)

            probs = self.policy(BWImg)
            probsArr = self.policy(BWImg).squeeze().cpu().detach().numpy()  # 2x8 matrix for 2 agents and 8 actions
            
            
            prob_list.append(probsArr) # collection os 2x8 matrices
            del probsArr
            
            categorical_dist = D.Categorical(probs.view(-1, 8))  # Flatten the probabilities for sampling
            actions = categorical_dist.sample().view(args.num_agents, -1).detach().cpu().numpy().flatten()


            # actions = torch.argmax(probs, dim=2).detach().numpy()
            action_list.append(actions) # collection of 1x2 
            actiInput = {('agent-' + str(j)): actions[j] for j in range(0, args.num_agents)}
            obs, rew, dones, info, = self.env.step(actiInput)

            reward = 0
            for agent in self.agents:
                if rew[agent.agent_id] >0 :
                    reward = reward + rew[agent.agent_id]
                    cumulativeReward = cumulativeReward + rew[agent.agent_id]
            reward_list.append(reward)

        return prob_list, state_list, action_list, reward_list, obsList, obsList2, cumulativeReward
    


def create_movie_clip(frames: list, output_file: str, fps: int = 10, scale_factor: int = 1):
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
    
    rewardsTensor = torch.tensor(rewards, dtype=torch.float, device=device).view(LENGTH_EPISODE, 1, 1)

    discount = discount**np.arange(len(rewards))

    rewards = np.asarray(rewards)*discount[:,np.newaxis]

    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int64, device=device) # torch.Size([200, 2])
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)


    
    

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)



    entropyOld = -torch.sum(old_probs * torch.log(old_probs + 1e-10), dim=2)
    entropyNew = -torch.sum(new_probs * torch.log(new_probs + 1e-10), dim=2)
    
    entropy  = (entropyOld + entropyNew)/2
    att1 = beta*entropy
    att1Copy = att1.view(LENGTH_EPISODE, 2, 1)
    ratio = new_probs/old_probs
    att2 = ratio * rewardsTensor
    outputtt = torch.mean(att1Copy + att2)
    return outputtt



SCALEFACTOR = 20
LENGTH_EPISODE = 500
NUM_EPISODE = 2000
totalFrames = []
totalFrames2 = []
beta = 0.01
discountRate = 0.99
mean_rewards = []

# widget bar to display progress
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ', 
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=NUM_EPISODE).start()


if __name__ == "__main__":
    control = REINFORCEController()
    policy = Policy().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    
    for epi in range(NUM_EPISODE):
        prob_list, state_list, action_list, reward_list, obsList, obsList2, cumulativeReward = control.collect_trajectories(horizon=LENGTH_EPISODE) # TODO Fix True condition
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



        # display some progress every 20 iterations
        if (epi+1)%20 ==0 :
            print("Episode: {0:d}, score: {1:f}".format(epi+1,np.mean(totalRewards)))


            torch.save(policy, 'REINFORCE.policy')

        timer.update(epi+1)

    timer.finish()
        
        # newProbs = states_to_prob(Policy().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), state_list)
        # print(action_list)
        # totalFrames = totalFrames + obsList
        # totalFrames2 = totalFrames2 + obsList2
        # print(cumulativeReward)
    # create_movie_clip(totalFrames,"REINFORCE_2_agents_Colour.mp4")
    # create_movie_clip(totalFrames2,"REINFORCE_2_agents_BW.mp4")