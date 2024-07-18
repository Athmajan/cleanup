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
import pandas as pd



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

def create_movie_clip(frames: list, output_file: str, fps: int = 30, scale_factor: int = 1):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width * scale_factor, height * scale_factor)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        # Upscale the frame
        upscaled_frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
        out.write(cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR))
    
    out.release()



def plot_rewards(rewards):
    """
    Plot the time steps taken for each episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Time Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('TIme Steps')
    plt.title('Time Steps per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()


 

class randomController():
    def __init__(self):
        self.env = CleanupEnv(num_agents=args.num_agents, render=True)
        self.env.reset()
        self.agents = list(self.env.agents.values())
        self.actionDim= self.agents[0].action_space.n

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

    
    def plotRewardHistory(self,reward_history,LENGTH_EPISODE):
        # Extract step and reward values
        steps = [entry['frame'] for entry in reward_history]
        rewards = [entry['reward'] for entry in reward_history]

        # Plotting the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, rewards, marker='o', markersize=1, linestyle='-', color='k', label='Reward')

        ax.set_title('Step vs. Cumulative Reward')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(0, LENGTH_EPISODE)

        # Convert the plot to an RGB array
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')  # Save the plot to a buffer
        plt.close(fig)  # Close the figure to prevent memory leak
        buffer.seek(0)
        image = Image.open(buffer)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        rgb_array = np.array(image)
        return rgb_array
    
    def combine_frames(self,full_frame, agent_views, num_agents, padding=10, title_height=0):
        full_height, full_width, _ = full_frame.shape
        agent_height, agent_width, _ = agent_views[0].shape
        
        # Determine the size of the final frame
        combined_height = max(full_height, agent_height + title_height) + padding
        combined_width = full_width + (agent_width + padding) * num_agents + padding

        # Create a white canvas for the combined frame
        combined_frame = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

        # Place the full frame on the left
        combined_frame[:full_height, :full_width, :] = full_frame

        # Place each agent view on the top, one next to the other, with titles
        for i, agent_view in enumerate(agent_views):
            x_offset = full_width + padding + i * (agent_width + padding)
            y_offset = title_height + padding

            combined_frame[y_offset:y_offset + agent_height, x_offset:x_offset + agent_width, :] = agent_view

        return combined_frame
    
    def combine_rewards(self,img1, img2):
        # Resize img2 while preserving aspect ratio
        target_height = 400
        scale_factor = target_height / img2.shape[0]
        target_width = int(img2.shape[1] * scale_factor)
        img2_resized = cv2.resize(img2, (target_width, target_height))

        # Create a white canvas for combined frame
        combined_height = img1.shape[0] + img2_resized.shape[0]
        combined_width = img1.shape[1]
        combined_frame = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

        # Place img1 at the top
        combined_frame[:img1.shape[0], :img1.shape[1], :] = img1

        # Place img2_resized below img1 on the left side
        combined_frame[img1.shape[0]:, :img2_resized.shape[1], :] = img2_resized

        return combined_frame

    def rollout(
            self,
            TERMINATION_REWARD,
            savePath,
            multiView,
            preprocess,
    ):
        cumulativeReward = 0
        rewardHistory = []
        frames = []
        stepsCount = 0
        
        while cumulativeReward<=TERMINATION_REWARD:
            randomAction = np.random.randint(self.actionDim,size=args.num_agents)
            obs, rew, dones, info, = self.env.step(
                        {('agent-' + str(j)): randomAction[j] for j in range(0, args.num_agents)})
            stepsCount += 1
            if preprocess:
                rgb_arr = self.env.render_preprocess()
            else:
                rgb_arr = self.env.map_to_colors()
            rgb_arr = self.resize_and_text(rgb_arr,SCALEFACTOR,"Full Frame")
            frames.append(rgb_arr)
            for agent in self.agents:
                if rew[agent.agent_id] >0 :
                    cumulativeReward = cumulativeReward + rew[agent.agent_id]
                    


            # TODO need to fix issues in this True condition
            if multiView:
                j = 0
                agent_views = []
                for agent in self.agents:
                    agentView = obs[agent.agent_id]
                    agentAction = agent.action_map(randomAction[j])
                    titleReWard = str(j) + "   R:" + str(rew[agent.agent_id]) + "  A:" + agentAction
                    if rew[agent.agent_id] >0 :
                        cumulativeReward = cumulativeReward + rew[agent.agent_id]

                    newimg = self.resize_and_text(agentView,SCALEFACTOR,titleReWard)
                    agent_views.append(newimg.astype(np.uint8))
                    j = j + 1

                rewardLog = {
                    "frame" : i,
                    "reward" : cumulativeReward
                }
                rewardHistory.append(rewardLog)

                rewardImg = self.plotRewardHistory(rewardHistory,LENGTH_EPISODE)
                newRewardImg = self.resize_and_text(rewardImg,SCALEFACTOR,"Cumulative Rewards")
                rgb_arrNew = self.resize_and_text(rgb_arr,SCALEFACTOR,"Full Frame View"+  " # :"+ str(i))
                combined_frame = self.combine_frames(rgb_arrNew.astype(np.uint8), agent_views, len(self.agents))
                newcombinedFrame = self.combine_rewards(combined_frame,newRewardImg)

                frames.append(newcombinedFrame)

                
        
        return frames, stepsCount


                
        
        

SCALEFACTOR = 20
TERMINATION_REWARD = 10
NUM_EPISODE = 300
PREPROCESS = False
EXPERIMENTCOUNT = 30
if __name__ == "__main__":
    dfTimeHistoryMaster = pd.DataFrame()
    pbar = tqdm(total=EXPERIMENTCOUNT*NUM_EPISODE, desc='Episodes')
    for experiment in range(EXPERIMENTCOUNT):
        timeStepsHistory = []
        control = randomController()
        frames = []
        for epi in range(NUM_EPISODE):
            control.env.reset()
            frames, stepsCount = control.rollout(TERMINATION_REWARD=TERMINATION_REWARD,savePath=args.vid_path,multiView=False,preprocess=PREPROCESS) # TODO Fix True condition
            timeStepsHistory.append({"Episode":epi,"TimeTaken":stepsCount})
            pbar.update(1)
            if (epi + 1) % 100 ==0:
                create_movie_clip(frames,f"finite_results/FINITE_randomRoll_2_agents_{experiment}_{epi}.mp4")

        dfTimeHistory = pd.DataFrame(timeStepsHistory)
        dfTimeHistory["Experiment"] = experiment
        dfTimeHistoryMaster = pd.concat([dfTimeHistoryMaster, dfTimeHistory], ignore_index=True)
    
    pbar.close()
    dfTimeHistoryMaster.to_csv("finite_results/time_history_master.csv", index=False)


