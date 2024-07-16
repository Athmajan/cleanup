from runBasePolicy import RuleBasedAgent
from itertools import product
from cleanup import CleanupEnv
from reinforce import rgb_to_grayscale, Policy
import torch
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from operator import itemgetter
import cv2
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



def load_policy(policy_path):
    """
    Load the saved policy from the given path.
    """
    policy = Policy()
    policy = torch.load(policy_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    policy.eval()  # Set the policy to evaluation mode
    return policy

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


class StdRolloutMultiAgent():
    def __init__(
            self,
            m_agents:int,
            action_space_n : int,
            n_sim_per_step : int,
            horizon :int ,

    ):
        self.m_agent = m_agents
        self.action_space_n = action_space_n
        self.n_sim_per_step = n_sim_per_step
        self.horizon = horizon

    def act_n(
            self,
            initObservation,
    ):
        available_moves = list(range(self.action_space_n))
        configs = list(product(available_moves, repeat=self.m_agent))

        sim_results = []
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = []
            for config in configs:
                futures.append(pool.submit(
                    self._simulate,
                    initObservation,
                    config,
                    self.m_agent,
                    self.action_space_n,
                    self.n_sim_per_step,
                    self.horizon,))
                
            for f in as_completed(futures):
                res = f.result()
                sim_results.append(res)
        best_config = max(sim_results, key=itemgetter(1))[0]


        return best_config
    
    @staticmethod
    def _simulate(
        initObservation, # initObservation = env.get_map_with_agents()
        initStep,
        m_agents,
        action_space_n,
        n_sim_per_step,
        horizon,
    ):
        env = CleanupEnv(num_agents=m_agents, render=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy = load_policy("REINFORCE.policy")
        agents = [RuleBasedAgent(i, m_agents,
                                 action_space_n)
                  for i in range(m_agents)]

        
        # run N simulations
        avg_total_reward = .0
        for _ in range(n_sim_per_step):
            env.reset()
            env.reset_from_observation(initObservation)
            
            # 1 step
            actiInput_init = {('agent-' + str(j)): initStep[j] for j in range(0, m_agents)}
            obs, rew, dones, info, = env.step(actiInput_init)
            reward = 0
            for agent in agents:
                agentID = 'agent-' + str(agent.id[0])
                if rew[agentID] >0 :
                    reward = reward + rew[agentID]

            avg_total_reward += np.sum(reward)


            for _ in range(horizon):
                rgb_arr = env.render_preprocess()
                BWImg = rgb_to_grayscale(rgb_arr)
                BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                probs = policy(BWImg) 
                for agent in agents:
                    bestAction, actions = agent.act_with_info(probs)
                    break # breaking because using base policy. all agents at once

                actiInput = {('agent-' + str(j)): actions[j] for j in range(0, m_agents)}
                obs, rew, dones, info, = env.step(actiInput)
                reward = 0
                for agent in agents:
                    agentID = 'agent-' + str(agent.id[0])
                    if rew[agentID] >0 :
                        reward = reward + rew[agentID]

                avg_total_reward += np.sum(reward)

        env.reset()
        avg_total_reward /= m_agents
        avg_total_reward /= n_sim_per_step

        return initStep, avg_total_reward







if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = CleanupEnv(num_agents=args.num_agents, render=True)
    env.reset()
    initObservation = env.get_map_with_agents()
    HORIZON = 600
    std_rollout_multiagent = StdRolloutMultiAgent(
        args.num_agents, 
        8,
        10,
        200)
    
    total_reward = .0
    frames = []
    obs_n = initObservation
    for timeStep in range(HORIZON):
        best_config = std_rollout_multiagent.act_n(obs_n)
        actiInput= {('agent-' + str(j)): best_config[j] for j in range(0, args.num_agents)}

        # update step
        obs, rew, dones, info, = env.step(actiInput)
        reward = 0
        for agentID in range(args.num_agents):
            if rew['agent-' + str(agentID)] >0 :
                reward = reward + rew[agentID]

        total_reward += np.sum(reward)

        obs_n = env.get_map_with_agents()

        rgb_arrScaled = resize_and_text(env.renderIMG(),20,"Full Frame")
        frames.append(rgb_arrScaled)
        print(timeStep)
        print(best_config)
    print(total_reward)
    create_movie_clip(frames,"RunningStandardRollout_AllAgentAtOnce.mp4")

    

