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
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from learnBasePolicy import create_movie_clip
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


def visualize_image(img: np.ndarray, pause_time: float = 1):
    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")
    
    fig, ax = plt.subplots()  # Create a figure and axis
    ax.imshow(img)
    ax.axis('off')
    plt.show(block=False)
    plt.pause(pause_time)
    plt.close(fig)  # Explicitly close the figure

class SeqRolloutAgent():
    def __init__(
            self,
            agent_id:  int,
            m_agents : int,
            action_space_n : int,
            n_sim_per_step : int,
            TERMINATION_REWARD :int ,
    ):
        self.id = agent_id
        self.m_agents = m_agents
        self.action_space_n = action_space_n
        self.n_sim_per_step = n_sim_per_step
        self.terminateR = TERMINATION_REWARD
        self.agents = self._create_agents()


    def _create_agents(
            self,
    ):
        agents = [RuleBasedAgent(
                i, self.m_agents, self.action_space_n,
            ) for i in range(self.m_agents)]

        return agents
    


    def act(
            self,
            obs,
            prev_actions,
            
    ):
        best_action, action_q_values = self.act_with_info(obs, prev_actions)
        return best_action
    
    def act_with_info(
            self,
            obs,
            prev_actions,
    ):
        sim_results = []
        # Simulate to all my action options.

        # for action_id in range(self.action_space_n):
        #     action_idDD, stepsTaken = self._simulate_action_par(
        #             self.id,
        #             action_id,
        #             self.n_sim_per_step,
        #             self.terminateR,
        #             obs,
        #             prev_actions,
        #             self.m_agents,
        #             self.agents,
        #         )
        #     print(action_idDD, stepsTaken)
        #     sim_results.append(stepsTaken)
            


        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = []
            for action_id in range(self.action_space_n):
                futures.append(pool.submit(
                    self._simulate_action_par,
                    self.id,
                    action_id,
                    self.n_sim_per_step,
                    self.terminateR,
                    obs,
                    prev_actions,
                    self.m_agents,
                    self.agents,
                ))
            for f in as_completed(futures):
                res = f.result()
                sim_results.append(res)

        np_sim_results = np.array(sim_results, dtype=np.float32)
        np_sim_results_sorted = np_sim_results[np.argsort(np_sim_results[:, 0])]
        action_q_values = np_sim_results_sorted[:, 1]
        best_action = np.argmax(action_q_values)
        return best_action, action_q_values
    
    


    @staticmethod
    def _simulate_action_par(
            agent_id: int,
            action_id: int,
            n_sims: int,
            terminateR,
            obs,
            prev_actions,
            m_agents,
            agents,
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        env = CleanupEnv(num_agents=m_agents, render=True)
        policy = load_policy("REINFORCE.policy")
        # roll first step
        first_step_prev_actions = dict(prev_actions)
        first_act_n = np.empty((m_agents,), dtype=np.int8)
        for i in range(m_agents):
            if i in prev_actions:
                # Previous best action is available
                # do not have to use the base policy
                first_act_n[i] = prev_actions[i]
            elif agent_id == i:
                # optimzation is done for me.
                # my action is fixed. -> options are simulated above paralelly
                first_act_n[i] = action_id
                first_step_prev_actions[i] = action_id
            else:
                # other agents actions are not available
                # therefore assume base policy for them
                rgb_arr = env.render_preprocess()
                # rgb_arrScaled = resize_and_text(env.renderIMG(),20,"Full Frame")
                BWImg = rgb_to_grayscale(rgb_arr)
                BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                probs = policy(BWImg) 

                underlying_agent = agents[i]
                assumed_action, _ = underlying_agent.act_with_info(probs)
                # assumed_action = underlying_agent.act(obs, prev_actions=first_step_prev_actions)
                first_act_n[i] = assumed_action
                first_step_prev_actions[i] = assumed_action

        # at this point Now first_act_n has been filled.
        # run N simulations
        stepsTaken = 0
        avg_total_reward = 0.
        initObservation = obs
        for j in range(n_sims):
            env.reset()
            env.reset_from_observation(initObservation)
            # make the fist step as calculated above
            actiInput_init = {('agent-' + str(j)): first_act_n[j] for j in range(0, m_agents)}
            obs, rew, dones, info, = env.step(actiInput_init)
            stepsTaken += 1
            # sim_obs = env.get_map_with_agents()

            reward = 0
            for agent in agents:
                agentID = 'agent-' + str(agent.id[0])
                if rew[agentID] >0 :
                    reward = reward + rew[agentID]

            avg_total_reward += np.sum(reward)

            while avg_total_reward <= terminateR:
                sim_act_n = []
                sim_prev_actions = {}
                for agent in agents:
                    rgb_arr = env.render_preprocess()
                    # rgb_arrScaled = resize_and_text(env.renderIMG(),20,"Full Frame")
                    BWImg = rgb_to_grayscale(rgb_arr)
                    BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    probs = policy(BWImg) 
                    sim_best_action, actions = agent.act_with_info(probs)
                    sim_act_n.append(sim_best_action)
                    sim_prev_actions[agent.id] = sim_best_action

                actiInput = {('agent-' + str(j)): sim_act_n[j] for j in range(0, m_agents)}
                obs, rew, dones, info, = env.step(actiInput)
                stepsTaken += 1
                reward = 0
                for agent in agents:
                    agentID = 'agent-' + str(agent.id[0])
                    if rew[agentID] >0 :
                        reward = reward + rew[agentID]


                # sim_obs = env.get_map_with_agents()

                avg_total_reward += np.sum(reward)
        
        env.reset()
        stepsTaken /= len(agents)
        stepsTaken /= n_sims

        return action_id, stepsTaken





def run_agent(device,env,policy):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # env = CleanupEnv(num_agents=args.num_agents, render=True)
    # policy = load_policy("REINFORCE.policy")
    stepsHistory = []
    for _ in tqdm(range(EPISODES)):
        frames = []
        cumulative_reward = 0
        stepsCount = 0

        env.reset()
        initObservation = env.get_map_with_agents()
        m_agents = 2
        action_space_n  = 8
        n_sim_per_step = 10
        
        agents = [SeqRolloutAgent(
            i,
            m_agents,
            action_space_n,
            n_sim_per_step, 
            TERMINATION_REWARD,
        ) for i in range(m_agents)]

        while cumulative_reward<=TERMINATION_REWARD:
            prev_actions = {}
            act_n = []
            print(act_n,"_____",cumulative_reward)
            for i, agent in enumerate(agents):
                action_id = agent.act(initObservation,prev_actions=prev_actions)
                prev_actions[i] = action_id
                act_n.append(action_id)

            actiInput= {('agent-' + str(j)): act_n[j] for j in range(0, m_agents)}
            print(actiInput)
            obs, rew, dones, info, = env.step(actiInput)
            stepsCount +=1 

            frames.append(resize_and_text(env.renderIMG(),20,"Full Frame"))
            visualize_image(env.renderIMG())
            initObservation = env.get_map_with_agents()
            reward = 0
            for agent in agents:
                agentID = 'agent-' + str(agent.id)
                if rew[agentID] >0 :
                    reward = reward + rew[agentID]
            cumulative_reward = cumulative_reward + reward

        stepsHistory.append(stepsCount)
        pbar.update(1)
    return stepsHistory, frames


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = CleanupEnv(num_agents=args.num_agents, render=True)
    policy = load_policy("REINFORCE.policy")
    EXPERIMENTCOUNT = 1
    EPISODES = 1
    TERMINATION_REWARD = 10
    dfTimeHistoryMaster = pd.DataFrame()
    pbar = tqdm(total=EPISODES*EXPERIMENTCOUNT, desc='Episodes')

    for experiment in range(EXPERIMENTCOUNT):
        stepsHistory, frames = run_agent(device,env,policy)
        dfTimeHistory = pd.DataFrame(stepsHistory)
        dfTimeHistory["Experiment"] = experiment
        dfTimeHistoryMaster = pd.concat([dfTimeHistoryMaster, dfTimeHistory], ignore_index=True)
        create_movie_clip(frames,f"finite_results/FINITE_SeqRollout.mp4")

    pbar.close()
    dfTimeHistoryMaster.to_csv("finite_results/SeqRolloutyResults.csv", index=False)




    # env.reset()
    # initObservation = env.get_map_with_agents()
    # visualize_image(env.renderIMG())
    # m_agents = 2
    # action_space_n  = 8
    # n_sim_per_step = 10
    # TERMINATION_REWARD = 10
    # N_EPISODES = 10
    # agents = [SeqRolloutAgent(
    #     i,
    #     m_agents,
    #     action_space_n,
    #     n_sim_per_step, 
    #     TERMINATION_REWARD,
    # ) for i in range(m_agents)]
    # cumulative_reward = 0
    # while cumulative_reward<=TERMINATION_REWARD:
    #     print(f"CUMMULATIVE REWARD_______#######################_________{cumulative_reward}")
    #     prev_actions = {}
    #     act_n = []
        
    #     for i, agent in enumerate(agents):
    #         print(f"Starting Sequence with Agent{agent.id}")
    #         action_id = agent.act(initObservation,prev_actions=prev_actions)
    #         visualize_image(env.renderIMG())
    #         prev_actions[i] = action_id
    #         act_n.append(action_id)

    #     print("",act_n)
    #     actiInput= {('agent-' + str(j)): act_n[j] for j in range(0, m_agents)}
    #     obs, rew, dones, info, = env.step(actiInput)
    #     initObservation = env.get_map_with_agents()

    #     reward = 0
    #     for agent in agents:
    #         agentID = 'agent-' + str(agent.id)
    #         if rew[agentID] >0 :
    #             reward = reward + rew[agentID]
    #     cumulative_reward = cumulative_reward + reward



            
            




