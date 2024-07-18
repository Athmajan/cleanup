from learnBasePolicy import REINFORCEController, create_movie_clip
from cleanup import CleanupEnv
import torch
from reinforce import rgb_to_grayscale, visualize_image1
import argparse
import os
import torch.distributions as D
import matplotlib.pyplot as plt
import pandas as pd
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



def load_policy(policy_path):
    """
    Load the saved policy from the given path.
    """
    policy = torch.load(policy_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    policy.eval()  # Set the policy to evaluation mode
    return policy

def plot_rewards(rewards):
    """
    Plot the cumulative rewards for each episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()



class RuleBasedAgent():
    def __init__(
         self,
         agent_id :int,
         m_agents: int,
         action_space_n : int,


    ):
        self.id = agent_id,
        self.m_agents = m_agents,
        self.action_space_n = action_space_n,
    
    def act(self):
        print("calling act in rule based")

        return "OK"

    def act_with_info(
        self,
        probs # processed BW image,
    ):
        # probs = self.policy(obs) 
        categorical_dist = D.Categorical(probs.view(-1, self.action_space_n[0]))  # Flatten the probabilities for sampling
        actions = categorical_dist.sample().view(self.m_agents, -1).detach().cpu().numpy().flatten()
        bestAction = actions[self.id]
        return bestAction, actions




if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    control = REINFORCEController()
    policy = load_policy("REINFORCE.policy")
    env = CleanupEnv(num_agents=args.num_agents, render=True)
    
    dfTimeHistoryMaster = pd.DataFrame()
    EPISODES = 300
    TERMINATION_REWARD = 10
    EXPERIMENTCOUNT = 30
    pbar = tqdm(total=EPISODES*EXPERIMENTCOUNT, desc='Episodes')
    for experiment in range(EXPERIMENTCOUNT):
        timeStepsHistory = []
        
        for epi in range(EPISODES):
            frames = []
            env.reset()
            cumulativeReward = 0
            stepsCount = 0
            
            while cumulativeReward<=TERMINATION_REWARD:
                rgb_arr = env.render_preprocess()
                rgb_arrScaled = control.resize_and_text(env.renderIMG(),20,"Full Frame")
                BWImg = rgb_to_grayscale(rgb_arr)
                BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                probs = policy(BWImg) 
                categorical_dist = D.Categorical(probs.view(-1, 8))  # Flatten the probabilities for sampling
                actions = categorical_dist.sample().view(args.num_agents, -1).detach().cpu().numpy().flatten()
                actiInput = {('agent-' + str(j)): actions[j] for j in range(0, args.num_agents)}
                obs, rew, dones, info, = env.step(actiInput)
                stepsCount += 1

                reward = 0
                frames.append(rgb_arrScaled)
                for agent in control.agents:
                    if rew[agent.agent_id] >0 :
                        reward = reward + rew[agent.agent_id]
                cumulativeReward = cumulativeReward + reward
            timeStepsHistory.append({"Episode":epi,"TimeTaken":stepsCount})
            pbar.update(1)
            # if (epi + 1) % 100 ==0:
            #     create_movie_clip(frames,f"finite_results/FINITE_BasePolicy_{experiment}_{epi}.mp4")
            
    
        dfTimeHistory = pd.DataFrame(timeStepsHistory)
        dfTimeHistory["Experiment"] = experiment
        dfTimeHistoryMaster = pd.concat([dfTimeHistoryMaster, dfTimeHistory], ignore_index=True)

    pbar.close()
    # dfTimeHistoryMaster.to_csv("finite_results/BasePolicyResults.csv", index=False)
             

    
