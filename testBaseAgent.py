from cleanup import CleanupEnv
from reinforce import rgb_to_grayscale, Policy
import torch
import argparse
import os
import torch.distributions as D
import matplotlib.pyplot as plt
from runBasePolicy import RuleBasedAgent
import argparse
import os

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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = load_policy("REINFORCE.policy")
    
    actionSpaceN = 8
    agents = [RuleBasedAgent(
                i, 
                args.num_agents, 
                actionSpaceN,
                ) for i in range(args.num_agents)]
    
    env = CleanupEnv(num_agents=args.num_agents, render=True)
    env.reset()
    rgb_arr = env.render_preprocess()
    BWImg = rgb_to_grayscale(rgb_arr)
    BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    probs = policy(BWImg) 
    for agent in agents:
        print(agent.id)
        print(agent.m_agents)
        print(agent.action_space_n)

        bestAction, actions = agent.act_with_info(probs)
        print(f"Whole Actions : {actions}")
        print(f"Agent ID : {agent.id}, Best Action :  {bestAction}")
