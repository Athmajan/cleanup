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

class SeqRolloutAgent():
    def __init__(
            self,
            agent_id:  int,
            m_agents : int,
            action_space_n : int,
            n_sim_per_step : int,
            horizon :int ,
    ):
        self.id = agent_id
        self.m_agents = m_agents
        self.action_space_n = action_space_n
        self.n_sim_per_step = n_sim_per_step
        self.horizon = horizon
        self.agents = self._create_agents()

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
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = []
            for action_id in range(self.action_space_n):
                futures.append(pool.submit(
                    self._simulate_action_par,
                    self.id,
                    action_id,
                    self.n_sim_per_step,
                    self.horizon,
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
    

    def _create_agents(
            self,
    ):
        agents = [RuleBasedAgent(
                i, self.m_agents, self.action_space_n,
            ) for i in range(self.m_agents)]

        return agents
    

    @staticmethod
    def _simulate_action_par(
            agent_id: int,
            action_id: int,
            n_sims: int,
            horizon,
            obs,
            prev_actions,
            m_agents,
            agents,
    ):
        env = CleanupEnv(num_agents=m_agents, render=True)
        
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

                # TODO : can use the trained agents policy for this later on.
                # Here the base policy is assumed to be all agents at once policy
                underlying_agent = agents[i]
                assumed_action = underlying_agent.act(obs, prev_actions=first_step_prev_actions)
                first_act_n[i] = assumed_action
                first_step_prev_actions[i] = assumed_action

        # Now first_act_n has been filled.
        # run N simulations
        avg_total_reward = 0.
        initObservation = obs
        for j in range(n_sims):
            env.reset()
            env.reset_from_observation(initObservation)
            # make the fist step as calculated above
            actiInput_init = {('agent-' + str(j)): first_act_n[j] for j in range(0, m_agents)}
            obs, rew, dones, info, = env.step(actiInput_init)
            sim_obs = env.get_map_with_agents()

            reward = 0
            for agent in agents:
                agentID = 'agent-' + str(agent.id[0])
                if rew[agentID] >0 :
                    reward = reward + rew[agentID]

            avg_total_reward += np.sum(reward)

            for _ in range(horizon):
                sim_act_n = []
                sim_prev_actions = {}
                for agent in agents:
                    sim_best_action = agent.act(sim_obs, prev_actions=sim_prev_actions )
                    sim_act_n.append(sim_best_action)
                    sim_prev_actions[agent.id] = sim_best_action

                actiInput = {('agent-' + str(j)): sim_act_n[j] for j in range(0, m_agents)}
                obs, rew, dones, info, = env.step(actiInput)
                reward = 0
                for agent in agents:
                    agentID = 'agent-' + str(agent.id[0])
                    if rew[agentID] >0 :
                        reward = reward + rew[agentID]


                sim_obs = env.get_map_with_agents()

                avg_total_reward += np.sum(reward)
        
        env.reset()
        avg_total_reward /= len(agents)
        avg_total_reward /= n_sims

        return action_id, avg_total_reward

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = CleanupEnv(num_agents=args.num_agents, render=True)
    policy = load_policy("REINFORCE.policy")


    env.reset()
    initObservation = env.get_map_with_agents()
    m_agents = 2
    action_space_n  = 8
    n_sim_per_step = 10
    horizon = 20
    agents = [SeqRolloutAgent(
        i,
        m_agents,
        action_space_n,
        n_sim_per_step,
        horizon,
    ) for i in range(m_agents)]

    rgb_arr = env.render_preprocess()
    BWImg = rgb_to_grayscale(rgb_arr)
    BWImg = torch.tensor(BWImg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    probs = policy(BWImg) 
    for _ in range(horizon):
        prev_actions = {}
        act_n = []
        i = 0

        for agent in agents:
            try:
                bestAction, action_q_values = agent.act_with_info(
                        initObservation, prev_actions=prev_actions)
            except:
                bestAction, actions = agent.act_with_info(probs)

            
            
            
            prev_actions[i] = bestAction
            act_n.append(bestAction)
            i = i +1

        actiInput= {('agent-' + str(j)): act_n[j] for j in range(0, m_agents)}
        obs, rew, dones, info, = env.step(actiInput)
        initObservation = env.get_map_with_agents()




            
            




