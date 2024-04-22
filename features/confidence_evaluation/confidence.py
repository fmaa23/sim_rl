# This script is used to automatically train multiple versions of the agent for different 
# numbers of training episodes and then evaluate the performance of each agent on the 
# simulation environment - using total reward over time as the metric for evaluation.


import sys
from pathlib import Path
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))
import torch 
import matplotlib.pyplot as plt
from rl_env.RL_Environment import *
from queue_env.queueing_network import * 
from foundations.supporting_functions import *
import numpy as np
import copy
import os 



class Confidence():
    def __init__(self, num_episodes, timesteps):
        self.num_episodes = num_episodes
        self.timesteps = timesteps 
        self.total_rewards = []
        
    def train(self, params, agent, env, num_episodes, best_params=None):
        """
        Conduct training sessions for a given agent and environment.

        Parameters:
        - params (dict): Hyperparameters for training.
        - agent: The agent to be trained.
        - env: The environment in which the agent operates.
        - num_episodes (int): The number of episodes to train the agent for.
        - best_params (dict, optional): If provided, these parameters will override the default training parameters.

        Returns:
        - Multiple values including lists that track various metrics through training.
        """
        if best_params is not None:
            for key in params.keys():
                if key not in best_params.keys():
                    best_params[key] = params[key]
            params = best_params

        next_state_list_all = []
        rewards_list_all = []
        critic_loss_list_all = []
        actor_loss_list_all = []
        reward_list = []
        actor_gradient_list_all = []
        action_dict = {}
        gradient_dict = {}
        transition_probas = init_transition_proba(env)

        _,batch_size, num_epochs, time_steps, target_update_frequency = get_params_for_train(params)

        agent.train()
        print(num_episodes)
        for episode in tqdm(range(num_episodes), desc="Training Progress"):  # num_episodes now comes directly as an argument
            env.reset()
            state = env.explore_state(agent, env.qn_net, episode)
            t = 0

            actor_loss_list= []
            critic_loss_list = []
            actor_gradient_list = []
            while t < time_steps:
                if type(state) == np.ndarray:
                    state = torch.from_numpy(state).to(device)
                action = agent.select_action(state).to(device) 
                
                action_list = action.cpu().numpy().tolist()
                for index, value in enumerate(action_list):
                    node_list = action_dict.setdefault(index, [])
                    node_list.append(value)
                    action_dict[index] = node_list
                                
                next_state = env.get_next_state(action)    
                next_state = torch.tensor(next_state).float().to(device)
                reward = env.get_reward()
        
                reward_list.append(reward)                               
                experience = (state, action, reward, next_state)        
                agent.store_experience(experience)                             
            
                if agent.buffer.get_current_size() > batch_size:
                    reward_loss_list, next_state_loss_list = agent.fit_model(batch_size=batch_size, epochs=num_epochs)
                    next_state_list_all += next_state_loss_list
                    rewards_list_all += reward_loss_list

                    transition_probas = update_transition_probas(transition_probas, env)
                    
                    batch = agent.buffer.sample(batch_size=batch_size)
                    critic_loss = agent.update_critic_network(batch)                   
                    actor_loss, gradient_dict = agent.update_actor_network(batch)    

                    actor_loss_list.append(actor_loss)
                    critic_loss_list.append(critic_loss)
                    agent.plan(batch)

                t += 1
                state = next_state

                if t % target_update_frequency == 0:
                    agent.soft_update(network="critic")
                    agent.soft_update(network="actor")

            actor_loss_list_all += actor_loss_list
            critic_loss_list_all += critic_loss_list
            actor_gradient_list_all += actor_gradient_list
    
        return rewards_list_all, next_state_list_all, critic_loss_list_all,\
            actor_loss_list_all, reward_list, action_dict, gradient_dict, transition_probas
            
    def evaluate_agent(self,agent, timesteps):
        total_reward = 0 
        state = self.env.reset()
        for _ in range(timesteps):
            state = env.get_state()
            action = agent.actor(state).detach()
            state = env.get_next_state(action)[0]
            reward = self.env.get_rewward()
            total_reward += reward
        return total_reward
    
    
    def save_reward_plot(self, file_path='reward_plot.png'):
        plt.plot(self.num_episodes, self.total_rewards)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward vs Number of Episodes trained')
        plt.savefig(file_path, dpi=1000)  # Save the plot with a resolution of 1000 DPI

        
    def start_train(self, config_file, param_file, save_file = True, 
                data_filename = 'data', image_filename = 'images', plot_curves = True ):
        """
        This is a modified version of the stanadard start train function that is used to train the agent for a given number of episodes and then evaluate the agent on the environment.

        Parameters:
        - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".
        - param_file (str, optional): The file path to the hyperparameters file. Defaults to "hyperparameter_file.yaml".
        - save_file (bool, optional): Flag indicating whether to save the training results to files. Defaults to True.

        This function orchestrates the loading of configurations, creation of environments and agents, and the training process.
        """
        for num_episode in self.num_episodes:
            # Load the hyperparameters and initialize the simulation environment and agent
            print(f"------ Initializing the {num_episode} episode agent ------")
            params, hidden = load_hyperparams(param_file)
            sim_environment = create_simulation_env(params, config_file)
            agent = create_ddpg_agent(sim_environment, params, hidden)
            
            # Train the agent for the specified number of episodes
            print(f"------ Training the agent for {num_episode} episodes ------") 
            print(num_episode)

            rewards_list_all, next_state_list_all, \
            critic_loss_list_all, actor_loss_list_all, \
            reward_list, action_dict, gradient_dict, \
            transition_probas = self.train(params, agent, sim_environment, num_episode)
            
            csv_filepath = os.getcwd() + '/Supporting_files/' + data_filename
            image_filepath = os.getcwd() + '/Supporting_files/' + image_filename
            
            if save_file:
                save_all(rewards_list_all, next_state_list_all, \
                critic_loss_list_all, actor_loss_list_all, \
                reward_list, action_dict, gradient_dict, \
                transition_probas, base_path=csv_filepath)
        
            if plot_curves:
                plot(csv_filepath, image_filepath)
           
            
            # Saving a copy of the trained agent in the current directory
            trained_agent = copy.deepcopy(agent)                                      
            torch.save(trained_agent.state_dict(), f"trained_agent_{num_episode}.pth")
            
            # Evaluate the agent on the environment
            print(f"------ Evaluating the {num_episode} episode agent  ------")
            total_reward = self.evaluate_agent(trained_agent, self.timesteps)
            self.total_rewards.append(total_reward)
        self.save_reward_plot()
            
                
        
         
        
        
# Logic for using this class 

# 1. Specify the data structures that will be needed to configre the class 
num_episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] # this list contains the number of episodes that the agent will be trained for 
timesteps = 1000 # this is the number of timesteps that the agent will be evaluated for

# 2. Specify the file path to the agent's configuration yaml file 
agent = 'user_config/eval_hyperparams.yml'
# 3. Speficy the file path for the environment's configuration yaml file
env = 'user_config/configuration.yml'
# 4. Intiialze the confidence class with the agent , environement and the number of episodes 
confidence = Confidence(num_episodes, timesteps)
# 5. Initialize the training and evaluation process - allow this to show updates to the user according to the status - training xxx episode agent , evaluating xxx epsiode agent 
confidence.start_train(env, agent,save_file = True, data_filename = 'data', image_filename = 'images')
# - also allow this fuinction to automatically save, the agent objects , the final graph of episodes vs reward for each environment 
# make these stored in attributes so that theu can be retrieved later 


