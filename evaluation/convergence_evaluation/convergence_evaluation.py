# This script is used to automatically train the agent for varying numbers of training episodes 
# and then evaluate the performance of each agent on the 
# simulation environment - using total reward over time as the metric for evaluation.

# Change this so that it mimics the gradient based approach 


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
from foundations.supporting_functions import Engine 
import numpy as np
import copy
import os 


class ConvergenceEvaluation(Engine):
    def __init__(self, num_episodes, timesteps , num_sim = 100):
        """_summary_

        Args:
            num_episodes (_type_): _description_
            timesteps (_type_): _description_
            num_sim (int, optional): _description_. Defaults to 100.
        """
        self.num_episodes = num_episodes
        self.timesteps = timesteps 
        self.total_rewards = []
        self.num_sim = num_sim        

    def train(self,params, agent, env, num_episodes, best_params = None , blockage_qn_net = None):
        """
        This is a modified version of the standard train function which takes the number of episodes as the argument 

        Parameters:
        - params (dict): Hyperparameters for training.
        - agent: The agent to be trained.
        - env: The environment in which the agent operates.

        Returns:
        - Multiple values including lists that track various metrics through training.
        """

        if best_params is not None:
            for key in params.keys():
                if key not in best_params.keys():
                    best_params[key] = params[key]
            params = best_params

        next_state_model_list_all = []
        reward_model_list_all =[]
        gradient_dict_all = {}
        action_dict = {}
        gradient_dict_all = {}
        transition_probas = init_transition_proba(env)
        actor_loss_list= []
        critic_loss_list = []
        reward_list = []
        reward_by_episode = {}
        _, _, num_epochs, time_steps, _, num_train_AC = get_params_for_train(params)
        latest_transition_proba = None

        for episode in tqdm(range(num_episodes), desc="Episode Progress"): 
            agent.train()
            if blockage_qn_net is None:
                env.reset()
            else:
                env.reset(blockage_qn_net)

            if latest_transition_proba is not None:
                env.net.set_transitions(latest_transition_proba)

            env.simulate()
            update = 0
            reward_list = []

            for _ in tqdm(range(time_steps), desc="Time Steps Progress"): 

                state = env.get_state()
                
                state_tensor = torch.tensor(state)
                action = agent.select_action(state_tensor).to(device) 
                action_list = action.cpu().numpy().tolist()

                for index, value in enumerate(action_list):
                    node_list = action_dict.setdefault(index, [])
                    node_list.append(value)
                    action_dict[index] = node_list

                next_state_tensor = torch.tensor(env.get_next_state(action)).float().to(device)
                reward = env.get_reward()
                reward_list.append(reward)                               
                experience = (state_tensor, action, reward, next_state_tensor) 
                agent.store_experience(experience)                             

            reward_model_loss_list, next_state_loss_list = agent.fit_model(batch_size=time_steps, epochs=num_epochs)
            next_state_model_list_all += next_state_loss_list
            reward_model_list_all += reward_model_loss_list
            transition_probas = update_transition_probas(transition_probas, env)

            for _ in tqdm(range(num_train_AC), desc="Train Agent"): 

                batch = agent.buffer.sample(batch_size=time_steps)
                critic_loss = agent.update_critic_network(batch)                   
                actor_loss, gradient_dict = agent.update_actor_network(batch)    
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

            agent.plan(batch)
            agent.soft_update(network="critic")
            agent.soft_update(network="actor")
            gradient_dict_all[update] = gradient_dict
            agent.buffer.clear()
            reward_by_episode[episode] = reward_list 
            latest_transition_proba = env.transition_proba
        
        save_agent(agent)
        return next_state_model_list_all, critic_loss_list, actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas
            
    def evaluate_agent(self,agent,env, timesteps): # make redundant when importing from core functions 
        total_reward = 0 
        state = env.reset()
        for _ in tqdm(range(timesteps),desc="Evaluate"):
            state = env.get_state()
            action = agent.actor(state).detach()
            state = env.get_next_state(action)[0]
            reward = env.get_reward()
            total_reward += reward
        return total_reward
    
    
    def save_reward_plot(self, file_path,filename='reward_plot.png'):
        plt.plot(self.num_episodes, self.total_rewards)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward vs Number of Episodes trained')
        file_save=os.path.join(file_path, filename)
        plt.savefig(file_save, dpi=1000)  # Save the plot with a resolution of 1000 DPI
      
        
        
    def start_train(self, config_file, eval_config_file, param_file, save_file = True, 
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
            eval_environment = create_simulation_env(params, eval_config_file)
            agent = create_ddpg_agent(sim_environment, params, hidden)
            
            # Train the agent for the specified number of episodes
            print(f"------ Training the agent for {num_episode} episodes ------") 
            print(num_episode)

            next_state_model_list_all, critic_loss_list,\
            actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas = self.train(params, agent, sim_environment,num_episode)
           
            supporting_files_dir = os.path.join(os.getcwd(), 'foundations')
            csv_filepath = os.path.join(supporting_files_dir, data_filename)
            if save_file:
                save_all(next_state_model_list_all, \
                critic_loss_list, actor_loss_list, \
                reward_by_episode, action_dict, gradient_dict, \
                transition_probas, base_path=csv_filepath)
            
            # Saving a copy of the trained agent in the current directory
            trained_agent = copy.deepcopy(agent)                                                  
            # Evaluate the agent on the environment
            print(f"------ Evaluating the {num_episode} episode agent  ------")
            total_reward = self.evaluate_agent(trained_agent, eval_environment,self.timesteps)
            total_reward = self.start_evaluation(eval_config_file, trained_agent, self.timesteps, self.num_sim)
        confidence_dir = os.path.join(os.getcwd(), 'features', 'confidence_evaluation') # evaluate 
        os.makedirs(confidence_dir, exist_ok=True)
        self.save_reward_plot(confidence_dir)
    
         
        
        
# Logic for using this class 

# 1. Specify the data structures that will be needed to configre the class 
num_episodes = [100,300,500,700,900] # this list contains the number of episodes that the agent will be trained for 
timesteps = 600 # this is the number of timesteps that the agent will be evaluated for


# 2. Specify the file path to the agent's configuration yaml file 
agent = 'user_config/eval_hyperparams.yml'

# 3. Speficy the file path for the training and evaluation environment's configuration yaml file
train_env = 'user_config/configuration.yml'
eval_env = 'user_config/configuration.yml'

# 4. Intiialze the confidence class with the agent , environement and the number of episodes 
confidence = ConvergenceEvaluation(num_episodes, timesteps)

# 5. Initialize the training and evaluation process - allow this to show updates to the user according to the status - training xxx episode agent , evaluating xxx epsiode agent 
confidence.start_train(train_env, eval_env, agent,save_file = True, data_filename = 'output_csv', image_filename = 'output_plots')


### CHANGES ###
# 1. Separate the training and evaluation process into two separate functions 
# 2. Make the mirror version that accepts the objects as opposed to the congiguration files - the rationale is that you want to completely decouple 
# the training and evaluation process from the configuration files which are used to create the objects inside the functions 
# 3. Change so that the evaluation to a gradient based approach 


