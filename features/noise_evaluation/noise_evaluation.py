# This class will be used to evaluate the effect of environmental noise on the performance of the agent
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
from queue_env.queueing_network import Queue_network

# Monkey patching for the Queue_network class
def get_arrival_f(self, max_rate_list):
    # compute the time of next arriva given arrival rate 
    self.arrivals_f = []
    for rate in max_rate_list:
        rate = lambda t: 2 + np.sin(2 * np.pi * t) + NoiseEvaluator.compute_increment()
        arrival_f = lambda t: poisson_random_measure(t, rate, rate)
        self.arrivals_f.append(arrival_f)
        
Queue_network.get_arrival_f = get_arrival_f
        
class NoiseEvaluator:
    def __init__(self,frequency,mean,variance):
        """
        Args:
            frequency(float ): the frequency at which noise is added to the environment - enforce that its between 0 and 1
            mean (float): Mean of the distribution from which the noise is sampled
            variance (float): Variance of the distribution from which the noise is sampled
        """
        self.frequency = frequency
        self.mean = mean 
        self.variance = variance  
        
        
    def train(params, agent, env, best_params = None):
        """
        Conduct training sessions for a given agent and environment.

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

        num_episodes, _, num_epochs, time_steps, _, num_train_AC = get_params_for_train(params)
        latest_transition_proba = None
        for episode in tqdm(range(num_episodes), desc="Episode Progress"): 

            agent.train()
            env.reset()

            if latest_transition_proba is not None:
                env.net.set_transitions(latest_transition_proba)

            env.simulate()

            update = 0
            reward_list = []

            for t in tqdm(range(time_steps), desc="Time Steps Progress"): 

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
    
    def start_train(config_file, param_file, save_file = True, 
                data_filename = 'data', image_filename = 'images', plot_curves = True):
        """
        Start the training process for a reinforcement learning environment and agent.

        Parameters:
        - config_file (str, optional): The file path to the environment configuration file. Defaults to "configuration_file.yaml".
        - param_file (str, optional): The file path to the hyperparameters file. Defaults to "hyperparameter_file.yaml".
        - save_file (bool, optional): Flag indicating whether to save the training results to files. Defaults to True.

        This function orchestrates the loading of configurations, creation of environments and agents, and the training process.
        """

        params, hidden = load_hyperparams(param_file)

        sim_environment = create_simulation_env(params, config_file)
        agent = create_ddpg_agent(sim_environment, params, hidden)

        next_state_model_list_all, critic_loss_list,\
            actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas = train(params, agent, sim_environment)

        csv_filepath = os.getcwd() + '/foundations/' + data_filename
        image_filepath = os.getcwd() + '/foundations/' + image_filename

        if save_file:

            save_all(next_state_model_list_all, critic_loss_list,\
            actor_loss_list, reward_by_episode, action_dict, gradient_dict, transition_probas)
        
        if plot_curves:
            plot(csv_filepath, image_filepath, transition_probas)

    def compute_increment(self): 
        """This function is main entry point for adding noise to the environment. This function samples from a normal distribution with mean and variance specified in the constructor and
        returns the noise increment to be added to the environment at a given time interval.
        Args:
        
        """
        if self.frequency > np.random.random():
            # Determines whether we are currently at a noise injection interval 
            noise = np.random.normal(self.mean,self.variance)
            return noise
        else:
            return 0