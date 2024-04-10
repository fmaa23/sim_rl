# from tensorflow import random
import numpy as np
from agents.ddpg import DDPGAgent
import torch
from Supporting_files.queueing_network import Queue_network
import os
import json
import matplotlib.pyplot as plt
import yaml
import matplotlib.pyplot as plt

class ExploreStateEngine():
    def __init__(self):
        """
        Initialize the ExploreStateEngine with default parameters and configurations.
        """
        self.eval_param_filepath = 'user_config/eval_hyperparams.yml'

        self.activate_features()
        self.load_params()
        self.init_track_reward()
        self.init_device()

    def activate_features(self):
        """
        Activate features based on loaded hyperparameters.
        """
        params = self.load_hyperparams()

        self.output_json_files = params['output_json']
        self.reset = params['reset'] 
        self.output_histogram = params['output_histogram']
        self.output_coverage_metric = params['output_coverage_metric']
    
    def init_device(self):
        """
        Initialize the computation device (CPU or CUDA) for PyTorch operations.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_hyperparams(self):
        """
        Load hyperparameters from a YAML file.

        Parameters:
        - param_filepath (str): The file path to the hyperparameters YAML file.

        Returns:
        - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
        """

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up one directory to the MScDataSparqProject directory
        project_dir = os.path.dirname(script_dir)

        # Build the path to the configuration file
        abs_file_path = os.path.join(project_dir, self.eval_param_filepath)
        
        with open(abs_file_path, 'r') as env_param_file:
            parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)
        params = parameter_dictionary['state_exploration_params']

        return params
    

    def get_param_for_state_exploration(self, params):
        """
        Extract parameters necessary for state exploration.

        Parameters:
        - params (dict): Hyperparameters including those needed for state exploration.

        Returns:
        - tuple: A tuple containing parameters specific to state exploration.
        """
        self.num_sample = params['num_sample']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.epsilon = params['epsilon_state_exploration']
        if self.reset == False:
            self.reset_frequency = None
        else:
            self.reset_frequency = params['reset_frequency']
        self.num_output = params['num_output']
        self.moa_coef = params['moa_window']
    
    def load_params(self):
        """
        Load parameters for state exploration from the hyperparameters file.
        """
        params = self.load_hyperparams()
        self.get_param_for_state_exploration(params)
    
    def init_track_reward(self):
        """
        Initialize a dictionary to keep track of rewards information.
        """
        self.reward_info = {}
    
    def get_top_key_states(self, reward_rankings):
        """
        Get the top and least rewarding states based on their rankings.

        Parameters:
        - reward_rankings (dict): A dictionary with states as keys and their rewards as values.
        """
        top_states = sorted(reward_rankings.items(), key=lambda item: item[1], reverse=True)[:self.num_output]
        top_states = [state[0] for state in top_states]
        least_states = sorted(reward_rankings.items(), key=lambda item: item[1], reverse=False)[:self.num_output]
        least_states = [state[0] for state in least_states]

        return top_states, least_states

    def get_top_peripheral_states(self, visit_rankings):
        """
        Get the most and least visited states based on their rankings.

        Parameters:
        - visit_rankings (dict): A dictionary with states as keys and their visit counts as values.
        """
        top_states = sorted(visit_rankings.items(), key=lambda item: item[1], reverse=True)[:self.num_output]
        top_states = [state[0] for state in top_states]
        least_states = sorted(visit_rankings.items(), key=lambda item: item[1], reverse=False)[:self.num_output]
        least_states = [state[0] for state in least_states]

        return top_states, least_states
    
    def load_states(self, reward_rankings, visit_rankings):
        """
        Load states based on their reward and visit rankings.

        Parameters:
        - reward_rankings (dict): Reward rankings of the states.
        - visit_rankings (dict): Visit rankings of the states.
        """
        top_rewards_states, least_reward_states = self.get_top_key_states(reward_rankings)
        top_visit_states, least_visit_states = self.get_top_peripheral_states(visit_rankings)

        keystates_dict = {}
        keystates_dict['top_states'] = top_rewards_states
        keystates_dict['least_states'] = least_reward_states

        peripheralstates_dict = {}
        peripheralstates_dict['top_states'] = top_visit_states
        peripheralstates_dict['least_states'] = least_visit_states

        return self.convert_format(keystates_dict), self.convert_format(peripheralstates_dict)

    def output_json(self, reward_rankings, visit_rankings, key_states_filename, peripheral_states_filename):
        """
        Output the rankings information to JSON files.

        Parameters:
        - reward_rankings (dict): Reward rankings of the states.
        - visit_rankings (dict): Visit rankings of the states.
        - key_states_filename (str): Filename for outputting key states.
        - peripheral_states_filename (str): Filename for outputting peripheral states.
        """
        keystates_dict, peripheralstates_dict = self.load_states(reward_rankings, visit_rankings)

        # Write the dictionary to a file as JSON
        with open(key_states_filename, 'w') as json_file:
            json.dump(keystates_dict, json_file)
        
        with open(peripheral_states_filename, 'w') as json_file:
            json.dump(peripheralstates_dict, json_file)
    
    def convert_format(self, states_dict):
        """
        Convert state format in the provided dictionary.

        Parameters:
        - states_dict (dict): A dictionary containing states information.
        """
        keys = list(states_dict.keys())
        for key in keys:
            for index, state in enumerate(states_dict[key]):
                convert_state = tuple(int(x) for x in state)
                states_dict['top_states'][index] = convert_state
                states_dict['least_states'][index] = convert_state
        
        return states_dict
    
    def get_keystates(self, reward_rankings):
        """
        Get key states based on reward rankings.

        Parameters:
        - reward_rankings (dict): Reward rankings of the states.
        """
        top_rewards_states, least_reward_states = self.get_top_key_states(reward_rankings)

        keystates_dict = {}
        keystates_dict['top_states'] = top_rewards_states
        keystates_dict['least_states'] = least_reward_states

        return self.convert_format(keystates_dict)

    def generate_path(self):
        """
        Generate the file paths for storing states information.
        """
        base_path = os.getcwd()
        base_path_key = base_path + "/features/feature_4_state_exploration/key_states/"
        base_path_peripheral = base_path + "/features/feature_4_state_exploration/peripheral_states/"

        os.makedirs(base_path_key, exist_ok=True)
        os.makedirs(base_path_peripheral, exist_ok=True)

        # Specify the filenames
        key_states_filename = os.path.join(base_path_key, "key_states.json")
        peripheral_states_filename = os.path.join(base_path_peripheral, "peripheral_states.json")

        return base_path_key, base_path_peripheral, key_states_filename, peripheral_states_filename

    def calculate_q_values(self, DDPG_agent, key_states_filename, peripheral_states_filename, visit_counts):
        """
        Calculate Q-values for the key and peripheral states.

        Parameters:
        - DDPG_agent: The DDPG agent instance.
        - key_states_filename (str): The filename for key states information.
        - peripheral_states_filename (str): The filename for peripheral states information.
        - visit_counts (dict): A dictionary of visit counts for the states.
        """
        with open(key_states_filename, 'r') as json_file:
            key_states = json.load(json_file)
        key_states = self.convert_format(key_states)

        with open(peripheral_states_filename, 'r') as json_file:
            peripheral_states = json.load(json_file)
        peripheral_states = self.convert_format(peripheral_states)

        all_key_states = key_states['top_states'] + key_states['least_states']
        q_value_list = []
        with torch.no_grad():
            for state in all_key_states:
                state_tensor = torch.tensor(state).float() 
                q_value = DDPG_agent.critic([state_tensor, DDPG_agent.actor(state_tensor)]).item()
                q_value_list.append(q_value)

        all_paripheral_states = peripheral_states['top_states'] + peripheral_states['least_states']
        visits_list = []
        for state in all_paripheral_states:
            if state not in list(visit_counts.keys()):
                visit = 0
            else:
                visit = visit_counts[state]

            visits_list.append(visit)
        
        return q_value_list, visits_list

    def update_reward_info(self, agent, keystates_dict):
        """
        Update reward information for the states.

        Parameters:
        - agent: The agent instance.
        - keystates_dict (dict): A dictionary containing key states information.
        """
        for key in keystates_dict.keys():
            for state in keystates_dict[key]:
                q_values_list = self.reward_info.setdefault(state, [])

                state_tensor = torch.tensor(state).float()
                with torch.no_grad():
                    q_values = agent.critic([state_tensor, agent.actor(state_tensor)]).item()
                q_values_list.append(q_values)

                self.reward_info[state] = q_values_list
    
    def track_reward(self, agent, reward_rankings, visit_rankings):
        """
        Track and update reward information based on current rankings.

        Parameters:
        - agent: The agent instance.
        - reward_rankings (dict): Reward rankings of the states.
        - visit_rankings (dict): Visit rankings of the states.
        """
        keystates_dict, _ = self.load_states(reward_rankings, visit_rankings)
        self.update_reward_info(agent, keystates_dict)

    def get_moving_average(self, reward_list):
        """
        Calculate the moving average of rewards.

        Parameters:
        - reward_list (list): A list of reward values.
        """
        if len(reward_list) < self.moa_coef:
            moving_average = np.mean(reward_list)
        latest_moa = reward_list[-self.moa_coef:] 
        moving_average = sum(latest_moa) / len(latest_moa)

        return moving_average
    
    def get_path(self, folder_name, file_name):
        """
        Get the path for a given folder and file name.

        Parameters:
        - folder_name (str): The name of the folder.
        - file_name (str): The name of the file.
        """
        base_path = os.getcwd()
        base_path = base_path + f"/features/feature_4_state_exploration/{folder_name}/"

        os.makedirs(base_path, exist_ok=True)

        # Specify the filenames
        file_path = os.path.join(base_path, file_name)

        return base_path, file_path

    def plot_coverage(self, metric_list, base_path, title):
        """
        Plot and save the coverage metric.

        Parameters:
        - metric_list (list): A list of metric values.
        - base_path (str): The base path for saving the plot.
        - title (str): The title of the plot.
        """
        if len(metric_list) > self.moa_coef:
            plt.figure()
            plt.plot(metric_list)
            plt.title(title)

            save_path = os.path.join(base_path, title)
            plt.savefig(save_path)
            plt.close()

    def output_metric(self, reward_rankings):
        """
        Output the metric based on the current reward rankings.

        Parameters:
        - reward_rankings (dict): Reward rankings of the states.
        """
        if len(self.reward_info) == 0:
            return None
        
        keystates_dict = self.get_keystates(reward_rankings)
        top_state = keystates_dict['top_states'][0]
        bottom_state = keystates_dict['least_states'][-1]

        top_reward_list = self.reward_info[top_state]
        bottom_reward_list = self.reward_info[bottom_state]

        folder_name = 'coverage_metric'
        file_name = 'coverate_metric.json'

        folder_path, file_path = self.get_path(folder_name, file_name)

        try:
            with open(file_path, 'r') as json_file:
                metric = json.load(json_file)
            metric_list = metric['coverage_metric']
            metric_list.append(self.get_moving_average(top_reward_list) - self.get_moving_average(bottom_reward_list))
            metric['coverage_metric'] = metric_list

            self.plot_coverage(metric_list, folder_path, title = "Coverage")
        except:
            metric = {'coverage_metric': [self.get_moving_average(top_reward_list) - self.get_moving_average(bottom_reward_list)]}
            # Write the dictionary to a file as JSON
        
        with open(file_path, 'w') as json_file:
            json.dump(metric, json_file)

    def reset_weights(self, episode, reset_frequency):
        """
        Reset the weights based on the episode and frequency.

        Parameters:
        - episode (int): The current episode number.
        - reset_frequency (int): The frequency at which weights should be reset.
        """
        if episode != 0 and episode % reset_frequency == 0:
            print()
            weights = input(f"{episode} episodes have passed. Please reset your weights:")
            weights = [float(x) for x in weights.split(',')]
            self.w1, self.w2 = weights

    def explore_state(self, agent, env, episode):
        """
        This function is used for state exploration for real word environment
        It helps the agent decide which state to explore next based on factors, including Q-values and visit counts. 
        
        Parameters:
        - DDPG_agent: the class of the DDPGAGent
        - queue_model: the class of the queue model
        - num_sample: number of states we sample from to get the lowest reward and times visited
        - visit_counts: a dictionary that stores the number of times being visited for each state;
        The key is the tuple of the state and the value is the number of time visited
        - w1: Weight for the influence of Q-values (default: 0.5)
        - w2: Weight for the influence of visit counts (default: 0.5)
        - epsilon: Probability of exploration (default: 1)

        Returns:
        - chosen_state: The selected state for exploration.
        """

        max_buffer_size = []
        for start_node in env.adja_list.keys():
            for end_node in env.adja_list[start_node]:
                
                edge_index = env.edge_list[start_node][end_node]

                if edge_index == 0:
                    continue

                if 'qbuffer' not in list(env.q_args[edge_index].keys()):
                    print(f"start node: {start_node}, end node: {end_node}, and edge index: {edge_index} ----- qbuffer not found")
                    print(f"q_args: {env.q_args}")

                max_buffer = env.q_args[edge_index]['qbuffer']
                max_buffer_size.append(max_buffer)
        
        sample_states = []
        for _ in range(self.num_sample):
            array = np.array([np.random.randint(0, max_val) for max_val in max_buffer_size])
            sample_states.append(array)

        states_array = np.array(sample_states)

        # explore state with lower Q values and being visited less frequently
        states_ordered_by_reward = self.rank_states_by_Q_values(agent, states_array, self.device)
        states_ordered_by_visits = self.rank_states_by_visits(agent.visited_count, states_array)

        reward_rankings = {state: rank for rank, state in enumerate(states_ordered_by_reward)}
        visit_rankings = {state: rank for rank, state in enumerate(states_ordered_by_visits)}

        # calculate weighted average of rankings for each state
        weighted_averages = {}
        for state in reward_rankings.keys():
            weighted_avg = self.w1 * reward_rankings[state]+ \
                            self.w2 * visit_rankings[state]
            weighted_averages[state] = weighted_avg

        # sort states by weighted average
        sorted_states = sorted(weighted_averages.items(), key=lambda x: x[1])

        # epsilon-greedy selection
        if np.random.rand() < self.epsilon:
            # exploit: choose the state with the lowest weighted average
            chosen_state = sorted_states[0][0]
        else:
            # explore: randomly choose any state
            index = np.random.randint(len(weighted_averages))
            chosen_state = list(weighted_averages.keys())[index]

        base_path_key, base_path_peripheral, key_states_filename, peripheral_states_filename = self.generate_path()
        q_values_list, visits_list = self.calculate_q_values(agent, key_states_filename, peripheral_states_filename, agent.visited_count)
        
        if self.output_json_files:
            self.output_json(reward_rankings, visit_rankings, key_states_filename, peripheral_states_filename)

        if self.output_histogram:
            self.plot_hist(q_values_list, base_path_key, title = "Q Values for Key States")
            self.plot_hist(visits_list, base_path_peripheral, title = "Visits for Peripheral States")
        
        self.track_reward(agent, reward_rankings, visit_rankings)
        if self.output_coverage_metric:
            self.output_metric(reward_rankings)
        
        if self.reset:
            self.reset_weights(episode, self.reset_frequency)

        return torch.tensor(chosen_state).to(self.device)

    def plot_hist(self, values_list, base_path, title):
        """
        the x-axis names that start from t represent the top states, 
        and the names that start from b represent the states with least metric.
        """
        x_labels = [f't{i+1}' for i in range(len(values_list)//2)] + [f'b{i+1}' for i in range(len(values_list)//2)]

        plt.figure()
        plt.bar(x_labels, values_list)
        plt.title(title)

        import os
        save_path = os.path.join(base_path, title)
        plt.savefig(save_path)
        plt.close()

    def rank_states_by_Q_values(self, DDPG_agent, states_array, device):
        """
        This function ranks states based on their Q-values. 
        The function first generates all possible states based on the maximum buffer sizes.
        Then calculates normalized Q-values for each state using the critic network.

        Parameters:
        - queue_model: The queue network class
        - critic_network: The critic neural network used to estimate Q-values.

        Returns:
        - sorted_state_q_value_pairs: A list of state-Q-value pairs sorted in ascending order of Q-values.
        """

        action_vector =  DDPG_agent.actor(torch.tensor(states_array).to(device))
        q_values = DDPG_agent.critic.forward((torch.tensor(states_array).float(), action_vector))

        state_q_value_pairs = list(zip(torch.tensor(states_array).float(), q_values))
        sorted_state_q_value_pairs = sorted(state_q_value_pairs, key=lambda x: x[1])

        sorted_dict = {}
        for info in sorted_state_q_value_pairs:
            state = info[0].numpy().tolist()
            q_value = info[1].item()

            sorted_dict[tuple(state)] = q_value

        return sorted_dict

    def rank_states_by_visits(self, visit_counts, states_array):
        """
        This function ranks states based on the number of visits they have received. 
        The function takes a matrix of visit counts and sorts the states in descending order of visit counts.

        Parameters:
        - visit_counts: A matrix containing visit counts for each state.

        Returns:
        - ordered_states_visit_dict: A dict of visited states and their corresponding visit counts, sorted in descending order of visit counts.
        """

        states_visit_dict = {}
        for state in states_array.tolist():
            if state in list(visit_counts.keys()):
                num_visited = visit_counts[state]
            else:
                num_visited = 0
            states_visit_dict[tuple(state)] = num_visited
        ordered_states_visit_dict = dict(sorted(states_visit_dict.items(), key=lambda item: item[1], reverse=True))
        return ordered_states_visit_dict