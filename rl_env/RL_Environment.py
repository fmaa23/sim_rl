import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import numpy as np
# from features.state_exploration.state_exploration import *
from queue_env.queueing_network import *
from foundations.base_functions import *
from queueing_tool.queues.queue_servers import *

transition_proba = {}

class RLEnv: 
    def __init__(self, qn_net, num_sim = 5000, start_state = None): 
        """
        Initializes the reinforcement learning environment.

        Args:
            qn_net: The queueing network object.
            num_sim (int): The number of simulations to run. Defaults to 5000.
            start_state: The initial state of the environment. If None, a default state is used.

        Initializes the queueing network parameters and starts the simulation.
        """

        self.qn_net = qn_net
        self.net = qn_net.queueing_network
        self.num_nullnodes = self.get_nullnodes()

        self.test_state_is_valid(start_state)

        self.initialize_qn_params(num_sim)

        # Starting simulation to allow for external arrivals
        self.net.start_collecting_data()

        # Simulation is specified by time in seconds, the number of events will depend on the arrival rate
        self.net.initialize(edge_type=1)

        self.initialize_params_for_visualization()

        self.num_entrynodes = self.get_entrynodes()
        self.departure_nodes =  self.num_nullnodes

        # self.ExploreStateEngine = ExploreStateEngine()

        self.num_entries = []

        self.temperature = 0.2
        self.growth_factor = (0.5/self.temperature)**(1/10)
    
    def get_entrynodes(self):
        """
        Returns the number of entry nodes in the network.

        Returns:
            int: The number of entry nodes.
        """
        return len(self.adja_list[0])

    def get_nullnodes(self):
        """
        Calculates and returns the number of null nodes in the network.

        Null nodes are defined as nodes with a specific edge type, indicating a specific condition in the network.

        Returns:
            int: The number of null nodes.
        """
        num_nullnodes = 0
        edge_list = self.qn_net.edge_list
        for start_node in edge_list.keys():
            connection = edge_list[start_node]
            edge_types = list(connection.values())
            for edge_type in edge_types:
                if edge_type == 0:
                    num_nullnodes += 1
        
        return num_nullnodes
    
    def initialize_params_for_visualization(self):
        """
        Initializes parameters required for visualization of the network.

        This method sets up various attributes needed for effectively visualizing the state and dynamics of the queueing network.
        """
        global transition_proba
        self.record_num_exit_nodes = []
    
    def initialize_qn_params(self, num_sim):
        """
        Returns the transition probabilities of the network.

        Returns:
            A data structure representing the transition probabilities between nodes in the network.
        """
        self.transition_proba = self.net.transitions(False)
        self.adja_list= self.qn_net.adja_list
        self.sim_n = num_sim
        self.iter = 0

        self.current_queue_id = 0
        self.current_source_vertex = self.net.edge2queue[self.current_queue_id].edge[0] 
        self.current_edge_tuple = self.net.edge2queue[self.current_queue_id].edge 
        self.current_queue = self.net.edge2queue[self.current_queue_id]

    def test_state_is_valid(self, start_state):
        if start_state is None: 
            self._state = np.zeros(self.net.num_edges-self.num_nullnodes)

    def get_net_connections(self):
        return self.transition_proba

    def explore_state(self, agent, env, episode):
        """
        Explores the state of the environment using the provided agent.

        Args:
            agent: The agent exploring the environment.
            env: The environment being explored.
            num_sample (int): The number of samples to take in exploration.
            device: The device to run the exploration on.
            w1 (float): Weight parameter for exploration.
            w2 (float): Another weight parameter for exploration.
            epsilon (float): The exploration rate.

        Returns:
            The result of exploring the state.
        """
        return self.ExploreStateEngine.explore_state(agent, env, episode)

    def get_state(self):
        """
        Retrieves the current state of the environment.

        Returns:
            The current state of the environment, represented as an array or a suitable data structure.
        """
        state = []
        for i in range(self.net.num_edges): 
            if isinstance(self.net.edge2queue[i], LossQueue):
                queue_data=self.net.get_queue_data(queues=i)
                if len(queue_data[queue_data[:, 2] == 0, 2]) > 0:
                    queue_data[queue_data[:, 2] == 0, 2] = self.net.current_time

                if len(queue_data)>0:
                    throughput = len(queue_data)
                    EtE_delay= queue_data[:,2]-queue_data[:,0]
                    tot_EtE_delay = EtE_delay.sum()
                    state.append(tot_EtE_delay / throughput)
                else:
                    state.append(0)
        return state

    def record_sim_data(self):
        for num in range(self.net.num_edges):
            self.num_entries.append(len(self.net.get_queue_data(queues = num)))
        

    def get_next_state(self, action):
        """
        Computes and returns the next state of the environment given an action.

        Args:
            action: The action taken in the current state.

        Returns:
            tuple: A tuple containing the next state and the transition probabilities.
        """
        # self.test_action_equal_nodes(action)
        action = self.get_correct_action_format(action)

        def softmax_with_temperature(logits, temperature):
            exp_logits = np.exp(logits / temperature)
            return exp_logits / np.sum(exp_logits)

        for i, node in enumerate(self.transition_proba.keys()):
            next_node_list = list(self.transition_proba[node].keys())

            if len(next_node_list) != 0:
                action_next_node_list = [x - 1 for x in next_node_list] 
                # Apply non-linear transformation here
                action_probs = softmax_with_temperature(action[action_next_node_list], temperature=self.temperature)
                
                for j, next_node in enumerate(next_node_list): 
                    self.test_nan(action_probs[j])
                    self.transition_proba[node][next_node] = action_probs[j]

        self.record_sim_data()

        self.net.set_transitions(self.transition_proba)
        current_state = self.simulate()

        return current_state
    
    def update_temperature(self):
        self.temperature *= self.growth_factor
        if self.temperature >= 0.5:
            self.temperature = 0.5
        
    def test_actions_equal_nodes(self, action):
        """
        Tests if the length of the action array is equal to the expected number of nodes minus null nodes.

        Args:
            action: The action array to test.

        Raises:
            ValueError: If the action space is incompatible with the dimensions expected.
        """
        if len(action) != self.net.num_nodes -  self.num_nullnodes:
            raise ValueError('The action space is incomatible with the dimensions')
    
    def test_nan(self, element):
        """
        Tests if the provided element is NaN (Not a Number).

        Args:
            element: The element to check.

        Raises:
            TypeError: If the element is NaN.
        """
        if np.isnan(element):
            TypeError("Encounter NaN")
    
    def get_correct_action_format(self, action):
        """
        Converts the action to the correct format for processing.

        Args:
            action: The action to format, which can be a list, NumPy array, or PyTorch tensor.

        Returns:
            The action formatted as a NumPy array.
        """
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        
        return action

    def convert_format(self, state):
        initial_states = {}
        for index, num in enumerate(state):
            initial_states[index] = num
        
        return initial_states

    def simulate(self):
        """
        Runs a simulation of the environment.

        Simulates the queueing network for a number of events determined by the initialized simulation parameters.

        Returns:
            The state of the environment after the simulation.
        """
        self.net.initialize(edge_type=1)
        
        self.iter +=1
        self.net.start_collecting_data()

        self.net.simulate(n = self.qn_net.sim_time) 

        return self.get_state()
        
    def inverted_adjacency(self,adjacency):
        inverted_dict = {}

        for key, values in adjacency.items():
            for value in values:
                if value in inverted_dict:
                    inverted_dict[value].append(key)
                else:
                    inverted_dict[value] = [key]

        return inverted_dict

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state of the environment.

        The reward calculation is based on the throughput and end-to-end delay of the queues.

        Returns:
            float: The calculated reward.
        """

        avg_delay = []

        for i in range(self.net.num_edges):
            if isinstance(self.net.edge2queue[i], LossQueue): 
                queue_data=self.net.get_queue_data(queues=i)[self.num_entries[i]:]
                ind_serviced = np.where(queue_data[:,2]!=0)[0]

                if len(ind_serviced)>0:
                    throughput = len(ind_serviced)
                    EtE_delay= queue_data[ind_serviced,2]-queue_data[ind_serviced,0]
                    tot_EtE_delay = EtE_delay.sum()
                    avg_delay.append(tot_EtE_delay / throughput)

        num_exits = 0
        for i in range(self.net.num_edges):
            if isinstance(self.net.edge2queue[i], NullQueue): 
                num_exits += len(self.net.get_queue_data(queues=i))
        
        throughput_ratio = num_exits / len(self.net.get_queue_data(queues=0))

        return -np.mean(avg_delay) / throughput_ratio

    def create_queueing_env(self, config_file):
        return create_queueing_env(config_file)
                
    def reset(self, qn_net = None): 
        self.net.clear_data()
        if qn_net is None:
            qn_net = self.create_queueing_env(config_file = 'user_config/configuration.yml')
            self.qn_net = qn_net
            self.net = qn_net.queueing_network
            self.num_entries = []
        else:
            self.qn_net = qn_net
            self.net = qn_net.queueing_network
            self.num_entries = []

    def return_queue(self,queue_index, metric):
        """
        Returns a specific metric for a given queue in the environment.

        Parameters:
        - queue_index: int
            The index of the queue to retrieve the metric for.
        - metric: str
            The metric to retrieve, either "waiting_time" or "throughput".

        Returns:
        float
            The value of the specified metric for the given queue.
        """
        if metric != "waiting_time" and metric != "throughput": 
            raise ValueError('Invalid metric...Try "waiting_time" or "throughput":') 
        queue_data = self.net.get_queue_data(queues=queue_index)
        ind_serviced = np.where(queue_data[:,2]!=0)[0]
        if len(ind_serviced)>0:
            throughput = len(ind_serviced)
            delay= np.sum((queue_data[ind_serviced,2]-queue_data[ind_serviced,0]))/len(queue_data)
        if metric == "waiting_time": 
            try:
                return delay 
            except UnboundLocalError: 
                return 0    
        if metric =="throughput": 
            try:
                return throughput
            except UnboundLocalError: 
                return 0 
            
if __name__=="__main__": 
    config_param_filepath = 'user_config/configuration.yml'
    eval_param_filepath = 'user_config/eval_hyperparams.yml'
    env = create_simulation_env({'num_sim':5000}, config_param_filepath)
    env.net.initialize(queues=0)
    breakpoint()



