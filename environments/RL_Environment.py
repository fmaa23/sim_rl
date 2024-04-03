import numpy as np
from Supporting_files.State_Exploration import *
from Supporting_files.queueing_network import *
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

        self.test_state_is_valid(start_state)

        self.initialize_qn_params(num_sim)

        # Starting simulation to allow for external arrivals
        self.net.start_collecting_data()

        # Simulation is specified by time in seconds, the number of events will depend on the arrival rate
        self.net.initialize(queues=0)
        self.net.simulate(n=num_sim)

        self.initialize_params_for_visualization()

        self.num_nullnodes = self.get_nullnodes()
        self.num_entrynodes = self.get_entrynodes()
        self.departure_nodes =  self.num_nullnodes
    
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
        self.sim_n = num_sim # Take next step (num_events)
        self.iter = 0

        self.current_queue_id = 0 # Edge Index, assuming we always starting as the 
        self.current_source_vertex = self.net.edge2queue[self.current_queue_id].edge[0] # Source Queue Vertex, the source node 
        self.current_edge_tuple = self.net.edge2queue[self.current_queue_id].edge # (source_vertex, target_vertex, edge_index, type)
        self.current_queue = self.net.edge2queue[self.current_queue_id]

    def test_state_is_valid(self, start_state):
        if start_state is None: 
            self._state = np.zeros(self.net.num_edges-1)

    def get_net_connections(self):
        return self.transition_proba

    def explore_state(self, agent, env, num_sample, device, w1 = 0.5, w2 = 0.5, epsilon = 1):
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
        return explore_state(agent, qn_model = env.net, qn_env = env.qn_net,
                            num_sample = num_sample, device = device, visit_counts = agent.visited_count,
                            w1 = w1, w2 = w2, epsilon = epsilon)

    def get_state(self):
        """
        Retrieves the current state of the environment.

        Returns:
            The current state of the environment, represented as an array or a suitable data structure.
        """
        for edge in range((self.net.num_edges-self.num_nullnodes)):
            
            edge_data = self.net.get_queue_data(queues=edge) # self.net.get_queue_data(edge_type=2)
            if len(edge_data) > 0:
                self._state[edge]=edge_data[-1][4]
            else:
                self._state[edge]=0

        return self._state

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

        for i, node in enumerate(self.transition_proba.keys()):
            next_node_list=list(self.transition_proba[node].keys())
            
            if len(next_node_list) != 0:
                action_next_node_list = [x - 1 for x in next_node_list] 
                action_probs= action[action_next_node_list]/sum(action[action_next_node_list])
                
                for j, next_node in enumerate(next_node_list): 
                    self.test_nan(action_probs[j])
                    self.transition_proba[node][next_node]=action_probs[j]
        
        current_state = self.simulate()
        return current_state
    
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

    def simulate(self):
        """
        Runs a simulation of the environment.

        Simulates the queueing network for a number of events determined by the initialized simulation parameters.

        Returns:
            The state of the environment after the simulation.
        """
        # Simulation is specified by time in seconds, the number of events will depend on the arrival rate
        self.net.initialize(queues=0)

        self.iter +=1
        self.net.clear_data()
        self.net.start_collecting_data()
        self.net.simulate(n = self.sim_n) 
        
        return self.get_state()

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state of the environment.

        The reward calculation is based on the throughput and end-to-end delay of the queues.

        Returns:
            float: The calculated reward.
        """
        reward = 0
        for i in range(self.net.num_edges): 
            queue_data=self.net.get_queue_data(queues=i)
            ind_serviced = np.where(queue_data[:,2]!=0)[0]
            if len(ind_serviced)>0:
                throughput = len(ind_serviced)
                EtE_delay= queue_data[ind_serviced,2]-queue_data[ind_serviced,0]
                tot_EtE_delay = EtE_delay.sum()
                reward += (throughput-tot_EtE_delay)
        return reward
                
    def reset(self): 
        self.net.clear_data()
        self.__init__(self.qn_net)