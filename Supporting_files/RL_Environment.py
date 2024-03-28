import numpy as np
from .State_Exploration import *
from .queueing_network import *
transition_proba = {}

class RLEnv: 
    def __init__(self, qn_net, num_sim = 5000, start_state = None): 

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
    
    def initialize_params_for_visualization(self):

        global transition_proba
        self.record_num_exit_nodes = []
    
    def initialize_qn_params(self, num_sim):
        self.transition_proba = self.net.transitions(False)
        self.adja_list= self.qn_net.adja_list
        self.sim_n = num_sim # Take next step (num_events)
        self.iter = 0

        self.departure_nodes = 1 # For now set to 0, should be changed to be got from the net structure 

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
        return explore_state(agent, queue_model = env.qn_net, 
                            num_sample = num_sample, device = device, visit_counts = agent.visited_count,
                            w1 = w1, w2 = w2, epsilon = epsilon)

    def get_state(self):

        for edge in range((self.net.num_edges-1)):
            
            edge_data = self.net.get_queue_data(queues=edge)
            if len(edge_data) > 0:
                self._state[edge]=edge_data[-1][4]
            else:
                self._state[edge]=0
        
        self.record_num_exit_nodes.append(len(self.net.get_queue_data(queues=12)))

        return self._state

    def get_next_state(self, action):

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
        return current_state, transition_proba
    
    def test_actions_equal_nodes(self, action):
        if len(action) != self.net.num_nodes - 1:
            raise ValueError('The action space is incomatible with the dimensions')
    
    def test_nan(self, element):
        if np.isnan(element):
            TypeError("Encounter NaN")
    
    def get_correct_action_format(self, action):
        if isinstance(action, list):
            action = np.array(action)
        elif isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        
        return action

    def simulate(self):


        # Simulation is specified by time in seconds, the number of events will depend on the arrival rate
        self.net.initialize(queues=0)

        self.iter +=1
        self.net.clear_data()
        self.net.start_collecting_data()
        self.net.simulate(n = self.sim_n) 
        
        return self.get_state()

    def get_reward(self):
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