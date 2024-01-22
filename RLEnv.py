import queueing_tool as qt 
from qt_framework import Queue_network as qn

class RLEnv: 
    def __init__(self, net, action, start_state=None, simulation_time=10): 
        # The queue network to optimize actions on 
        self.net = net 
        self._t = 0 
        self.action = action

        # Getting the queue type and ID
        self.current_queue_id = 0 # Edge Index
        self.current_source_vertex = 0 # Source Queue Vertex  
        self.current_queue = net.queueing_network.edge2queue[self.current_queue_id]

        # Starting simulation to allow for external arrivals
        self.net.queueing_network.initialize(queues=0)
        self.net.queueing_network.start_collecting_data()

        # Simulation is specified by time in seconds, the number of events will depend on the arrival rate
        self.net.queueing_network.simulate(t=simulation_time)

        # The starting queue length (backlog) at the first node 
        if start_state is None: 
            self._state = self.current_queue.number_queued()
        
        # Time since simulation start 
        self._t += simulation_time 

    def get_state(self):
        # Returns the queue length (backlog) in the current state queue ()
        self.current_queue = self.net.queueing_network.edge2queue[self.current_queue_id]
        self._state = self.current_queue.number_queued()
        return self._state     

    def get_next_state(self, state, action):
        # If reached departure reset the system 
        if isinstance(self.current_queue, qt.NullQueue): 
            self.reset()
        
        i = 0 # Given that actions are flattened list we sttart with index 0 a the first transition prob 
        
        transitions_mat = {}
        edges_ind_mat = {}
        for edge_id in range(self.net.queueing_network.num_edges):
            if isinstance(self.net.queueing_network.edge2queue[edge_id], qt.NullQueue):
                continue

            # (Source Queue, Target Queue, Edge Idx, Edge Type)
            connections_tuple = self.net.queueing_network.edge2queue[edge_id].edge

            if connections_tuple[0] not in transitions_mat.keys():
                edge_transitions = {}
            else: 
                edge_transitions = transitions_mat[connections_tuple[0]]
            edge_transitions[connections_tuple[1]] = action[i]
            i +=1 # Next transition prob 
            transitions_mat[connections_tuple[0]]=edge_transitions 
            edges_ind_mat[edge_id]=(connections_tuple(1), connections_tuple(0))

        # Getting the transition probablities as provided by the user 
        self.net.queueing_network.set_transition(transitions_mat)

        # Possible target indices for next destination 
        possible_target_vertices = list(transitions_mat[self.current_queue_id].keys())

        # Corresponding edge index 
        possible_edges_indices=[]
        for target_edge in possible_target_vertices: 
            for edge_ind in edges_ind_mat.keys():
                if  edges_ind_mat[edge_ind][0]==self.current_queue_id and edges_ind_mat[edge_ind][1]==target_edge:
                    possible_edges_indices.append(edge_ind)
        
        # Corresponding edge types 
        possible_edges_types = []
        for edge_i in possible_edges_indices:
            possible_edges_types.appned(self.net.queueing_networkedge2queue[edge_i].edge[3])

        destination_tuple = (self.current_source_vertex,possible_target_vertices, possible_edges_indices, possible_edges_types)

        new_state_edge_idx = self.current_queue.Agent.desired_destination(self.net,destination_tuple)

        self.current_queue_id = new_state_edge_idx
        self.current_source_vertex = self.net.queueing_network.edge2queue[new_state_edge_idx][0]
        self.current_queue = self.net.queueing_network.edge2queue[new_state_edge_idx]

    def get_reward(self):
        if isinstance(self.current_queue, qt.NullQueue):
            pass
        else: 
            self.reward = 1/self.net.queueing_network.get_queue_data(queues=self.current_queue_id)[4]

    def reset(self): 
        pass

    # Questions to ask: 
    # How often do we reset 
    # What is a good simulation time
    # The reward is calculated as a cost of how many states or should I make the reward function time dependent?
