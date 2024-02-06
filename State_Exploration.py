from __future__ import absolute_import

from collections import deque
import tensorflow
import random
from qt_MollyPart_runnable import Queue_network
import numpy as np
import itertools

class Critic:
    def _init__(self):
        pass

    def train(self, states, actions, target_q_value):
        #TODO: update the weights of the critic target network
        pass

    def update_target_model(self):
        """
        target_weights = tau * main_weights + (1 - tau) * target_weights
        """
        # TODO: update the weights of target model

class Actor:
    def __init__(self):
        pass

    def train(self, states, action_gradients):
        #TODO: update the weights of the target actor network
        pass

    def update_target_model(self):
        """
        target_weights = tau * main_weights + (1 - tau) * target_weights
        """
        # TODO: update the weights of target model

class DDPG:
    def __init__ (self, model):
        self.memory = deque()
        self.batch_size = 100

        self.critic_network = Critic()
        self.actor_network = Actor()

        self.queue_model = model
        self.max_size_per_dimension = [10, 10, 10]
        self.visit_counts = np.zeros(self.max_size_per_dimension, dtype=int)
        self.epsilon = 1
        pass

    def get_action(self, state):
        #TODO: return the best action predicted by the actor network 
        pass

    def store_to_buffer(self, current_state, action, reward, next_state, done):
        #TODO: store the given state, action, reward etc in the Agent's memory
        pass
    
    def train(self):
        states, actions, rewards, done, next_states = self.sample_from_buffer()
        self.train_critic(states, actions, rewards, done, next_states)
        self.train_actor(states)

        for state, action, reward, next_state in zip(states, actions, rewards, done, next_states):
            #TODO: do planning using Dyna-Q
            pass

        self.update_target_networks()
    
    def sample_from_buffer(self):
        sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, done, next_states = zip(*sample)
        return states, actions, rewards, done, next_states

    def train_critic(self, states, actions, rewards, done, next_states):
        target_q_value = self.get_target_q_values(next_state, done, rewards)
        self.critic_network.train(states, actions, target_q_value)
    
    def train_actor(self, states):
        #TODO: train the actor network 
        gradients = self.get_gradients(states)
        self.actor_network.train(states, gradients)
    
    def get_target_q_values(self, next_state, done, rewards):
        #TODO: compute the q values 
        pass

    def get_gradients(self, states):
        #TODO: compute the policy gradient for the Actor
        pass

    def update_target_networks(self):
        #TODO: update target models
        self.critic_network.update_target_model()
        self.actor_network.update_target_model()

    def explore_state(self, w1 = 0.5, w2 = 0.5):

        #TODO: explore state with lower Q values and being visited less frequently
        states_ordered_by_reward = self.rank_states_by_reward()
        states_ordered_by_visits = self.rank_states_by_visits()

        reward_rankings = {state[0]: rank for rank, state in enumerate(states_ordered_by_reward)}
        visit_rankings = {state[0]: rank for rank, state in enumerate(states_ordered_by_visits)}

        # Calculate weighted average of rankings for each state
        weighted_averages = {}
        for state in reward_rankings.keys():
            weighted_avg = w1 * (reward_rankings[state] / len(states_ordered_by_reward)) + \
                           w2 * (visit_rankings[state] / len(states_ordered_by_visits))
            weighted_averages[state] = weighted_avg

        # Sort states by weighted average
        sorted_states = sorted(weighted_averages.items(), key=lambda x: x[1])

        # Epsilon-greedy selection
        if np.random.rand() < self.epsilon:
            # Exploit: choose the state with the lowest weighted average
            chosen_state = sorted_states[0][0]
        else:
            # Explore: randomly choose any state
            chosen_state = np.random.choice(list(weighted_averages.keys()))

        return chosen_state


    def rank_states_by_reward(self):

        max_buffer_size = []
        for q_buffer in self.queue_model.q_args.keys():
            max_buffer = self.queue_model.q_args[q_buffer]
            max_buffer_size.append(max_buffer)
        
        queue_ranges = [range(max_buffer_size[i]) for i in range(len(max_buffer))]
        all_states = np.array(list(itertools.product(*queue_ranges)))
        
        all_q_values = []
        batch_size = 64
        for i in range(0, len(all_states), batch_size):
            batch_states = all_states[i:i+batch_size]
            batch_q_values = self.critic_network.forward(batch_states)
            all_q_values.append(batch_q_values)
        
        max_q_value = np.max(all_q_values)  
        normalized_q_values = all_q_values / max_q_value  

        state_q_value_pairs = list(zip(all_states, normalized_q_values))
        sorted_state_q_value_pairs = sorted(state_q_value_pairs, key=lambda x: x[1])

        return sorted_state_q_value_pairs

    def track_state(self, state_indices):
        
        # Increment the visit count for a given state
        self.visit_counts[state_indices] += 1
    
    def rank_states_by_visits(self):

        flattened_visit_counts = self.visit_counts.flatten()
        sorted_indices = np.argsort(flattened_visit_counts)[::-1]
        sorted_visit_counts = flattened_visit_counts[sorted_indices]
        multi_dim_indices = np.unravel_index(sorted_indices, self.visit_counts.shape)
        ranked_states_and_counts = list(zip(zip(*multi_dim_indices), sorted_visit_counts))
        visited_states_and_counts = [item for item in ranked_states_and_counts if item[1] > 0]
        
        return visited_states_and_counts

    
if __name__ == "__main__":
    Queue_model = Queue_network()
    DDPG_agent = DDPG(model = Queue_model)
    num_stored_experiences = 0
    batch_size = 100
    
    while True:
        current_state = DDPG_agent.explore_state()
        action = DDPG_agent.get_action(current_state)
        #TODO: need to modify Queue_model.get_next_state() such that it returns reward, next_state, done
        reward, next_state, done = DDPG_agent.model.get_next_state(action)
        DDPG_agent.store_to_buffer(current_state, action, reward, next_state)
        num_stored_experiences += 1

        if num_stored_experiences > batch_size:
            DDPG_agent.train()