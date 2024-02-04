from __future__ import absolute_import

from collections import deque
import tensorflow
import random

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

        self.model = model
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

    def initialize_state(self):
        #TODO: randomly intialize state
        pass
    

Queue_model = Q_network()
DDPG_agent = DDPG(model = Queue_model)
num_stored_experiences = 0
batch_size = 100
current_state = DDPG_agent.initialize_state()
while True:
    action = DDPG_agent.get_action(current_state)
    #TODO: need to modify Queue_model.get_next_state() such that it returns reward, next_state, done
    reward, next_state, done = DDPG_agent.model.get_next_state(action)
    DDPG_agent.store_to_buffer(current_state, action, reward, next_state)
    num_stored_experiences += 1

    if num_stored_experiences > batch_size:
        DDPG_agent.train()