import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import Actor, Critic, RewardModel, NextStateModel
from buffer import ReplayBuffer
import random

class DDPGAgent():
    def __init__(self, n_states, n_actions, params):
        self.state_size = n_states
        self.action_size = n_actions

        # hyperparameters
        self.batch_size = params['batch_size']
        self.tau = params['tau']
        self.lr = params['lr']
        self.discount= params['discount']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']

        # create buffer to replay experiences
        self.buffer = ReplayBuffer(size=1000)

        # actor networks + optimizer
        self.actor = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # critic networks + optimizer
        self.critic = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # model networks + optimizer (separate reward and next state)
        self.reward_model = RewardModel(self.state_size,self.action_size)
        self.reward_model_optim = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr)
        self.next_state_model = NextStateModel(self.state_size,self.action_size)
        self.next_state_model_optim = torch.optim.Adam(self.next_state_model.parameters(), lr=self.lr)

        # hard update to ensure same weights
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        
    def update_actor_network(self, experience, soft_update=True):

        state, action, reward, next_state = experience

        self.actor.zero_grad()
        policy_loss = -self.critic(state, self.actor(state))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        if soft_update:
            self.soft_update(self.critic_target, self.critic, self.tau)
        else:
            self.hard_update(self.critic_target, self.critic, self.tau)

    def update_critic_network(self, experience, soft_update=True):

        state, action, reward, next_state = experience

        next_q_value = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + self.discount*next_q_value
        actual_q = self.critic(state, action)

        self.critic.zero_grad()
        q_loss = nn.MSELoss(actual_q, target_q)
        q_loss.backward()
        self.critic_optim.step()

        if soft_update:
            self.soft_update(self.actor_target, self.actor, self.tau)
        else:
            self.hard_update(self.actor_target, self.actor, self.tau)


    def fit_model(self, batch_size, epochs=3):
        
        data = self.buffer.get_items()
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                state, action, reward, next_state = batch

                # update network for Model(s,a) = r 
                pred_reward = self.reward_model([state,action])
                self.reward_model.zero_grad()
                loss1 = nn.MSELoss(pred_reward, reward)
                loss1.backward()
                self.reward_model_optim.step()

                # update network for Model(s,a) = s'
                pred_next_state = self.next_state_model([state,action])
                self.next_state_model.zero_grad()
                loss2 = nn.MSELoss(pred_next_state, next_state)
                loss2.backward()
                self.next_state_model_optim.step()


        # for experience in self.buffer:
        # state, action, reward, next_state = experience

    # def step(environment, action):
        

    def plan(self, experience):

        state, action, reward, next_state = experience

        for _ in range(self.planning_steps):
            action2 = action + self.epsilon
            # reward2, next_state2 = 
    
    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )
    
    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)



if __name__ == "__main__":
    
    params = {
            'batch_size': 1,
            'tau': 0.1,
            'lr': 0.001,
            'discount': 0.1,
            'epsilon_i': 1,
            'epsilon_f': 0.1
            }

    n_states = 4
    n_actions = 4

    test = DDPGAgent(n_states, n_actions, params)

    

        
