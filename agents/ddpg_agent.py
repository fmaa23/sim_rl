import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from foundations.model import Actor, Critic, RewardModel, NextStateModel
from foundations.buffer import ReplayBuffer
torch.autograd.set_detect_anomaly(True)

gradient_dict = {} 

class DDPGAgent():
    def __init__(self, n_states, n_actions, hidden, params):
        """
        Creates a DDPG agent in an environment. This class has six neural networks as attributes:

        Actor Networks (x2):
            - Policy Network and Target Network: Used for action selection given a state vector as input
        Critic Networks (x2):
            - Policy Network and Target Network: Outputs Q-value given a state-action pair as input
        Reward Network (x1):
            - Agent's prediction of reward given a state-action pair, based on its internal model of the environment
        Next State Network (x1):
            - Agent's prediction of the next state given a state-action pair, based on its internal model of the environment

        Parameters:
            n_states (int): Dimensions of the state vector
            n_actions (int): Dimensions of the action vector
            hidden (dict of list): Each value is a list specifying the size of the hidden layers
                                   in the four unique neural networks that define this class. Example:
                                   hidden = {
                                       'actor': [64, 64],
                                       'critic': [64, 64],
                                       'reward_model': [10, 10],
                                       'next_state_model': [10, 10]
                                   }
            params (dict of float): Dictionary of hyperparameters

        """
        self.state_size = n_states
        self.action_size = n_actions
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyperparameters
        self.tau = params['tau']
        self.lr = params['critic_lr']
        self.actor_lr = params["actor_lr"]
        self.discount= params['discount']
        self.epsilon = params['epsilon']
        self.planning_steps = params['planning_steps']

        # create buffer to replay experiences
        self.buffer = ReplayBuffer(max_size=params['buffer_size'])

        # actor networks + optimizer
        self.actor = Actor(n_states, n_actions, hidden['actor']).to(self.device)
        self.actor_target = Actor(n_states, n_actions,hidden['actor']).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, gamma=0.8)

        # critic networks + optimizer
        self.critic = Critic(n_states, n_actions, hidden['critic']).to(self.device)
        self.critic_target = Critic(n_states, n_actions, hidden['critic']).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # hard update to ensure same weights
        self.hard_update(network="actor")
        self.hard_update(network="critic")

        # model networks + optimizer (separate reward and next state)
        self.reward_model = RewardModel(n_states, n_actions, hidden['reward_model']).to(self.device)
        self.reward_model_optim = torch.optim.Adam(self.reward_model.parameters(), lr=self.lr)
        self.next_state_model = NextStateModel(n_states, n_actions, hidden['next_state_model']).to(self.device)
        self.next_state_model_optim = torch.optim.Adam(self.next_state_model.parameters(), lr=self.lr)

        # loss function
        self.loss = nn.MSELoss()

        # set to training mode
        self.train()

        # current state and action
        self.s_t = None 
        self.a_t = None

        # record state visits
        self.visited_count = {}

        global gradient_dict

        self.num_select_action = 0

    def update_actor_network(self, batch):
        total_policy_loss = torch.zeros(1, requires_grad=True).to(self.device)
        self.actor_optim.zero_grad()

        for experience in batch:
            state, action, reward, next_state = experience
            # self.actor.zero_grad()                                                      # TO CHANGE
            policy_loss = -self.critic([state, self.actor(state)])                      # TO CHANG
            policy_loss = policy_loss.mean().to(torch.float32)                          # TO CHANGE
            total_policy_loss = total_policy_loss + policy_loss
        
        mean_policy_loss = total_policy_loss / len(batch)
        mean_policy_loss.backward() # retain_graph=True

        self.actor_scheduler.step()
        
        if True:
            
            for name, parameter in self.actor.named_parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad.data
                    param_gradient = gradient_dict.setdefault(name, [])
                    param_gradient.append(gradient.flatten().cpu().numpy().tolist())
                    gradient_dict[name] = param_gradient
                                                    
            self.actor_optim.step()
            mean_policy_loss = mean_policy_loss.detach()
            if gradient is None:
                ValueError("gradient is None")
            else:
                return mean_policy_loss.item(), gradient_dict
        else:
            return mean_policy_loss.item(), gradient_dict

    def update_critic_network(self, batch):
        total_critic_loss = torch.zeros(1, requires_grad=True).to(self.device)
        self.critic_optim.zero_grad()
        
        for experience in batch:
            state, action, reward, next_state = experience
            next_q_value = self.critic_target([next_state, self.actor_target(next_state).detach()]).detach()
            target_q = reward + self.discount * next_q_value
            
            actual_q = self.critic([state, action])
            q_loss = nn.MSELoss()(actual_q.to(torch.float32), target_q.to(torch.float32))
            # q_loss.backward(retain_graph=True) # needed cause called multiple times in the same update in plan
            total_critic_loss = total_critic_loss + q_loss
        
        mean_critic_loss = total_critic_loss / len(batch)
        mean_critic_loss.backward()
        self.critic_optim.step()
        mean_critic_loss = mean_critic_loss.detach()
        return mean_critic_loss.item()

    def fit_model(self, batch_size, epochs=5):
        """
        Fits the agent's model of the environment M with all the data in the buffer B. This involves training
        two separate neural networks for the number of epochs specified:
            1. M1(s,a) = \hat{r}    --> predicts the reward for a given (s,a) pair
            2. M2(s,a) = \hat{s'}   --> predicts the next state for a given (s,a) pair
        
        Parameters:
            batch_size (int): The size of each batch of data to use during fitting of M.
            threshold (int): The minimum number of samples needed in the buffer before training.
            epochs (int): The number of epochs to train the two neural networks. Defaults to 5

        Returns:
            None            
        """
        # not sure if we need to reset Model(s,a) to be a new network
        # here we just take Model(s,a) from the previous iteration but re-train it
        reward_loss_list = []
        next_state_list = []
        if self.buffer.current_size < batch_size:
            raise Exception('Number of transitions in buffer fewer than chosen threshold value')
        
        data = self.buffer.get_items()
        #dataset = TensorDataset(data)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            for batch in dataloader:
                state, action, reward, next_state = batch

                # update network for Model(s,a) = r 
                self.reward_model_optim.zero_grad()
                pred_reward = self.reward_model([state.to(torch.float32), action.to(torch.float32)])
                # self.reward_model.zero_grad()
                loss1 = nn.MSELoss()(pred_reward.to(torch.float32), reward.to(torch.float32))
                reward_loss_list.append(loss1.item())
                loss1=loss1.to(torch.float32)
                
                # loss1.backward(retain_graph=True)
                loss1.backward()
                self.reward_model_optim.step()

                # update network for Model(s,a) = s'
                self.next_state_model_optim.zero_grad()
                pred_next_state = self.next_state_model([state.to(torch.float32), action.to(torch.float32)])
                # self.next_state_model.zero_grad()
                loss2 = nn.MSELoss()(pred_next_state.to(torch.float32), next_state.to(torch.float32))
                # print(f"next state model epoch {epoch}: loss {loss2.item()}; reward model epoch {epoch}: loss {loss1.item()}")
                # loss2.backward(_graretainph=True)
                next_state_list.append(loss2.item())
                loss2.backward()
                self.next_state_model_optim.step()
        return reward_loss_list, next_state_list

    def store_experience(self, experience):
        """
        Store an experience in the agent's buffer, given that it is in training mode.

        Parameters:
            experience (tuple): tuple of (state, action, reward, next_state) where each element is
                                of type torch.Tensor
        
        Returns:
            None
        """
        if self.training:
            self.buffer.push(experience)
        else:
            raise Exception("Agent is not in training mode. Use the .train() method to set the agent \
                                to training mode to push experiences to the buffer.")

    def plan(self, batch):
        """
        Implementing lines 12 to 16 in Dyna-DDPG.

        Given an experience (s,a,r,s'), plan for P steps. This involves
        - perturbing the action by a small value epsilon
        - obtaining reward and next state from the agent's model of the environment
        - updating the critic network using this experience (s, \hat{a}, \hat{r}, \hat{s'})
    
        Parameters:
            experience (tuple): tuple of (state, action, reward, next_state) where each element is
                                of type torch.Tensor  

        Returns:
            None
        
        """

        for num in tqdm(range(len(batch)), desc="Planning Progress"): 
            experience = batch[num]
            state, action, reward, next_state = experience
            experiences = []
            for _ in range(self.planning_steps):
                action_hat = (action + (torch.randn(len(action)) * self.epsilon).to(self.device))
                action_hat = torch.clamp(action_hat, min=0, max=1)
                reward_hat = self.reward_model([state, action_hat])
                next_state_hat = self.next_state_model([state, action_hat])
                experience_hat = (state, action_hat, reward_hat, next_state_hat)
                experiences.append(experience_hat)

            self.update_critic_network(experiences)


    def soft_update(self, network):
        """
        Perform a soft update of target network weights using the policy network weights. This involves updating
        the target network parameters slowly by interpolating between the current target network parameters and
        the current policy network parameters.

        Parameters:
            network (str): Specifies which network to update. Should be either 'actor' or 'critic'.

        Raises:
            Exception: If the input network parameter is neither 'actor' nor 'critic'.

        Returns:
            None

        """
        if network == "actor":
            target = self.actor_target
            source = self.actor
        elif network == "critic":
            target = self.critic_target
            source = self.critic
        else:
            raise Exception("Invalid input. Parameter should be either 'actor' or 'critic', depending \
                                on the network to be updated.")
        
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )


    def hard_update(self, network):
        """
        Perform a hard update of target network weights using the policy network weights. This directly
        copies the parameters of the source network to the target network. 

        Parameters:
            network (str): Specifies which network to update. Should be either 'actor' or 'critic'.

        Raises:
            Exception: If the input network parameter is neither 'actor' nor 'critic'.

        Returns:
            None
        """
        if network == "actor":
            target = self.actor_target
            source = self.actor
        elif network == "critic":
            target = self.critic_target
            source = self.critic
        else:
            raise Exception("Invalid input. Parameter should be either 'actor' or 'critic', depending \
                                on the network to be updated.")
        
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


    def convert_state(self, state_tensor):
        state_list = state_tensor.tolist()
        state_tuple = tuple([int(x) for x in state_list])
        return state_tuple

    def select_action(self, state):
  
        self.s_t = state
        try:
            self.a_t = self.actor(state.float()).detach() # modify
        except:
            print("wrong")

        # record visited states
        state_tuple = tuple(state.tolist())
        state_tuple = tuple(int(x) for x in state_tuple)
        self.visited_count[state_tuple] = self.visited_count.setdefault(state_tuple,0) + 1 
        self.num_select_action += 1
        return self.a_t


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.reward_model.eval()
        self.next_state_model.eval()
        self.training = False


    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.actor_target.train()
        self.reward_model.train()
        self.next_state_model.train()
        self.training = True
    

if __name__ == "__main__":
    pass
