import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=4, hidden2=3):
        """
        Neural network representing the actor (denoted \mu in literature). Given
        a state vector as input, return the action vector as output.

        Parameters:
        - n_states (int): Number of nodes in the system. Represents the size of the state vector.
        - n_actions (int): Number of nodes in the system. Represents the size of the action vector.

        State vector: Vector of shape (N, 1), where each element represents the
        queue length at each node.
        
        Action vector: Vector of shape (N, 1), where each element represents the
        probability of a job arriving at the node.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
    

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=4, hidden2=3):
        """
        Neural network representing the critic (denoted Q in literature). Given a
        state vector and action vector as input, return the Q value of this
        state-action pair.

        Parameters:
        - n_states (int): Number of nodes in the system. Represents the size of the state vector.
        - n_actions (int): Number of nodes in the system. Represents the size of the action vector.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1+n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self,xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have shape
                        (N,1)

        Returns:
        - torch.Tensor: Output Q value.
        """
        x,a = xa
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class RewardModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=4, hidden2=3):
        """
        Neural network representing the critic (denoted Q in literature). Given a
        state vector and action vector as input, return the Q value of this
        state-action pair.

        Parameters:
        - n_states (int): Number of nodes in the system. Represents the size of the state vector.
        - n_actions (int): Number of nodes in the system. Represents the size of the action vector.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1+n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self,xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have shape
                        (N,1)

        Returns:
        - torch.Tensor: Output Q value.
        """
        x,a = xa
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
class NextStateModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=4, hidden2=3):
        """
        Neural network representing the critic (denoted Q in literature). Given a
        state vector and action vector as input, return the Q value of this
        state-action pair.

        Parameters:
        - n_states (int): Number of nodes in the system. Represents the size of the state vector.
        - n_actions (int): Number of nodes in the system. Represents the size of the action vector.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1+n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, n_states)
        self.relu = nn.ReLU()

    def forward(self,xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have shape
                        (N,1)

        Returns:
        - torch.Tensor: Output Q value.
        """
        x,a = xa
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
    