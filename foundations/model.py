import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        if x.size(-1) != out.size(-1):
            # Adapt identity if needed (e.g., during dimension change)
            identity = nn.Linear(x.size(-1), out.size(-1))(identity)
        out += identity  # Skip connection
        return out

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden):
        super(Actor, self).__init__()
        check_validity(hidden)
        
        layers = [nn.Linear(n_states, hidden[0]), nn.LayerNorm(hidden[0]), nn.LeakyReLU(0.2)]
        
        for i in range(1, len(hidden)):
            layers.append(ResidualBlock(hidden[i-1], hidden[i]))
        
        layers.append(nn.Linear(hidden[-1], n_actions))
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        
        self.layers = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.layers(state_tensor)
        #action = self.layers(state.float().to(self.device))
        return action

class Actor_Old(nn.Module):
    def __init__(self, n_states, n_actions, hidden=[3,3]):
        """
        Neural network representing the actor (denoted \mu in literature). Given
        a state vector as input, return the action vector as output.

        Parameters:
            n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
            n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
            hidden (list of ints):  Number of neurons in each hidden layer. len(hidden) = number of 
                                    hidden layers within the network.

        State vector: Vector of shape (N, 1), where each element represents the
        queue length at each node.
        
        Action vector: Vector of shape (N, 1), where each element represents the
        probability of a job arriving at the node.

        """
        super(Actor, self).__init__()
        check_validity(hidden)
        
        layers = [nn.Linear(n_states, hidden[0]), nn.ReLU()]
        layers.append(nn.LayerNorm(hidden[0]))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(1, len(hidden)):
            layers.append(nn.Linear(hidden[i-1], hidden[i]))
            layers.append(nn.LayerNorm(hidden[i]))
            layers.append(nn.LeakyReLU(0.2))
            
        layers.append(nn.Linear(hidden[-1], n_actions))
        # layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Sigmoid())                             # clip logits to [0,1] in masked action vector
        self.layers = nn.Sequential(*layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        if type(x)==np.ndarray: 
            x = torch.tensor(x)
        x = x.float()
        #if np.isnan(1/self.layers(x)[-1].item()):
        #    print('nan')
        # return self.layers(x.to("cuda:0"))
        return self.layers(x.to(self.device))

class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden):
        """
        Neural network representing the critic (denoted Q in literature). Given a
        state vector and action vector as input, return the Q value of this
        state-action pair.

        Parameters:
            n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
            n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
            hidden (list of ints):  Number of neurons in each hidden layer. len(hidden) = number of 
                                    hidden layers within the network.

        """
        super(Critic, self).__init__()
        check_validity(hidden)

        self.layer1 = nn.Sequential(nn.Linear(n_states, hidden[0]), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(hidden[0]+n_actions, hidden[1]), nn.LeakyReLU(0.2))

        layers = []
        for i in range(2, len(hidden)):
            layers.append(nn.Linear(hidden[i-1], hidden[i]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden[-1], 1))
        # layers.append(nn.ReLU())
        self.layer3 = nn.Sequential(*layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have shape
                        (N,1)

        Returns:
        - torch.Tensor: Output Q value.

        """
        x, a = xa
        if type(x) == np.ndarray: 
            x = torch.tensor(x)
        if type(a) == np.ndarray: 
            a = torch.tensor(x)
        x = x.float()

        # out = self.layer1(x.to("cuda:0"))
        out = self.layer1(x.to(self.device))

        if len(a.shape) == 1: 
            out = self.layer2(torch.cat([out,a]))
        else: 
            out = self.layer2(torch.cat([out,a],1))
        out = self.layer3(out)
        return out


class RewardModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden=[3,3]):
        """
        Neural network representing the DDPG agent's internal model of the environment.
        Given a state vector and action vector as input, returns the predicted reward
        of this state-action pair.

        Parameters:
            n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
            n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
            hidden (list of ints):  Number of neurons in each hidden layer. len(hidden) = number of 
                                    hidden layers within the network.

        """
        super().__init__()
        check_validity(hidden)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layer1 = nn.Sequential(nn.Linear(n_states, hidden[0]), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(hidden[0]+n_actions, hidden[1]), nn.LeakyReLU(0.2))

        layers = []
        for i in range(2, len(hidden)):
            layers.append(nn.Linear(hidden[i-1], hidden[i]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden[-1], 1))             # scalar output
        layers.append(nn.LeakyReLU())
        self.layer3 = nn.Sequential(*layers)


    def forward(self,xa):
        """
        Parameters:
            xa (list) : [state vector, action vector] where both vectors have
                        shape (N,1)

        Returns:
            torch.Tensor : predicted reward of state-action pair

        """

        x, a = xa
        if type(x) == np.ndarray: 
            x = torch.tensor(x)
        if type(a) == np.ndarray: 
            a = torch.tensor(x)
        x = x.float()

        out = self.layer1(x.to(self.device))
        if len(a.shape) == 1: 
            out = self.layer2(torch.cat([out,a]).to(self.device))
        else: 
            out = self.layer2(torch.cat([out,a],1).to(self.device))
        out = self.layer3(out)
        return out


class NextStateModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden=[3,3]):
        """
.       Neural network representing the DDPG agent's internal model of the environment.
        Given a state vector and action vector as input, returns the predicted next
        state of this state-action pair.

        Parameters:
            n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
            n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
            hidden (list of ints):  Number of neurons in each hidden layer. len(hidden) = number of 
                                    hidden layers within the network.

        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        check_validity(hidden)

        self.layer1 = nn.Sequential(nn.Linear(n_states, hidden[0]), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(hidden[0]+n_actions, hidden[1]), nn.LeakyReLU(0.2))

        layers = []
        for i in range(2, len(hidden)):
            layers.append(nn.Linear(hidden[i-1], hidden[i]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden[-1], n_states))             # vector output
        layers.append(nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(*layers)


    def forward(self,xa):
        """
        Parameters:
            xa (list) : [state vector, action vector] where both vectors have 
                        shape (N,1)

        Returns:
            torch.Tensor (n_states,) : predicted next state

        """
        x, a = xa
        if type(x) == np.ndarray: 
            x = torch.tensor(x.to(self.device))
        if type(a) == np.ndarray: 
            a = torch.tensor(x.to(self.device))
        x = x.float()

        out = self.layer1(x)
        if len(a.shape) == 1: 
            out = self.layer2(torch.cat([out,a]))
        else: 
            out = self.layer2(torch.cat([out,a],1))
        out = self.layer3(out)
        return out
    
    
def check_validity(hidden):
    """
    Helper function that checks the validity of the input 'hidden' to the 
    constructors of the neural networks

    """
    if type(hidden) != list or not all(isinstance(x,int) for x in hidden):
        raise Exception("The argument 'hidden' should be a list of integers.")
    if len(hidden) < 2:
        raise Exception("The list/tuple should have a length >= 2")