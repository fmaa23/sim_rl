import random
import torch
from torch.utils.data import TensorDataset

class ReplayBuffer():
    def __init__(self, max_size):
        """
        Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.current_size = 0
        self.buffer = []

    
    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """ 
        # state, action, reward, next_state = transition
        
        """if self.current_size > self.max_size:
            self.buffer.pop(0)
      
        self.buffer += [transition]
        self.current_size += 1"""

        if len(self.buffer) == self.max_size:
            self.buffer.pop(0)  # Remove oldest transition
        self.buffer.append(transition)  # Add new transition
            

    def sample(self, batch_size):
        """
        Get a random sample from the replay buffer.
        
        Args:
            batch_size (int): size of sample

        Returns:
            iterable list) with objects sampled from buffer without replacement
        """
         # Check if there are enough transitions to sample the requested batch size
        if batch_size > len(self.buffer):
            raise ValueError("Sample size larger than buffer")
        return random.sample(self.buffer, batch_size)
    
    def get_items(self):
        """
        Returns a 4-tuple containing all items in the buffer (as torch.Tensors)
        
        """
        state = torch.stack([item[0] for item in self.buffer], dim=0).to(self.device)
        action = torch.stack([item[1] for item in self.buffer], dim=0).to(self.device)
        reward = torch.stack([torch.tensor(item[2]).view(-1) for item in self.buffer], dim=0).to(self.device)
        next_state = torch.stack([item[3] for item in self.buffer], dim=0).to(self.device)
        return TensorDataset(state, action, reward, next_state)
        #return torch.cat(torch.tensor(self.buffer[:,0]), dim=0)
