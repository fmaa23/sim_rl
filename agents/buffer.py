import random
import torch
from torch.utils.data import TensorDataset

class ReplayBuffer():
    def __init__(self, max_size = 1000):
        """
        Replay buffer initialization.

        Args:
            max_size (int): Maximum number of transitions stored by the replay buffer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_size = 1000
        self.buffer = []
        self.current_size = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Push a transition to the replay buffer. If the buffer exceeds `max_size`, the oldest transition is removed.

        Args:
            transition: Transition to be stored in the replay buffer. Can be of any type.
        """
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove the oldest transition
        self.buffer.append(transition)
        self.current_size += 1

    def sample(self, batch_size):
        """
        Get a random sample from the replay buffer.

        Args:
            batch_size (int): size of sample

        Raises:
            ValueError: If the requested batch_size exceeds the current buffer size or the buffer is empty.

        Returns:
            list: A list of randomly sampled transitions.
        """
        if batch_size > len(self.buffer):
            raise ValueError("Requested sample size exceeds current buffer size.")
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        
        return random.sample(self.buffer, batch_size)

    def get_items(self):
        """
        Returns a 4-tuple containing all items in the buffer (as torch.Tensors).

        Returns:
            TensorDataset: A dataset containing states, actions, rewards, and next states.
        """
        states, actions, rewards, next_states = zip(*self.buffer)
        state = torch.stack(states, dim=0).to(self.device)
        action = torch.stack(actions, dim=0).to(self.device)
        reward = torch.stack([torch.tensor(r).view(-1) for r in rewards], dim=0).to(self.device)
        next_state = torch.stack(next_states, dim=0).to(self.device)
        return TensorDataset(state, action, reward, next_state)