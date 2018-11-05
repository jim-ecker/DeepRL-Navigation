import numpy as np
import random
from collections import namedtuple, deque
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): CPU, or some GPU
        """
        self.action_size = action_size
        self.memory      = deque(maxlen=buffer_size)
        self.batch_size  = batch_size
        self.experience  = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed        = random.seed(seed)
        self.device      = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences if e is not None])).pin_memory().float().to(self.device, non_blocking=True)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences if e is not None])).pin_memory().long().to(self.device, non_blocking=True)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences if e is not None])).pin_memory().float().to(self.device, non_blocking=True)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).pin_memory().float().to(self.device, non_blocking=True)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences if e is not None]).astype(np.uint8)).pin_memory().float().to(self.device, non_blocking=True)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        for key, val in self.__dict__.items():
            if key == 'experience':
                format_string += '  {0}: {1} (\n    fields: {2}\n  )\n'.format(key, val.__name__, val._fields)
            elif key == 'memory':
                format_string += '  {0}: {1} (\n    maxlen: {2}\n  )\n'.format(key, val.__class__.__name__, val.maxlen)
            else:
                format_string += '  {0}: {1}\n'.format(key, val)
        format_string += ')'
        return format_string