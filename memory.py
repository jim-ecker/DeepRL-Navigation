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
        """
        self.action_size = action_size
        self.memory      = deque(maxlen=buffer_size)
        self.batch_size  = batch_size
        self.experience  = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed        = random.seed(seed)
        self.device      = device

    def add(self, state, action, reward, next_state, done, error=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(np.vstack([e.state      for e in experiences if e is not None])).float().to(self.device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences if e is not None])).long().to(self.device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

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


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, action_size, buffer_size, batch_size, seed, device, alpha=0.6, beta=0.4, beta_ips=0.001, epsilon=0.01):
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed, device)
        assert alpha >= 0
        self._alpha    = alpha
        self._beta     = beta
        self._beta_ips = beta_ips
        self._epsilon  = epsilon

        tree_size     = 1
        while tree_size < batch_size:
            tree_size *= 2

        self.tree         = SumTree(tree_size)
        self.max_priority = 1.0

    def _priority(self, error):
        return (error + self._epsilon) ** self._alpha

    def add(self, state, action, reward, next_state, done, error=None):
        self.tree.add((state, action, reward, next_state, done), self._priority(error) if error is not None else 0)

    def sample(self, batch_size):
        batch      = []
        indexes    = []
        segment    = self.tree.segment(batch_size)
        priorities = []

        self.beta = np.min([1., self._beta + self._beta_ips])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)

            index, priority,  data = self.tree.get(s)
            priorities.append(priority)
            batch.append(data)
            indexes.append(index)

        sampling_distribution = self.tree.get_distribution(priorities)
        weight  = np.power(self.tree.entries * sampling_distribution, -self.beta)
        weight /= weight.max()
        return batch, indexes, weight

    def update(self, index, error):
        priority = self._priority(error)
        self.tree.update(index, priority)


class SumTree:
    write = 0

    def __init__(self, tree_size):
        self.size    = tree_size
        self.tree    = np.zeros(2 * self.size - 1)
        self.data    = np.zeros(self.size, dtype=object)
        self.entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def _total(self):
        return self.tree[0]

    def segment(self, size):
        return self._total() / size

    def get_distribution(self, priorities):
        return priorities / self._total()

    # store priority and sample
    def add(self, data, error):
        index = self.write + self.size - 1

        self.data[self.write] = data
        self.update(index, error)

        self.write += 1
        if self.write >= self.size:
            self.write = 0

        if self.entries < self.size:
            self.entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.size + 1

        return idx, self.tree[idx], self.data[dataIdx]
