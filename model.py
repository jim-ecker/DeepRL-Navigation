import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, nodes=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            nodes (dict): keys = layer, val = number of nodes
        """
        if nodes is None:
            nodes = {
                'fc1' : 64,
                'fc2' : 64
            }
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, nodes["fc1"])
        self.fc2 = nn.Linear(nodes["fc1"] , nodes["fc2"])
        self.fc3 = nn.Linear(nodes["fc2"], action_size)

    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        pass
