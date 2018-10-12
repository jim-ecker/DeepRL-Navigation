import numpy as np
import random

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import ReplayBuffer

BUFFER_SIZE  = int(1e5)  # replay buffer size
BATCH_SIZE   = 64        # minibatch size
GAMMA        = 0.99      # discount factor
TAU          = 1e-3      # for soft update of target parameters
LR           = 5e-4      # learning rate
UPDATE_EVERY = 4         # how often to update the network

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, prioritized=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.seedval     = seed
        self.seed        = random.seed(seed)

        # Q-Network
        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_local.apply(self.weights_init)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.prioritized     = prioritized
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight)

    def __repr__(self):
        import pandas as pd
        from tabulate import tabulate
        agent_table = []
        for key, val in self.__dict__.items():
            if key == 'seed':
                agent_table.append((key, str(self.seedval)))
            elif key != 'seedval':
                agent_table.append((key, str(val)))
        return tabulate(pd.DataFrame.from_records(agent_table), tablefmt='fancy_grid', showindex='never')

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # return np.argmax(action_values.cpu().data.numpy())
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.prioritized:
            states, actions, rewards, next_states, dones, weights, indexes = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target      = rewards + (gamma * Q_target_next * (1 - dones))

        Q_E = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_E, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
