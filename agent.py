import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import QNetwork
from memory import ReplayBuffer


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, prioritized=False, cpu=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        if cpu:
            self.device      = torch.device("cpu")
        else:
            self.device      = torch.device("cuda:0")
            
        self.state_size      = state_size
        self.action_size     = action_size
        self.seedval         = seed
        self.seed            = random.seed(seed)
        self.buffer_size     = buffer_size
        self.batch_size      = batch_size
        self.gamma           = gamma
        self.tau             = tau
        self.lr              = lr
        self.update_every    = update_every
        
        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_local.apply(self.weights_init)
        
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.prioritized     = prioritized
        self.memory          = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed, self.device)
        self.t_step          = 0


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
        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()

                self.learn(experiences)

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.uniform() > eps:
            # return torch.argmax(action_values[0]).item()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(0, self.action_size)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target      = rewards + (self.gamma * Q_target_next * (1 - dones))

        Q_E = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_E, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return abs(Q_E - Q_target)

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
