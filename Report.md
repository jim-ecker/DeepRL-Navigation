## Report
This project is an implementation of a Deep Q Network using [PyTorch](https://pytorch.org/) and [Python](https://www.python.org/) 3

### Deep Q Network

An implementation of the Deep Q Learning neural network introduced by Deepmind in the following Nature article:

[**_Human-level control through deep reinforcement learning_**](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), V. Mnih, K. Kavukcuoglu, D. Silver, A. Rusu, J. Veness, M. Bellemare, A. Graves, M. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. **Nature 518 (7540): 529--533** (February 2015)

<Nature cover here>

The **Deep Q Network (DQN)** implemented in this repo is a multilayer Fully Connected neural network that approximates the Q-function for the state-space over which the agent is performing some set of actions using the agent's velocity and ray data for objects in its forward path as input.

<DQN network diagram here>

The DQN achieves stability via two main features:

  * **Experience Replay**
    
    Upon entering each state, the agent selects an action via its action selection policy. It then sends this action to the environment.This generates an "experience," represented as the tuple (s,a,r,s'). This experience gives the agent the information it needs to evaluate its performance: s_t, the action selected via the action selection policy, \alpha, reward yielded by the pair s_t/\alpha, and the resultant next state, s'. The experience generated is then stored in the agent's "experience replay memory," E.

    Experiences exhibit high correlation given temporal locality since we are working with trajectories with temporal reward dependence. One should decorrelate the agent's experience in order to avoid overfitting the agent's action selection. This is achieved by sampling a finite batch of experiences b, representing a subset of E, via the uniform distrubtion. The agent uses then uses b as its data in the learning phase.
    
    I have implemented the agent's experience reply as a Python [list](https://docs.python.org/3/tutorial/datastructures.html) turned [ring buffer](https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html), bounded by the Buffer Size hyperparameter. 
    
    Each experience is stored as a [namedtuple](https://docs.python.org/2/library/collections.html#collections.namedtuple) of the form:
    
    state | action | reward | next_state | done
    ------|--------|--------|------------|------
    
    The [ReplayBuffer](memory.py) class exposes the following methods:
    
    Method | Arguments| Description
    -------|----------|------------
    add | state, action, reward, next_state, done | Add a new experience to memory
    sample | | Sample batch, of size ReplayBuffer.batch_size, uniformly from memory
    
  * **Target Networks**
    
    Since the agent is evaluating the value of each state/action pair via a Bellman equation, the agent is affecting its own network's weight structure upon each learning phase. This means that, effectively, the ground is moving under the agent as it's learning - introducing significant instability into the network. Target networks alleviate this instability by holding a network "frozen in time" separate from the network upon which the learning is happening. This way, intermittent updates happen to the agent asynchronously without introducing the instability of online network updates.

  The agent uses two networks, [DQNAgent](agent.py).qnetwork_local and [DQNAgent](agent.py).qnetwork_target. The latter of these two networks acts as the target network for the agent, while the former acts as the network under evaluation.

#### Performance

