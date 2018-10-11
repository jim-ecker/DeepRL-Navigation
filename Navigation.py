from python.unityagents import UnityEnvironment
import numpy as np
import torch
from dqn_agent import Agent
import click
import time

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

# get default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


def timeit(method):
    def timed(*args, **kw):
        import datetime
        from timeit import default_timer as timer
        t_0     = timer()
        result  = method(*args, **kw)
        t_end   = timer()
        seconds = datetime.timedelta(seconds=t_end - t_0)
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = t_end - t_0
        else:
            print('\n\nExecution Time')
            print('\r{}\t{}'.format(method.__name__, seconds))
        return result

    return timed

@timeit
def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, **kwargs):
    from collections import deque
    if 'agent' in kwargs:
        agent = kwargs.get('agent')
    scores        = []
    scores_window = deque(maxlen=100)
    averages      = {0: 0, 12: 12, 19:19}
    eps           = eps_start
    i_episode     = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        state = env_info.vector_observations[0]                 # get the current state
        score = 0                                               # initialize the score
        while True:
            action = agent.act(state, eps)                      # select an action
            env_info = env.step(action)[brain_name]             # send the action to the environment
            next_state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]                       # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                     # update the score
            state = next_state                                  # roll over the state to next time step
            if done:                                            # exit loop if episode finished
                break

        # save most recent scores
        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay * eps)                     # decay epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            averages[str(i_episode)] = np.mean(scores_window)
        # Banana environment is solved if avg score over 100 episodes is >= 13
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    env.close()
    if 'report' in kwargs:
        kwargs['report'].add_plot(scores)
    return scores, averages


class Report:

    def __init__(self, agent=None):
        if agent is not None:
            self.agent      = agent
        self.plots      = []
        self.averages   = {}
        self.scores     = []
        self.wall_time  = {}
    def from_dict(*args, **kwargs):
        import pandas as pd
        report = []
        if kwargs is not None:
            for key, val in kwargs.items():
                report.append((key, val))
        return pd.DataFrame.from_items(report)

    def add_plot(self, data, labels=('Episode', 'Score')):
        import matplotlib.pyplot as plt
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if labels is not None:
            x_label, y_label = [label for label in labels]
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        plt.plot(np.arange(1, len(data) + 1, step=1), data)
        self.plots.append(plt)

    def show_plots(self):
        for plot in self.plots:
            plot.show()

    def run(self, exe, **kwargs):
        self.scores, self.averages = exe(**dict(kwargs, agent=self.agent, log_time=self.wall_time, report=self))

        self.show_plots()
        return self

    def __str__(self):
        from tabulate import tabulate
        import pandas as pd
        import datetime
        avg_table = pd.DataFrame.from_records([(key, val) for key, val in self.averages.items()])
        report =  [('Averages', tabulate(avg_table, tablefmt='fancy_grid', headers=["Ep", "Score"], showindex='never'))]
        wall_time_table = pd.DataFrame.from_records([(key, datetime.timedelta(seconds=val)) for key, val in self.wall_time.items()] + [('Total', datetime.timedelta(seconds=sum(self.wall_time.values())))])
        report += [('Wall Time', tabulate(wall_time_table, tablefmt='fancy_grid', headers=["Fn", "Time"], showindex='never'))]
        report += [('Episodes', len(self.scores))]
        report_table = pd.DataFrame.from_records(report)
        return tabulate(report_table, tablefmt='fancy_grid', showindex='never')


if __name__ == '__main__':
    report    = Report(Agent(state_size=state_size, action_size=action_size, seed=1337)).run(dqn, n_episodes=5)

    print(report)
