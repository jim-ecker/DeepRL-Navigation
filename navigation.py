import numpy as np
import torch
import click
from python.unityagents import UnityEnvironment
from agent              import DQNAgent
from utilities          import timeit, Report

@timeit
def dqn(env, brain_name, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, **kwargs):
    from collections import deque
    if 'agent' in kwargs:
        agent = kwargs.get('agent')
    scores        = []
    scores_window = deque(maxlen=100)
    averages      = {0: 0, 12: 12, 19:19}
    eps           = eps_start
    i_episode     = 0
    print()
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


@click.command()
@click.option(
    "--novis",
    is_flag=True,
    help="Run without visualization"
)
@click.option (
    "--env-dir",
    default="Banana_Linux",
    help="Set directory where environment is implemented"
)
@click.option (
    "--env-file",
    default="Banana.x86_64",
    help="Set file for environment"
)
@click.option(
    "--n-episodes",
    default=2000,
    help="Set number of episodes to run agent"
)
@click.option(
    "--seed",
    default=1337,
    help="Set random seed for reproducibility"
)
@click.option(
    "--prioritized",
    is_flag=True,
    help="Use prioritized replay memory"
)
def run(novis, env_dir, env_file, n_episodes, seed, prioritized):
    if novis:
        env_dir = "{}_NoVis".format(env_dir)

    env = UnityEnvironment(file_name="environments/{}/{}".format(env_dir, env_file))

    # get default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    # print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    # print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    # print('States look like:', state)
    state_size = len(state)
    # print('States have length:', state_size)

    report = Report(DQNAgent(state_size=state_size, action_size=action_size, seed=seed, prioritized=prioritized)).run(dqn, env=env, brain_name=brain_name, n_episodes=n_episodes)
    print(report)


if __name__ == '__main__':
    run()
