import itertools
import numpy as np

from collections import defaultdict
from functools import partial

from lib.plotting import EpisodeStats
from utils import ConstantLearningRate, epsilon_greedy_policy


def sarsa(env, num_episodes, use_expected=False, discount_factor=0.95,
          epsilon=0.1, alpha=ConstantLearningRate(0.2), debug_callback=None):

    # We start with an all-zero Q-function estimate.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keep track of useful statistics.
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Define behavior policy.
    policy = partial(epsilon_greedy_policy, Q=Q, epsilon=epsilon,
                     num_actions=env.action_space.n)

    def sample_action(state):
        action_probs = policy(state)
        action = np.random.choice(env.action_space.n, p=action_probs)
        return action, action_probs

    # Run episodes.
    for episode_idx in xrange(num_episodes):
        state = env.reset()
        action, _ = sample_action(state)

        for t in itertools.count():
            # Act according to the behavior policy.
            next_state, reward, done, _ = env.step(action)

            # Update reward statistics.
            stats.episode_rewards[episode_idx] += reward

            td_target = reward
            if use_expected:
                _, action_probs = sample_action(next_state)
                td_target += discount_factor * action_probs.dot(Q[next_state])
            else:
                next_action, _ = sample_action(next_state)
                td_target += discount_factor * Q[next_state][next_action]

            Q[state][action] += alpha(state, action) * (td_target -
                                                        Q[state][action])

            # import pprint
            # print '======= Step {}'.format(t)
            # print 'State:       {}'.format(state)
            # print 'Action:      {}'.format(action)
            # print 'Next state:  {}'.format(next_state)
            # print 'Next action: {}'.format(next_action)
            # pprint.pprint(dict(Q))
            # if t == 5:
            #     return Q, stats

            if use_expected:
                next_action, _ = sample_action(next_state)

            state, action = next_state, next_action

            if done:
                stats.episode_lengths[episode_idx] = t + 1
                if debug_callback is not None:
                    debug_callback(episode_idx, stats)
                break

    return Q, stats
