import itertools
import numpy as np

from collections import defaultdict
from functools import partial


class DynamicEpsilonGreedyPolicy(object):
    def __init__(self, Q, num_actions):
        self.Q = Q
        self.num_actions = num_actions
        self.reset()

    def reset(self):
        self.num_seen = defaultdict(lambda: 0)

    def __call__(self, observation):
        self.num_seen[observation] += 1
        epsilon = 1.0 / np.sqrt(self.num_seen[observation])
        return epsilon_greedy_policy(observation,
                                     Q=self.Q,
                                     epsilon=epsilon,
                                     num_actions=self.num_actions)


class ConstantLearningRate(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, observation, action):
        return self.alpha


class PolynomialLearningRate(object):
    def __init__(self, alpha=1.0, power=0.8):
        self.alpha = alpha
        self.power = power
        self.reset()

    def reset(self):
        self.num_updates = defaultdict(lambda: 0)

    def __call__(self, observation, action):
        self.num_updates[(observation, action)] += 1
        alpha = (self.alpha /
                 pow(self.num_updates[(observation, action)], self.power))
        return alpha


def epsilon_greedy_policy(observation, Q, epsilon, num_actions):
    A = np.ones(num_actions, dtype=np.float) * epsilon / num_actions
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A


def simulate_greedy_policy(env, Q):
    policy = partial(epsilon_greedy_policy,
                     Q=Q,
                     epsilon=0.0,
                     num_actions=env.action_space.n)

    state = env.reset()

    for t in itertools.count():
        action_probs = policy(state)
        action = np.random.choice(env.action_space.n, p=action_probs)
        state, reward, done, _ = env.step(action)

        env.render()
        print ""
        if done:
            break
