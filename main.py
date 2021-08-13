#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gym

import agents.reinforce_agent
import agents.reinforce_agent_with_baseline
from utils.network_architectures import Actor
from utils.network_architectures import Baseline

from utils.wrappers import make_cartpole_swing_up

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v1")
# env = make_cartpole_swing_up("CartPole-v1", max_episode_steps=1000)

n_features = 4
n_actions = 2
n_neurons = 32

# REINFORCE
actor_1 = Actor(n_features=n_features,
                n_actions=n_actions,
                n_neurons=n_neurons)

agent_1 = agents.reinforce_agent.ReinforceAgent(device=device,
                                                n_actions=n_actions,
                                                actor=actor_1,
                                                lr_actor=0.0005,
                                                gamma=0.99)

# REINFORCE with baseline
actor_2 = Actor(n_features=n_features,
                n_actions=n_actions,
                n_neurons=n_neurons)
baseline = Baseline(n_features=n_features,
                    n_neurons=n_neurons)

agent_2 = agents.reinforce_agent_with_baseline.ReinforceAgent(device=device,
                                                            n_actions=n_actions,
                                                            actor=actor_2,
                                                            baseline=baseline,
                                                            lr_actor=0.0005,
                                                            lr_baseline=0.0005,
                                                            gamma=0.99)


# Training
step_list_1, return_list_1 = agent_1.train(env, 1000)
step_list_2, return_list_2 = agent_2.train(env, 1000)

# Plot
def mean_window(return_list, window=50):
    return [sum(return_list[i:i+window])/window for i in range(len(return_list) - window)]

plt.plot(mean_window(return_list_1), label='REINFORCE')
plt.plot(mean_window(return_list_2), label='REINFORCE with baseline')
plt.legend()
plt.xlabel("episodes")
plt.ylabel("returns")