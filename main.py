#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gym

from agents.reinforce_agent import ReinforceAgent
from utils.network_architectures import Actor, ActorWithoutBaseline, ActorWithSharedLayers

from utils.wrappers import make_cartpole_swing_up

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v1")

n_episodes = 10

n_features = 4
n_actions = 2
n_neurons = 32

# REINFORCE
actor_1 = ActorWithoutBaseline(n_features=n_features,
                               n_actions=n_actions,
                               n_neurons=n_neurons)

agent_1 = ReinforceAgent(device=device,
                         n_actions=n_actions,
                         actor=actor_1,
                         lr=0.001,
                         gamma=0.99)

# REINFORCE with baseline
actor_2 = Actor(n_features=n_features,
                n_actions=n_actions,
                n_neurons=n_neurons)

agent_2 = ReinforceAgent(device=device,
                         n_actions=n_actions,
                         actor=actor_2,
                         gamma=0.99)

# Training
step_list_1, return_list_1 = agent_1.train(env, n_episodes)
step_list_2, return_list_2 = agent_2.train(env, n_episodes)

# Plot
def mean_window(return_list, window=50):
    return [sum(return_list[i:i+window])/window for i in range(len(return_list) - window)]

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(mean_window(return_list_1), color='dodgerblue', label="REINFORCE")
ax.plot(mean_window(return_list_2), color='black', label="REINFORCE with baseline")

plt.legend()
plt.title("Learning curve for CartPole")
plt.xlabel("Episodes")
plt.ylabel("Return")

plt.savefig('reinforce-cartpole3.png', dpi=1000)
plt.show()