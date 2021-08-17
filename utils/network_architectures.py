#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Policy and actor-critic network architectures.

Examples of network architectures suited for classic control and Atari 2600
environments.

REINFORCE: Networks must implement a forward method as well as a baseline
method, eventually returning 0 in case of REINFORCE without baseline (see
architecture `ReinforceNetworkWithoutBaseline`).

A2C:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ReinforceNetwork(nn.Module):
    """Classic control policy network architecture with baseline."""
    
    def __init__(self, n_features, n_actions, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_actions: int, number of neurons of the last layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(ReinforceNetwork, self).__init__()
        self.policy_fc1 = nn.Linear(n_features, n_neurons)
        self.policy_fc2 = nn.Linear(n_neurons, n_neurons)
        self.policy_fc3 = nn.Linear(n_neurons, n_actions)
        
        self.baseline_fc1 = nn.Linear(n_features, n_neurons)
        self.baseline_fc2 = nn.Linear(n_neurons, n_neurons)
        self.baseline_fc3 = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        x = F.relu(self.policy_fc1(x))
        x = F.relu(self.policy_fc2(x))
        x = self.policy_fc3(x)
        return F.softmax(x, dim=1)
    
    def baseline(self, x):
        x = F.relu(self.baseline_fc1(x))
        x = F.relu(self.baseline_fc2(x))
        x = self.baseline_fc3(x)
        return x

class ReinforceNetworkWithSharedLayers(nn.Module):
    """Classic control policy network architecture with baseline with shared
    layers."""
    
    def __init__(self, n_features, n_actions, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_actions: int, number of neurons of the last layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(ReinforceNetworkWithSharedLayers, self).__init__()
        self.common_fc1 = nn.Linear(n_features, n_neurons)
        
        self.policy_fc2 = nn.Linear(n_neurons, n_neurons)
        self.policy_fc3 = nn.Linear(n_neurons, n_actions)
        
        self.baseline_fc2 = nn.Linear(n_neurons, n_neurons)
        self.baseline_fc3 = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        x = F.relu(self.common_fc1(x))
        x = F.relu(self.policy_fc2(x))
        x = self.policy_fc3(x)
        return F.softmax(x, dim=1)
    
    def baseline(self, x):
        x = F.relu(self.common_fc1(x))
        x = F.relu(self.baseline_fc2(x))
        x = self.baseline_fc3(x)
        return x

class ReinforceNetworkWithoutBaseline(nn.Module):
    """Classic control policy network architecture without baseline."""
    
    def __init__(self, n_features, n_actions, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_actions: int, number of neurons of the last layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(ReinforceNetworkWithoutBaseline, self).__init__()
        self.policy_fc1 = nn.Linear(n_features, n_neurons)
        self.policy_fc2 = nn.Linear(n_neurons, n_neurons)
        self.policy_fc3 = nn.Linear(n_neurons, n_actions)
    
    def forward(self, x):
        x = F.relu(self.policy_fc1(x))
        x = F.relu(self.policy_fc2(x))
        x = self.policy_fc3(x)
        return F.softmax(x, dim=1)
    
    def baseline(self, x):
        return torch.tensor([[0]])


class A2CNetwork(nn.Module):
    def __init__(self, n_features, n_actions, n_neurons):
        super(A2CNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(n_features, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 1))
        
        self.actor = nn.Sequential(
            nn.Linear(n_features, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_actions),
            nn.Softmax(dim=1))
    
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value