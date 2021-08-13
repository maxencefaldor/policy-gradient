#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Policy network architectures.

Examples of network architectures suited for classic control environments. Note
that the networks must implement a forward method as well as a baseline method,
eventually returning 0 in case of REINFORCE without baseline (see architecture
`ActorWithoutBaseline`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Classic control policy network architecture with baseline."""
    
    def __init__(self, n_features, n_actions, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_actions: int, number of neurons of the last layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(Actor, self).__init__()
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

class ActorWithSharedLayers(nn.Module):
    """Classic control policy network architecture with baseline with shared
    layers."""
    
    def __init__(self, n_features, n_actions, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_actions: int, number of neurons of the last layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(ActorWithSharedLayers, self).__init__()
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

class ActorWithoutBaseline(nn.Module):
    """Classic control policy network architecture without baseline."""
    
    def __init__(self, n_features, n_actions, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_actions: int, number of neurons of the last layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(ActorWithoutBaseline, self).__init__()
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
