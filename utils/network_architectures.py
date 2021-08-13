#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Classic control network architecture."""
    
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
