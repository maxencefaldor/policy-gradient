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
        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_actions)
    
    def forward(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=1)
    
    def logits(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Baseline(nn.Module):
    """Classic baseline network architecture."""
    
    def __init__(self, n_features, n_neurons):
        """Creates the layers.
        
        Args:
            n_features: int, number of neurons of the first layer.
            n_neurons: int, number of neurons of the hidden layers.
        """
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(n_features, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
