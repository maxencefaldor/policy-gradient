#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.optim as optim


class A2CAgent(object):
    """Implementation of an A2C agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 network,
                 lr=0.001,
                 gamma=0.99):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take at any state.
            network: `torch.nn`, neural network used to approximate the policy.
            lr: float, learning rate.
            gamma: float, discount rate.
        """
        self._device = device
        self.n_actions = n_actions
        self.network = network
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
    
    def act(self, state):
        """Sample an action according to the actor.
        
        Args:
            state: `torch.Tensor`, state of the agent.
        
        Returns:
            int, action sampled from the policy.
        """
        prob = self.network.actor(state)
        m = Categorical(prob)
        action = m.sample()
        self.log_probs.append(m.log_prob(action).squeeze(0))
        return action.item()