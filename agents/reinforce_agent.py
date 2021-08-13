#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import count
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ReinforceAgent(object):
    """Implementation of a REINFORCE agent."""
    
    def __init__(self,
                 device,
                 n_actions,
                 actor,
                 lr=0.001,
                 gamma=0.99):
        """Initializes the agent.
        
        Args:
            device: `torch.device`, where tensors will be allocated.
            n_actions: int, number of actions the agent can take at any state.
            actor: `torch.nn`, neural network used to approximate the policy.
            lr: float, learning rate.
            gamma: float, discount rate.
        """
        self._device = device
        self.n_actions = n_actions
        self.actor = actor
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []
        self.baselines = []
        self.step = 0
    
    def act(self, state):
        """Sample an action according to the policy.
        
        Args:
            state: `torch.Tensor`, state of the agent.
        
        Returns:
            int, action sampled from the policy.
        """
        prob = self.actor(state)
        m = Categorical(prob)
        action = m.sample()
        self.log_probs.append(m.log_prob(action).squeeze(0))
        return action.item()
    
    def learn(self):
        """Learns the policy from an episode."""
        G = 0
        baseline_loss = []
        policy_loss = []
        returns = []
        
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        
        baselines = torch.cat(self.baselines)
        deltas = returns - baselines
        deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-9)
        
        for G, baseline, log_prob, delta in zip(returns, baselines,
                                                self.log_probs, deltas):
            baseline_loss.append(F.mse_loss(baseline, G.detach(),
                                            reduction='none'))
            policy_loss.append(-log_prob * delta.detach())
        
        baseline_loss = torch.stack(baseline_loss).mean()
        policy_loss = torch.stack(policy_loss).sum()
        loss = baseline_loss + policy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        del self.rewards[:]
        del self.log_probs[:]
        del self.baselines[:]
    
    def train(self, env, n_episodes):
        """Trains the agent in the environment for n_episodes episodes.
        
        Args:
            env: Gym environment.
            n_episodes: int, number of episodes to train for.
        
        Returns:
            list of ints, list of steps.
            list of floats, list of returns.
        """
        step_list = []
        return_list = []
        for i_episode in range(1, n_episodes+1):
            episode_return = 0
            state = env.reset()
            for t in count():
                state = torch.tensor(state, dtype=torch.float32).to(
                    self._device).unsqueeze(0)
                self.baselines.append(self.actor.baseline(state).squeeze(0))
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.rewards.append(reward)
                
                state = next_state
                episode_return += reward
                self.step += 1
                
                if done:
                    self.learn()
                    
                    step_list.append(self.step)
                    return_list.append(episode_return)
                    print("Episode {:4d} : {:4d} steps "
                          "| return = {:.1f}".format(i_episode, t+1,
                                                     episode_return))
                    
                    if return_list and episode_return >= max(return_list):
                        self.save("model.pt")
                    
                    break
        
        return step_list, return_list
    
    def test(self, env, n_steps=np.inf, agent_name=None):
        """Tests the agent in the environment for one episode and at most
        n_steps time steps.
        
        If agent_name is not None, record a video of the episode and write it
        to a disk file `videos/{env.spec.id}-{agent_name}.mp4`.
        
        Args:
            env: Gym environment.
            n_steps: int, maximum number of steps.
            agent_name: str, filename of the recorded video. If None, the
                episode is not recorded.
        """
        self.actor.eval()
        if agent_name:
            if not os.path.exists("videos"):
                os.mkdir("videos")
            recorder = VideoRecorder(env,
                                     base_path="videos/{}-{}"
                                     .format(env.spec.id, agent_name))
            
        episode_return = 0
        state = env.reset()
        for t in count():
            if agent_name:
                recorder.capture_frame()
            else:
                env.render()
            
            state = torch.tensor(state, dtype=torch.float32).to(
                self._device).unsqueeze(0)
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            state = next_state

            if done or t+1 >= n_steps:
                if agent_name:
                    recorder.close()
                env.close()
                self.actor.train()
                return episode_return
    
    def save(self, path):
        """Saves the actor to a disk file.
        
        Args:
            path: str, path of the disk file.
        """
        torch.save(self.actor.state_dict(), path)
    
    def load(self, path, map_location='cpu'):
        """Loads a saved actor from a disk file.
        
        Args:
            path: str, path of the disk file.
            map_location: str, string specifying how to remap storage
                locations.
        """
        self.actor.load_state_dict(torch.load(path,
                                              map_location=map_location))
