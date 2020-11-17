import os
import random
import sys
import time
import math
import queue
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import matplotlib.pyplot as plt

import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

from gym import logger as gymlogger
from gym.wrappers import Monitor
# from pyvirtualdisplay import Display

display = Display(visible=0, size=(400, 300))
display.start()


GAMMA = 0.99
NUM_EPISODES = 100
NUM_STEPS = 10 ** 4
BATCH_SIZE = 64

class PolicyNet():
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p = 0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.dropout(self.affine1(x)))
        probs = F.softmax(self.affine2(x), dim = 1)
        return probs



def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r — r.mean()


def train():
    env = gym.make('CartPole-v1')
    policy_net = PolicyNet()
    optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)

    reward_list, batch_rewards, batch_actions, batch_states = [], [], [], []
    batch_counter = 1

    for episode in range(NUM_EPISODES):
        prev_state = env.reset()
        episode_reward = 0

        states, rewards, actions = [], [], []
        terminal = False

        while terminal:
            action_probs = policy_net(prev_state).detach()
            action = np.random.choice(
                np.arange(env.action_space.n),
                p = action_probs
            )

            state, reward, terminal, info = env.step(action)
            episode_reward += reward
            states.append(prev_state)
            rewards.append(reward)
            actions.append(action)
            prev_state = state

            if terminal:
                break
        
        batch_rewards.extend(discount_rewards(rewards, GAMMA))
        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_counter += 1
        reward_list.append(sum(rewards))

        if batch_counter == BATCH_SIZE:
            optimizer.zero_grad()
            state_tensor = torch.FloatTensor(batch_states)
            reward_tensor = torch.FloatTensor(batch_rewards)
            action_tensor = torch.LongTensor(batch_actions)

            logprob = torch.log(policy_net(state_tensor))
            loss = (-(reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze())).mean()

            loss.backward()
            optimizer.step()

            batch_rewards, batch_actions, batch_states = [], [], []
            batch_counter = 1
        
        avg_rewards = np.mean(reward_list[-100:])
        print("\rEp: {} Average of last 100:" +   
                    "{:.2f}".format(
                    episode + 1, avg_rewards), end="")

        


