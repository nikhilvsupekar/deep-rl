from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
from collections import deque
from model import Model
from utils import *

import torch

def preprocess_X_for_conv(X, h):
    zeros = np.zeros((h-1, 96, 96))
    
    new_X_train = np.concatenate((zeros, X))
    new_X_list = []
    
    for i in range(X.shape[0]):
        new_X_list.append(new_X_train[i: i+h])
    
    new_X_train = np.array(new_X_list)
    
    return new_X_train

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    h = 12

    episode_reward = 0
    step = 0

    state = env.reset()
    q = deque(h * [np.zeros((96, 96))], h)
    
    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py

        if step < h:
            zeros = np.zeros((h - step - 1, 96, 96))

        state = rgb2gray(state)
        q.append(state)
        state_torch = torch.Tensor(np.array(list(q))).unsqueeze(0)

        # state_torch = torch.Tensor(state).view(1, 1, 96, 96)
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        agent.model.eval()

        with torch.no_grad():
            a = agent.model(torch.Tensor(state_torch))

        a = np.array(a)
        a = a[0, :]

        

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = Model()
    agent.load("models/agent_h12.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
