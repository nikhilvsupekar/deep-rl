from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
# from tensorboard_evaluation import Evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)
    
    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')
    
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.
    # 
    # # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
 
    print("... train model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size
    )

    agent = Model()
    agent.model = agent.model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(agent.model.parameters(), lr = lr)


    for epoch in range(5):
        print(f'-- running epoch {epoch + 1} --')
        
        total_loss = 0
        count = 0
        for X, y in train_loader:
            X = X.unsqueeze(1)
            X = X.to(device)
            y = y.to(device)
            out = agent_model(X)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            count += 1
        
        avg_loss = total_loss / count

        print(f'epoch = {epoch}, avg_loss = {avg_loss}')
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=1000, batch_size=64, lr=0.0001)
 
