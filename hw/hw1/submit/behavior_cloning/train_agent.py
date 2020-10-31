import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import pickle

from model import Model
from utils import *

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

def preprocess_X_for_conv(X, h):
    zeros = np.zeros((h-1, 96, 96))
    
    new_X_train = np.concatenate((zeros, X))
    new_X_list = []
    
    for i in range(X.shape[0]):
        new_X_list.append(new_X_train[i: i+h])
    
    new_X_train = np.array(new_X_list)
    
    return new_X_train

def preprocessing(X_train, y_train, X_valid, y_valid, h=1):
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
    
    X_train = preprocess_X_for_conv(X_train, h)
    X_valid = preprocess_X_for_conv(X_valid, h)
    
    return X_train, y_train, X_valid, y_valid




def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="/scratch/ns4486/deep_rl/models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
 
    print("... train model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.Tensor(X_train).float(), torch.Tensor(y_train).float())
    val_dataset = TensorDataset(torch.Tensor(X_valid).float(), torch.Tensor(y_valid).float())
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size
    )

    agent = Model()
    agent.model = agent.model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(agent.model.parameters(), lr = lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    train_losses = []
    val_losses = []
    for epoch in range(n_minibatches):
        # print(f'-- running epoch {epoch + 1} --')
        
        total_train_loss = 0
        count = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            out = agent.model(X)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             scheduler.step()

            total_train_loss += float(loss) * X.shape[0]
            count += X.shape[0]
        
        avg_train_loss = total_train_loss / count


        with torch.no_grad():
            total_val_loss = 0
            count = 0
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                out = agent.model(X)
                loss = criterion(out, y)

                total_val_loss += float(loss) * X.shape[0]
                count += X.shape[0]
            
            avg_val_loss = total_val_loss / count


        print(f'epoch = {epoch}, train_loss = {avg_train_loss}, val_loss = {avg_val_loss}')
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 50 == 0:
            agent.save(os.path.join(model_dir, f"agent_{epoch}.ckpt"))
        
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "final_agent.ckpt"))
    print("Model saved in file: %s" % model_dir)
    
    with open('train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    
    with open('val_losses.pkl', 'wb') as f:
        pickle.dump(val_losses, f)

if __name__ == "__main__":

    X_train, y_train, X_valid, y_valid = read_data("/scratch/prs392/data")
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, h=12)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=0.0001)


 
