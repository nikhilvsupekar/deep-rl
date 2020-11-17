import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

samples = {
    "state": [],
    "next_state": [],
    "reward": [],
    "action": [],
    "terminal" : [],
}

env = gym.make('CarRacing-v0').unwrapped

env.reset()

action_map = {
    0 : [-1, 0, 0],
    1 : [1, 0, 0],
    2 : [0, 1, 0],
    3 : [0, 0, 1]
}



class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 4
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out





def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data



def train(q_net, target_net, start):
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=1e-6)
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    replay_memory = []

    steps_done = 0

    for episode in range(600):
        # state shape is 96 x 96 x 3
        state = env.reset()

        epsilon = q_net.initial_epsilon
        epsilon_decrements = np.linspace(
            q_net.initial_epsilon, 
            q_net.final_epsilon, 
            q_net.number_of_iterations
        )

        game_action = np.array(action_map[0]).astype('float32')

        next_state, reward, terminal, info = env.step(game_action)
        next_state = resize_and_bgr2gray(next_state)
        next_state = image_to_tensor(next_state)
        state = torch.cat((next_state, next_state, next_state, next_state)).unsqueeze(0)

        iteration = 0
        c_reward = 0

        while True:
            output = q_net(state)[0]

            action = torch.zeros([q_net.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():
                action = action.cuda()

            # epsilon greedy exploration
            random_action = random.random() <= epsilon
            action_index = [torch.randint(q_net.number_of_actions, torch.Size([]), dtype=torch.int)
                            if random_action
                            else torch.argmax(output)][0]

            if torch.cuda.is_available():
                action_index = action_index.cuda()

            action[action_index] = 1
            action = action.unsqueeze(0)
            game_action = np.array(action_map[action_index.item()]).astype('float32')

            for i in range(q_net.game_step):
              next_state_1, reward, terminal, info = env.step(game_action)
            
            next_state_1 = resize_and_bgr2gray(next_state_1)
            next_state_1 = image_to_tensor(next_state_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], next_state_1)).unsqueeze(0)

            # action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # save transition to replay memory
            replay_memory.append((state, action, reward, state_1, terminal))

            if len(replay_memory) > q_net.replay_memory_size:
                replay_memory.pop(0)

            # epsilon annealing
            # epsilon = epsilon_decrements[iteration]
            epsilon = q_net.final_epsilon + (q_net.initial_epsilon - q_net.final_epsilon) * \
                              math.exp(-1. * steps_done / 200)
            steps_done += 1

            minibatch = random.sample(replay_memory, min(len(replay_memory), q_net.minibatch_size))

            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            if torch.cuda.is_available():
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # output_1_batch = q_net(state_1_batch)
            output_1_batch = target_net(state_1_batch)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                    else reward_batch[i] + q_net.gamma * torch.max(output_1_batch[i])
                                    for i in range(len(minibatch))))

            q_value = torch.sum(q_net(state_batch) * action_batch, dim=1)


            optimizer.zero_grad()
            y_batch = y_batch.detach()
            loss = criterion(q_value, y_batch)
            loss.backward()
            optimizer.step()

            state = state_1
            c_reward += reward.numpy()[0][0]

            # print(
            #     "epoch: ", epoch, 
            #     "iteration: ", iteration, 
            #     "elapsed time: ", time.time() - start, 
            #     "epsilon: ", epsilon, 
            #     "action: ", action_index, 
            #     "reward:", c_reward, 
            #     "Q max:", np.max(output.cpu().detach().numpy())
            # )

            if iteration % 10 == 0:
              print(
                  "episode: ", episode, 
                  "iteration: ", iteration, 
                  "reward:", c_reward, 
                  "Q max:", np.max(output.cpu().detach().numpy())
              )
            
            if episode % q_net.target_update_freq == 0:
              target_net.load_state_dict(q_net.state_dict())

            if terminal or c_reward < -200:
              break
            
            iteration += 1
        
        if (episode + 1) % 20 == 0:
            torch.save(q_net, "models/current_model_" + str(episode) + ".pth")
        
        env.close()
        env.reset()


def test(model):
    state = env.reset()

    # initial action is do nothing
    # action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    # action[0] = 1

    game_action = np.array(action_map[0]).astype('float32')

    next_state, reward, terminal, info = env.step(game_action)
    next_state = resize_and_bgr2gray(next_state)
    next_state = image_to_tensor(next_state)
    state = torch.cat((next_state, next_state, next_state, next_state)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output).item()
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        game_action = action_map[action_index]

        next_state_1, reward, terminal, info = env.step(game_action)
        next_state_1 = resize_and_bgr2gray(next_state_1)
        next_state_1 = image_to_tensor(next_state_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], next_state_1)).unsqueeze(0)

        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])