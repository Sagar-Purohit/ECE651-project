# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from base_layer import Network,ReplayMemory

# Implementation of Experience Replay
class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # Pushes a new event to the memory and nd checking whether it is full or not.
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Sampling of moves from the memory
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementation of Deep Q Learning
class DeepQNetwork():

    def __init__(self, input_size, number_of_actions, gamma):
        self.gamma = gamma
        self.reward_average = []

        # Creates the Neural Network
        self.model = Network(input_size, number_of_actions)

        # Creates Experience Replay with capacity of 100,000
        self.memory = ExperienceReplay(100000)

        # Chooses best optimization algorithm to reduce the Loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.last_state = torch.Tensor(input_size).unsqueeze(0)

        self.last_action = 0
        self.last_reward = 0

    # Decides what the next action should be
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_average.append(reward)

        if len(self.reward_average) > 1000:
            del self.reward_average[0]

        return action

    def score(self):
        return sum(self.reward_average) / (len(self.reward_average) + 1.)


