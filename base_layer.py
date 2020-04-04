# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:36:26 2020

@author: Dell
"""

# ***** Creating the architecture of the Neural Network****

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Implementation of Neural Network specifications

class Network(nn.Module):
    
    def __init__(self, input_size, no_of_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.no_of_actions = no_of_actions

        # Creation of 1 hidden layer with 30 neurons (input to hidden and hidden to output)
        # Each layer is fully connected to other layer due to nn.Line

        self.full_connection1 = nn.Linear(input_size, 30)
        self.full_connection2 = nn.Linear(30, no_of_actions)
        
    # forward propagation
    def forward(self, state):
        #Applying activation function on the hidden layer neurons and returning the Q-values. 
        temp = F.relu(self.full_connection1(state))
        q_val = self.full_connection2(temp)
        return q_val
    

# Implementation Replay Memory

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    # Adding a new move to the memory and checking whether it is full or not.
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    # Sampling of moves from the memory
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
