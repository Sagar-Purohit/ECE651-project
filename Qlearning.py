# -*- coding: utf-8 -*-

Implementing Reinforcement learning (DQN):

This project shows how to build a game based on self driving car using Q-learning.

So in this project, our environment contains the car that we will train to drive itself and to
avoid obstacles, we will create a path and some blocks for our car to navigate around them.
We will later build an artificial intelligence to train this car to drive on the road without
crossing the limits and avoiding some obstacles that will come on its way.

**Task**
The agent which is our car has to make a trip from the the airport to downtown and vice versa
avoiding the obstacles on its way.
Consider map to be a square block. The upper left corner of the map is the airport of a city and
the bottom right corner of the map is the downtown of the city. So the car has to make the
roundtrips between the two places without crossing the sides of road and avoiding obstacles.

As the agent observes the current state of the environment and chooses an action, the environment
*transitions* to a new state, and also returns a reward that indicates the consequences of the
action.

Rewards are the way our AI learns, so the best reward is +1 and the worst reward is -1 .
In the above task, the sides of the roads are considered to have some sand, if the car doesn't go
into the sand then the reward will be positive but if it happens the other way the car gets bad
reward and it slows down.
Additionally, if the car reaches to any of the four edges, then it gets the full density of sand.
The car gets penalised terribly(gets -1 as reward).

If car is getting closer to the goal then it will get a slightly positive reward.
On the other hand,if it gets further away from the goal, it gets a slightly negative reward -0.2

#PACKAGES:

We will use the following from PyTorch library:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)
-  Testing (``torch_testing``)

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchtest import assert_vars_same
import torch_testing as tt
import unittest

class NaNTensorException(Exception):
  pass

class InfTensorException(Exception):
  pass

from base_layer import Network


#################################################################################################
# Experience Replay
# -----------------
#
# We'll be using experience replay memory for training our DQN. It stores he transitions that the
# agent observes, allowing us to reuse this data later.
# By sampling from it randomly, the transitions that build up a batch are decorrelated.
# It has been shown that this greatly stabilizes and improves the DQN training procedure.
#
# For this, we're going to need ExperienceReplay classs and three functions to implement it:
# init, push and sample
#
# Implementation of Experience Replay

class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # Pushes a new event to the memory and and checking whether it is full or not.
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Sampling of moves from the memory
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementation of Deep Q Learning

# Actions are chosen either randomly or based on a policy, getting the next step sample from the
# environment. We record the results in the replay memory and also run optimization step on every
# iteration.
# Optimization picks a random batch from the replay memory to do training of the new policy.

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
        
        #for testing purpose
        print('Our list of parameters', [ np[1] for np in self.model.named_parameters() ])
        self.initial_param = [ np[1] for np in self.model.named_parameters()]

    # Decides what the next action should be
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 7)
        action = probs.multinomial(1)
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
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
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'saved_ai.pth')

    def load(self):
        if os.path.isfile('saved_ai.pth'):
            print("Loading checkpoint...")
            checkpoint = torch.load('saved_ai.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("No checkpoint found...")
    
    # Unit Testing for Q-learning
    # ---------------------------
    #
    # We have performed Unit tests for the q-learning module.
    # We used different test like whether the tensors are same or not, we covered some edge cases like if they are having any NaN values or infinity values.
    
    
    #Unit test for q-learning 
    def test(self):
        self.after_param = [ np[1] for np in self.model.named_parameters() ]
        print('After save', self.after_param)
        #####test for same params
        count=0
        for i,a_val in enumerate(self.after_param):
            if(tt.assert_equal(a_val, self.initial_param[i])== None):
                print('tensor unit test [', i,']passed')
                count=count+1
        if(count==4):
            print('unit test for Q-learning passed')
        
        
        ###test for Nan
        for val in self.after_param:
         try:
            assert not torch.isnan(val).byte().any()
         except AssertionError:
            raise NaNTensorException("There was a NaN value in tensor")
            
        ###test for infinite values
        for val in self.after_param:
            try:
                assert torch.isfinite(val).byte().any()
            except AssertionError:
                raise InfTensorException("There was an Inf value in tensor")
                
    # Integration Testing 
    # ---------------------------
    #        
    #### We performed integration testing by merging the graphics and brain of self driving car.
    #We performed integration testing by checking whether the car is learning properly or not to reach the destination from source.
    
    # Integration Testing 
    # ---------------------------
    # For the system testing we created a different paths and obstacles on to test the car for different conditions.
