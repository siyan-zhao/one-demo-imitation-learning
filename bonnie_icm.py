"""
Usage
======
self.icm = ICM(state_size = 12, action_size = 2)
def update_rewards_v5(self):
    # ICM
    state_size = 12
    action_size = 2
    obs = self.all_obs
    a = self.all_actions

    all_rews = np.zeros((self.buffer_size, self.time_dim, 1))
    states, actions, next_states = ([None] * (self.time_dim - 1) for _ in range(3))
    for t in range(self.time_dim - 1):
        rew = self.icm.compute_intrinsic_reward(obs[:, t, :], a[:, t, :], obs[:, t+1, :])
        rew = np.expand_dims(rew, -1)
        # print ("Intrinsic Reward: ", rew)
        all_rews[:, t, :] = rew
        states[t] = obs[:, t, :]
        actions[t] = a[:, t, :]
        next_states[t] = obs[:, t+1, :]
    self.all_rewards = all_rews
    # train ICM
    states = np.stack(states).transpose([1, 0, 2]).reshape([-1, state_size])
    next_states = np.stack(next_states).transpose([1, 0, 2]).reshape([-1, state_size])
    actions = np.stack(actions).transpose([1, 0, 2]).reshape([-1, action_size])
    self.icm.batch_train(states, actions, next_states)

buffer = BonnieUnsupervisedBuffer(buffer_size=20, obs_dim=env.obs_dim,
                                  action_dim=env.action_dim, time_dim=traj_horizon)
buffer.update_rewards_v5()
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

def swish(x):
    return x * F.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ICMModel(nn.Module):
    """ICM model for non-vision based tasks"""
    def __init__(self, input_size, output_size):
        super(ICMModel, self).__init__()

        self.net_size = 32

        self.input_size = input_size
        self.output_size = output_size
        self.resnet_time = 4
        self.device = device

        self.feature = nn.Sequential(
            nn.Linear(self.input_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(self.net_size * 2, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(self.output_size + self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
        ).to(self.device)] * 2 * self.resnet_time

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.output_size + self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size)
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.output_size + self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size),
            Swish(),
            nn.Linear(self.net_size, self.net_size)
        )

    def forward(self, state, next_state, action):
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)
        # residual
        for i in range(self.resnet_time):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

class ICM():
    """Intrinsic Curisity Module"""
    def __init__(
        self,
        state_size,
        action_size,
        num_epoch = 3, # training iters for icm
        batch_size = 64, # training batches for icm
        eta = 1, # intrinsic rewardd scale (1, 0.1, 0.01)
        learning_rate = 1e-4):
        self.model = ICMModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

        self.input_size = state_size
        self.output_size = action_size
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.device = device

        self.eta = eta
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic rewards for parallel transitions
        :param: (ndarray) state, action, next_state
        :return: (list) intrinsic_reward eg. [rew, rew, ... , rew]
        """
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)

        action_onehot = action

        real_next_state_feature, pred_next_state_feature, pred_action = self.model(
            state, next_state, action_onehot)
        intrinsic_reward = self.eta * (real_next_state_feature - pred_next_state_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def sample(self, indices, batch_size):
        """Shuffle, chop + yield"""
        indices = np.asarray(np.random.permutation(indices))
        batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
        for batch in batches:
            yield batch
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]

    def batch_train(self, states, actions, next_states):
        """
        Convert transitions into batches + call train()
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        for _ in range(self.num_epoch):
            sampler = self.sample(np.arange(states.shape[0]), self.batch_size)
            for batch_indices in sampler:
                obs = states[batch_indices]
                act = actions[batch_indices]
                next_obs = next_states[batch_indices]

                self.train(obs, act, next_obs)
        print ("icm: batch train complete")

    def train(self, states, actions, next_states):
        """
        Train the ICM model with given minibatches
        :param: (float tensors) states, actions, next_states
        """
        action_onehot = actions
        real_next_state_feature, pred_next_state_feature, pred_action = self.model(
            states, next_states, action_onehot)

        inverse_loss = self.mse(
            pred_action, actions)

        forward_loss = self.mse(
            pred_next_state_feature, real_next_state_feature.detach())

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        loss = inverse_loss + forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
