import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

logger = logging.getLogger('')
logger.setLevel(logging.INFO)


# 假设 client_nums = 5, participants_nums = 3
# [1, 1, 1, 0, 0, 0.66, 1, 0, 1, 0, 1, 0.70]
# action = 0, 2, 3
# [1, 0, 1, 1, 0, 0.77] -> [action + reward]
# [1, 0, 1, 0, 1, 0.70, 1, 0, 1, 1, 0, 0.77] - > update obs

class Model2D(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.body_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.body_2 = nn.Sequential(
            nn.Linear(32 * 101 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )
    def forward(self, obs):
        x = self.body_1(obs)
        x = x.view(x.size(0), -1)
        x = self.body_2(obs)
        return F.softmax(x, dim=1)

class Model(nn.Module):
    def __init__(self, obs_dim=20, act_dim=5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # hid1_size = act_dim * 10 * 2 # 52 * 10 * 2 = 1024
        # hid2_size = hid1_size * 10 * 2 # 1024 * 10 * 2 = 20480
        # hid1_size = 1024 * 3
        # hid2_size = 10240 * 3

        # hid1_size = 2048
        # hid2_size = 20480
        # hid3_size = hid2_size

        hid1_size = 20480 * 2
        hid2_size = 20480
        hid3_size = 2048
        self.body = nn.Sequential(
            nn.Linear(obs_dim,hid1_size),
            # nn.BatchNorm1d(hid1_size),
            nn.ReLU(),
            nn.Linear(hid1_size,hid2_size),
            # nn.BatchNorm1d(hid2_size),
            nn.ReLU(),
            nn.Linear(hid2_size, hid3_size),
            # nn.BatchNorm1d(hid3_size),
            nn.ReLU(),
            nn.Linear(hid3_size,act_dim),
            nn.Softmax(dim=1),
        )
        # for idx, layer in enumerate(self.body):
        #     if isinstance(layer, nn.Linear):
        #         if idx == len(self.body)-1:
        #             nn.init.xavier_uniform_(layer.weight)
        #         else:
        #             nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, obs):
        return self.body(obs)

class PolicyGradient:
    def __init__(self, model, lr=0.01, device='cuda'):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=10e-4)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        self.device=device

    def eval(self):
        self.model.eval()
    
    def predict(self, obs):
        # print(next(self.model.parameters()).device, obs.device, self.device)
        return self.model(obs.to(self.device))
    
    def learn_by_batch(self, obses, actions, rewards, one_hots):
        self.optimizer.zero_grad()
        self.model.train()
        # logger.info('batch: {}'.format(obses.shape))
        pred = self.model(obses.to(self.device)).cpu()
        # cost = torch.mean(torch.sum(torch.log(pred) * one_hots * rewards, dim=1))
        cost = torch.sum(torch.log(pred) * one_hots * rewards)
        cost /= pred.shape[0]
        cost.backward()
        clip_value = 1.0
        nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        self.optimizer.step()

    
    def batch_learn(self, obs_list, action_list, reward_list):
        self.optimizer.zero_grad()
        self.model.train()
        torch.cuda.empty_cache()
        cost = 0
        for i in range(len(obs_list)):
            obs = obs_list[i].detach()
            action = action_list[i]
            reward = reward_list[i]

            act_prob = self.predict(obs).squeeze().cpu()
            act_dim = self.model.act_dim
            one_hot = torch.zeros(act_dim).scatter(0, torch.LongTensor(action), torch.ones(act_dim))
            # cost += torch.sum(-1 * torch.log(act_prob) * one_hot * reward)
            cost += torch.sum(torch.log(act_prob) * one_hot * reward)
        cost = cost / len(obs_list)
        cost.backward()
        self.optimizer.step()
    
    def learn(self, obs, action, reward):
        self.optimizer.zero_grad()
        act_prob = self.predict(obs).squeeze() # 根据输入的obs做一次评估，得到当前obs对应的评分
        # action = [1, 3, 4] -> onehot -> [0, 1, 0, 1, 1]
        # act_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
        # loss = onehot * act_prob * reward
        # loss.backward()
        act_dim = self.model.act_dim
        one_hot = torch.zeros(act_dim).scatter(0, torch.LongTensor(action), torch.ones(act_dim))

        cost = -1 * torch.log(act_prob) * one_hot * reward
        cost.backward()
        self.optimizer.step()

class Agent:
    def __init__(self, algo, obs_dim, act_dim, participant_nums, client_nums, device='cuda',lr=0.1):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.model = Model(obs_dim, act_dim)
        self.algo = algo
        self.participant_nums = participant_nums
        self.client_nums = client_nums
        self.lr = lr
        self.device = device

    def eval(self):
        self.algo.eval()
    
    def sample(self, obs):
        episode = 0.05
        self.algo.eval()
        act_prob =  self.predict(obs).detach().cpu().squeeze().numpy()
        # logger.info('act_prob: {}'.format(act_prob))
        if np.random.uniform(0, 1, 1)[0] < episode: #以episode的概率随机选择
            return np.random.choice(range(self.act_dim), size=self.participant_nums, replace=False), act_prob
        if np.count_nonzero(act_prob) < self.act_dim:
            act = np.random.choice(range(self.act_dim), size=self.participant_nums, replace=False)
        else:
            act = np.random.choice(range(self.act_dim), size=self.participant_nums, replace=False, p=act_prob)
        return act, act_prob

    def predict(self, obs):
        return self.algo.predict(obs)
    
    def learn_by_batch(self, obs_list, action_list, reward_list):
        batch_size = len(obs_list)
        obs = torch.Tensor(batch_size,  self.obs_dim)
        actions = torch.LongTensor(batch_size, self.participant_nums)
        rewards = torch.Tensor(batch_size, 1)
        for i in range(batch_size):
            obs[i] = torch.Tensor(obs_list[i])
            actions[i] = torch.LongTensor(action_list[i])
            rewards[i] = reward_list[i]
        one_hots = torch.zeros(batch_size, self.act_dim).scatter(1, actions, torch.ones(batch_size, self.act_dim))
        self.algo.learn_by_batch(obs, actions, rewards, one_hots)
    
    def learn(self, obs_list, action_list, reward_list):
        self.algo.batch_learn(obs_list, action_list, reward_list)
        # for i in range(len(obs)):
        #     self.algo.learn(obs[i], action[i], reward[i])


# input = torch.zeros(1,10)
# index = [1, 3, 6]
# tensor_index = torch.Tensor(index).reshape(1, 3)
# value = torch.ones_like(tensor_index, dtype=input.dtype).reshape(3)
# out = input.scatter(dim=0, index=index, src=value)
