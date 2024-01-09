import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as  np
import collections
import pickle
import scipy

import logging

logger = logging.getLogger('')
logger.setLevel(logging.INFO)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

criterion = nn.CrossEntropyLoss()

def diff(origin_net, update_net):
    grad = {}
    ori_params = origin_net.state_dict()
    new_params = update_net.state_dict()
    for k in new_params:
        grad[k] = new_params[k] - ori_params[k]
    return grad

def update(net, grad):
    params = net.state_dict()
    for k in params:
        params[k] = params[k] + grad[k]
    net.load_state_dict(params, strict=True)
    return net

def aggregate(net, clientGrads, clientN, samples):
    sum_grad = {}
    sample_nums = 0.0
    for i in samples:
        sample_nums += i
    weights = [i/sample_nums for i in samples]
    for k in clientGrads[0]:
        sum = 0
        # for cId in range(clientN):
        #     sum = sum + clientGrads[cId][k]
        # sum_mean = sum/clientN
        for cId in range(clientN):
            sum = sum + clientGrads[cId][k]*weights[cId]
        sum_mean = sum
        sum_grad[k] = sum_mean
    return update(net, sum_grad)

class LogSaver(object):
    def __init__(self):
        self.size = 0
        self.rank_pool = []
        self.history_pool = []
        self.rank = None
        self.history = []
        self.history_dim = 0
    
    def finishInit(self):
        self.history_dim = len(self.history)

    def flush(self):
        self.rank_pool.append(self.rank)
        self.history_pool.append(self.history)
        self.rank = None
        self.history = []
        self.size += 1

    def setRank(self, rank):
        self.rank = rank
    
    def updateLog(self, log):
        self.history.append(log)

class Log(object):
    def __init__(self, participants, loss, acc):
        self.participants = participants
        self.loss = loss
        self.acc = acc
    def encoding(self, dim):
        t = torch.zeros(dim)
        t[self.participants] = 1.0
        return torch.cat((t, torch.tensor([self.loss])))
    def action(self):
        return self.participants
    def reward(self):
        return self.loss


class ReplayMemory(object):
    
    def __init__(self, latest_size=12):
        self.latest_size = latest_size
        self.buffer = collections.deque()

    def append(self, participants, loss, acc):
        self.buffer.append(Log(participants, loss, acc))

    def isHistoryReady(self, hdim, rl_ddl, batch_size=2):
        return len(self.__usabel_indexes(hdim=hdim, rl_ddl=rl_ddl)) >= batch_size
        return len(self.buffer) > hdim and len(self.buffer)-hdim >= batch_size
    
    def latestObs(self, hdim, client_nums):
        first = len(self.buffer)-hdim
        log_list = []
        for idx in range(first, first+hdim):
            log_list.append(self.buffer[idx].encoding(client_nums))
        obs = torch.stack(log_list, dim=0)
        return obs.unsqueeze(0).unsqueeze(0)
    
    def __sample_single_history2D(self, hdim, client_nums, first=-1):
        if first == -1:
            first = random.choice(0, len(self.buffer)-1)
        log_list = []
        for idx in range(first, first+hdim):
            log_list.append(self.buffer[idx].encoding(client_nums))
        history = torch.stack(log_list, dim=0)
        action = self.buffer[first+hdim].action()
        # 改变reward, 让d=lossi - loss_i-1, d越小好
        reward = self.buffer[first+hdim].reward() - self.buffer[first+hdim-1].reward()
        return history.unsqueeze(0), action, reward
    
    def __usabel_indexes(self, hdim, rl_ddl):
        forbidded = []
        # 50是随便取的一个比较大的数
        for n in range(1, 50):
            forbidded += [i for i in range((rl_ddl+hdim-1)*n - hdim+1, (rl_ddl+hdim-1)*n + 1)]
        forbidded = set(forbidded)
        indexes = list(range(0, len(self.buffer)-hdim))
        indexes = [item for item in indexes if item not in forbidded][-self.latest_size:]
        return indexes

    def sample2D(self, hdim, client_nums, rl_ddl, batch_size=1):
        batch_size = min(batch_size, len(self.buffer)-hdim)
        assert batch_size > 0

        indexes = self.__usabel_indexes(hdim=hdim, rl_ddl=rl_ddl)
        indexes = random.sample(indexes, batch_size)
        obs_list = []
        action_list = []
        reward_list = []
        for i in indexes:
            obs, action, reward = self.__sample_single_history2D(hdim, client_nums, i)
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
        return obs_list, action_list, reward_list

    def __len__(self):
        return len(self.buffer)

def calc_contribution_with_QI(result):
    client_nums = result['client_nums']
    participant_nums = result['participant_nums']
    ddl = result['rl_ddl']
    logs = result['logs']
    contirbution = []
    for r in range(ddl):
        contirbution.append([0 for c in range(client_nums)])
    # contirbution[i][j] = client j's contribution in round i
    _, pp_acc = logs[0]
    p_participants, p_acc = logs[1]

    for r in range(2, ddl):
        for c in range(client_nums):
            contirbution[r][c] = contirbution[r-1][c]
        participants, acc = logs[r]
        if acc - p_acc > p_acc - pp_acc:
            for c in p_participants:
                contirbution[r][c] -= 1
            for c in participants:
                contirbution[r][c] += 1
        
        if acc < p_acc:
            for c in participants:
                contirbution[r][c] -= 1
    
    return contirbution

def calc_spearman_with_QI(result):
    client_nums = result['client_nums']
    participant_nums = result['participant_nums']
    ddl = result['rl_ddl']
    contribution = calc_contribution_with_QI(result)
    rank = result['rank']
    spearmans = []
    y = []
    for c in range(client_nums):
        y.append(c)

    for r in range(ddl):
        x = []
        for i in range(len(rank)):
            x.append(contribution[r][rank[i]])
        spearman=scipy.stats.spearmanr(x,y)[0]
        spearmans.append(spearman)

    return spearmans

def calc_spearman_with_QI_File(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    spearmans = calc_spearman_with_QI(result)
    for i in range(len(spearmans)):
        print(i, spearmans[i])

if __name__ == '__main__':
    calc_spearman_with_QI_File('train_data/qi_data_seed1_archMOBILENETV2_datasetCIFAR10_nums25_partiid')
