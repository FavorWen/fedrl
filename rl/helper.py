import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as  np
import collections
import pickle
import scipy

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

class ReplayMemory(object):
    def __init__(self, max_size=12):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size=6):
        batch_size = min(batch_size, self.__len__())

        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

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
