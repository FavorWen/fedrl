import part
import torchvision
from model import CustomSubset, Classifier
from server import Server, RecieveSendService
from client import Client
import random
import torch
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import FMNISTPartitioner
from fedlab.utils.dataset import MNISTPartitioner

class Env:
    def __init__(self, obs_dim, arch_name, client_nums, participant_nums, dataset_name, partition, seed, device, criterion, optimizer="SGD", lr=0.01, epoch=1, rl_ddl = 100, batch_size=32, dst_path='/HARD-DRIVE/QI/data'):
        self.obs_dim = obs_dim # x1的维度
        self.arch_name = arch_name # 联邦学习目标模型的名字
        self.client_nums = client_nums # 客户端的总数
        self.participant_nums = participant_nums # 一轮训练选多少客户端参与
        self.dataset_name = dataset_name # 联邦学习数据集的名字
        self.partition = partition
        self.seed = seed
        self.device = device
        self.criterion = criterion
        self.optimizer="SGD"
        self.lr=lr
        self.epoch=epoch
        self.batch_size=batch_size
        self.state = torch.zeros(1, obs_dim)
        self.tick = 0
        self.rl_ddl = rl_ddl
        self.dst_path = dst_path
        assert obs_dim % (client_nums+1) == 0
    def init_state(self):
        _state = self.state[0]
        r = int(self.obs_dim / (self.client_nums+1))
        base = self.client_nums+1
        for i in range(0, r):
            participants = self.server.run()
            acc, loss = self.validate(self.valset)
            _state[i*base + self.client_nums] = loss
            for j in participants:
                _state[i*base + j] = 1
        self.state[0] = _state
        return self.state
    def update_state(self, action, loss):
        r = int(self.obs_dim / (self.client_nums+1))
        base = self.client_nums+1
        _state = self.state[0]
        for i in range(0, (r-1)*base):
            _state[i] = _state[i+base]
        for j in range(0, base):
            if j in action:
                _state[(r-1)*base+j] = 1
            else:
                _state[(r-1)*base+j] = 0
        self.state[0] = _state
        return self.state

    def init_dataset(self):
        #划分数据，训练集，（测试集, 验证集）
        #训练集划分为client_nums份，形成data_loader
        partitioner = part.Partitioner(self.dst_path, self.dataset_name, self.client_nums, self.partition, self.seed)
        self.trainset, self.testset, self.valset, self.id2num = partitioner.make_part()
        self.clients = []
        self.global_net = Classifier(self.arch_name, self.dataset_name).to(self.device)
        self.rsService = RecieveSendService(self.global_net)
        self.rank = [i for i in range(self.client_nums)]
        self.scores = [0 for i in range(self.client_nums)]
        #数量偏斜，则按照数量来进行真实贡献度排序，升序
        if self.partition == "quantity-skew":
            x = [(i, len(self.id2num[i])) for i in self.id2num]
            sorted(x, key = lambda t: t[1])
            self.rank = [i[0] for i in x]
        #否则，随机进行真实贡献度排序
        else:
            random.shuffle(self.rank)
        for i in range(self.client_nums):
            if self.partition == "iid" or self.partition == "label-skew":
                #按照rank，位次越低，注入噪声越多，贡献度越低
                idx = self.rank.index(i)
                threshold = len(self.id2num[i]) * (self.client_nums - idx)/(self.client_nums)
            else:
                threshold = 0
            # threshold /= 1 # reduce noise
            css = CustomSubset(self.trainset, self.id2num[i], i, threshold=threshold)
            dloader = DataLoader(css, batch_size=self.batch_size)
            c = Client(id=i, dst_loader=dloader, server_addr=self.rsService, device=self.device, criterion=self.criterion, optimizer=self.optimizer, lr=self.lr, epoch=self.epoch)
            self.clients.append(c)
        self.server = Server(client_nums=self.client_nums, participant_nums=self.participant_nums, clients=self.clients, reciever=self.rsService, device=self.device, criterion=self.criterion, optimizer=self.optimizer, lr=self.lr)

    def reset(self):
        self.tick = 0
        self.init_dataset()
        return self.init_state()
    
    def reset_light(self):
        self.global_net = Classifier(self.arch_name, self.dataset_name).to(self.device)
        self.rsService.set_net(self.global_net)
        self.tick = 0
        return self.init_state()

    def validate(self, set):
        val_acc = 0.0
        val_loss = 0.0
        dloader = DataLoader(set, batch_size=self.batch_size)
        with torch.no_grad():
            for i, data in enumerate(dloader):
                if data[0].size()[0] == 1:
                    continue
                val_pred = self.global_net(data[0].to(self.device))
                batch_loss = self.criterion(val_pred, data[1].to(self.device))

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()
        return val_acc/set.__len__(), val_loss/set.__len__()
    def step(self, action=None):
        # action是对每一个client的最新评分
        # 选出前participant_num个与后participant_num个，分别进行一次学习，然后计算在valset上的loss之差
        # 优化目标是使得-loss最小
        participants = self.server.run(action)
        acc, loss = self.validate(self.valset)
        if type(action) != type(None):
            self.update_state(action, loss)
        self.tick += 1
        return self.state, loss, self.tick >= self.rl_ddl, participants, acc