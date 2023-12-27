from distutils.command.config import config
import numpy as np
import pandas as pd
from pprint import pprint

from PIL import Image

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
import scipy.stats

import random
import math
from torch.utils.tensorboard import SummaryWriter

import copy
from typing import Any, Callable, Optional, Tuple

import os

from resnet import resnet18
from mobilenet import mobilenet_v2

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MlpNet(nn.Module):
    def __init__(self,dstName):
        super().__init__()
        if dstName == "MINIST":
            num_in = 28 * 28
            num_hid = 64
            num_out = 10
        else:
            num_in = 32 * 32 * 3
            num_hid = 64
            num_out = 10
        self.body = nn.Sequential(
            nn.Linear(num_in,num_hid),
            nn.ReLU(),
            nn.Linear(num_hid,num_hid),
            nn.ReLU(),
            nn.Linear(num_hid,num_out)
        )
    def forward(self,x):
        x = x.view(x.size(0), -1)
        return self.body(x)

class CnnNet(nn.Module):
    def __init__(self, dstName):
        super().__init__()
        if dstName == 'CIFAR10':
            # input [3, 32, 32]
            self.body = nn.Sequential(
                nn.Conv2d(3, 10, kernel_size=5, padding=1, stride=1), # [10, 32, 32]
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.MaxPool2d(2,2,0),# [10, 16, 16]

                nn.Conv2d(10, 20, kernel_size=5, padding=1, stride=1), # [20, 16, 16]
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.MaxPool2d(2,2,0),#[20, 6, 6]
            )
            self.fc = nn.Sequential(
                nn.Linear(20*6*6, 84),
                nn.Linear(84, 10)
            )
        if dstName == 'MINIST':
            # input [1, 28, 28]
            self.body = nn.Sequential(
                nn.Conv2d(1, 5, kernel_size=5, padding=1, stride=1), # [5, 28, 28]
                nn.BatchNorm2d(5),
                nn.ReLU(),
                nn.MaxPool2d(2,2,0),# [5, 14, 14]

                nn.Conv2d(5, 10, kernel_size=5, padding=1, stride=1), # [10, 14, 14]
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.MaxPool2d(2,2,0),#[10, 7, 7]
            )
            self.fc = nn.Sequential(
                nn.Linear(250, 84),
                nn.Linear(84, 10)
            )
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)

class ResNet18(nn.Module):
    def __init__(self, dstName):
        super().__init__()
        if dstName == 'CIFAR10':
            self.body = resnet18(pretrained=False,n_classes=10,input_channels=3)
        if dstName == 'MINIST':
            self.body = resnet18(pretrained=False,n_classes=10,input_channels=1)
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return out

class MobileNetV2(nn.Module):
    def __init__(self, dstName):
        super().__init__()
        if dstName == 'CIFAR10':
            self.body = mobilenet_v2(pretrained=False,n_class=10,i_channel=3,input_size=32)
        if dstName == 'MINIST':
            self.body = mobilenet_v2(pretrained=False,n_class=10,i_channel=1,input_size=28)
    def forward(self,x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return out

class Classifier(nn.Module):
    def __init__(self, netName, dstName):
        super().__init__()
        if netName == 'CNN':
            self.body = CnnNet(dstName)
        if netName == 'MLP':
            self.body = MlpNet(dstName)
        if netName == 'RESNET18':
            self.body = ResNet18(dstName)
        if netName == 'MOBILENETV2':
            self.body = MobileNetV2(dstName)
    def forward(self,x):
        return self.body(x)

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

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def aggregate(net, clientGrads, clientN):
    sum_grad = {}
    for k in clientGrads[0]:
        sum = 0
        for cId in range(clientN):
            sum = sum + clientGrads[cId][k]
        sum_mean = sum/clientN
        sum_grad[k] = sum_mean
    return update(net, sum_grad)

def selectParticipants(participantsNumber, clientsNumber, scores, goodRate):
    if len(goodRate) != 3:
        return np.random.choice(clientsNumber, participantsNumber, False).tolist()
    bad = int(participantsNumber * goodRate[0])
    mid = int(participantsNumber * goodRate[1])
    good = int(participantsNumber * goodRate[2])
    badIndexBegin = 0
    midIndexBegin = int(clientsNumber * 0.5)
    goodIndexBegin = int(clientsNumber * 0.7)
    result = np.random.choice(range(goodIndexBegin, clientsNumber), good, False).tolist()
    result += np.random.choice(range(badIndexBegin, midIndexBegin), bad, False).tolist()
    result += np.random.choice(range(midIndexBegin, goodIndexBegin), mid, False).tolist()

    return result

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

class CustomSubset(torch.utils.data.dataset.Subset):
    def __init__(self,  dataset, indices, id, threshold):
        self.id = id
        self.threshold = threshold
        super().__init__(dataset, indices)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.dataset[self.indices[index]]
        # Add noise to client dataset
        # see Eq. 4 in Reference paper
        dice = np.random.uniform(0, len(self.indices),1)[0]
        uid = int(np.random.uniform(0,10,1)[0])
        if dice < self.threshold:
            target = uid

        return img, target

def initDst(DST_NAME,CLIENT_N, ISIID=True, NOISE=True):
    dstPath = "./data"
    if DST_NAME == 'MINIST':
        trainDst = datasets.MNIST(dstPath, train=True, download=True, transform=transform)
    elif DST_NAME == 'CIFAR10':
        trainDst = datasets.CIFAR10(dstPath, train=True, download=True, transform=transform)
    else:
        raise RuntimeError('MUST specific a dataset')

    indices = [i for i in range(trainDst.__len__())]
    random.shuffle(indices)
    if ISIID:
        numPerClient = trainDst.__len__() // CLIENT_N
        splitList = [numPerClient] * CLIENT_N
    else:
        total = (1+CLIENT_N)*CLIENT_N / 2
        splitList = [int((i+1)*1.0 / total * len(indices)) for i in range(CLIENT_N)]
    clientTrainDstSet = []
    start_idx = 0
    for c in range(CLIENT_N):
        if NOISE:
            threshold = splitList[c] * (CLIENT_N-1 - c)*1.0/(CLIENT_N-1)
        else:
            threshold = -1
        clientTrainDstSet.append(CustomSubset(trainDst, indices[start_idx:start_idx+splitList[c]], c, int(threshold)))
        start_idx += splitList[c]
    
    if DST_NAME == 'MINIST':
        testDst = datasets.MNIST(dstPath, train=False, download=False, transform=transform)
    elif DST_NAME == 'CIFAR10':
        testDst = datasets.CIFAR10(dstPath, train=False, download=False, transform=transform)
    else:
        raise RuntimeError('MUST specific a dataset')
    return clientTrainDstSet, testDst

def trainWithoutUpdate(args):
    globalNet = args['globalNet']
    Epoch = args['Epoch']
    BatchSize = args['BatchSize']
    ParticipantSet= args['ParticipantSet']
    Round= args['Round']
    testDst= args['testDst']
    clientTrainDstSet= args['clientTrainDstSet']
    device= args['device']
    writer = args['writer']
    goodRate = args['goodRate']
    client_n = args['Client_n']
    net_name = args['net_name']
    dst_name = args['dst_name']
    iid = args['iid']
    noise = args['noise']
    
    acc_logs = []
    participants = []
    scores = []
    for c in range(client_n):
        scores.append([0 for r in range(Round)])

    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(testDst, batch_size=BatchSize, shuffle=False)
    for round in range(Round):
        gradsFromClient = []
        participants = ParticipantSet[round]
        for cId in participants:
            cNet = copy.deepcopy(globalNet)
            optimizer = torch.optim.SGD(cNet.parameters(), lr=0.01)
            train_loader = DataLoader(clientTrainDstSet[cId], batch_size=BatchSize)
            cNet.train()
            for epoch in range(Epoch):
                train_acc = 0.0
                train_loss = 0.0

                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    train_pred = cNet(data[0].to(device))
                    batch_loss = criterion(train_pred, data[1].to(device))
                    batch_loss.backward()
                    optimizer.step()
            originNet = copy.deepcopy(globalNet)
            gradsFromClient.append(diff(originNet, cNet))
        globalNet = aggregate(globalNet, gradsFromClient, len(participants))
        test_acc = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                test_pred = globalNet(data[0].to(device))
                batch_loss = criterion(test_pred, data[1].to(device))

                test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                test_loss += batch_loss.item()
            print('[%03d/%03d round] Test Acc: %3.6f Loss: %3.6f' % \
                (round, Round, test_acc/testDst.__len__(), test_loss/testDst.__len__()))
            # writer.add_scalars(f'Learnning/Acc-{net_name}-{dst_name}-{client_n}-{noise}-{iid}',{f'{goodRate}-{noise}-{iid}':test_acc/testDst.__len__()},round)
        acc_logs.append(test_acc/testDst.__len__())
    result = {
        'goodRate':goodRate,
        'dst_name':dst_name,
        'net_name':net_name,
        'client_n':client_n,
        'net':globalNet,
        'acc_logs':acc_logs,
        'ParticipantSet':ParticipantSet,
        'iid':iid,
        'noise':noise,
        'globalNet':globalNet
    }
    return result

def produceData(config):
    seed = config['seed']
    id = config['id']
    arch = config['arch']
    datasets = config['datasets']
    clientSetting = config['clientSetting']
    Round = config['round']

    setup_seed(seed)
    log_dir = 'acc_data' + str(id)
    writer = SummaryWriter(log_dir)
    e_name = 'random'
    # clientSetting = [100, 25, 5]#[100, 25, 5]
    participantSetting = {5:2, 25:5, 100:10}
    # roundSetting = {5: 34, 25: 101, 100:201}
    roundSetting = {5: 101, 25: 101, 100:101}
    BatchSize = 32
    # Round = 101
    Epoch = 1
    goodRates = [[],[],[]]
    for net_name in arch:
        for dst_name in datasets:
            for client_n in clientSetting:
                for noise in [True]:
                    for iid in [True, False]:
                        noise_str = 'noise' if noise else 'nonise'
                        iid_str = 'iid' if iid else 'noniid'
                        if not iid:
                            continue
                        setup_seed(seed)
                        Round = roundSetting[client_n]
                        start = time.time()
                        ThisExpName = f'{net_name}-{dst_name}-{client_n}-{noise_str}-{iid_str}'
                        print(ThisExpName)

                        participant_n = participantSetting[client_n]
                        clientTrainDstSet, testDst = initDst(dst_name, client_n, iid, noise)
                        globalNet = Classifier(net_name, dst_name).to(device)
                        ParticipantSet = [selectParticipants(participant_n, client_n, 0, goodRates[r%3]) for r in range(Round)]

                        args = {
                        'net_name':net_name,
                        'dst_name':dst_name,
                        'Client_n':client_n,
                        'globalNet': copy.deepcopy(globalNet),
                        'Epoch':Epoch,
                        'BatchSize':BatchSize,
                        'ParticipantSet':ParticipantSet,
                        'Round':len(ParticipantSet),
                        'testDst':testDst,
                        'clientTrainDstSet':clientTrainDstSet,
                        'device':device,
                        'writer':writer,
                        'goodRate':e_name,
                        'goodRates':goodRates,
                        'noise':noise_str,
                        'iid':iid_str
                        }
                        res = trainWithoutUpdate(args)
                        torch.save(res, f'{log_dir}/{net_name}-{dst_name}-{client_n}-{noise_str}-{iid_str}.save')
                        stop = time.time()
                        print("Exp",ThisExpName,"using",stop-start,"seconds")
                        # done.append(f'{net_name}-{dst_name}-{client_n}-{noise_str}-{iid_str}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu",type=int,default='1')
    parser.add_argument("--start",type=int)
    parser.add_argument("--stop",type=int)
    parser.add_argument("--arch",nargs='*',type=str,default=["CNN","MOBILENETV2","RESNET18","MLP"])
    parser.add_argument("--datasets",nargs='*',type=str,default=["CIFAR10","MINIST"])
    parser.add_argument("--clientSetting",nargs='*',type=int,default=[100,25,5])
    parser.add_argument("--round",type=int,default=101)
    args = parser.parse_args()
    config = vars(args)
    for i in range(args.start,args.stop+1):
        config['seed'] = i
        config['id'] = i
        start = time.time()
        produceData(config)
        stop = time.time()
        print("Data",i,"using",stop-start,"seconds")