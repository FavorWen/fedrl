import numpy as np
import pandas as pd

from PIL import Image
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

class MlpNet(nn.Module):
    def __init__(self,dstName):
        super().__init__()
        if dstName == "MINIST" or dstName == "MNIST" or dstName == 'FMNIST':
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
                # nn.Linear(20*6*6, 84),
                # nn.Linear(84, 10)
                nn.Linear(20*6*6, 10)
            )
        if dstName == 'MINIST'  or dstName == "MNIST" or dstName == 'FMNIST':
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
                # nn.Linear(250, 84),
                # nn.Linear(84, 10)
                nn.Linear(250, 10)
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
        if dstName == 'MINIST'  or dstName == "MNIST" or dstName == 'FMNIST':
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
        if dstName == 'MINIST'  or dstName == "MNIST" or dstName == 'FMNIST':
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

class CustomSubset(torch.utils.data.dataset.Subset):
    def __init__(self,  dataset, indices, id, threshold):
        self.id = id
        # if threshold < len(indices)/2:
        #     threshold *= 0.5
        # else:
        #     threshold *= 1.5
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