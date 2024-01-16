from fedlab.utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner
from fedlab.utils.dataset import FMNISTPartitioner
from fedlab.utils.dataset import MNISTPartitioner

import torchvision
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Dataset

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class Partitioner:
    def __init__(self, dst_path, dst_name, client_nums, partition, seed):
        self.dst_path = dst_path
        self.dst_name = dst_name
        self.client_nums = client_nums
        self.partition = partition
        self.seed = seed

    def make_part(self):
        rate = 0.7
        if self.dst_name == "MNIST":
            trainset = torchvision.datasets.MNIST(root=self.dst_path, train=True, download=True, transform=transform)
            alltestset =  torchvision.datasets.MNIST(root=self.dst_path, train=False, download=True, transform=transform)
            partitioner = MNISTPartitioner
            client_dict = self._part(trainset, partitioner)
        elif self.dst_name == "FMNIST":
            trainset = torchvision.datasets.FashionMNIST(root=self.dst_path, train=True, download=True, transform=transform)
            alltestset =  torchvision.datasets.FashionMNIST(root=self.dst_path, train=False, download=True, transform=transform)
            partitioner = FMNISTPartitioner
            client_dict = self._part(trainset, partitioner)
        elif self.dst_name == "CIFAR10":
            trainset = torchvision.datasets.CIFAR10(root=self.dst_path, train=True, download=True, transform=transform)
            alltestset =  torchvision.datasets.CIFAR10(root=self.dst_path, train=False, download=True, transform=transform)
            partitioner = CIFAR10Partitioner
            client_dict = self.cifar_part(trainset, partitioner)
        elif self.dst_name == "CIFAR100":
            trainset = torchvision.datasets.CIFAR100(root=self.dst_path, train=True, download=True, transform=transform)
            alltestset =  torchvision.datasets.CIFAR100(root=self.dst_path, train=False, download=True, transform=transform)
            partitioner = CIFAR100Partitioner
            client_dict = self.cifar_part(trainset, partitioner)
        else:
            exit("No such dataset" + self.dst)
        n1 = int(rate * alltestset.__len__())
        n2 = alltestset.__len__() - n1
        testset, valset = torch.utils.data.random_split(alltestset, [n1, n2])
        return trainset, valset, testset, client_dict
        

    def cifar_part(self, trainset, partitioner):
        if self.partition == "iid":
            part = partitioner(trainset.targets,
                                      self.client_nums,
                                      balance=True,
                                      partition="iid",
                                      seed=self.seed)
        elif self.partition == "label-skew":
            part = partitioner(trainset.targets,
                                      self.client_nums,
                                      balance=None,
                                      partition="dirichlet",
                                      dir_alpha=0.5,
                                      unbalance_sgm=0.3,
                                    #   balance=False,
                                    #   partition="iid",
                                    #   dir_alpha=0.5,
                                    #   unbalance_sgm=0.3,
                                      seed=self.seed)
        elif self.partition == "quantity-skew":
            part = partitioner(trainset.targets,
                                      self.client_nums,
                                      balance=False,
                                      partition="iid",
                                      unbalance_sgm=0.7,
                                      seed=self.seed)
        else:
            exit("No such partition")
        # x = [(i, len(part.client_dict[i])) for i in part.client_dict]
        # x = sorted(x, key = lambda t: t[1])
        return part.client_dict

    def _part(self, trainset, partitioner):
        if self.partition == "iid":
            part = partitioner(trainset.targets,
                                  self.client_nums,
                                  partition="iid",
                                  seed=self.seed)
        elif self.partition == "label-skew":
            part = partitioner(trainset.targets,
                                  self.client_nums,
                                #   partition="noniid-#label",
                                #   major_classes_num=5,
                                #   dir_alpha=0.7,
                                  partition="noniid-labeldir", 
                                  dir_alpha=0.5,
                                  seed=self.seed)
        elif self.partition == "quantity-skew":
            part = partitioner(trainset.targets,
                                  self.client_nums,
                                  partition="unbalance",
                                  dir_alpha=0.5,
                                  seed=self.seed)
        else:
            exit("No such partition")
        return part.client_dict