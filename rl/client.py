import copy
import torch
from helper import *

class Client:
    def __init__(self, id, dst_loader, server_addr, device, criterion, optimizer="SGD", lr=0.01, epoch=1):
        self.id = id
        self.dst_loader = dst_loader
        self.server_addr = server_addr
        self.device = device
        self.criterion = criterion
        self.optimizer = "SGD"
        self.lr = lr
        self.epoch = epoch
    
    def recv_net(self):
        return self.server_addr.get_net().to(self.device)
    
    def train(self, net, dst_loader):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
        net.train()
        for epoch in range(self.epoch):
            for i, data in enumerate(dst_loader):
                if data[0].size()[0] == 1:
                    continue
                optimizer.zero_grad()
                train_pred = net(data[0].to(self.device))
                batch_loss = self.criterion(train_pred, data[1].to(self.device))
                batch_loss.backward()
                optimizer.step()
        origin_net = self.recv_net()
        with torch.no_grad():
            grad = diff(origin_net, net)
        return grad
    
    def send_grad(self, grad):
        self.server_addr.put_grad((self.id, grad, self.dst_loader.dataset.__len__()))
    
    def recv_task(self):
        net = self.recv_net()
        grad = self.train(net, self.dst_loader)
        self.send_grad(grad)
        net = ""
        