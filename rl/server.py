import numpy as np
import copy
from helper import *

class RecieveSendService:
    def __init__(self,global_net):
        self.recieved = []
        self.global_net = global_net

    def empty(self):
        return len(self.recieved) == 0
    
    def get_grad(self):
        msg = self.recieved[0]
        self.recieved = self.recieved[1:]
        return msg
    
    def put_grad(self, msg):
        self.recieved.append(msg)

    def get_net(self):
        return copy.deepcopy(self.global_net)
    
    def set_net(self, global_net):
        self.global_net = global_net

class Server:
    def __init__(self, client_nums, participant_nums, clients, reciever:RecieveSendService, device, criterion, optimizer="SGD", lr=0.01):
        self.clients = clients
        self.client_nums = client_nums
        self.participant_nums = participant_nums
        self.reciever = reciever
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        return
    
    def select_clients(self):
        return np.random.choice(self.client_nums, self.participant_nums, False).tolist()
    
    def run(self, participants=None):
        if type(participants) == type(None):
            participants = self.select_clients()
        grads = []
        samples = []
        state = []
        for i in participants:
            self.clients[i].recv_task()
        while not self.reciever.empty():
            id, grad, sample_nums = self.reciever.get_grad()
            if grad != None:
                state.append((id, True))
                grads.append(grad)
                samples.append(sample_nums)
            else:
                state.append((id, False))
                samples.append(sample_nums)
        # new_global_net = aggregate(self.global_net, grads, self.participant_nums, samples)
        # self.reciever.set_net(new_global_net)
        aggregate(self.reciever.global_net, grads, self.participant_nums, samples)
        return participants