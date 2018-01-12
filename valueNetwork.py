#!/usr/bin/env python
"""
Value Network based on Giraffe
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class EvalNet(nn.Module):
    """
    Value Network Layers, Architecture and forward pass
    """
    def __init__(self):
        'initialise all the layers and activation functions needed'
        super(EvalNet, self).__init__()

        # three layers
        self.fc1 = nn.Linear(144, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)

        # if cuda, use GPU
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.cuda()
        
    def forward(self, inputLayer):
        'forward pass'
        out = F.relu(self.fc1(inputLayer))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))
        return out


def forward_pass(network, board):
    'do a forward pass of the network'
    x = torch.FloatTensor(board)
    if network.use_gpu:
        x = x.cuda()
    return network(Variable(x)).data[0]


loss_fn = torch.nn.MSELoss(size_average=False)


def train_step(network, boards, values, LEARNING_RATE):
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    for b, v in zip(boards, values):
        x = Variable(torch.FloatTensor(b))
        y = Variable(torch.FloatTensor([v]), requires_grad=False)
        if network.use_gpu:
            x = x.cuda()
            y = y.cuda()
        y_pred = network(x)

        loss = loss_fn(y_pred, y)
        #print(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    'create network and verify it works with previous giraffe weights'
    evalNet = EvalNet()

    # save weights
    #torch.save(evalNet.state_dict(), "weights.t7")
