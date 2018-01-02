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

from boardRepresentation import *


class EvalNet(nn.Module):
    """
    Value Network Layers, Architecture and forward pass
    """
    def __init__(self):
        'initialise all the layers and activation functions needed'
        super(EvalNet, self).__init__()

        # three layers
        self.fc1 = nn.Linear(216, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)

        # if cuda, use GPU
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.cuda()

        
    def forward(self, inputLayer):
        'forward pass'
        # slice input layer
        out = F.relu(self.fc1(inputLayer))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


def forward_pass(network, board):
    'do a forward pass of the network'
    x = torch.FloatTensor(board_to_feature_vector(board))
    if network.use_gpu:
        x = x.cuda()
    return network(Variable(x)).data[0]


if __name__ == "__main__":
    'create network and verify it works with previous giraffe weights'
    evalNet = EvalNet()

    # save weights
    #torch.save(evalNet.state_dict(), "weights.t7")
