#!/usr/bin/env python
"""
train deepfish using reinforcement learning
"""
from __future__ import print_function

from deepfish import Searcher as DeepSearcher
from randomfish import Searcher as RandomSearcher
from slimfish import *
from valueNetwork import *

import torch, random


LEARNING_RATE = 0.5
DISCOUNT_RATE = 0.7


def self_play(searchers):
    'return 0 if white wins and 1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    boards = [pos.board]

    for _ in range(200):
        # if no possible white moves, black checkmate, else white ply
        whiteMove = random.choice(searchers).search_white(pos, secs=None)
        if whiteMove == None:
            # if no possible moves and in check, then mate
            if check(pos.rotate()):
                return 1, boards, pos
            else:
                return
        pos = whiteMove
        boards.append(pos.board)
        
        # if no possible black moves, white checkmate, else black ply
        blackMove = random.choice(searchers).search_black(pos, secs=None)
        if blackMove == None:
            # if no possible moves and in check, then mate
            if check(pos):
                return 0, boards, pos
            else:
                return
        pos = blackMove
        boards.append(pos.board)


def train(numGames, searchers):
    'train the searcher using td learning'
    weightsNum = 1
    for n in range(numGames):
        output = self_play(searchers)
        if output != None:
            # if winner, print final outcome and identity
            winner, boards, pos = output
            print_pos(pos)
            print(n, winner)
            
            posValues = [DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            negValues = [-DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            if winner == 0:
                outputs = zip(posValues, negValues)
            else:
                outputs = zip(negValues, posValues)
            trainingPairs = zip(boards, outputs)
            random.shuffle(trainingPairs)
            train_step(searchers[0].network, trainingPairs, LEARNING_RATE)

        else:
            print(n, 'draw')
            
        if (n % 100 == 0):
            name = "weights/" +  str((n / 100) + 1) + ".t7"
            torch.save(searchers[0].network.state_dict(), name)
        

def validate(searchers):
    'get % wins against random agents'
    learn = 0
    for i in range(100):
        if play(searchers[0], searchers[1]) == 0:
            learn += 1
        if play(searchers[1], searchers[0]) == 1:
            learn += 1
    with open("learn.txt", "a") as file:
        file.write(str(learn) + "\n")

        
if __name__ == "__main__":
    searchers = (DeepSearcher(), RandomSearcher())
    while True:
        train(1000, searchers)
        validate(searchers)
    
