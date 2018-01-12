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
    'return 1 if white wins and -1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    boards = [pos]

    for _ in range(200):
        # if no possible white moves, black checkmate, else white ply
        whiteMove = random.choice(searchers).search_white(pos, secs=None)
        if whiteMove == None:
            # if no possible moves and in check, then mate
            if check(pos.rotate()):
                return -1, boards
            else:
                return 0, boards
        pos = whiteMove
        boards.append(pos)
        
        # if no possible black moves, white checkmate, else black ply
        blackMove = random.choice(searchers).search_black(pos, secs=None)
        if blackMove == None:
            # if no possible moves and in check, then mate
            if check(pos):
                return 1, boards
            else:
                return 0, boards
        pos = blackMove
        boards.append(pos)
    # otherwise a draw
    return 0, boards


def train(numGames, searchers):
    'train the searcher using td learning'
    global weightsNum
    for n in range(numGames):
        winner, positions = self_play(searchers)

        boards = [p.board for p in positions]

        # compute temporal difference values
        differences = []
        for i in range(len(positions)-1)]:
            positions[i+1].score - positions[i].score

        # then add the final board values to the differences
        if winner == 1:
            differences.append(np.array([1.0, -1.0]) - positions[-1].score)
        elif winner == -1:
            differences.append(np.array([-1.0, 1.0]) - positions[-1].score)
        else:
            differences.append(np.array([0.0, 0.0]) - positions[-1].score)

        
        if output != None:
            # if winner, print final outcome and identity
            winner, boards = output
            print(n, winner)
            print_pos(boards[-1])
            
            posValues = [DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            negValues = [-DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            if winner == 1:
                outputs = zip(posValues, negValues)
            elif winner == -1:
                outputs = zip(negValues, posValues)
            trainingPairs = zip(boards, outputs)
            random.shuffle(trainingPairs)
            train_step(searchers[0].network, trainingPairs, LEARNING_RATE)

        else:
            print(n, 'draw')
            
        if (n % 100 == 0):
            name = "tdWeights/" +  str((n / 100) + 1) + ".t7"
            torch.save(searchers[0].network.state_dict(), name)
        

def validate(searchers):
    'get % wins against random agents'
    learn = 0
    for i in range(100):
        if play(searchers[0], searchers[1]) == 1:
            learn += 1
        if play(searchers[1], searchers[0]) == -1:
            learn += 1
    with open("tdLearn.txt", "a") as file:
        file.write(str(learn) + "\n")

        
if __name__ == "__main__":
    searchers = (DeepSearcher(), RandomSearcher())
    weightsNum = 0
    while True:
        train(1000, searchers)
        validate(searchers)
    
