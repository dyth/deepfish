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

    # train for 800 steps
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
    for n in range(numGames):
        winner, positions = self_play(searchers)
        plys = len(positions)
        
        # compute temporal difference values
        diffs = [positions[i+1].score-positions[i].score for i in range(plys-1)]
        
        # add the final board values to differences and print final position
        if winner == 1:
            diffs.append(np.array([1.0, -1.0]) - positions[-1].score)
            print(n, "White won")
        elif winner == -1:
            diffs.append(np.array([-1.0, 1.0]) - positions[-1].score)
            print(n, "Black won")
        else:
            diffs.append(np.array([0.0, 0.0]) - positions[-1].score)
            print(n, "Draw")
        print_pos(positions[-1])
            
        # iterate over the discounted sum of future rewards from current state
        outputs = []
        for i in range(plys):
            o = sum([diffs[i+j] * DISCOUNT_RATE**j for j in range(plys - i)])
            outputs.append(o)

        # create and train on (board, TD(DISCOUT_RATE)) pairs
        boards = [p.board for p in positions]
        trainingPairs = zip(boards, outputs)
        train_step(searchers[0].network, trainingPairs, LEARNING_RATE)

    # save weights in directory tdWeights/
    global weightsNum
    name = "tdWeights/" +  str(weightsNum + 1) + ".t7"
    torch.save(searchers[0].network.state_dict(), name)
        

def validate(searchers):
    'get % wins against random agents'
    whiteWins, blackWins, stalemate = 0, 0, 0
    for i in range(50):
        # agent as white, random as black
        firstPlay = play(searchers[0], searchers[1])
        if firstPlay == 1:
            whiteWins += 1
        elif firstPlay == 0.5:
            stalemate += 1
        # random as white, agent as black
        secondPlay = play(searchers[1], searchers[0])
        if secondPlay == -1:
            blackWins += 1
        elif secondPlay == 0.5:
            stalemate += 1
    with open("tdLearn.txt", "a") as file:
        l = str(whiteWins) + " " + str(blackWins) + " " + str(stalemate) + "\n"
        file.write(l)

        
if __name__ == "__main__":
    searchers = (DeepSearcher(), RandomSearcher())
    weightsNum = 0
    while True:
        train(1000, searchers)
        validate(searchers)
        weightsNum += 1
