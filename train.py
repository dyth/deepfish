#!/usr/bin/env python
"""
train deepfish using reinforcement learning
"""
from __future__ import print_function

from deepfish import Searcher as DeepSearcher
from randomfish import Searcher as RandomSearcher
from sunfish import initial, print_pos, Position
from valueNetwork import *
from boardRepresentation import *

import torch, random


LEARNING_RATE = 0.0003
DISCOUNT_RATE = 0.7


def play(white, black):
    'return 0 if white wins and 1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    boards = [pos.board]

    for _ in range(200):
        # if no possible white moves, black checkmate, else white ply
        whiteMoves = white.search(pos, secs=None)
        if whiteMoves == None:
            return 1
        pos = pos.move(whiteMoves[0])
        
        # if no possible black moves, white checkmate, else black ply
        blackMoves = white.search(pos, secs=None)
        if blackMoves == None:
            return 0
        pos = pos.move(blackMoves[0])
        

def self_play(searchers):
    'return 0 if white wins and 1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    boards = [pos.board]

    for _ in range(200):
        # if no possible white moves, black checkmate, else white ply
        whiteMoves = random.choice(searchers).search(pos, secs=None)
        if whiteMoves == None:
            return 1, boards, pos.rotate()
        
        pos = pos.move(whiteMoves[0])
        boards.append(pos.rotate().board)
        
        # if no possible black moves, white checkmate, else black ply
        blackMoves = random.choice(searchers).search(pos, secs=None)
        if blackMoves == None:
            return 0, boards, pos

        pos = pos.move(blackMoves[0])
        boards.append(pos.board)


def train(numGames, searchers):
    'train the searcher using td learning'
    num = 0
    # train numGames
    for _ in range(numGames):
        output = self_play(searchers)
        # if there is a winner
        if output != None:
            # print winner and final outcome
            winner, boards, finalBoard = output
            print_pos(finalBoard)
            print(num, winner)
            
            # determine winner
            if winner == 0:
                positive, negative = boards[1:][::2], boards[::2]
            else:
                positive, negative = boards[::2], boards[1:][::2]

            # generate values: expected outputs
            positive = [board_to_small_feature_vector(b) for b in positive]
            posValues = [DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            #negValues = [-DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            
            train_step(searchers[0].network, positive, posValues, LEARNING_RATE)
            if (num % 10 == 0):
                name = "weights/" +  str(num) + ".t7"
                torch.save(searchers[0].network.state_dict(), name)
        else:
            print(num, 'no winner')
        num += 1


def validate(searchers):
    'get % wins against random agents'
    play(searchers[0], searchers[1])
    play(searchers[1], searchers[0])
        
if __name__ == "__main__":
    searchers = (DeepSearcher(), RandomSearcher())
    train(500, searchers)
    validate(searchers)
    
