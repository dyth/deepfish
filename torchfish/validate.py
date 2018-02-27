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
        blackMoves = black.search(pos, secs=None)
        if blackMoves == None:
            return 0
        pos = pos.move(blackMoves[0])

def validate(searchers):
    'get % wins against random agents'
    learn = 0
    for i in range(50):
        if play(searchers[0], searchers[1]) == 0:
            learn += 1
        if play(searchers[1], searchers[0]) == 1:
            learn += 1
    with open("learn.txt", "w") as file:
        file.write(str(learn) + "\n")


if __name__ == "__main__":
    searchers = (DeepSearcher(), RandomSearcher())
    searchers[0].network.load_state_dict(torch.load("weights/10.t7"))
    validate(searchers)
