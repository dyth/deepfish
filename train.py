#!/usr/bin/env python
"""
train deepfish using reinforcement learning
"""
from __future__ import print_function

from deepfish import *
from sunfish import initial, print_pos
from valueNetwork import *
import torch


LEARNING_RATE = 0.0001
DISCOUNT_RATE = 0.7


def observe_self_play():
    'return 0 if white wins and 1 if black wins'
    # initialise searcher and print initial board
    searcher = Searcher()
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    print_pos(pos)
    # in the range of 90 plys, 10 plys more than an average game length of 80
    for _ in range(90):
        # check black checkmate if there are no possible white moves
        whiteMoves = searcher.search(pos, secs=None)
        if whiteMoves == None:
            print("Black won")
            return 1
        
        # white ply
        pos = pos.move(whiteMoves[0])
        a = raw_input()
        print_pos(pos.rotate())

        # check white checkmate if there are no possible black moves
        blackMoves = searcher.search(pos, secs=None)
        if blackMoves == None:
            print("White won")
            return 0

        #black ply
        pos = pos.move(blackMoves[0])
        a = raw_input()
        print_pos(pos)
        

def self_play():
    'return 0 if white wins and 1 if black wins'
    # initialise searcher and print initial board
    searcher = Searcher()
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    # in the range of 90 plys, 10 plys more than an average game length of 80
    for _ in range(90):
        # if no possible white moves, black checkmate, else white ply
        whiteMoves = searcher.search(pos, secs=None)
        if whiteMoves == None:
            return 1, searcher, pos
        pos = pos.move(whiteMoves[0])

        # if no possible black moves, white checkmate, else black ply
        blackMoves = searcher.search(pos, secs=None)
        if blackMoves == None:
            return 0, searcher, pos
        pos = pos.move(blackMoves[0])


def train(numGames):
    'train the searcher using td learning'
    number = 0
    # train numGames
    for _ in range(numGames):
        output = self_play()
        # if there is a winner
        if output != None:
            winner, searcher, pos = output
            print(number, winner)
            print_pos(pos)
            # determine winner
            if winner == 0:
                boards = searcher.boards[::2]
            else:
                boards = searcher.boards[1:][::2]
            # generate values: expected outputs
            values = [DISCOUNT_RATE ** i for i in range(len(boards))][::-1]
            train_step(searcher.network, boards, values, LEARNING_RATE)
        else:
            print(number, 'no winner')
        number += 1

            
if __name__ == "__main__":
    train(500)
    #print(self_play())
