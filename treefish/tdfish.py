#!/usr/bin/env python
"""
2-ply searcher trained with temporal difference reinforcement learning 
"""
from __future__ import print_function
import torch, random, re, sys
import numpy as np

from randomfish import Searcher as RandomSearcher
from slimfish import *
from valueNetwork import *
from minimax import *


LEARNING_RATE = 0.5
DISCOUNT_RATE = 0.7


class TreeStrapSearcher:
    
    def __init__(self):
        'set network to evalnet and minimax search depth'
        self.network = EvalNet()
        self.depth = 4


    def train_white(self, pos, secs=None):
        'train white on minimax values to self.depth'
        currentNode = node(pos, self.depth, 0, self.network, 'train')
        return currentNode.bestNode, currentNode.depth_first_pairs()

    
    def train_black(self, pos, secs=None):
        'train black on minimax values to self.depth'
        currentNode = node(pos, self.depth, 1, self.network, 'train')
        return currentNode.bestNode, currentNode.depth_first_pairs()


    def search_white(self, pos, secs=None):
        'train white on minimax values to self.depth'
        return node(pos, self.depth, 0, self.network, 'search').bestNode
        
    
    def search_black(self, pos, secs=None):
        'train black on minimax values to self.depth'
        return node(pos, self.depth, 1, self.network, 'search').bestNode
         
    
def play(white, black):
    'return 1 if white wins and -1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)

    for _ in range(40):
        # if no possible white moves, black checkmate, else white ply
        whiteMove = white.search_white(pos, secs=None)
        if whiteMove == None:
            # if no possible moves and in check, then mate
            if check(pos.rotate()):
                return -1
            else:
                return 0.5
        pos = whiteMove

        # if no possible black moves, white checkmate, else black ply
        blackMove = black.search_black(pos, secs=None)
        if blackMove == None:
            # if no possible moves and in check, then mate
            if check(pos):
                return 1
            else:
                return 0.5
        pos = blackMove
    print('cows')
    return 0


def self_play(searcher):
    'return 1 if white wins and -1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    boards = []

    # train for 800 steps
    for _ in range(40):
        # if no possible white moves, black checkmate, else white ply
        move, pairs = searcher.train_white(pos, secs=None)
        if move == None:
            # if no possible moves and in check, then mate
            if check(pos.rotate()):
                return -1, boards
            else:
                return 0, boards
        pos = move
        boards += pairs
        
        # if no possible black moves, white checkmate, else black ply
        move, pairs = searcher.train_black(pos, secs=None)
        if move == None:
            # if no possible moves and in check, then mate
            if check(pos):
                return 1, boards
            else:
                return 0, boards
        pos = move
        boards += pairs
    # otherwise a draw
    return 0, boards

        
def train(numGames, searcher):
    'train the searcher using td learning'
    for n in range(numGames):
        winner, positions = self_play(searcher)

        boards = [p[0] for p in positions]
        outputs = [p[1] for p in positions]
        trainingPairs = zip(boards, outputs)
        
        # add the final board values to differences and print final position
        if winner == 1:
            print(n, "White won")
        elif winner == -1:
            print(n, "Black won")
        else:
            print(n, "Draw")
        print(boards[-1])
        
        # create and train on (board, TD(DISCOUT_RATE)) pairs
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
    searchers = (TreeStrapSearcher(), RandomSearcher())
    weightsNum = 0
    while True:
        #searchers[0].network.load_state_dict(torch.load("tdWeights/11.t7"))
        #train(1000, searchers[0])
        validate(searchers)
        weightsNum += 1
