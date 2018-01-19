#!/usr/bin/env python
"""
2-ply searcher trained with TD-Leaf reinforcement learning 
"""
from __future__ import print_function
import torch, random, re, sys
import numpy as np

from randomTwoPlyFish import randomTwoPlySearcher as RandomSearcher
from slimfish import *
from valueNetwork import *


LEARNING_RATE = 0.5
DISCOUNT_RATE = 0.7


class TDLeafSearcher:

    def __init__(self):
        self.network = EvalNet()

    def search_max_white(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.move(m) for m in pos.gen_moves()]
        base_score, nextPosition = [None, None], None
        for nP in newPositions:
            # if move is valid and scored higher than existing value, replace
            if check_valid_move(nP):
                score = forward_pass(self.network, nP.board)
                if (score[0] > base_score[0]) or (nextPosition == None):
                    base_score, nextPosition = score, nP
        if nextPosition == None:
            # if no move possible, return None
            return None
        else:
            return nextPosition._replace(score = base_score)


    def search_white(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.move(m) for m in pos.gen_moves()]
        base_score, nextPosition, leaf = [None, None], None, None
        for nP in newPositions:
            if check_valid_move(nP):
                # no opponent move scores, use current board values
                opponentMove = self.search_max_black(nP)
                if opponentMove != None:
                    score = opponentMove.score
                    #else: #REMOVE INDENTATION
                    #    score = forward_pass(self.network, nP.board)
                    # select largest score
                    if (score[0] > base_score[0]) or (nextPosition == None):
                        base_score, nextPosition, leaf = score, nP, opponentMove
                        leaf._replace(score = base_score)
        if nextPosition == None:
            # if no move possible return None, otherwise position and score
            return None, None
        else:
            return nextPosition, leaf
        
        
    def search_max_black(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.rotate().move(m) for m in pos.rotate().gen_moves()]
        base_score, nextPosition = [None, None], None
        for nP in newPositions:
            # if move is valid and scored higher than existing value, replace
            if check_valid_move(nP):
                score = forward_pass(self.network, nP.rotate().board)
                if (score[1] > base_score[1]) or (nextPosition == None):
                    base_score, nextPosition = score, nP
        if nextPosition == None:
            # if no move possible, return None
            return None
        else:
            return nextPosition._replace(score = base_score).rotate()

        
    def search_black(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.rotate().move(m) for m in pos.rotate().gen_moves()]
        base_score, nextPosition, leaf = [None, None], None, None
        for nP in newPositions:
            if check_valid_move(nP):
                # no opponent move scores, use current board values
                opponentMove = self.search_max_black(nP)
                if opponentMove != None:
                    score = opponentMove.score
                    #else: #REMOVE INDENTATION
                    #    score = forward_pass(self.network, nP.rotate().board)
                    # select largest score
                    if (score[1] > base_score[1]) or (nextPosition == None):
                        base_score, nextPosition, leaf = score, nP, opponentMove
                        leaf._replace(score = base_score)
        if nextPosition == None:
            # if no move possible return None, otherwise position and score
            return None, None
        else:
            return nextPosition.rotate(), leaf

    
def self_play(searchers):
    'return 1 if white wins and -1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    boards = []

    # train for 800 steps
    for _ in range(200):
        # if no possible white moves, black checkmate, else white ply
        whiteMove = random.choice(searchers).search_white(pos, secs=None)
        if whiteMove[1] == None:
            # if no possible moves and in check, then mate
            if check(pos.rotate()):
                return -1, boards
            else:
                return 0, boards
        pos = whiteMove[0]
        boards.append(whiteMove[1])
        
        # if no possible black moves, white checkmate, else black ply
        blackMove = random.choice(searchers).search_black(pos, secs=None)
        if blackMove[1] == None:
            # if no possible moves and in check, then mate
            if check(pos):
                return 1, boards
            else:
                return 0, boards
        pos = blackMove[0]
        boards.append(blackMove[1])
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
    name = "tdLeafWeights/" +  str(weightsNum + 1) + ".t7"
    torch.save(searchers[0].network.state_dict(), name)

    
def play(white, black):
    'return 1 if white wins and -1 if black wins'
    # initialise searcher and initial board
    pos = Position(initial, 0, (True,True), (True,True), 0, 0)

    for _ in range(200):
        # if no possible white moves, black checkmate, else white ply
        whiteMove = white.search_white(pos, secs=None)[0]
        if whiteMove == None:
            # if no possible moves and in check, then mate
            if check(pos.rotate()):
                return -1
            else:
                return 0.5
        pos = whiteMove

        # if no possible black moves, white checkmate, else black ply
        blackMove = black.search_black(pos, secs=None)[0]
        if blackMove == None:
            # if no possible moves and in check, then mate
            if check(pos):
                return 1
            else:
                return 0.5
        pos = blackMove
    return 0

    
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
    with open("tdleafLearn.txt", "a") as file:
        l = str(whiteWins) + " " + str(blackWins) + " " + str(stalemate) + "\n"
        file.write(l)

        
if __name__ == "__main__":
    searchers = (TDLeafSearcher(), RandomSearcher())
    weightsNum = 0
    while True:
        #searchers[0].network.load_state_dict(torch.load("tdWeights/11.t7"))
        train(1000, searchers)
        validate(searchers)
        weightsNum += 1
