#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning one-ply lookahead
"""
import re, sys
import numpy as np

from slimfish import *
from valueNetwork import *

    
class Searcher:

    def __init__(self):
        self.network = EvalNet()

    def search_white(self, pos, secs=None):
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
            return nextPosition._replace(score = np.array(base_score))

        
    def search_black(self, pos, secs=None):
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
            return nextPosition._replace(score = np.array(base_score)).rotate()

                
if __name__ == '__main__':
    play(Searcher(), Searcher())
