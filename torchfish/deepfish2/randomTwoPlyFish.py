#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
"""
randomly choose next move out from all next possible valid moves
"""
import re, sys
import numpy as np

from slimfish import *

    
class randomTwoPlySearcher:

    def search_max_white(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.move(m) for m in pos.gen_moves()]
        base_score, nextPosition = [None, None], None
        for nP in newPositions:
            # if move is valid and scored higher than existing value, replace
            if check_valid_move(nP):
                score = 2.0 * np.random.rand(2) - 1.0
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
                score = 2.0 * np.random.rand(2) - 1.0
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


if __name__ == '__main__':
    play(Searcher(), Searcher())
    
