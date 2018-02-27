#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
"""
randomly choose next move out from all next possible valid moves
"""
import re, sys, random

from slimfish import *

    
class Searcher:

    def select_move(self, newPositions):
        base_score, nextPosition = None, None
        for nP in newPositions:
            # if move is valid and scored higher than existing value, replace
            if check_valid_move(nP):
                score = 2.0 * random.random() - 1
                if (score > base_score) or (base_score == None):
                    base_score, nextPosition = score, nP
        if nextPosition == None:
            # if no move possible, return None
            return None
        else:
            nextPosition._replace(score = base_score)
            return nextPosition

        
    def search_white(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.move(m) for m in pos.gen_moves()]
        return self.select_move(newPositions)

        
    def search_black(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        newPositions = [pos.rotate().move(m) for m in pos.rotate().gen_moves()]
        nextPosition = self.select_move(newPositions)
        if nextPosition != None:
            return nextPosition.rotate()


if __name__ == '__main__':
    play(Searcher(), Searcher())
    
