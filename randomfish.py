#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict, namedtuple
import re, sys, random
from sunfish import Position, main
from valueNetwork import *

    
class Searcher:

    # network stays constant
    network = EvalNet()
    
    def check_valid_move(self, pos):
        'check move for validity'
        for opponentMove in pos.gen_moves():
            opponentBoard = pos.move(opponentMove).board
            if ('k' not in opponentBoard) or ('K' not in opponentBoard):
                return False
        return True


    def search(self, pos, secs):
        'generate list of all moves, then select one at random'
        base, move, moves = 0.0, None, pos.gen_moves()
        for m in moves:
            newPos = pos.move(m)
            if self.check_valid_move(newPos):
                score = random.random()
                if score > base:
                    base, move = score, m
        return move, 0.0


if __name__ == '__main__':
    main(Searcher())

    
    
    
