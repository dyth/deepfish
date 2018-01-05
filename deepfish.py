#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning one-ply lookahead
"""
from __future__ import print_function
from collections import OrderedDict, namedtuple
import re, sys

from sunfish import Position, main
from valueNetwork import *
from boardRepresentation import *

    
class Searcher:

    def __init__(self):
        self.network = EvalNet()
        self.boards = []

        
    def check_valid_move(self, pos):
        'check move for validity'
        for opponentMove in pos.gen_moves():
            opponentBoard = pos.move(opponentMove).board
            if ('k' not in opponentBoard) or ('K' not in opponentBoard):
                return False
        return True


    def search(self, pos, secs=None):
        'generate all valid moves, then greedily select largest one'
        base, move, newPosition, moves = 0.0, None, None, pos.gen_moves()
        for m in moves:
            newPos = pos.move(m)
            # check whether king is in check
            if self.check_valid_move(newPos):
                board = board_to_small_feature_vector(newPos.board)
                score = forward_pass(self.network, board)
                if score > base:
                    base, move, newPosition = score, m, newPos
        if newPosition == None:
            # if no move possible, return None
            return None
        else:
            # append new board position to boards, return move
            self.boards.append(board_to_small_feature_vector(newPosition.board))
            return move, newPosition.score


if __name__ == '__main__':
    main(Searcher())
