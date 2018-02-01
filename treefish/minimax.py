#!/usr/bin/env python
"""
minimax algorithm and search tree
"""
from valueNetwork import *
from slimfish import *


class node:
    # node value: self.board
    # children list: self.moves
    # minimax value self.score

    def gen_all_valid_moves(self, pos, player):
        'generate all possible moves, depending on the player'
        if player == 0:
            nextPos = [pos.move(m) for m in pos.gen_moves()]
        else:
            rotPos = pos.rotate()
            nextPos = [rotPos.move(m).rotate() for m in rotPos.gen_moves()]
        return [nP for np in nextPos if check_valid_move(nP)]


    def select_minimax(self, moves, player):
        'select the minimax value'
        self.bestNode, self.score = None, None
        for m in moves:
            if (self.score == None) or (m.score[player] > self.score[player]):
                self.bestNode, self.score = m, m.score


    def __init__(self, pos, depth, player, network):
        'searches depth nodes, player is about to move on board'
        # white = 0, black = 1
        self.board = pos.board
        self.score = None
        self.moves = []
        if depth == 0:
            # if depth is 0, leaf reached, thus evaluate current board
            self.score = forward_pass(network, self.board)
        else:
            nextPs = self.gen_all_valid_moves(pos, player)
            # if no available moves, leaf reached, thus evaluate current board 
            if nextPs == []:
                self.score = forward_pass(network, self.board)
            # otherwise do recursion and select the minimax value
            else:
                for np in nextPs:
                    self.moves.append(node(pos, depth-1, (player+1)%2, network))
                self.select_minimax(self.moves, player)
            
            
    def depth_first_pairs(self):
        'depth first search (board, score) pairs'
        pairs = [(self.board, self.score)]
        for m in self.moves:
            pairs += m.depth_first_pairs()
        return pairs
        
