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
        return [nP for nP in nextPos if check_valid_move(nP)]


    def select_minimax(self, moves, player):
        'select the minimax value'
        self.bestNode, self.score = None, None
        for m in moves:
            if (self.bestNode==None) or (m.score[player] > self.score[player]):
                self.bestNode, self.score = m.pos, m.score

                
    def is_checkmate(self, pos):
        'if in checkmate, set score to reward values'
        # if white
        if player == 0:
            # if in check, then mate, otherwise stalemate draw
            if check(pos.rotate()):
                self.score = [1.0, -1.0]
            else:
                self.score = [0.0, 0.0]
        # otherwise black
        else:
            # if in check, then mate, otherwise stalemate draw
            if check(pos.rotate()):
                self.score = [-1.0, 1.0]
            else:
                self.score = [0.0, 0.0]
                

    def __init__(self, pos, depth, player, network, state):
        'searches depth nodes, player is about to move on board'
        # white = 0, black = 1
        self.pos = pos
        self.board = pos.board
        self.bestNode = None
        self.self_score = forward_pass(network, self.board) 
        self.score = None
        self.moves = []
        if depth == 0:
            # if depth is 0, leaf reached, score is evaluated value
            self.score = self.self_score
        else:
            nextPs = self.gen_all_valid_moves(pos, player)
            # if no available moves, terminal game state reached
            if nextPs == []:
                # if training, set reward values, otherwise score is evaluated
                if state == 'train':
                    self.is_checkmate()
                else:
                    self.score = self.self_score
            # otherwise do recursion and select the minimax value
            else:
                for np in nextPs:
                    nextP = node(np, depth-1, (player+1) % 2, network, state)
                    self.moves.append(nextP)
                self.select_minimax(self.moves, player)
            
            
    def depth_first_pairs(self):
        'depth first search (board, score) pairs'
        if self.moves == []:
            return []
        else:
            pairs = [(self.board, self.score - self.self_score)]
            for m in self.moves:
                pairs += m.depth_first_pairs()
            return pairs
        
