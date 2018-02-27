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
            return [nP for nP in nextPos if check_valid_move(nP)]
        else:
            rotPos = pos.rotate()
            nextPos = [rotPos.move(m) for m in rotPos.gen_moves()]
            return [nP.rotate() for nP in nextPos if check_valid_move(nP)]


    def select_minimax(self, player):
        'select the minimax value'
        self.bestNode, self.score = None, None
        for m in self.moves:
            if (self.bestNode==None) or (m.score[player] > self.score[player]):
                self.bestNode, self.score = m.pos, m.score

                
    def is_checkmate(self, pos, player):
        'if in checkmate, set score to reward values'
        # if white
        if player == 0:
            # if in check, then mate, otherwise stalemate draw
            if check(pos.rotate()):
                self.score = [1000.0 * 1.0, 1000.0 * -1.0]
            else:
                self.score = [0.0, 0.0]
        # otherwise black
        else:
            # if in check, then mate, otherwise stalemate draw
            if check(pos):
                self.score = [1000.0 * -1.0, 1000.0 * 1.0]
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
        self.noTrain = False
        
        # if no available moves, terminal game state reached
        nextPs = self.gen_all_valid_moves(pos, player)
        if nextPs == []:
            # if training, set reward values, otherwise network score
            if state == 'train':
                self.is_checkmate(pos, player)
            else:
                self.score = self.self_score
        else:
            # if depth 0, leaf reached, score is evaluated value
            if depth == 0:
                self.score = self.self_score
                self.noTrain = True # do not backup leaf states
            # otherwise recursively select minimax value
            else:
                for np in nextPs:
                    nextP = node(np, depth-1, (player+1)%2, network, state)
                    self.moves.append(nextP)
                self.select_minimax(player)
            
            
    def depth_first_pairs(self):
        'depth first search (board, score) pairs'
        pairs = [(self.board, self.score - self.self_score)]
        if self.noTrain == True:
            return []
        else:
            for m in self.moves:
                pairs += m.depth_first_pairs()
            return pairs
        
