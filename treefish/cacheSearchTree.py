#!/usr/bin/env python
"""
Create a file searchTreeCache.csv which is DFS of depth 50 of chess game
"""
from slimfish import *
from valueNetwork import board_to_feature_vector
import csv, numpy as np


def next_moves(pos, player):
    'generate all possible moves, depending on the player'
    if player == 0:
        nextPos = [pos.move(m) for m in pos.gen_moves()]
        return [nP for nP in nextPos if check_valid_move(nP)]
    else:
        rotPos = pos.rotate()
        nextPos = [rotPos.move(m) for m in rotPos.gen_moves()]
        return [nP.rotate() for nP in nextPos if check_valid_move(nP)]


def evaluate(pos, player):
    'if in checkmate, set score to reward values'
    # if white
    if player == 0:
        # if in check, then mate, otherwise stalemate draw
        if check(pos.rotate()):
            score = [1.0, -1.0]
        else:
            score = [0.0, 0.0]
    # otherwise black
    else:
        # if in check, then mate, otherwise stalemate draw
        if check(pos):
            score = [-1.0, 1.0]
        else:
            score = [0.0, 0.0]
    return score


def depth_first_search(depth, node, w, w1):
    'depth first search to depth d'
    # if not at depth limit, continue
    player = depth % 2
    if depth != 0:
        nextBoards = next_moves(node, player)
        # if possible moves, create local stack and evaluate
        if len(nextBoards) != 0:
            depth -= 1
            scores = [depth_first_search(depth, nB, w, w1) for nB in nextBoards]
            score = max(scores)
        # otherwise evaluate
        else:
            score = evaluate(node, player)
    # otherwise evaluate leaf nodes
    else:
        score = evaluate(node, player)
    w.writerow(score + list(board_to_feature_vector(node.board)))
    w1.writerow(score + [node.board.replace("\n", "newline")])
    return score


writeFile = open('searchTreeCache.csv', 'wb')
writeFile1 = open('searchTreeCacheVerbose.csv', 'wb')
w = csv.writer(writeFile)
w1 = csv.writer(writeFile1)
pos = Position(initial, 0, (True,True), (True,True), 0, 0)
depth_first_search(100, pos, w, w1)
