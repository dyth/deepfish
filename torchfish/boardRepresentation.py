#!/usr/bin/env python
"""
represent board as coordinate vector
"""
import numpy as np


def set_piece_position_small(index, vector, f, r):
    'set normalised file, rank, 8-rank for a piece'
    while vector[index] != -1.0:
        index += 3
    # normalise values
    vector[index] = f / 8.0
    vector[index+1] = r / 8.0
    # due to chessboard symmetry, also include reflected rank
    vector[index+2] = (8.0 - r) / 8.0
    

def board_to_small_feature_vector(board):
    'promote to queen only, resulting in 144 length vector'
    # create {piece -> vectorposition} dictionary
    # {white, black} 8*Pawn, 2*Knight, 2*Bishop, 2*Rook, 9*Queen, 1*King
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    count = [8, 2, 2, 2, 9, 1, 8, 2, 2, 2, 9, 1]
    countSum = [3*sum(count[:i]) for i in range(len(count))]
    index = dict(zip(pieces, countSum))
    # create input vector for network
    inputVector = np.full(3*sum(count), -1.0)
    board = board[20:-20]
    for f in range(8):
        for r in range(8):
            square = board[(10*f) + r+1]
            # if occupied, set places in the vector
            if square != '.':
                set_piece_position_small(index[square], inputVector, f, r)
    return inputVector
