#!/usr/bin/env python
"""represent board as coordinate vector"""
import numpy as np

def create_blank_input_vector():
    'without any rank file information'
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    index = dict(zip(pieces, range(len(pieces))))
    # 9 due to a possibility of a maximum of 8 promotions -- length 216
    inputVector = np.full(9*2*len(pieces), -1.0)
    return index, inputVector


def set_piece_position(index, vector, f, r):
    'set count, rank, file for a piece'
    vectorIndex = 9*2*index
    # if black to move, do 8 - rank due to reflectional symmetry
    if index > 5:
        r = 8.0 - r
    while vector[vectorIndex] != -1.0:
        vectorIndex += 2
    # normalise values
    vector[vectorIndex] = f / 8.0
    vector[vectorIndex+1] = r / 8.0
    
    
def board_to_feature_vector(board):
    'output 36 length vector of (number, rank, file) piece information'
    # order of 8*Pawn, 2*Knight, 2*Bishop, 2*Rook, 1*Queen, 1*King, 8*promotion
    index, inputVector = create_blank_input_vector()
    board = board[20:-20]
    for f in range(8):
        for r in range(8):
            square = board[(10*f) + r+1]
            # if occupied, set places in the vector
            if square != '.':
                set_piece_position(index[square], inputVector, f, r)
    return inputVector
