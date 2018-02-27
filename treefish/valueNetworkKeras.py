#!/usr/bin/env python
"""multilayer perceptron for chess agent"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np


# create model
def create_model():
    np.random.seed(1729)
    model = Sequential()
    model.add(Dense(units=96, input_dim=96, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=2, activation='tanh'))
    model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])
    return model


def train_model(model, pairs):
    # train model
    data = np.array([board_to_feature_vector(p[0]) for p in pairs])
    values = np.array([p[1][0] for p in pairs])
    model.fit(data, values, epochs=1, batch_size=1)


def forward_pass(model, board):
    KerasRegressor(build_fn=model, epochs=1, batch_size=1, verbose=0)
    board = board_to_feature_vector(board)
    return model.predict(np.array([board]))

    
def set_piece_position_small(index, vector, f, r):
    'set normalised file, rank, 8-rank for a piece'
    while vector[index] != -1.0:
        index += 2
    # normalise values
    vector[index] = f / 8.0
    vector[index+1] = r / 8.0
    

def board_to_feature_vector(board):
    'promote to queen only, resulting in 144 length vector'
    # create {piece -> vectorposition} dictionary
    # {white, black} 8*Pawn, 2*Knight, 2*Bishop, 2*Rook, 9*Queen, 1*King
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    count = [8, 2, 2, 2, 9, 1, 8, 2, 2, 2, 9, 1]
    countSum = [2 * sum(count[:i]) for i in range(len(count))]
    index = dict(zip(pieces, countSum))
    # create input vector for network
    inputVector = np.full(2*sum(count), -1.0)
    board = board[20:-20]
    for f in range(8):
        for r in range(8):
            square = board[(10*f) + r+1]
            # if occupied, set places in the vector
            if square != '.':
                set_piece_position_small(index[square], inputVector, f, r)
    return inputVector
