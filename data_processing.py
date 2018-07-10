# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import pickle
import random
import lstm

## configuration of network architecture, defined in lstm.py
IN_TIMESTEPS = lstm.IN_TIMESTEPS
OUT_TIMESTEPS_RANGE = lstm.OUT_TIMESTEPS_RANGE
OUTPUT_DIM = lstm.OUTPUT_DIM

# robot (0:3) , hand (3:6) + elbow (6:9)

def rnn_data_X(data):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> chose input from sequence, [[1, 2], [2, 3], [3, 4]], 
        -> chose output from sequence, [3, 4, 5]
    """
    
    in_timesteps = IN_TIMESTEPS
    out_timesteps_min = OUT_TIMESTEPS_RANGE[0]
    out_timesteps_max = OUT_TIMESTEPS_RANGE[1]
    
    # chose input from sequence considering the maximal timestep of output layer
    rnn_df = []
    for i in range(data.shape[0] - in_timesteps - (out_timesteps_max-1)):
        X = data[i: i+in_timesteps, 3:9]  #hand (3:6) + elbow (6:9)
        rnn_df.append(X if len(X.shape) > 1 else [[item] for item in X])
    # print('X shape:', rnn_df[0].shape)
    
    return rnn_df

def rnn_data_Y(data):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> chose input from sequence, [[1, 2], [2, 3], [3, 4]], 
        -> chose output from sequence, [3, 4, 5]
    """
    in_timesteps = IN_TIMESTEPS
    out_timesteps_min = OUT_TIMESTEPS_RANGE[0]
    out_timesteps_max = OUT_TIMESTEPS_RANGE[1]

    ## chose multiple outputs from sequence for inputs
    rnn_df = []
    for i in range(data.shape[0] - in_timesteps - (out_timesteps_max-1)):
    
        Y_list = []  # multiple outputs for one input timestep
        for out_timesteps in range(out_timesteps_min, out_timesteps_max+1):
            Y = data[i+in_timesteps: i+in_timesteps + out_timesteps, 0:6]  # robot (0:3) human_hand (3:6)
            Y = Y.reshape((out_timesteps,OUTPUT_DIM))
            Y_list.append(Y)
            # print('input_timestep[{0}] and output_timestep[{1}] Y shape: {2}'.format(i, out_timesteps, Y.shape))
        
        Y_list = np.concatenate(Y_list, axis=0)
        # print("Y_list shape:", Y_list.shape)
        
        rnn_df.append(Y_list)
    
    return rnn_df


def prepare_seqs_data(seqs):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    print('length of seqs:', len(seqs))
    
    seqs_x = []
    seqs_y = []
    for i, seq in enumerate(seqs):
        print('shape of seq:', seq.shape)
        
        seq_x = rnn_data_X(seq)
        seq_y = rnn_data_Y(seq)
        
        seqs_x += seq_x
        seqs_y += seq_y
    
    seqs_x = np.array(seqs_x, dtype=np.float32)
    seqs_y = np.array(seqs_y, dtype=np.float32)
    print("shape of seqs_x and seqs_y:", seqs_x.shape, seqs_y.shape)
    
    return seqs_x, seqs_y


def prepare_test_seqs_data(seqs):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    print('length of seqs:', len(seqs))

    seqs_x = []
    seqs_y = []
    for i, seq in enumerate(seqs):
        print('shape of seq:', seq.shape)

        seq_x = rnn_data_X(seq)
        seq_y = rnn_data_Y(seq)
        seq_x = np.array(seq_x, dtype=np.float32)
        seq_y = np.array(seq_y, dtype=np.float32)
        seqs_x.append(seq_x)
        seqs_y.append(seq_y)

    print("length of seqs_x and seqs_y:", len(seqs_x), len(seqs_y))

    return seqs_x, seqs_y


def split_data(data, train_pos=0.7, val_pos=0.8, test_pos=1.0):
    """
    splits data to training, validation and testing parts
    """
    random.shuffle(data)

    num = len(data)
    train_pos = int(num * train_pos)
    val_pos = int(num * val_pos)
    
    train_data = data[:train_pos]
    val_data = data[train_pos:val_pos]

    test_data = data[val_pos:]

    return train_data, val_data, test_data


def generate_data(file_name):
    """generates data with based on a function func"""

    pkl_file = open(file_name,'rb')
    datasets = pickle.load(pkl_file)
    print('length of tasks:', len(datasets))

    seqs = []
    for i, task in enumerate(datasets):
        print('\nTask {0} has {1} seqs'.format(str(i), len(task)))

        for j, seq in enumerate(task):
            print('seq {0} has shape {1}'.format(str(j), seq.shape))
            seqs.append(seq)
    print('num of seqs:', len(seqs))


    train_seqs, val_seqs, test_seqs = split_data(seqs)
    
    print('\ntrain_seqs info:')
    train_x, train_y = prepare_seqs_data(train_seqs)
    
    print('\nval_seqs info:')
    val_x, val_y = prepare_seqs_data(val_seqs)
    
    print('\ntest_seqs info:')
    test_x, test_y = prepare_test_seqs_data(test_seqs)

    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)
