# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers


## configuration of network architecture
IN_TIMESTEPS = 20  # timestep of input layer
OUT_TIMESTEPS_RANGE = [5, 6]  # minimal/maximal timestep of output layer
OUTPUT_DIM = 6  # dimensionality of each timestep output; True: hand(0:3)
 
RNN_LAYERS = [{'num_units': 50}]  # number list of hidden units in each rnn layer.

DENSE_LAYER_RANGE = list(range(OUT_TIMESTEPS_RANGE[0], OUT_TIMESTEPS_RANGE[1]+1))
DENSE_LAYER_SUM = sum(DENSE_LAYER_RANGE)
DENSE_LAYER_DIM = DENSE_LAYER_SUM * OUTPUT_DIM # dimensionality of output layer.
print(DENSE_LAYER_DIM)


def lstm_model():
    
    def lstm_cells(layers):
        """
        param RNN_LAYERS: list of int or dict
                          * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                          * list of dict: [{steps: int, keep_prob: int}, ...]
        return: the model definition
        """
        
        cells = []
        
        if not isinstance(layers[0], dict):
            for step in layers:
                cell = tf.contrib.rnn.BasicLSTMCell(step, state_is_tuple=True)
                cells.append(cell)
            
        if isinstance(layers[0], dict):
            for layer in layers:
                cell = tf.contrib.rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True)
                if layer.get('keep_prob'):
                  cell = tf.contrib.rnn.DropoutWrapper(cell, layer['keep_prob'])
                cells.append(cell)
                
        return cells


    def dnn_layer(input_layer, layer):
        return tflayers.stack(input_layer, tflayers.fully_connected, layer)


    def _lstm_model(X, Y):
        """
        Creates a deep model based on:
        * stacked lstm cells
        * an dense layer
        """
        
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(RNN_LAYERS), state_is_tuple=True)
        
        X_ = tf.unstack(X, axis=1)
        print('X_ length:', len(X_))
        print('X_[0].shape:', X_[0].get_shape())
        
        
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, X_, dtype=dtypes.float32)
        print('output shape of rnn:', output[-1].get_shape())
        
        ## output predictions via dense layers
        Y_pred = dnn_layer(output[-1], [DENSE_LAYER_DIM])
        
        ## reshape and split predictions
        Y_pred = tf.reshape(Y_pred, [-1, DENSE_LAYER_DIM//OUTPUT_DIM, OUTPUT_DIM])
        Y_pred_splits = tf.split(Y_pred, DENSE_LAYER_RANGE, axis=1)
        print('Y_pred_splits length:', len(Y_pred_splits), Y_pred_splits[0].get_shape())

        ## reshape and split true output data
        Y = tf.reshape(Y, [-1, DENSE_LAYER_DIM//OUTPUT_DIM, OUTPUT_DIM])
        Y_splits = tf.split(Y, DENSE_LAYER_RANGE, axis=1)
        print('Y_splits length:', len(Y_splits), Y_splits[0].get_shape())
        

        ## calculate losses for outputs based on linear regression
        predictions = []
        losses = []
        for i, out_timesteps in enumerate(DENSE_LAYER_RANGE):
            with tf.variable_scope('DENSE_LAYER_RANGE_' + str(i)) as scope:
                y_pred = tf.reshape(Y_pred_splits[i], [-1, out_timesteps*OUTPUT_DIM])
                y = tf.reshape(Y_splits[i], [-1, out_timesteps*OUTPUT_DIM])
                print('shape of y_pred and y:', y_pred[i].get_shape(), y[i].get_shape())
                
                prediction, loss = tflearn.models.linear_regression(y_pred, y)
                prediction = tf.reshape(prediction, [-1, out_timesteps, OUTPUT_DIM])
                print('shape of prediction:', prediction.get_shape())
                
                predictions.append(prediction)
                losses.append(loss)
                
        total_prediction = tf.concat(predictions, axis=1)
        print('shape of total_prediction:', total_prediction.get_shape())
        
        ## calculate the mean loss for optimization 
        total_loss = tf.reduce_mean(tf.stack(losses))
        train_op = tf.contrib.layers.optimize_loss(total_loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad', learning_rate=0.1)
        
        return total_prediction, total_loss, train_op

    return _lstm_model