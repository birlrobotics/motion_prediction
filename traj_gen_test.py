'''
generate trajectory
'''

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import csv
import pickle
from utils import restore_data
from utils import calculate_mse
from utils import traj_gen

import tensorflow as tf
from tensorflow.contrib import learn

import lstm
from lstm import lstm_model
from data_processing import generate_data
import logging

## optimization hyper-parameters
TRAINING_STEPS = 100
VALIDATION_STEPS = 1000
BATCH_SIZE = 100
LOG_DIR = './ops_logs/lstm/model_20_5_6'

## Save log to a local file
# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

## load model and dataset
X=pickle.load(open("./dataset/x_set"+str(lstm.IN_TIMESTEPS)+str(lstm.OUT_TIMESTEPS_RANGE[-1])+".pkl", "rb"))
Y=pickle.load(open("./dataset/y_set"+str(lstm.IN_TIMESTEPS)+str(lstm.OUT_TIMESTEPS_RANGE[-1])+".pkl", "rb"))

## build the lstm model
model_fn = lstm_model()
config = tf.contrib.learn.RunConfig(log_step_count_steps=200, save_checkpoints_steps=VALIDATION_STEPS // 2)

estimator = learn.Estimator(model_fn=model_fn, model_dir=LOG_DIR, config=config)
regressor = learn.SKCompat(estimator)


## create a validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], Y['val'], every_n_steps=VALIDATION_STEPS)

## fit the train dataset
regressor.fit(X['train'], Y['train'], monitors=None, batch_size=BATCH_SIZE, steps=1) #load model


# start test
raw_true_trajs=[]
raw_pred_trajs=[]

for X_test,Y_test in zip(X['test'], Y['test']):

    ## predict using test datasets and  calculate rmse with reshaping correctly
    y_true = Y_test.reshape((-1,lstm.DENSE_LAYER_DIM))

    y_pred = regressor.predict(X_test)
    y_pred = y_pred.reshape((-1,lstm.DENSE_LAYER_DIM))

    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    print("rmse: %f" % rmse)

    ## reshape for human-friendly illustration
    y_true = y_true.reshape((-1, lstm.DENSE_LAYER_SUM, lstm.OUTPUT_DIM))
    y_pred = y_pred.reshape((-1, lstm.DENSE_LAYER_SUM, lstm.OUTPUT_DIM))

    begin = 0
    end = lstm.DENSE_LAYER_RANGE[0]

    ## split data
    for i, out_timesteps in enumerate(lstm.DENSE_LAYER_RANGE):
        y_true_split = y_true[:, begin:end]
        y_pred_split = y_pred[:, begin:end]

        ## restore to origin data (0-1 to original range)
        y_true_restore = restore_data.restore_dataset(y_true_split)
        y_pred_restore = restore_data.restore_dataset(y_pred_split)

        # update iteration and check error
        if (i + 1) < len(lstm.DENSE_LAYER_RANGE):
            begin += lstm.DENSE_LAYER_RANGE[i]
            end = begin + lstm.DENSE_LAYER_RANGE[i + 1]

    raw_true_trajs.append(traj_gen.traj_generation(y_true_restore))
    raw_pred_trajs.append(traj_gen.traj_generation(y_pred_restore))

print("finish prediction, now check error")
#for debug
pickle.dump(raw_true_trajs,open("./results/raw_true_trajs.pkl","wb"))
pickle.dump(raw_pred_trajs, open("./results/raw_pred_trajs.pkl", "wb"))
print("save restored trajs successfully!")

