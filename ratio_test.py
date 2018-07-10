from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import csv
import pickle
from utils import restore_data
from utils import calculate_mse

import tensorflow as tf
from tensorflow.contrib import learn

import lstm
from lstm import lstm_model
from data_processing import generate_data
import logging

## optimization hyper-parameters
TRAINING_STEPS = 10000
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
TRAINING_STEPS = 1
regressor.fit(X['train'], Y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)


#todo: test average predicted error each step
#todo: add experiment, joint angles error
#todo: add experiment, joing angles error in Cartersian space


## prepare for testing
step = 0.05
ratio = [step * i for i in range(1, int(1 / step) + 1)]
# mse_array = [0] * len(ratio)
mse_array = np.zeros(len(ratio))
count = 0

# start test
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

        # for debug
        pickle.dump(y_true_split,open("test_y_true.pkl","wb"))
        pickle.dump(y_pred_split, open("test_y_predicted.pkl", "wb"))
        print("save successfully!")

        ## restore to origin data (0-1 to original range)
        y_true_restore = restore_data.restore_dataset(y_true_split)
        pred_restore = restore_data.restore_dataset(y_pred_split)

        #todo: save true and pred into new file

        # update iteration and check error
        if (i + 1) < len(lstm.DENSE_LAYER_RANGE):
            begin += lstm.DENSE_LAYER_RANGE[i]
            end = begin + lstm.DENSE_LAYER_RANGE[i + 1]

        ## write mse against ratio to csv file
        for i, r in enumerate(ratio):
            index = int(r * len(y_true_restore) - 1)
            # imse = calculate_mse.mse(pred_restore[index],y_true_restore[index])
            dmse,_ = calculate_mse.dist(pred_restore[index], y_true_restore[index])
            mse_array[i] = mse_array[i] + dmse

        count+=1
        print("count: ", count)

print("test trajectories: ",count)
mse_array = np.divide(mse_array,count)

# mse_array[:]=[x/len(X['test']) for x in mse_array]

#write to csv file
CSV_PATH = './csv/'+str(lstm.IN_TIMESTEPS)+str(lstm.OUT_TIMESTEPS_RANGE[-1])
with open(CSV_PATH+'ratio_result.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(ratio)
    spamwriter.writerow(mse_array)