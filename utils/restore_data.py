'''
test read data
'''

from __future__ import print_function
import pickle
import numpy as np
import os


FILE_PATH = os.path.dirname(__file__)
#start testing
min_max_path = open(os.path.join(FILE_PATH,'../pkl/min_max_scaler.pkl'),'rb')
MIN_MAX_SCALAR = pickle.load(min_max_path)
#scalar of left_hand, then left_joint

def restore_dataset(dataset):
    restore_set = []
    for data in dataset:
        temp=[]
        for step_data in data:
            r_data = step_data[0:3]
            h_data=step_data[3:6]
            origin_rdata = restore_single_rdata(r_data)
            origin_hdata = restore_single_hdata(h_data)

            origin_data = np.hstack([origin_rdata,origin_hdata])
            temp.append(origin_data)
        restore_set.append(temp)
    restore_set = np.asarray(restore_set)
    return restore_set


def restore_single_hdata(data):
    global MIN_MAX_SCALAR

    #check length. expand our data to scalar size
    data_len = len(data)
    scalar_len = len(MIN_MAX_SCALAR.scale_)

    if data_len!=scalar_len:
        #expand the data size to our size
        z = np.zeros((scalar_len-data_len,), dtype=data.dtype)
        exp_data = np.concatenate((data, z), axis=0)
    else:
        exp_data=data

    #transform data from 0-1 to origin
    restore_data = MIN_MAX_SCALAR.inverse_transform(exp_data.reshape(1,-1))

    origin_data = restore_data[0][0:len(data)] #human data

    return origin_data

def restore_single_rdata(data):
    global MIN_MAX_SCALAR

    # check length. expand our data to scalar size
    data_len = len(data)
    scalar_len = len(MIN_MAX_SCALAR.scale_)

    if data_len != scalar_len:
        # expand the data size to our size
        z = np.zeros((scalar_len - data_len,), dtype=data.dtype)
        exp_data = np.concatenate((z, data), axis=0)
    else:
        exp_data = data

    # transform data from 0-1 to origin
    restore_data = MIN_MAX_SCALAR.inverse_transform(exp_data.reshape(1, -1))
    # extract our feature
    # if len(data) == 3:

    origin_data = restore_data[0][-len(data):]  # robot only

    return origin_data


def main():
    # read x_test, y_true and predict
    # X_test = pickle.load(open("test_x.pkl", "rb"), encoding='iso-8859-1')
    y_true = pickle.load(open("../test_y_true.pkl","rb"))
    predicted = pickle.load(open("../test_y_predicted.pkl","rb"))

    y_true_restore = restore_dataset(y_true)
    pred_restore =restore_dataset(predicted)

if __name__ == '__main__':
    main()





