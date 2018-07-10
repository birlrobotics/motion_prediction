from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import ConfigParser


# the current file path
FILE_PATH = os.path.dirname(__file__)
TASK_NAME_LIST = []

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(FILE_PATH, '../cfg/models.cfg'))
# read models params
MODEL_NAME = cp_models.get('model', 'model_name')

def mse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

def dist(pred,true):
    '''
    calculate Cartesian distance error along a trajectory
    '''
    dist = []
    for p1, p2 in zip(pred, true):
        dist.append(np.sqrt(np.sum((p1-p2)**2)))
    dist = np.asarray(dist)
    return dist.mean(),dist


def plot_3D_result(pred_traj,true_traj,human_flag=False):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    if human_flag==True:#print human predict trajectory
        ax.plot(pred_traj[:,3],pred_traj[:,4],pred_traj[:,5], '-', linewidth=5, color='blue', label='human predict result')
        ax.plot(true_traj[:,3], true_traj[:,4], true_traj[:, 5], '--', linewidth=5, color='blue', label='human ground truth result')
    else:#print robot trajectory
        ax.plot(pred_traj[:,0],pred_traj[:,1],pred_traj[:,2], '-', linewidth=5, color='red', label='robot predict result')
        ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:, 2], '--', linewidth=5, color='red', label='robot ground truth result')



def main():
    data_path = os.path.join(FILE_PATH, '../model', MODEL_NAME)
    y_true = pickle.load(open(data_path+"/raw_true_trajs.pkl", "rb"))
    predicted = pickle.load(open(data_path+"/raw_pred_trajs.pkl","rb"))

    for true_traj,pred_traj in zip(y_true,predicted):
        plot_3D_result(pred_traj,true_traj,human_flag=True)
        plot_3D_result(pred_traj, true_traj)
        plt.show()


if __name__ == '__main__':
     main()
