from __future__ import print_function
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot_result(traj,pred_flag=False):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    if pred_flag==True:#print predict trajectory
        ax.plot(traj[:,0],traj[:,1],traj[:,2], '-', linewidth=5, color='red', label='predict result')
    else:#print true trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', linewidth=5, color='blue', label='ground truth result')


# def seperate_data(y_true,predicted):
#     for set1, set2 in zip(y_true, predicted):
#         len1 = len(set1)
#         len2 = len(set1[0])
#         sample1= np.reshape(set1, (len1 * len2))
#         sample2 = np.reshape(set2,(len1 * len2))
#         print(mse(sample1,sample2))
#
# def traj_dist(pred,true):
#     dist = []
#     for sample1,sample2 in zip(pred,true):
#         dist.append(dist(sample1,sample2))
#     dist_mean=np.mean(dist)
#     return dist_mean


def main():
    y_true = pickle.load(open("../results/raw_true_trajs.pkl", "rb"))
    predicted = pickle.load(open("../results/raw_pred_trajs.pkl","rb"))

    for true_traj,pred_traj in zip(y_true,predicted):
        # plot_result(pred_traj,pred_flag=True)
        plot_result(true_traj)
        plt.show()


if __name__ == '__main__':
     main()
