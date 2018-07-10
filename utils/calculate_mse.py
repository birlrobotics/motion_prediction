from __future__ import print_function
import numpy as np
import pickle

def mse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

def dist(pred,true):
    '''
    calculate square root distance error along a trajectory
    '''
    dist = []
    for p1, p2 in zip(pred, true):
        dist.append(np.sqrt(np.sum((p1-p2)**2)))
    dist = np.asarray(dist)
    return dist.mean(),dist



def main():
    y_true = pickle.load(open("../results/raw_true_trajs.pkl", "rb"))
    predicted = pickle.load(open("../results/raw_pred_trajs.pkl","rb"))

    for true_traj,pred_traj in zip(y_true,predicted):
        mean_dist,dist_list = dist(pred_traj,true_traj)



if __name__ == '__main__':
     main()
