from __future__ import print_function
import numpy as np
import pickle

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
        mean_dist,dist_list = dist(pred_traj,true_traj)



if __name__ == '__main__':
     main()
