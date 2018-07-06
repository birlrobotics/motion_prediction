from __future__ import print_function
import numpy as np
import pickle

def mse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

def dist(pred,true):
    dist = []
    for p1, p2 in zip(pred, true):
        dist.append(np.sqrt(np.sum((p1-p2)**2)))
    dist = np.asarray(dist)
    return dist.mean()


def seperate_data(y_true,predicted):
    for set1, set2 in zip(y_true, predicted):
        len1 = len(set1)
        len2 = len(set1[0])
        sample1= np.reshape(set1, (len1 * len2))
        sample2 = np.reshape(set2,(len1 * len2))
        print(mse(sample1,sample2))



def main():
    y_true = pickle.load(open("y_true_restore.pkl", "rb"))
    predicted = pickle.load(open("pred_restore.pkl","rb"))


    seperate_data(y_true,predicted)

if __name__ == '__main__':
     main()
