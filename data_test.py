'''
This script only works in python2 environment
'''
from __future__ import print_function
import pickle
import numpy as np

def main():
    reg_data = pickle.load(open("./pkl/datasets_raw.pkl", "rb"))
    print('1')

    # for true_traj,pred_traj in zip(y_true,predicted):
    #     # plot_result(pred_traj,pred_flag=True)
    #     plot_result(true_traj)
    #     plt.show()


if __name__ == '__main__':
     main()
