'''
generate whole trajectory based on output

'''

from __future__ import print_function
import pickle
import numpy as np


def traj_expand(previous_traj, future_steps,step=1):
    for i in range(0,len(future_steps)):
        if (-(len(future_steps)-step)+i) < 0:
            # print(-(len(future_steps)-step)+i)
            # temp = previous_traj[-(len(future_steps)-step)+i]
            temp2 = np.add(previous_traj[-(len(future_steps)-step)+i],future_steps[i])
            previous_traj[-(len(future_steps) - step) + i] = np.divide(temp2,2)
        else:
            previous_traj = np.append(previous_traj,[future_steps[i]],axis=0)

    return previous_traj

def traj_generation(traj_steps,step=1):
    for i,time_step in enumerate(traj_steps):
        if i==0:
            traj = np.copy(time_step)
        else:
            traj = traj_expand(traj,time_step,step)
    return traj

def combine_steps(steps):
    #todo: combine multiple time steps of prediction
    print(steps)

def main():
    predict_steps = pickle.load(open("y_pred_restore.pkl","rb"))
    true_steps = pickle.load(open("y_true_restore.pkl", "rb"))

    true_traj=traj_generation(true_steps)
    predict_traj = traj_generation(predict_steps)

    print("generate traj successfully!")


if __name__=='__main__':
    main()