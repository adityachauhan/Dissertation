import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk
def trans_prob_matrix(obs):
    trans = np.empty((9,9),dtype = 'int')
    trans.fill(0)
    for i in range(obs[:,0].size):
        for j in range(obs[0].size-1):
            if(obs[i][j] > 0 and obs[i][j+1]>0):
                trans[obs[i][j]-1][obs[i][j+1]-1] = trans[obs[i][j]-1][obs[i][j+1]-1] + 1

    sum_trans = np.empty((9,1),dtype='int')
    sum_trans.fill(0)
    for r in range(9):
        for c in range(9):
            sum_trans[r] = sum_trans[r] + trans[r][c]

    trans_prob = np.empty((9,9),dtype = 'float')
    trans_prob.fill(0)
    for r in range(9):
        for c in range(9):
            trans_prob[r][c] = trans[r][c]/sum_trans[r]

    return trans_prob


def emission_prob_matrix(obs):
    ems = np.empty((9,37),dtype='int')
    ems.fill(0)
    for r in range(obs[:,0].size):
        for c in range(obs[0].size):
            if(obs[r][c]>0):
                ems[obs[r][c]-1][c] = ems[obs[r][c]-1][c] + 1

    sum_ems = np.zeros((1,37),dtype='int')
    for c in range(ems[0].size):
        for r in range(ems[:,0].size):
            sum_ems[0][c] = sum_ems[0][c] + ems[r][c]

    ems_prob = np.zeros((9,37), dtype='float')
    for r in range(ems[:,0].size):
        for c in range(ems[0].size):
            ems_prob[r][c] = ems[r][c]/sum_ems[0][c]

    return ems_prob

def emission_prob_matrix_time(obs, obs_time):
    ems = np.empty((9,37),dtype='float')
    ems.fill(0)
    for r in range(obs[:,0].size):
        for c in range(obs[0].size):
            if(obs[r][c]>0 and obs_time[r][c]>0):
                ems[obs[r][c]-1][c] = ems[obs[r][c]-1][c] + obs_time[r][c]

    #print(ems)
    sum_ems = np.zeros((1,37),dtype='float')
    for c in range(ems[0].size):
        for r in range(ems[:,0].size):
            sum_ems[0][c] = sum_ems[0][c] + ems[r][c]

    ems_prob = np.zeros((9,37), dtype='float')
    for r in range(ems[:,0].size):
        for c in range(ems[0].size):
            ems_prob[r][c] = ems[r][c]/sum_ems[0][c]

    return ems_prob
