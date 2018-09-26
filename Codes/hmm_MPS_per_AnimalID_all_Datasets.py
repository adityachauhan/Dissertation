import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk
import matplotlib.pyplot as plt
import string
import hmm_train

def main():
    r = np.genfromtxt('datasets/RawData_first.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    r2 = np.genfromtxt('datasets/RawData_second.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    r3 = np.genfromtxt('datasets/RawData_third.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    r4 = np.genfromtxt('datasets/RawData_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    t = np.genfromtxt('datasets/RawData_time_first.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    t2 = np.genfromtxt('datasets/RawData_time_second.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    t3 = np.genfromtxt('datasets/RawData_time_third.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    t4 = np.genfromtxt('datasets/RawData_time_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    obs1 = np.zeros((r['Behaviours__1'].size,37),dtype='int')
    obs1.fill(-1)
    obs2 = np.zeros((r2['Behaviours__1'].size,37),dtype='int')
    obs2.fill(-1)
    obs3 = np.zeros((r3['Behaviours__1'].size,37),dtype='int')
    obs3.fill(-1)
    obs4 = np.zeros((r4['Behaviours__1'].size,37),dtype='int')
    obs4.fill(-1)
    obs_time1 = np.zeros((t['Time__1'].size,37),dtype='float')
    obs_time2 = np.zeros((t2['Time__1'].size,37),dtype='float')
    obs_time3 = np.zeros((t3['Time__1'].size,37),dtype='float')
    obs_time4 = np.zeros((t4['Time__1'].size,37),dtype='float')
    animalID = np.zeros(obs1[:,0].size, int)
    targetID = np.zeros(obs1[:,0].size, int)
    for ro in range(obs1[:,0].size):
        for col in range(36):
            obs1[ro][col] = r[ro][col+5]

    
    for row in range(obs2[:,0].size):
        for col in range(36):
            obs2[row][col] = r2[row][col+5]

    for row in range(obs3[:,0].size):
        for col in range(36):
            obs3[row][col] = r3[row][col+5]

    for row in range(obs4[:,0].size):
        for col in range(36):
            obs4[row][col] = r4[row][col+5]
    for row in range(obs1[:,0].size):
        animalID[row] = r[row][0]
        targetID[row] = r[row][4]

    for row in range(obs1[:,0].size):
        for col in range(36):
            if(obs1[row][col]== -1):
                obs1[row][col] = 9
            if(obs2[row][col]== -1):
                obs1[row][col] = 9
            if(obs3[row][col]== -1):
                obs1[row][col] = 9
            if(obs4[row][col]== -1):
                obs1[row][col] = 9


    for row in range(obs_time1[:,0].size):
        for col in range(36):
            obs_time1[row][col] = t[row][col+5]

    
    for row in range(obs_time2[:,0].size):
        for col in range(36):
            obs_time2[row][col] = t2[row][col+5]

    for row in range(obs_time3[:,0].size):
        for col in range(36):
            obs_time3[row][col] = t3[row][col+5]

    for row in range(obs_time4[:,0].size):
        for col in range(36):
            obs_time4[row][col] = t4[row][col+5]


    pos = 0
    error_matrix = np.zeros(54, dtype = 'float')
    e=0
    group1 = np.zeros(27, float)
    group2 = np.zeros(27, float)
    g1=0
    g2=0
    plot_val = np.arange(27)
    with open('Results/MPS_per_AnimalID_all_Datasets.csv','w') as csvfile:
        fieldnames = ['AnimalID','TargetID','PATH']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while pos!= obs1[:,0].size:
            g = 0
            obs_set = np.zeros((48,37), dtype = 'int')
            obs_time_set = np.zeros((48,37), dtype='float')
            obs_set.fill(-1)
            for i in range(pos, pos+12):
                for j in range(36):
                    obs_set[i-pos][j] = obs1[i][j]
                    obs_set[(i-pos)+12][j] = obs2[i][j]
                    obs_set[(i-pos)+24][j] = obs3[i][j]
                    obs_set[(i-pos)+36][j] = obs4[i][j]
                    obs_time_set[i-pos][j] = obs_time1[i][j]
                    obs_time_set[(i-pos)+12][j] = obs_time2[i][j]
                    obs_time_set[(i-pos)+24][j] = obs_time3[i][j]
                    obs_time_set[(i-pos)+36][j] = obs_time4[i][j]
    
            T = obs_set[0].shape[0]
            num_states = 9
            trans_mat = hmm_train.trans_prob_matrix(obs_set)
            trans_mat = np.log(trans_mat)
            emi_mat_norm = hmm_train.emission_prob_matrix(obs_set)
            emi_mat_norm[:,36] = 0
            emi_mat = np.log(emi_mat_norm)
            emi_mat_time = hmm_train.emission_prob_matrix_time(obs_set, obs_time_set)
            emi_mat_time[:,36] = 0
            emi_mat_time = np.log(emi_mat_time)
            path_set = np.empty(T, dtype='int')
            path_set.fill(-1)
            for t in range(T):
                if(emi_mat_norm[8, t] > 0.8):
                    path_set[t] = -2
                elif(emi_mat_norm[8, t] > 0.7 and emi_mat_norm[8, t] < 0.8):
                    for s in range(num_states-1):
                        path_set[t] = np.argmax(emi_mat[:,t] + trans_mat[:, s])
                elif(emi_mat_norm[8, t] > 0.5 and emi_mat_norm[8, t] < 0.7):
                    for s in range(num_states-1):
                        path_set[t] = np.argmin(emi_mat[:,t-1] + trans_mat[s, s])
                else:
                    for s in range(num_states-1):
                        path_set[t] = np.argmax(emi_mat[:,t-1] + trans_mat[s,s] + emi_mat_time[:,t])

            path_set[36] = -2
            writer.writerow({'AnimalID' : str(animalID[pos]), 'TargetID' : str(targetID[pos]), 'PATH' : str(path_set+1)})
            for r in range(37):
                if(path_set[r]+1==-1):
                    g = g+1
            if(targetID[pos]==1):
                val = (36-g)/36
                group1[g1] = val
                g1 = g1 + 1
            if(targetID[pos]==2):
                val = (36-g)/36
                group2[g2] = val
                g2 = g2 + 1

            pos = pos+12

    plt.plot(plot_val, group1,'r',label="group1")
    plt.plot(plot_val, group2, 'b', label="group2")
    plt.legend(('group1', 'group2'))
    plt.show()


if __name__ == '__main__':
    main()
    
