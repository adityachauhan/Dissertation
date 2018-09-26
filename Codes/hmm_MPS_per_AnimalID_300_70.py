import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk
import hmm_train
import matplotlib.pyplot as plt


def main():
 
    r = np.genfromtxt('datasets/RawData_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    t = np.genfromtxt('datasets/RawData_time_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    obs1 = np.zeros((r['Behaviours__1'].size,37),dtype='int')
    obs_time1 = np.zeros((t['Time__1'].size,37),dtype='float')
    n = obs1[:,0].size
    obs = np.zeros((n,37),dtype='int')
    obs.fill(-1)
    obs_time = np.zeros((n,37),dtype='float')
    animalID = np.zeros(n,int)
    targetID = np.zeros(n,int)
    for ro in range(obs1[:,0].size):
        for col in range(36):
            obs[ro][col] = r[ro][col+5]

    for row in range(obs[:,0].size):
        for col in range(36):
            if(obs[row][col]== -1):
                obs[row][col] = 9

    for row in range(obs_time1[:,0].size):
        for col in range(36):
            obs_time[row][col] = t[row][col+5]

    for row in range(obs[:,0].size):
        animalID[row] = r[row][0]
        targetID[row] = r[row][4]

    pos = 0
    count = 0
    e=0
    group1 = np.zeros(27,float)
    group2 = np.zeros(27, float)
    g1=0
    g2=0
    plot_val = np.arange(27)
    error_matrix = np.zeros(54, dtype='float')
    with open('Results/MPS_per_AnimalID_300_70.csv','w') as csvfile:
        fieldnames = ['AnimalID','TargetID', 'PATH']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        while pos != obs[:,0].size:
            g=0
            obs_set = np.zeros((12,37), dtype='int')
            obs_time_set = np.zeros((12,37), dtype='float')
            obs_set.fill(-1)
            for i in range(pos, pos+12):
                for j in range(36):
                    obs_set[i-pos][j] = obs[i][j]
                    obs_time_set[i-pos][j] = obs_time[i][j]
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
            '''for r in range(37):
                if(path_set[r]+1==-1):
                    g = g+1
            if(targetID[pos]==1):
                val = (36-g)/36
                group1[g1] = val
                g1 = g1 + 1
            if(targetID[pos]==2):
                val = (36-g)/36
                group2[g2] = val
                g2 = g2 + 1'''
            pos = pos+12
    #plt.plot(plot_val, group1,'r',label="group1")
    #plt.plot(plot_val, group2, 'b', label="group2")
    #plt.legend(('group1', 'group2'))
    #plt.show()


if __name__ == '__main__':
    main()
            
                    









    
