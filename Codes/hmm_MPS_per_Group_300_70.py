import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk
import hmm_train


def main():
 
    r = np.genfromtxt('datasets/RawData_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    t = np.genfromtxt('datasets/RawData_time_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    obs1 = np.zeros((r['Behaviours__1'].size,37),dtype='int')
    obs_time1 = np.zeros((t['Time__1'].size,37),dtype='float')
    
    n = obs1[:,0].size 
    m = int(n/2)
    obs = np.zeros((m,37),dtype='int')
    obsg2 = np.zeros((m, 37), int)
    obs.fill(-1)
    obsg2.fill(-1)
    obs_time = np.zeros((m,37),dtype='float')
    obsg2_time = np.zeros((m,37), float)
    T = obs[0].shape[0]
    num_states = 9

    
    for ro in range(m):
        for col in range(36):
            obs[ro][col] = r[ro][col+5]

    for ro in range(m, n):
        for col in range(36):
            obsg2[ro-m][col] = r[ro][col+5] 

    


    for row in range(obs[:,0].size):
        for col in range(36):
            if(obs[row][col]== -1):
                obs[row][col] = 9
            if(obsg2[row][col]==-1):
                obsg2[row][col] = 9

                
    for row in range(m):
        for col in range(36):
            obs_time[row][col] = t[row][col+5]

    for row in range(m, n):
        for col in range(36):
            obsg2_time[row-m][col] = t[row][col+5]

    



    
    trans_mat_norm1 = hmm_train.trans_prob_matrix(obs)
    trans_mat1 = np.log(trans_mat_norm1)
    emi_mat_norm1 = hmm_train.emission_prob_matrix(obs)
    emi_mat_norm1[:,36] = 0
    emi_mat1 = np.log(emi_mat_norm1)
    emi_mat_time_norm1 = hmm_train.emission_prob_matrix_time(obs, obs_time)
    emi_mat_time1 = np.log(emi_mat_time_norm1)
    emi_mat_time1[:, 36] = 0

    
    trans_mat_norm2 = hmm_train.trans_prob_matrix(obsg2)
    trans_mat2 = np.log(trans_mat_norm2)
    emi_mat_norm2 = hmm_train.emission_prob_matrix(obsg2)
    emi_mat_norm2[:,36] = 0
    emi_mat2 = np.log(emi_mat_norm2)
    emi_mat_time_norm2 = hmm_train.emission_prob_matrix_time(obsg2, obsg2_time)
    emi_mat_time2 = np.log(emi_mat_time_norm2)
    emi_mat_time2[:, 36] = 0

    
    path1 = np.empty(T,dtype='int')
    path1.fill(-1)
    path2 = np.empty(T,dtype='int')
    path2.fill(-1)
    
    

    for t in range(T):
        if(emi_mat_norm1[8, t] > 0.8):
            path1[t] = -2
        elif(emi_mat_norm1[8, t] > 0.7 and emi_mat_norm1[8, t] < 0.8):
            for s in range(num_states-1):
                path1[t] = np.argmax(emi_mat1[:,t] + trans_mat1[:, s])
        elif(emi_mat_norm1[8, t] > 0.5 and emi_mat_norm1[8, t] < 0.7):
            for s in range(num_states-1):
                path1[t] = np.argmin(emi_mat1[:,t-1] + trans_mat1[s, s])
        else:
            for s in range(num_states-1):
                path1[t] = np.argmax(emi_mat1[:,t-1] + trans_mat1[s,s] + emi_mat_time1[:,t])


    for t in range(T):
        if(emi_mat_norm2[8, t] > 0.8):
            path2[t] = -2
        elif(emi_mat_norm2[8, t] > 0.7 and emi_mat_norm2[8, t] < 0.8):
            for s in range(num_states-1):
                path2[t] = np.argmax(emi_mat2[:,t] + trans_mat2[:, s])
        elif(emi_mat_norm2[8, t] > 0.5 and emi_mat_norm2[8, t] < 0.7):
            for s in range(num_states-1):
                path2[t] = np.argmin(emi_mat2[:,t-1] + trans_mat2[s, s])
        else:
            for s in range(num_states-1):
                path2[t] = np.argmax(emi_mat2[:,t-1] + trans_mat2[s,s] + emi_mat_time2[:,t])
        
    for i in range(37):
        if(path1[i]==8):
            path1[i] = -2
        if(path2[i]==8):
            path2[i] = -2
    
    path1[36] = -2
    path2[36] = -2
    count1 = 0
    count2 = 0
    for i in range(37):
        if(path1[i]+1==-1):
            count1=count1+1;

    for i in range(37):
        if(path2[i]+1==-1):
            count2=count2+1;

    count1 = float((36-count1)/36)
    count2 = float((36-count2)/36)
    
    print('=============== MOST PROBABLE STATE SEQUENCE =================')
    print('MPS Group1 : ' , path1+1)
    print('==============================================================')
    print('MPS Group2 : ' , path2+1)
    print('==============================================================')
    return count1, count2, path1+1, path2+1

if __name__ == '__main__':
    main()





    
