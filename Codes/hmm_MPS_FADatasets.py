'''
HMM Model to calcuate the most probable path sequence for all
the data sets.

RawData_first -> segments length 200cm with 70% overlap
RawData_second -> segment length 250cm with 70% overlap
RawData_third -> segment length 250cm with 90% overlap
RawData_fourth ->segment length 300cm with 70% overlap(this file introduce lot of error in model)

'''

import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk
import hmm_train
def main():
    # Reading csv data file and retreiving content in variables.

    r = np.genfromtxt('datasets/RawData_first.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    r2 = np.genfromtxt('datasets/RawData_second.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    r3 = np.genfromtxt('datasets/RawData_third.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    r4 = np.genfromtxt('datasets/RawData_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'int')
    t = np.genfromtxt('datasets/RawData_time_first.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    t2 = np.genfromtxt('datasets/RawData_time_second.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    t3 = np.genfromtxt('datasets/RawData_time_third.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')
    t4 = np.genfromtxt('datasets/RawData_time_fourth.csv',delimiter=',', names=True,case_sensitive=True,dtype = 'float')

    #Declaring variables for size and storing particular data needed in analysis
    
    obs1 = np.zeros((r['Behaviours__1'].size,37),dtype='int')
    obs2 = np.zeros((r2['Behaviours__1'].size,37),dtype='int')
    obs3 = np.zeros((r3['Behaviours__1'].size,37),dtype='int')
    obs4 = np.zeros((r4['Behaviours__1'].size,37),dtype='int')
    obs_time1 = np.zeros((t['Time__1'].size,37),dtype='float')
    obs_time2 = np.zeros((t2['Time__1'].size,37),dtype='float')
    obs_time3 = np.zeros((t3['Time__1'].size,37),dtype='float')
    obs_time4 = np.zeros((t4['Time__1'].size,37),dtype='float')

    # Setting total size of observation sequences.
    # Declaring observation sequence variables.
    
    n = obs1[:,0].size + obs2[:,0].size + obs3[:,0].size + obs4[:,0].size
    obs = np.zeros((n,37),dtype='int')
    obs.fill(-1)
    obs_time = np.zeros((n,37),dtype='float')
    T = obs[0].shape[0]
    num_states = 9

    #Populating observation sequence variables usinf data from four datasets.
    
    for ro in range(obs1[:,0].size):
        for col in range(36):
            obs[ro][col] = r[ro][col+5]

    
    for row in range(obs2[:,0].size):
        for col in range(36):
            obs[row+obs1[:,0].size][col] = r2[row][col+5]

    for row in range(obs3[:,0].size):
        for col in range(36):
            obs[row+obs1[:,0].size+obs2[:,0].size][col] = r3[row][col+5]

    for row in range(obs4[:,0].size):
        for col in range(36):
            obs[row+obs1[:,0].size+obs2[:,0].size + obs3[:,0].size][col] = r4[row][col+5]

    for row in range(obs[:,0].size):
        for col in range(36):
            if(obs[row][col]== -1):
                obs[row][col] = 9

                
    for row in range(obs_time1[:,0].size):
        for col in range(36):
            obs_time[row][col] = t[row][col+5]

    
    for row in range(obs_time2[:,0].size):
        for col in range(36):
            obs_time[row+obs_time1[:,0].size][col] = t2[row][col+5]

    for row in range(obs_time3[:,0].size):
        for col in range(36):
            obs_time[row+obs_time1[:,0].size+obs_time2[:,0].size][col] = t3[row][col+5]

    for row in range(obs_time4[:,0].size):
        for col in range(36):
            obs_time[row+obs_time1[:,0].size+obs_time2[:,0].size + obs_time3[:,0].size][col] = t4[row][col+5]

    # Training model based on observation sequence

    #Calculating transition probability matrix based on observation
    
    trans_mat_norm = hmm_train.trans_prob_matrix(obs)
    trans_mat = np.log(trans_mat_norm)

    #Calculating Emission probability matrix for model 
    emi_mat_norm = hmm_train.emission_prob_matrix(obs)
    emi_mat_norm[:,36] = 0
    emi_mat = np.log(emi_mat_norm)

    # Calculating Average time for model
    emi_mat_time_norm = hmm_train.emission_prob_matrix_time(obs, obs_time)
    emi_mat_time = np.log(emi_mat_time_norm)
    emi_mat_time[:, 36] = 0

    #All the values for transition, emission and average have log applied
    #to them to avoid float underflow as probability values sometimes go very low

    # Declaring variable to store most probable state sequence
    
    path = np.empty(T,dtype='int')
    path.fill(-1)


    # Looping over complete observation sequence for all the states to get most
    #probable state sequence
    
    for t in range(T):
        if(emi_mat_norm[8, t] > 0.8):
            path[t] = -2
        elif(emi_mat_norm[8, t] > 0.7 and emi_mat_norm[8, t] < 0.8):
            for s in range(num_states-1):
                path[t] = np.argmax(emi_mat[:,t] + trans_mat[:, s])
        elif(emi_mat_norm[8, t] > 0.5 and emi_mat_norm[8, t] < 0.7):
            for s in range(num_states-1):
                path[t] = np.argmin(emi_mat[:,t-1] + trans_mat[s, s])
        else:
            for s in range(num_states-1):
                path[t] = np.argmax(emi_mat[:,t-1] + trans_mat[s,s] + emi_mat_time[:,t])

    path[36] = -2


    print('==================== MOST PROBABLE STATE SEQUENCE ================')
    print(path+1)
    print('==================================================================')
    return(path+1)
    


if __name__ == '__main__':
    main()





    
