'''
Code to estimate the efficiency of Model using K-Fold Cross Validatoion

'''

import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk

import matplotlib.pyplot as plt
import hmm_MPS_FADatasets
import hmm_MPS_200_07
import hmm_MPS_250_07
import hmm_MPS_250_90
import hmm_MPS_300_07

# function to compare the most probable state sequeces with the remaining datasets
# most probable state sequence

def cross_validation(obs_row, best_path):
    count = 0;
    for pos in range(37):
        if(obs_row[pos] != best_path[pos]):
            count = count+1

    error = count/36
    return error

def main():
    path = hmm_MPS_FADatasets.main()
    path1 = hmm_MPS_200_07.main()
    path2 = hmm_MPS_250_07.main()
    path3 = hmm_MPS_250_90.main()
    path4 = hmm_MPS_300_07.main()
    error_matrix = np.zeros(4, dtype='float')
    error_matrix[0] = cross_validation(path, path1)
    error_matrix[1] = cross_validation(path, path2)
    error_matrix[2] = cross_validation(path, path3)
    error_matrix[3] = cross_validation(path, path4)
    val = np.arange(4)
    plt.plot(val+1, error_matrix, 'r')
    plt.show()
    

if __name__ == '__main__':
    main()
