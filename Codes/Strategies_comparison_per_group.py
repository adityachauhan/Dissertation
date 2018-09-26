import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import hmm_MPS_per_Group_200_70
import hmm_MPS_per_Group_250_70
import hmm_MPS_per_Group_250_90
import hmm_MPS_per_Group_300_70

def main():
    err_g1 = np.empty(4, float)
    err_g2 = np.empty(4, float)
    group1 = np.zeros(8, float)
    group2 = np.zeros(8, float)
    err_g1[0], err_g2[0], path1_g1, path1_g2 = hmm_MPS_per_Group_200_70.main()
    err_g1[1], err_g2[1], path2_g1, path2_g2 = hmm_MPS_per_Group_250_70.main()
    err_g1[2], err_g2[2], path3_g1, path3_g2 = hmm_MPS_per_Group_250_90.main()
    err_g1[3], err_g2[3], path4_g1, path4_g2 = hmm_MPS_per_Group_300_70.main()
    val = np.arange(8)
    for i in range(37):
        if(path1_g1[i] > 0):
            group1[path1_g1[i]-1] = group1[path1_g1[i]-1] + 1
        if(path1_g2[i] > 0):
            group2[path1_g2[i]-1] = group2[path1_g2[i]-1] + 1
        if(path2_g1[i] > 0):
            group1[path2_g1[i]-1] = group1[path2_g1[i]-1] + 1
        if(path2_g2[i] > 0):
            group2[path2_g2[i]-1] = group2[path2_g2[i]-1] + 1
        if(path3_g1[i] > 0):
            group1[path3_g1[i]-1] = group1[path3_g1[i]-1] + 1
        if(path3_g2[i] > 0):
            group2[path3_g2[i]-1] = group2[path3_g2[i]-1] + 1
        if(path4_g1[i] > 0):
            group1[path4_g1[i]-1] = group1[path4_g1[i]-1] + 1
        if(path4_g2[i] > 0):
            group2[path4_g2[i]-1] = group2[path4_g2[i]-1] + 1

    normalg1 = group1/np.sqrt((np.sum(group1**2)))
    normalg2 = group2/np.sqrt((np.sum(group2**2)))
    
    plt.bar((val+1)-0.1, normalg1, width=0.2, color = 'b', label='Group1')
    plt.bar((val+1)+0.1, normalg2, width=0.2, color = 'g', label='Group2')
    
    plt.legend(('Group1','Group2'))
    plt.show()
    

if __name__ == '__main__':
    main()
