import numpy as np
import csv as csv
from sklearn.mixture import GaussianMixture
import tkinter as tk

import matplotlib.pyplot as plt
import hmm_MPS_each_trial_all_Datasets


def main():
    group1, group2 = hmm_MPS_each_trial_all_Datasets.main()
    normalg1 = group1/np.sqrt((np.sum(group1**2)))
    normalg2 = group2/np.sqrt((np.sum(group2**2)))
    val = np.arange(12)
    for i in range(8):
        plt.bar((val+1)-0.1, normalg1[i], width=0.2, color='b', label='Group1')
        plt.bar((val+1)+0.1, normalg2[i], width=0.2, color='g', label='Group2')
        plt.legend(('Group1', 'Group2'))
        plt.show()
        






if __name__ == '__main__':
    main()
