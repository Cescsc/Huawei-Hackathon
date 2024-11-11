# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:29:14 2024

@author: l00416959
"""

import numpy as np
import itertools

cfg_path = 'D:\share\channel\AlgoDataSet\Round0CfgData1.txt'
inputdata_path = 'D:\share\channel\AlgoDataSet\Round0InputData1.txt'

# func to read in slices
def read_slice_of_file(file_path, start, end):
    with open(file_path, 'r') as file:
        # use itertools.islice to get slices
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines

# read RoundYCfgDataX.txt 
slice_lines = read_slice_of_file(cfg_path, 1, 6)
info = np.loadtxt(slice_lines)
tol_samp_num = int(info[0])
port_num = int(info[2])
ant_num = int(info[3])
sc_num = int(info[4])

# read RoundYInputDataX. in slices 
H = []
slice_samp_num = 1000   #number of samples
slice_num = int(tol_samp_num / slice_samp_num) #number of slices
for slice_idx in range(slice_num):
    print(slice_idx)
    slice_lines = read_slice_of_file(inputdata_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
    Htmp = np.loadtxt(slice_lines)
    Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
    Htmp = Htmp[:, 0, :, :, :] + 1j*Htmp[:, 1, :, :, :]
    Htmp = np.transpose(Htmp, (0,3,2,1))
    
    if np.size(H) == 0:
        H = Htmp
    else:
        H = np.concatenate((H, Htmp), axis=0)
