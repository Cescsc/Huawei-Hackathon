# Huawei Hackaton 2024

import os
import time
import numpy as np
import itertools


# This function calculates the positions of all channels, should be implemented by the participants
def calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num):
    '''
    H: channel data
    anch_pos: anchor ID and its coordinates
    bs_pos: coordinate of the base station
    tol_samp_num: total number of channel data points
    anch_samp_num: total number of anchor points
    port_num: number of SRS Ports (number of antennas for the UE)
    ant_num: number of antennas for the base station
    sc_num: number of subcarriers
    '''
    #########The following should be implemented by the participants################
    #########In this example, we use zeros for the predictions##########
    loc_result = np.zeros([tol_samp_num, 2], 'float')
    return loc_result


def read_cfg_file(file_path):
    """Reads the configuration file"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    info = line_fmt
    bs_pos = list([float(info[0][0]), float(info[0][1]), float(info[0][2])])
    tol_samp_num = int(info[1][0])
    anch_samp_num = int(info[2][0])
    port_num = int(info[3][0])
    ant_num = int(info[4][0])
    sc_num = int(info[5][0])
    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num


def read_anch_file(file_path, anch_samp_num):
    """Reads the info related to the anchor points"""
    anch_pos = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    for line in line_fmt:
        tmp = np.array([int(line[0]), float(line[1]), float(line[2])])
        if np.size(anch_pos) == 0:
            anch_pos = tmp
        else:
            anch_pos = np.vstack((anch_pos, tmp))
    return anch_pos


def _read_slice_of_file(file_path, start, end):
    """The channel file is large, reads channels in smaller slices"""
    with open(file_path, 'r') as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines


def full_read(directory, scene):
    """
    Reads and processes full dataset 0/1/2

    >> directory: path to directory with data of form DatasetX
    >> scene = 1, 2 or 3, specifies a scene from dataset X
    """

    files = os.listdir(directory)
    files.sort()

    for f in files:
        if f[-5] != str(scene):
            files.remove(f)

    if "DataSet0" in directory:
        demo = True
    else:
        demo = False

    cfg_files = []
    anchor_files = []
    channel_files = []
    ground_files = []
    for f in files:
        if "Cfg" in f:
            cfg_files.append(f)
        if "Pos" in f:
            anchor_files.append(f)
        if "InputData" in f:
            channel_files.append(f)
        if demo:
            if "Ground" in f:
                ground_files.append(f)

    print('Loading cfg files')
    for f in cfg_files:
        cfg_file_path = directory + '/' + f
        bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_file_path)

    print('Loading anchor files')
    for f in anchor_files:
        anch_file_path = directory + '/' + f
        anch_pos = read_anch_file(anch_file_path, anch_samp_num)

    print('Loading channel data files')
    for f in channel_files:
        slice_samp_num = 1000  # number of samples in each slice
        slice_num = int(tol_samp_num / slice_samp_num)  # total number of slices
        csi_file_path = directory + '/' + f
        H = []

        for slice_idx in range(slice_num):  # range(slice_num): # Reads a channel per loop

            print('Loading input CSI data of slice ' + str(slice_idx) + "of " + str(slice_num))
            slice_lines = _read_slice_of_file(csi_file_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
            Htmp = np.loadtxt(slice_lines)
            Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
            Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
            Htmp = np.transpose(Htmp, (0, 3, 2, 1))
            if np.size(H) == 0:
                H = Htmp
            else:
                H = np.concatenate((H, Htmp), axis=0)

        H = H.astype(np.complex64)  # trunc to complex64 to reduce storage

        csi_file = directory + '/' + f[:-4] + '.npy'
        np.save(csi_file, H)  # save file as binary npy
        # H = np.load(csi_file) # if saved in npy, you can load npy file instead of txt


if __name__ == "__main__":
    pass

