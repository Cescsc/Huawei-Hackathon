import os
import time
import numpy as np
import itertools


# This funcation calculates the positions of all channels, should be implemented by the participants
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

# Read in the configuration file
def read_cfg_file(file_path):
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

# Read in the info related to the anchor points
def read_anch_file(file_path, anch_samp_num):
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

# The channel file is large, read in channels in smaller slices
def read_slice_of_file(file_path, start, end):
    with open(file_path, 'r') as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines

if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")
    ## For ease of data managenment, input data for different rounds are stored in different folders. Feel free to define your own
    PathSet = {0: "./Test", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}

    Ridx = 1  # Flag defining the round of the competition, used for define where to read data。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    
    # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中
    ### Get all files in the folder related to the competition. Data for other rounds should be kept in a different folder  
    files = os.listdir(PathRaw)
    names = []
    for f in sorted(files):
        if f.find('CfgData') != -1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])
    
    
    for na in names:
        FileIdx = int(na)
        print('Processing Round ' + str(Ridx) + ' Case ' + str(na))
        
        # Read in the configureation file: RoundYCfgDataX.txt
        print('Loading configuration data file')
        cfg_path = PathRaw + '/' + Prefix + 'CfgData' + na + '.txt'
        bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)
                
        # Read in info related to the anchor points: RoundYInputPosX.txt
        print('Loading input position file')
        anch_pos_path = PathRaw + '/' + Prefix + 'InputPos' + na + '.txt'
        anch_pos = read_anch_file(anch_pos_path, anch_samp_num)

        # Read in channel data:  RoundYInputDataX.txt
        slice_samp_num = 1000  # number of samples in each slice
        slice_num = int(tol_samp_num / slice_samp_num)  # total number of slices
        csi_path = PathRaw + '/' + Prefix + 'InputData' + na + '.txt'
        H = []
        for slice_idx in range(2): # range(slice_num): # Read in channel data in a loop. In each loop, only one slice of channel is read in
            print('Loading input CSI data of slice ' + str(slice_idx))
            slice_lines = read_slice_of_file(csi_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
            Htmp = np.loadtxt(slice_lines)
            Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
            Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
            Htmp = np.transpose(Htmp, (0, 3, 2, 1))
            if np.size(H) == 0:
                H = Htmp
            else:
                H = np.concatenate((H, Htmp), axis=0)
        H = H.astype(np.complex64) # trunc to complex64 to reduce storage
        
        csi_file = PathRaw + '/' + Prefix + 'InputData' + na + '.npy'
        np.save(csi_file, H) # After reading the file, you may save txt file into npy, which is faster for python to read 
        # H = np.load(csi_file) # if saved in npy, you can load npy file instead of txt
        
        tStart = time.perf_counter()
        
        
        print('Calculating localization results')
        result = calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num) # This function should be implemented by yourself
        
        # Replace the position information for anchor points with ground true coordinates
        for idx in range(anch_samp_num):
            rowIdx = int(anch_pos[idx][0] - 1)
            result[rowIdx] = np.array([anch_pos[idx][1], anch_pos[idx][2]])

        # Output, be careful with the precision
        print('Writing output position file')
        with open(PathRaw + '/' + Prefix + 'OutputPos' + na + '.txt', 'w') as f:
            np.savetxt(f, result, fmt='%.4f %.4f')

        # This help to evaluate the running time, can be removed!
        tEnd = time.perf_counter()
        print("Total time consuming = {}s\n\n".format(round(tEnd - tStart, 3)))

