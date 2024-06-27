# ----------------------------------------------------------------------------------------------------------------------
# RandomSamples_Generation.py
# -to generate random field of soil properties with specified parameters and transform data to image data (*.imd)
# -to automatically pad free area with additional data information in a rectangle that covers the landslide model
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch
from RadomField_Gene_Fuctions import solve_wii
from RadomField_Gene_Fuctions import solve_ramdai
from RadomField_Gene_Fuctions import solve_h_2d
from RadomField_Gene_Fuctions import solve_h_3d


# ----------------------------------------------------------------------------------------------------------------------
def random_samples_generate_kl(proj_path):  # generating random field of landslides using the K-L expansion method
    # setting the problem parameters and random field parameters
    if os.sep == "/":
        input_path = proj_path + r'/input.dat'  # linux platform
    else:
        input_path = proj_path + r'\input.dat'  # windows platform
    if os.path.exists(input_path):
        input_file = open(input_path, 'r')  # open the input file: input.dat
        # ------------------------------------------------------------------------------------------------------------------
        # read data from input.dat
        input_data_rl = input_file.readlines()
        flline_0 = input_data_rl[14].split()
        flline_1 = input_data_rl[15].split()
        flline_2 = input_data_rl[16].split()
        ndim = int(flline_0[0])  # setting the number of dimensions
        ntotal = int(flline_1[3])  # setting the total number of particles
        dr = float(flline_2[1])  # setting the particle spacing
        parti_type = np.zeros(ntotal, 'i4').reshape(ntotal, 1)
        n_nonb = 0
        for k in range(253, ntotal + 253):
            flline = input_data_rl[k].split()
            if ndim == 2:
                parti_type[k - 253] = int(flline[4])
            else:
                parti_type[k - 253] = int(flline[6])
            if parti_type[k - 253] > 0:
                n_nonb = n_nonb + 1
        parti_x = np.zeros(n_nonb * 3, 'f4').reshape(n_nonb, 3)
        parti_para = np.zeros(n_nonb * 4, 'f4').reshape(n_nonb, 4)
        parti_nonb_type = np.zeros(n_nonb, 'i4').reshape(n_nonb, 1)
        k_1 = 0
        for k in range(253, ntotal + 253):
            if parti_type[k - 253] > 0:
                flline = input_data_rl[k].split()
                parti_x[k_1][0] = float(flline[0])
                parti_x[k_1][1] = float(flline[1])
                parti_x[k_1][2] = float(flline[2])
                parti_nonb_type[k_1] = parti_type[k - 253]
                k_1 = k_1 + 1
        # ------------------------------------------------------------------------------------------------------------------
        # read random field parameters from input.dat
        randf_flag = np.zeros(6, 'i4').reshape(6, 1)
        randf_para = np.zeros(5, 'f4').reshape(5, 1)
        randf_svf = np.zeros(2 * 6, 'i4').reshape(6, 2)
        randf_svp = np.zeros(6 * 6, 'f4').reshape(6, 6)
        flline_3 = input_data_rl[87].split()
        flline_4 = input_data_rl[88].split()
        flline_5 = input_data_rl[89].split()
        randf_flag[0] = int(flline_3[0])  # flag_fai of soil
        randf_flag[1] = int(flline_3[1])  # flag_c of soil
        randf_flag[2] = int(flline_3[2])  # flag_cop of soil
        randf_flag[3] = int(flline_3[3])  # flag_ds of soil
        randf_flag[4] = int(flline_4[0])  # flag_fai of fluid
        randf_flag[5] = int(flline_4[1])  # flag_c of fluid
        randf_para[0] = float(flline_5[0])  # random field parameter: coefficient of cell size, *dr
        randf_para[1] = float(flline_5[1])  # random field parameter: correlation coefficient of fai and c
        randf_para[2] = float(flline_5[2])  # random field parameter: type of auto correlation function
        randf_para[3] = float(flline_5[3])  # random field parameter: beta, inclination angle of x direction
        randf_para[4] = float(flline_5[4])  # random field parameter: beta1, inclination angle of y direction
        for k in range(95, 100):
            flline_6 = input_data_rl[k].split()
            randf_svf[k - 95][0] = int(flline_6[0])  # Line no.
            randf_svf[k - 95][1] = int(flline_6[1])  # type distribution: type
            randf_svp[k - 95][0] = float(flline_6[2])  # mean
            randf_svp[k - 95][1] = float(flline_6[3])  # stand deviation:d original x
            randf_svp[k - 95][2] = float(flline_6[4])  # correlation length: x
            randf_svp[k - 95][3] = float(flline_6[5])  # correlation length: y
            randf_svp[k - 95][4] = float(flline_6[6])  # correlation length: z
            randf_svp[k - 95][5] = float(flline_6[7])  # coe_depth
        # ------------------------------------------------------------------------------------------------------------------
        # building the auto correlation function matrix for the random field
        # determine the range of non-boundary particles
        range_min = (np.min(parti_x, axis=0)).T  # the left and lower corner of rectangle or box
        range_max = (np.max(parti_x, axis=0)).T  # the right and upper corner of rectangle or box
        range_min = range_min - 0.2 * (range_max - range_min)
        range_max = range_max + 0.2 * (range_max - range_min)
        if ndim == 2:
            range_min[2] = 0.0
            range_max[2] = 0.0
        x_trans = (range_min + range_max) * 0.5
        a_x = (range_max - range_min) * 0.5
        samples_num = int(input("Input the directory of input.dat: \n"))  # number of random field samples
        covar_mat = np.zeros(2 * 2, 'f4').reshape(2, 2)
        klterm = 200
        if ndim == 2:
            if randf_svf[0] == 1 and randf_svf[1] == 1:  # correlated fai and c for soil
                # setting parameters
                type_dist = randf_svf[0][1]
                mean_0 = randf_svp[0][0]
                std_0 = randf_svp[0][1]
                mean_1 = randf_svp[1][0]
                std_1 = randf_svp[1][1]
                # calculate the updated correlation coefficient
                if type_dist == 2:
                    temp1 = randf_para[1]
                    coe_corr = np.log(1.0 + temp1 * (std_0 / mean_0) * (std_1 / mean_1))
                    coe_corr = coe_corr / np.sqrt(
                        np.log(1.0 + std_0 * std_0 / mean_0 / mean_0) * np.log(1.0 + std_1 * std_1
                                                                               / mean_1 / mean_1))
                else:
                    coe_corr = randf_para[1]
                # correlation matrix C
                covar_mat[0][0] = 1.0
                covar_mat[0][1] = coe_corr
                covar_mat[1][0] = 0.0
                covar_mat[1][1] = np.sqrt(1.0 - coe_corr * coe_corr)

            elif randf_svf[4] == 1 and randf_svf[5] == 1:   # correlated fai and c for fluid

            else:   # generating random field for each parameter one by one

        else:

    else:
        return 2


# ----------------------------------------------------------------------------------------------------------------------
def random_samples_generate_midp(proj_path):  # generating random field of landslides using the Mid-Point method
    # setting the problem parameters and random field parameters
    if os.sep == "/":
        input_path = proj_path + r'/input.dat'  # linux platform
    else:
        input_path = proj_path + r'\input.dat'  # windows platform
    if os.path.exists(input_path):
        input_file = open(input_path, 'r')  # open the input file: input.dat
        # ------------------------------------------------------------------------------------------------------------------
        # read data from input.dat
        input_data_rl = input_file.readlines()
        flline_0 = input_data_rl[14].split()
        flline_1 = input_data_rl[15].split()
        flline_2 = input_data_rl[16].split()
        ndim = int(flline_0[0])  # setting the number of dimensions
        ntotal = int(flline_1[3])  # setting the total number of particles
        dr = float(flline_2[1])  # setting the particle spacing
        parti_type = np.zeros(ntotal, 'i4').reshape(ntotal, 1)
        n_nonb = 0
        for k in range(253, ntotal + 253):
            flline = input_data_rl[k].split()
            if ndim == 2:
                parti_type[k - 253] = int(flline[4])
            else:
                parti_type[k - 253] = int(flline[6])
            if parti_type[k - 253] > 0:
                n_nonb = n_nonb + 1
        parti_x = np.zeros(n_nonb * 3, 'f4').reshape(n_nonb, 3)
        parti_para = np.zeros(n_nonb * 4, 'f4').reshape(n_nonb, 4)
        parti_nonb_type = np.zeros(n_nonb, 'i4').reshape(n_nonb, 1)
        k_1 = 0
        for k in range(253, ntotal + 253):
            if parti_type[k - 253] > 0:
                flline = input_data_rl[k].split()
                parti_x[k_1][0] = float(flline[0])
                parti_x[k_1][1] = float(flline[1])
                parti_x[k_1][2] = float(flline[2])
                parti_nonb_type[k_1] = parti_type[k - 253]
                k_1 = k_1 + 1
        # ------------------------------------------------------------------------------------------------------------------
        # read random field parameters from input.dat
        randf_flag = np.zeros(6, 'i4').reshape(6, 1)
        randf_para = np.zeros(5, 'f4').reshape(5, 1)
        randf_svf = np.zeros(2 * 6, 'i4').reshape(6, 2)
        randf_svp = np.zeros(6 * 6, 'f4').reshape(6, 6)
        flline_3 = input_data_rl[87].split()
        flline_4 = input_data_rl[88].split()
        flline_5 = input_data_rl[89].split()
        randf_flag[0] = int(flline_3[0])  # flag_fai of soil
        randf_flag[1] = int(flline_3[1])  # flag_c of soil
        randf_flag[2] = int(flline_3[2])  # flag_cop of soil
        randf_flag[3] = int(flline_3[3])  # flag_ds of soil
        randf_flag[4] = int(flline_4[0])  # flag_fai of fluid
        randf_flag[5] = int(flline_4[1])  # flag_c of fluid
        randf_para[0] = float(flline_5[0])  # random field parameter: coefficient of cell size, *dr
        randf_para[1] = float(flline_5[1])  # random field parameter: correlation coefficient of fai and c
        randf_para[2] = float(flline_5[2])  # random field parameter: type of auto correlation function
        randf_para[3] = float(flline_5[3])  # random field parameter: beta, inclination angle of x direction
        randf_para[4] = float(flline_5[4])  # random field parameter: beta1, inclination angle of y direction
        for k in range(95, 100):
            flline_6 = input_data_rl[k].split()
            randf_svf[k - 95][0] = int(flline_6[0])  # Line no.
            randf_svf[k - 95][1] = int(flline_6[1])  # type distribution: type
            randf_svp[k - 95][0] = float(flline_6[2])  # mean
            randf_svp[k - 95][1] = float(flline_6[3])  # stand deviation:d original x
            randf_svp[k - 95][2] = float(flline_6[4])  # correlation length: x
            randf_svp[k - 95][3] = float(flline_6[5])  # correlation length: y
            randf_svp[k - 95][4] = float(flline_6[6])  # correlation length: z
            randf_svp[k - 95][5] = float(flline_6[7])  # coe_depth
        # ------------------------------------------------------------------------------------------------------------------
        # building the auto correlation function matrix for the random field
        # determine the range of non-boundary particles
        range_min = (np.min(parti_x, axis=0)).T  # the left and lower corner of rectangle or box
        range_max = (np.max(parti_x, axis=0)).T  # the right and upper corner of rectangle or box
        range_min = range_min - 0.1 * (range_max - range_min)
        range_max = range_max + 0.1 * (range_max - range_min)
        cell_dr = randf_para[0] * dr
        grid_dim = np.zeros(3, 'i4')
        grid_dim[0] = int((range_max[0] - range_min[0]) / cell_dr) + 1
        grid_dim[1] = int((range_max[1] - range_min[1]) / cell_dr) + 1
        grid_dim[2] = int((range_max[2] - range_min[2]) / cell_dr) + 1
        x_base = np.zeros(3, 'i4')
        x_base[0] = -(grid_dim[0] * cell_dr - (range_max[0] - range_min[0])) / 2.0 + 0.01 * dr + range_min[0]
        x_base[1] = -(grid_dim[1] * cell_dr - (range_max[1] - range_min[1])) / 2.0 + 0.01 * dr + range_min[1]
        x_base[2] = -(grid_dim[2] * cell_dr - (range_max[2] - range_min[2])) / 2.0 + 0.01 * dr + range_min[2]
        # ------------------------------------------------------------------------------------------------------------------
        # calculate the coordinate of each cell
        cell_total = grid_dim[0] * grid_dim[1] * grid_dim[2]
        x_cell = np.zeros(cell_total * 3, 'f4').reshape(cell_total, 3)
        for k in range(0, cell_total):
            if ndim == 2:
                np_x = k % grid_dim[0]
                np_y = int(k / grid_dim[0])
                x_cell[k][0] = x_base[0] + (float(np_x) + 0.5) * cell_dr
                x_cell[k][1] = x_base[1] + (float(np_y) + 0.5) * cell_dr
            else:
                np_z = int(k / (grid_dim[0] * grid_dim[1]))
                np_y = int((k % (grid_dim[0] * grid_dim[1])) / grid_dim[0])
                np_x = (k % (grid_dim[0] * grid_dim[1])) % grid_dim[0]
                x_cell[k][0] = x_base[0] + (float(np_x) + 0.5) * cell_dr
                x_cell[k][1] = x_base[1] + (float(np_y) + 0.5) * cell_dr
                x_cell[k][2] = x_base[2] + (float(np_z) + 0.5) * cell_dr
        # ------------------------------------------------------------------------------------------------------------------
        # calculate the auto correlation matrix and calculate the lower triangular matrix
        mat_corr = np.zeros(6 * cell_total * cell_total, 'f4')  # .reshape((6, cell_total, cell_total))
        mat_corr_l = np.zeros(6 * cell_total * cell_total, 'f4').reshape((6, cell_total, cell_total))
        # loop for svf and svp
        for id_svp in range(1, 6):
            if randf_flag[id_svp] > 0:
                lx = randf_svp[id_svp][4]
                ly = randf_svp[id_svp][5]
                lz = randf_svp[id_svp][6]
                for i in range(0, cell_total):
                    for j in range(0, cell_total):
                        dx = np.abs(x_cell[i] - x_cell[j])
                        if randf_para[2] == 1:
                            acf = np.exp(-2.0 * (dx[0] / lx + dx[1] / ly + dx[2] / lz))
                        else:
                            acf = np.exp(
                                -np.pi * (np.pow(dx[0] / lx, 2) + np.pow(dx[1] / ly, 2) + np.pow(dx[2] / lz, 2)))
                        mat_corr[id_svp][i][j] = acf
                # Cholesky decomposition to get the lower triangular matrix
                mat_corr_l[id_svp] = np.linalg.cholesky(mat_corr[id_svp])
        # ------------------------------------------------------------------------------------------------------------------
        # generating random field one by one or correlated random field
        covar_mat = np.zeros(2 * 2, 'f4').reshape(2, 2)
        grid_para = np.zeors(cell_total * 6, 'f4').reshape(cell_total, 6)
        mean_value = np.zeros(cell_total * 2, 'f4').reshape(cell_total, 2)
        std_value = np.zeros(cell_total * 2, 'f4').reshape(cell_total, 2)
        x_id = np.zeros(3, "i4").reshape((3, 1))
        samples_num = int(input("Input the directory of input.dat: \n"))  # number of random field samples
        for steps in range(0, samples_num):
            if randf_flag[0] == 1 and randf_flag[1] == 1:  # correlated strength parameters for soil
                # setting parameters
                type_dist = randf_svf[0][1]
                mean_0 = randf_svp[0][0]
                std_0 = randf_svp[0][1]
                mean_1 = randf_svp[1][0]
                std_1 = randf_svp[1][1]
                coe_depth = randf_svp[0][5]
                # calculate the updated correlation coefficient
                if type_dist == 2:
                    temp1 = randf_para[1]
                    coe_corr = np.log(1.0 + temp1 * (std_0 / mean_0) * (std_1 / mean_1))
                    coe_corr = coe_corr / np.sqrt(
                        np.log(1.0 + std_0 * std_0 / mean_0 / mean_0) * np.log(1.0 + std_1 * std_1
                                                                               / mean_1 / mean_1))
                else:
                    coe_corr = randf_para[1]
                # correlation matrix C
                covar_mat[0][0] = 1.0
                covar_mat[0][1] = coe_corr
                covar_mat[1][0] = 0.0
                covar_mat[1][1] = np.sqrt(1.0 - coe_corr * coe_corr)
                # generating standard normal distributed variables (cell_total * 2)
                zeta_rnd = np.random.randn(cell_total, 2)
                zeta_rnd_1 = zeta_rnd @ covar_mat
                vec_e = mat_corr_l[0] @ zeta_rnd_1
                # generating random field for cells
                if type_dist == 1:  # normal distribution
                    for k in range(0, cell_total):
                        if ndim == 2:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                        else:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                    grid_para[:][0] = mean_value[:][0] + std_0 * vec_e[:][0]
                    grid_para[:][1] = mean_value[:][1] + std_1 * vec_e[:][1]
                    grid_para[:][3] = 1.0
                else:  # log normal distribution
                    for k in range(0, cell_total):
                        if ndim == 2:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                        else:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                    std_value[:][0] = np.sqrt(np.log(1.0 + (std_0 / mean_value[:][0]) * (std_0 / mean_value[:][0])))
                    std_value[:][1] = np.sqrt(np.log(1.0 + (std_1 / mean_value[:][1]) * (std_1 / mean_value[:][1])))
                    mean_value[:][0] = np.log(mean_value[:][0]) - 0.5 * std_value[:][0] * std_value[:][0]
                    mean_value[:][0] = np.log(mean_value[:][1]) - 0.5 * std_value[:][1] * std_value[:][1]
                    grid_para[:][0] = mean_value[:][0] + std_value[:][0] * vec_e[:][0]
                    grid_para[:][1] = mean_value[:][1] + std_value[:][1] * vec_e[:][1]
                    grid_para[:][3] = 1.0
                # setting parameters in cell to particles
                for pid in range(0, n_nonb):
                    if parti_nonb_type[pid] == 2:  # soil particle
                        np_x = int(np.abs(parti_x[pid][0] - x_base[0]) / cell_dr)
                        np_y = int(np.abs(parti_x[pid][1] - x_base[1]) / cell_dr)
                        np_z = int(np.abs(parti_x[pid][2] - x_base[2]) / cell_dr)
                        cell_id = np_z * grid_dim[0] * grid_dim[1] + np_y * grid_dim[0] + np_x
                        parti_para[pid][0] = grid_para[cell_id][0]
                        parti_para[pid][1] = grid_para[cell_id][1]
                        parti_para[pid][3] = grid_para[cell_id][3]
            elif randf_flag[4] == 1 and randf_flag[5] == 1:  # correlated strength parameters for fluid
                # setting parameters
                type_dist = randf_svf[4][1]
                mean_0 = randf_svp[4][0]
                std_0 = randf_svp[4][1]
                mean_1 = randf_svp[5][0]
                std_1 = randf_svp[5][1]
                coe_depth = randf_svp[4][5]
                # calculate the updated correlation coefficient
                if type_dist == 2:
                    temp1 = randf_para[1]
                    coe_corr = np.log(1.0 + temp1 * (std_0 / mean_0) * (std_1 / mean_1))
                    coe_corr = coe_corr / np.sqrt(
                        np.log(1.0 + std_0 * std_0 / mean_0 / mean_0) * np.log(1.0 + std_1 * std_1
                                                                               / mean_1 / mean_1))
                else:
                    coe_corr = randf_para[1]
                # correlation matrix C
                covar_mat[0][0] = 1.0
                covar_mat[0][1] = coe_corr
                covar_mat[1][0] = 0.0
                covar_mat[1][1] = np.sqrt(1.0 - coe_corr * coe_corr)
                # generating standard normal distributed variables (cell_total * 2)
                zeta_rnd = np.random.randn(cell_total, 2)
                zeta_rnd_1 = zeta_rnd @ covar_mat
                vec_e = mat_corr_l[4] @ zeta_rnd_1
                # generating random field for cells
                if type_dist == 1:  # normal distribution
                    for k in range(0, cell_total):
                        if ndim == 2:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                        else:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                    grid_para[:][4] = mean_value[:][0] + std_0 * vec_e[:][0]
                    grid_para[:][5] = mean_value[:][1] + std_1 * vec_e[:][1]
                    grid_para[:][3] = 1.0
                else:  # log normal distribution
                    for k in range(0, cell_total):
                        if ndim == 2:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                        else:
                            mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                            mean_value[k][1] = mean_1 + coe_depth * np.abs(range_max[2] - x_cell[k][2])
                    std_value[:][0] = np.sqrt(np.log(1.0 + (std_0 / mean_value[:][0]) * (std_0 / mean_value[:][0])))
                    std_value[:][1] = np.sqrt(np.log(1.0 + (std_1 / mean_value[:][1]) * (std_1 / mean_value[:][1])))
                    mean_value[:][0] = np.log(mean_value[:][0]) - 0.5 * std_value[:][0] * std_value[:][0]
                    mean_value[:][0] = np.log(mean_value[:][1]) - 0.5 * std_value[:][1] * std_value[:][1]
                    grid_para[:][4] = mean_value[:][0] + std_value[:][0] * vec_e[:][0]
                    grid_para[:][5] = mean_value[:][1] + std_value[:][1] * vec_e[:][1]
                    grid_para[:][3] = 1.0
                # setting parameters in cell to particles
                for pid in range(0, n_nonb):
                    if parti_nonb_type[pid] == 1:  # fluid particle
                        np_x = int(np.abs(parti_x[pid][0] - x_base[0]) / cell_dr)
                        np_y = int(np.abs(parti_x[pid][1] - x_base[1]) / cell_dr)
                        np_z = int(np.abs(parti_x[pid][2] - x_base[2]) / cell_dr)
                        cell_id = np_z * grid_dim[0] * grid_dim[1] + np_y * grid_dim[0] + np_x
                        parti_para[pid][0] = grid_para[cell_id][4]
                        parti_para[pid][1] = grid_para[cell_id][5]
                        parti_para[pid][3] = grid_para[cell_id][3]
            else:  # other case one by one
                for id_svp in range(1, 6):
                    if randf_flag[id_svp] > 0:
                        # setting parameters
                        type_dist = randf_svf[4][1]
                        mean_0 = randf_svp[4][0]
                        std_0 = randf_svp[4][1]
                        coe_depth = randf_svp[4][5]
                        # generating standard normal distributed variables (cell_total * 2)
                        zeta_rnd = np.random.randn(cell_total, 1)
                        vec_e = mat_corr_l[4] @ zeta_rnd
                        # generating random field for cells
                        if type_dist == 1:  # normal distribution
                            for k in range(0, cell_total):
                                grid_para[k][3] = 1.0
                                if ndim == 2:
                                    mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                                else:
                                    mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[2] - x_cell[k][2])

                                if id_svp == 0:  # friction angle for soil
                                    grid_para[k][0] = mean_value[k][0] + std_0 * vec_e[k]
                                elif id_svp == 1:  # cohesion for soil
                                    grid_para[k][1] = mean_value[k][0] + std_0 * vec_e[k]
                                elif id_svp == 2:  # coefficient of permeability for soil
                                    grid_para[k][2] = mean_value[k][0] + std_0 * vec_e[k]
                                elif id_svp == 3:  # soil skeleton size
                                    grid_para[k][3] = mean_value[k][0] + std_0 * vec_e[k]
                                elif id_svp == 4:  # friction angle for fluid model
                                    grid_para[k][4] = mean_value[k][0] + std_0 * vec_e[k]
                                else:  # cohesion for fluid model
                                    grid_para[k][5] = mean_value[k][0] + std_0 * vec_e[k]

                        else:  # log normal distribution
                            for k in range(0, cell_total):
                                grid_para[k][3] = 1.0
                                if ndim == 2:
                                    mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[1] - x_cell[k][1])
                                else:
                                    mean_value[k][0] = mean_0 + coe_depth * np.abs(range_max[2] - x_cell[k][2])

                                std_value[k][0] = np.sqrt(np.log(1.0 + (std_0 / mean_value[k][0]) * (std_0
                                                                                                     / mean_value[k][
                                                                                                         0])))
                                mean_value[k][0] = np.log(mean_value[k][0]) - 0.5 * std_value[k][0] * std_value[k][0]

                                if id_svp == 0:  # friction angle for soil
                                    grid_para[k][0] = np.exp(mean_value[k][0] + std_value[k][0] * vec_e[k])
                                elif id_svp == 1:  # cohesion for soil
                                    grid_para[k][1] = np.exp(mean_value[k][0] + std_value[k][0] * vec_e[k])
                                elif id_svp == 2:  # coefficient of permeability for soil
                                    grid_para[k][2] = np.exp(mean_value[k][0] + std_value[k][0] * vec_e[k])
                                elif id_svp == 3:  # soil skeleton size
                                    grid_para[k][3] = np.exp(mean_value[k][0] + std_value[k][0] * vec_e[k])
                                elif id_svp == 4:  # friction angle for fluid model
                                    grid_para[k][4] = np.exp(mean_value[k][0] + std_value[k][0] * vec_e[k])
                                else:  # cohesion for fluid model
                                    grid_para[k][5] = np.exp(mean_value[k][0] + std_value[k][0] * vec_e[k])
                # setting parameters in cell to particles
                for pid in range(0, n_nonb):
                    np_x = int(np.abs(parti_x[pid][0] - x_base[0]) / cell_dr)
                    np_y = int(np.abs(parti_x[pid][1] - x_base[1]) / cell_dr)
                    np_z = int(np.abs(parti_x[pid][2] - x_base[2]) / cell_dr)
                    cell_id = np_z * grid_dim[0] * grid_dim[1] + np_y * grid_dim[0] + np_x
                    if parti_nonb_type[pid] == 1:  # fluid particle
                        parti_para[pid][0] = grid_para[cell_id][4]
                        parti_para[pid][1] = grid_para[cell_id][5]
                        parti_para[pid][2] = grid_para[cell_id][2]
                        parti_para[pid][3] = grid_para[cell_id][3]
                    elif parti_nonb_type[pid] == 2:  # soil particle
                        parti_para[pid][0] = grid_para[cell_id][0]
                        parti_para[pid][1] = grid_para[cell_id][1]
                        parti_para[pid][2] = grid_para[cell_id][2]
                        parti_para[pid][3] = grid_para[cell_id][3]
            # --------------------------------------------------------------------------------------------------------------
            # output the random field to parameters.txt
            if os.sep == "/":
                output_txt_path = proj_path + r'/Para-' + f"{steps:0>5}" + r'.txt'  # linux platform
            else:
                output_txt_path = proj_path + r'\Para-' + f"{steps:0>5}" + r'.txt'  # windows platform
            with open(output_txt_path, 'w') as out_txt_file:
                out_txt_file.write(r"No.---- X----Y----Z----fai----c----cop----ds----type----matype----")
                for pid in range(0, n_nonb):
                    out_txt_file.write(f"{pid:>9} {parti_x[pid][0]:8.6e} {parti_x[pid][1]:8.6e} {parti_x[pid][2]:8.6e} "
                                       f"{parti_para[pid][0]:8.6e} {parti_para[pid][1]:8.6e} {parti_para[pid][2]:8.6e} "
                                       f"{parti_para[pid][3]:8.6e} {parti_nonb_type[pid]:>3}   0")
                out_txt_file.write(r"No.---- X----Y----Z----fai----c----cop----ds----type----matype----")
            # --------------------------------------------------------------------------------------------------------------
            # write to imd file
            # calculate the grid in the rectangle that covers the landslide model and find particles in each grid
            range_min = (np.min(parti_x, axis=0)).T  # the left and lower corner of rectangle or box
            range_max = (np.max(parti_x, axis=0)).T  # the right and upper corner of rectangle or box

            grid_dim[0] = int((range_max[0] - range_min[0]) / dr) + 1
            grid_dim[1] = int((range_max[1] - range_min[1]) / dr) + 1
            grid_dim[2] = int((range_max[2] - range_min[2]) / dr) + 1

            grid_para_1 = np.resize(grid_para, (grid_dim[0] * grid_dim[1] * grid_dim[2], 4))

            cell_num = grid_dim[0] * grid_dim[1] * grid_dim[2]
            for part_id in range(0, n_nonb):  # calculate grid_id for each particle
                x_id[0] = int((parti_x[part_id][0] - range_min[0]) / dr + 0.2)
                x_id[1] = int((parti_x[part_id][1] - range_min[1]) / dr + 0.2)
                x_id[2] = int((parti_x[part_id][2] - range_min[2]) / dr + 0.2)
                if ndim == 2:
                    grid_id = x_id[1] * grid_dim[0] + x_id[0]
                else:
                    grid_id = x_id[0] * grid_dim[2] * grid_dim[1] + x_id[2] * grid_dim[1] + x_id[1]

                if 0 <= grid_id < cell_num:
                    grid_para_1[grid_id] = parti_para[part_id]
            # --------------------------------------------------------------------------------------------------------------
            # out put to results file with designated name
            if ndim == 2:
                feat_ten = torch.from_numpy(grid_para_1.reshape((4, grid_dim[0], grid_dim[1])))
            else:
                feat_ten = torch.from_numpy(grid_para_1.reshape((4 * grid_dim[0], grid_dim[1], grid_dim[2])))
            if os.sep == "/":  # linux platform
                file_feature_out = proj_path + r'/features-' + f"{steps:0>5}" + r'.imd'
            else:  # Windows platform
                file_feature_out = proj_path + r'\\features-' + f"{steps:0>5}" + r'.imd'
            torch.save(feat_ten, file_feature_out)
    # ----------------------------------------------------------------------------------------------------------------------
    else:
        exit(2)
