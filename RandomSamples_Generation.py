# ----------------------------------------------------------------------------------------------------------------------
# RandomSamples_Generation.py
# -to generate random field of soil properties with specified parameters and transform data to image data (*.imd)
# -to automatically pad free area with additional data information in a rectangle that covers the landslide model
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import scipy as sp
from RadomField_Gene_Fuctions import kl_corr_cfai
from RadomField_Gene_Fuctions import kl_nocorr_cfai
from RadomField_Gene_Fuctions import data_to_imd
from RadomField_Gene_Fuctions import cell_coordinates
from RadomField_Gene_Fuctions import auto_correlation_matrix
from RadomField_Gene_Fuctions import corr_random_field_midp
from RadomField_Gene_Fuctions import non_random_field_midp


# ----------------------------------------------------------------------------------------------------------------------
def sparse_cholesky(A):  # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = sp.linalg.splu(A, diag_pivot_thresh=0)  # sparse LU decomposition

    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():  # check the matrix A is positive definite.
        return LU.L.dot(sp.sparse.diags(LU.U.diagonal() ** 0.5))
    else:
        exit('The matrix is not positive definite')


# ----------------------------------------------------------------------------------------------------------------------
def random_samples_generate_kl(proj_path):  # generating random field of landslides using the K-L expansion method
    # setting the problem parameters and random field parameters
    if os.sep == "/":
        input_path = proj_path + r'/input.dat'  # linux platform
    else:
        input_path = proj_path + r'\input.dat'  # windows platform
    if os.path.exists(input_path):
        input_file = open(input_path, 'r')  # open the input file: input.dat
        # --------------------------------------------------------------------------------------------------------------
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
        parti_para[:][3] = 1.0
        # --------------------------------------------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------------------------------------------
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
        klterm = 200
        if randf_svf[0] == 1 and randf_svf[1] == 1:  # correlated fai and c for soil
            kl_corr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_para,randf_svf, randf_svp, klterm,
                         samples_num, 0, n_nonb, parti_nonb_type, ndim, dr, proj_path)
        elif randf_svf[4] == 1 and randf_svf[5] == 1:   # correlated fai and c for fluid
            kl_corr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_para, randf_svf, randf_svp, klterm,
                         samples_num, 4, n_nonb, parti_nonb_type, ndim, dr, proj_path)
        else:   # generating random field for each parameter one by one
            kl_nocorr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_para, randf_svf, randf_svp, klterm,
                         samples_num, n_nonb, parti_nonb_type, ndim, dr, proj_path)
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
        # --------------------------------------------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------------------------------------------
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
        # --------------------------------------------------------------------------------------------------------------
        # calculate the coordinate of each cell
        cell_total = grid_dim[0] * grid_dim[1] * grid_dim[2]
        x_cell = np.zeros(cell_total * 3, 'f4').reshape(cell_total, 3)
        grid_para = np.zeros(cell_total * 4, 'f4').reshape(cell_total, 4)
        cell_coordinates(x_cell, x_base, cell_dr, grid_dim, cell_total, ndim)
        # --------------------------------------------------------------------------------------------------------------
        # calculate the auto correlation matrix and calculate the lower triangular matrix
        # sparse matrix for auto correlation matrix and Cholesky decomposition
        if randf_flag[0] > 0:
            mat_corr_fais = auto_correlation_matrix(randf_svp[0], randf_para, x_cell, cell_dr, cell_total, ndim)
            mat_corr_fais_l = sparse_cholesky(mat_corr_fais)
        if randf_flag[1] > 0:
            mat_corr_cs = auto_correlation_matrix(randf_svp[1], randf_para, x_cell, cell_dr, cell_total, ndim)
            mat_corr_cs_l = sparse_cholesky(mat_corr_cs)
        if randf_flag[2] > 0:
            mat_corr_cops = auto_correlation_matrix(randf_svp[2], randf_para, x_cell, cell_dr, cell_total, ndim)
            mat_corr_cops_l = sparse_cholesky(mat_corr_cops)
        if randf_flag[3] > 0:
            mat_corr_ds = auto_correlation_matrix(randf_svp[3], randf_para, x_cell, cell_dr, cell_total, ndim)
            mat_corr_ds_l = sparse_cholesky(mat_corr_ds)
        if randf_flag[4] > 0:
            mat_corr_faif = auto_correlation_matrix(randf_svp[4], randf_para, x_cell, cell_dr, cell_total, ndim)
            mat_corr_faif_l = sparse_cholesky(mat_corr_faif)
        if randf_flag[5] > 0:
            mat_corr_cf = auto_correlation_matrix(randf_svp[5], randf_para, x_cell, cell_dr, cell_total, ndim)
            mat_corr_cf_l = sparse_cholesky(mat_corr_cf)
        # loop for the number of random fields
        samples_num = int(input("Input the directory of input.dat: \n"))  # number of random field samples
        for steps in range(0, samples_num):
            if randf_flag[0] == 1 and randf_flag[1] == 1:  # correlated strength parameters for soil
                corr_random_field_midp(0, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                       mat_corr_fais_l, mat_corr_cs_l, grid_para, ndim)
                # setting parameters in cell to particles: type == 2
                for pid in range(0, n_nonb):
                    if parti_nonb_type[pid] == 2:  # soil particle
                        np_x = int(np.abs(parti_x[pid][0] - x_base[0]) / cell_dr)
                        np_y = int(np.abs(parti_x[pid][1] - x_base[1]) / cell_dr)
                        np_z = int(np.abs(parti_x[pid][2] - x_base[2]) / cell_dr)
                        cell_id = np_z * grid_dim[0] * grid_dim[1] + np_y * grid_dim[0] + np_x
                        parti_para[pid]= grid_para[cell_id]
            elif randf_flag[4] == 1 and randf_flag[5] == 1:  # correlated strength parameters for fluid
                corr_random_field_midp(4, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                       mat_corr_faif_l, mat_corr_cf_l, grid_para, ndim)
                # setting parameters in cell to particles: type == 2
                for pid in range(0, n_nonb):
                    if parti_nonb_type[pid] == 1:  # fluid particle
                        np_x = int(np.abs(parti_x[pid][0] - x_base[0]) / cell_dr)
                        np_y = int(np.abs(parti_x[pid][1] - x_base[1]) / cell_dr)
                        np_z = int(np.abs(parti_x[pid][2] - x_base[2]) / cell_dr)
                        cell_id = np_z * grid_dim[0] * grid_dim[1] + np_y * grid_dim[0] + np_x
                        parti_para[pid]= grid_para[cell_id]
            else:    # other case one by one
                if randf_flag[0] > 0:
                    non_random_field_midp(0, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                          mat_corr_fais_l, grid_para, ndim)
                if randf_flag[1] > 0:
                    non_random_field_midp(1, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                          mat_corr_cs_l, grid_para, ndim)
                if randf_flag[2] > 0:
                    non_random_field_midp(2, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                          mat_corr_cops_l, grid_para, ndim)
                if randf_flag[3] > 0:
                    non_random_field_midp(3, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                          mat_corr_ds_l, grid_para, ndim)
                if randf_flag[4] > 0:
                    non_random_field_midp(4, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                          mat_corr_faif_l, grid_para, ndim)
                if randf_flag[5] > 0:
                    non_random_field_midp(5, randf_para, randf_svf, randf_svp, range_max, x_cell, cell_total,
                                          mat_corr_cf_l, grid_para, ndim)
            # ----------------------------------------------------------------------------------------------------------
            # output the random field to parameters.txt
            # write to imd file
            data_to_imd(parti_x, parti_para, dr, ndim, n_nonb, proj_path, steps, parti_nonb_type)
    # ------------------------------------------------------------------------------------------------------------------
    else:
        exit(2)
