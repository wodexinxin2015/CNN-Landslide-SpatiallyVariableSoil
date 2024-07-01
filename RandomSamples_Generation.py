# ----------------------------------------------------------------------------------------------------------------------
# RandomSamples_Generation.py
# -to generate random field of soil properties with specified parameters and transform data to image data (*.imd)
# -to automatically pad free area with additional data information in a rectangle that covers the landslide model
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
from RandomField_Gene_Functions import kl_corr_cfai
from RandomField_Gene_Functions import kl_nocorr_cfai


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
        samples_num = int(input("Input the number of random field sample: \n"))  # number of random field samples
        klterm = 150
        if randf_flag[0] == 1 and randf_flag[1] == 1:  # correlated fai and c for soil
            kl_corr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_para, randf_svf, randf_svp, klterm,
                         samples_num, 0, n_nonb, parti_nonb_type, ndim, dr, proj_path)
        elif randf_flag[4] == 1 and randf_flag[5] == 1:  # correlated fai and c for fluid
            kl_corr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_para, randf_svf, randf_svp, klterm,
                         samples_num, 4, n_nonb, parti_nonb_type, ndim, dr, proj_path)
        else:  # generating random field for each parameter one by one
            kl_nocorr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_flag, randf_svf, randf_svp, klterm,
                           samples_num, n_nonb, parti_nonb_type, ndim, dr, proj_path)
    else:
        return 2
