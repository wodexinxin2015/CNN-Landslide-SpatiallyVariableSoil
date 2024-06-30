# ----------------------------------------------------------------------------------------------------------------------
# RandomField_Gene_Fuctions.py
# - functions used in the genrating process of landslide random fields
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
import torch
import os


# ----------------------------------------------------------------------------------------------------------------------
# define the function of radmai
def equation_ramdai(dirac, wi_v):
    self = 4.0 * dirac / (4.0 + wi_v * wi_v * dirac * dirac)
    return self


# ----------------------------------------------------------------------------------------------------------------------
# define the function of faii
def equation_faii(x, a, wi_v, i):
    if (i + 1) % 2 == 1:
        alphai = 1.0 / np.sqrt(np.sin(2.0 * wi_v * a) / 2.0 / wi_v + a)
        faii = alphai * np.cos(wi_v * x)
    else:
        alphai = 1.0 / np.sqrt(-np.sin(2.0 * wi_v * a) / 2.0 / wi_v + a)
        faii = alphai * np.sin(wi_v * x)
    return faii


# ----------------------------------------------------------------------------------------------------------------------
# define the function to solve wi
def solve_wii(wi, a_x, dirac_r, ndim, klterm):
    for d in range(0, ndim):
        a = a_x[d]
        dira = dirac_r[d]
        for i in range(0, klterm):
            if (i + 1) % 2 == 1:  # odd term
                x1 = i * np.pi / a + 1.0e-6
                x2 = (i + 0.5) * np.pi / a - 1.0e-6
                xm = (x1 + x2) * 0.5
                fm = 2.0 / dira - xm * np.tan(xm * a)
                while np.fabs(fm) > 1.0e-6:
                    f1 = 2.0 / dira - x1 * np.tan(x1 * a)
                    fm = 2.0 / dira - xm * np.tan(xm * a)
                    if (f1 * fm) < 0.0:
                        x2 = xm
                        xm = (x1 + x2) * 0.5
                    else:
                        x1 = xm
                        xm = (x1 + x2) * 0.5
            else:  # even term
                x1 = (i + 0.5) * np.pi / a + 1.0e-6
                x2 = (i + 1.0) * np.pi / a - 1.0e-6
                xm = (x1 + x2) * 0.5
                fm = 2.0 / dira * np.tan(xm * a) + xm
                while np.fabs(fm) > 1.0e-6:
                    f1 = 2.0 / dira * np.tan(x1 * a) + x1
                    fm = 2.0 / dira * np.tan(xm * a) + xm
                    if (f1 * fm) < 0.0:
                        x2 = xm
                        xm = (x1 + x2) * 0.5
                    else:
                        x1 = xm
                        xm = (x1 + x2) * 0.5
            wi[i][d] = xm


# ----------------------------------------------------------------------------------------------------------------------
# define the function to solve ramdai
def solve_ramdai(eigval, wi, dirac_r, order_id, ndim, klterm):
    # initialization of order_id
    for i in range(0, ndim):
        order_id[i][:] = i
    # calculating ramda i for each idx in klterm
    ramda = np.zeros(klterm * 3, 'f4').reshape(klterm, 3)
    for d in range(0, ndim):
        for i in range(0, klterm):
            ramda[i][d] = equation_ramdai(dirac_r[d], wi[i][d])

    if ndim == 2:
        # calculate the value of eigval_2d
        eigval_2d = np.zeros(klterm * klterm, 'f4').reshape((klterm, klterm))
        for i in range(0, klterm):
            for j in range(0, klterm):
                eigval_2d[i][j] = ramda[i][0] * ramda[j][1]
        # sorting the eigval_2d and find the order
        for i in range(0, klterm):
            eigval[i] = -1.0
            for m in range(0, klterm):
                for n in range(0, klterm):
                    if eigval[i] < eigval_2d[m][n]:
                        eigval[i] = eigval_2d[m][n]
                        order_id[i][0] = m
                        order_id[i][1] = n
            m = order_id[i][0]
            n = order_id[i][1]
            eigval_2d[m][n] = 0.0
    else:
        # calculate the value of eigval_3d
        eigval_3d = np.zeros(klterm * klterm * klterm, 'f4').reshape((klterm, klterm, klterm))
        for i in range(0, klterm):
            for j in range(0, klterm):
                for k in range(0, klterm):
                    eigval_3d[i][j][k] = ramda[i][0] * ramda[j][1] * ramda[k][2]
        # sorting the eigval_3d and find the order
        for i in range(0, klterm):
            eigval[i] = -1.0
            for m in range(0, klterm):
                for n in range(0, klterm):
                    for k in range(0, klterm):
                        if eigval[i] < eigval_3d[m][n][k]:
                            eigval[i] = eigval_3d[m][n][k]
                            order_id[i][0] = m
                            order_id[i][1] = n
                            order_id[i][2] = k
            m = order_id[i][0]
            n = order_id[i][1]
            k = order_id[i][2]
            eigval_3d[m][n][k] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# define the function to solve h_2d
def solve_h_2d(part_x, part_para, wi_v, eigval, uu, x_trans, order_id, randf_svf, randf_svp, soil_max, water_max,
               parti_nonb_type, type_no, ntotal, klterm):
    x_a = np.zeros(2, 'f4').reshape((2, 1))
    fai = np.zeros(2, 'f4').reshape((2, 1))
    eigvect = np.zeros(klterm, 'f4').reshape((klterm, 1))
    for i in range(0, ntotal):
        # x value
        x_a[0] = part_x[i][0] - x_trans[0]
        x_a[1] = part_x[i][1] - x_trans[1]
        # solve faii
        for j in range(0, klterm):
            idx = order_id[j][0]
            idy = order_id[j][1]
            fai[0] = equation_faii(x_a[0], randf_svp[2], wi_v[idx][0], idx)
            fai[1] = equation_faii(x_a[1], randf_svp[3], wi_v[idx][1], idy)
            eigvect[j] = fai[0] * fai[1]
        # solve H value
        cof_dep = randf_svp[5]
        if randf_svf[1] == 1:  # normal distribution
            # setting the mean value considering the increasing of parameters with depths and standard deviation
            if parti_nonb_type[i] == 2:
                temp_mean = randf_svp[0] + (soil_max - part_x[i][1]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[i][1]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = randf_svp[1]
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j]) * eigvect[j] * uu[j]
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[i][0] = temp_mean + temp_h * temp_std
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[i][1] = temp_mean + temp_h * temp_std
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[i][2] = temp_mean + temp_h * temp_std
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[i][3] = temp_mean + temp_h * temp_std
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[i][0] = temp_mean + temp_h * temp_std
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[i][1] = temp_mean + temp_h * temp_std
        else:  # log normal distribution
            # setting the mean value considering the increasing of parameters with depths and standard deviation
            if parti_nonb_type[i] == 2:
                temp_mean = randf_svp[0] + (soil_max - part_x[i][1]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[i][1]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = np.sqrt(np.log(1.0 + (randf_svp[1] / temp_mean) * (randf_svp[1] / temp_mean)))
            temp_mean = np.log(temp_mean) - 0.5 * temp_std * temp_std
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j]) * eigvect[j] * uu[j]
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[i][0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[i][1] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[i][2] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[i][3] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[i][0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[i][1] = np.exp(temp_mean + temp_h * temp_std)


# ----------------------------------------------------------------------------------------------------------------------
# define the function to solve h_3d
def solve_h_3d(part_x, part_para, wi_v, eigval, uu, x_trans, order_id, randf_svf, randf_svp, soil_max, water_max,
               parti_nonb_type, type_no, ntotal, klterm):
    x_a = np.zeros(3, 'f4').reshape((3, 1))
    fai = np.zeros(3, 'f4').reshape((3, 1))
    eigvect = np.zeros(klterm, 'f4').reshape((klterm, 1))
    for i in range(0, ntotal):
        # x value
        x_a[0] = part_x[i][0] - x_trans[0]
        x_a[1] = part_x[i][1] - x_trans[1]
        x_a[2] = part_x[i][2] - x_trans[2]
        # solve faii
        for j in range(0, klterm):
            idx = order_id[j][0]
            idy = order_id[j][1]
            idz = order_id[j][2]
            fai[0] = equation_faii(x_a[0], randf_svp[2], wi_v[idx][0], idx)
            fai[1] = equation_faii(x_a[1], randf_svp[3], wi_v[idx][1], idy)
            fai[2] = equation_faii(x_a[2], randf_svp[4], wi_v[idx][2], idz)
            eigvect[j] = fai[0] * fai[1] * fai[2]
        # solve H value
        cof_dep = randf_svp[5]
        if randf_svf[1] == 1:  # normal distribution
            # setting the mean value considering the increasing of parameters with depths and standard deviation
            if parti_nonb_type[i] == 2:
                temp_mean = randf_svp[0] + (soil_max - part_x[i][2]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[i][2]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = randf_svp[1]
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j]) * eigvect[j] * uu[j]
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[i][0] = temp_mean + temp_h * temp_std
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[i][1] = temp_mean + temp_h * temp_std
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[i][2] = temp_mean + temp_h * temp_std
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[i][3] = temp_mean + temp_h * temp_std
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[i][0] = temp_mean + temp_h * temp_std
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[i][1] = temp_mean + temp_h * temp_std
        else:  # log normal distribution
            # setting the mean value considering the increasing of parameters with depths and standard deviation
            if parti_nonb_type[i] == 2:
                temp_mean = randf_svp[0] + (soil_max - part_x[i][2]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[i][2]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = np.sqrt(np.log(1.0 + (randf_svp[1] / temp_mean) * (randf_svp[1] / temp_mean)))
            temp_mean = np.log(temp_mean) - 0.5 * temp_std * temp_std
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j]) * eigvect[j] * uu[j]
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[i][0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[i][1] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[i][2] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[i][3] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[i][0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[i][1] = np.exp(temp_mean + temp_h * temp_std)


# ----------------------------------------------------------------------------------------------------------------------
# define the function that writes data to *imd files
def data_to_imd(parti_x, parti_para, dr, ndim, n_nonb, proj_path, no_file, parti_nonb_type):
    # ------------------------------------------------------------------------------------------------------------------
    if os.sep == "/":
        output_txt_path = proj_path + r'/Para-' + f"{no_file + 1:0>5}" + r'.txt'  # linux platform
    else:
        output_txt_path = proj_path + r'\Para-' + f"{no_file + 1:0>5}" + r'.txt'  # windows platform
    with open(output_txt_path, 'w') as out_txt_file:
        out_txt_file.write(r"No.---- X----Y----Z----fai----c----cop----ds----type----matype---- \n")
        for pid in range(0, n_nonb):
            out_txt_file.write("%9d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %3d  0\n" % (pid, parti_x[pid][0],
                                                                                    parti_x[pid][1],
                                                                                    parti_x[pid][2],
                                                                                    parti_para[pid][0],
                                                                                    parti_para[pid][1],
                                                                                    parti_para[pid][2],
                                                                                    parti_para[pid][3],
                                                                                    parti_nonb_type[pid]))
        out_txt_file.write(r"No.---- X----Y----Z----fai----c----cop----ds----type----matype----\n")
    # ------------------------------------------------------------------------------------------------------------------
    x_id = np.zeros(3, "i4").reshape((3, 1))
    grid_dim = np.zeros(3, 'i4')
    # calculate the grid in the rectangle that covers the landslide model and find particles in each grid
    range_min = (np.min(parti_x, axis=0)).T  # the left and lower corner of rectangle or box
    range_max = (np.max(parti_x, axis=0)).T  # the right and upper corner of rectangle or box

    grid_dim[0] = int((range_max[0] - range_min[0]) / dr) + 1
    grid_dim[1] = int((range_max[1] - range_min[1]) / dr) + 1
    grid_dim[2] = int((range_max[2] - range_min[2]) / dr) + 1

    grid_para = np.zeros(grid_dim[0] * grid_dim[1] * grid_dim[2] * 4, 'f4').reshape(
        (grid_dim[0] * grid_dim[1] * grid_dim[2], 4))

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
            grid_para[grid_id] = parti_para[part_id]
    # ----------------------------------------------------------------------------------------------------------
    # out put to results file with designated name
    if ndim == 2:
        feat_ten = torch.from_numpy(grid_para.reshape((4, grid_dim[0], grid_dim[1])))
    else:
        feat_ten = torch.from_numpy(grid_para.reshape((4 * grid_dim[0], grid_dim[1], grid_dim[2])))
    if os.sep == "/":  # linux platform
        file_feature_out = proj_path + r'/features-' + f"{no_file + 1:0>5}" + r'.imd'
    else:  # Windows platform
        file_feature_out = proj_path + r'\\features-' + f"{no_file + 1:0>5}" + r'.imd'
    torch.save(feat_ten, file_feature_out)


# ----------------------------------------------------------------------------------------------------------------------
# define the K-L expansion procedure of correlated friction angle and cohesion
def kl_corr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_para, randf_svf, randf_svp, klterm, samples_num,
                 para_no, n_nonb, parti_nonb_type, ndim, dr, proj_path):
    rl_val = np.zeros(3, 'f4').reshape(3, 1)
    covar_mat = np.zeros(2 * 2, 'f4').reshape(2, 2)
    order_id = np.zeros(klterm * 3, 'i4').reshape(klterm, 3)
    wi_v = np.zeros(klterm * 3, 'f4').reshape(klterm, 3)
    eigval = np.zeros(klterm, 'f4').reshape(klterm, 1)
    uu = np.zeros(klterm, 'f4').reshape(klterm, 1)
    # setting parameters
    type_dist = randf_svf[para_no][1]
    mean_0 = randf_svp[para_no][0]
    std_0 = randf_svp[para_no][1]
    mean_1 = randf_svp[para_no + 1][0]
    std_1 = randf_svp[para_no + 1][1]
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
    if ndim == 2:
        for step_s in range(0, samples_num):
            # generate random variables of standard normal distribution
            uu_cfai = np.random.randn(klterm, 2)
            # calculate the correlated random variables of standard normal distribution
            zeta = uu_cfai @ covar_mat
            # solve H-2D for friction angle of soil
            # calculate wi_v
            rl_val[0] = randf_svp[para_no][2]
            rl_val[1] = randf_svp[para_no][3]
            solve_wii(wi_v, a_x, rl_val, ndim, klterm)
            # calculate ramda
            solve_ramdai(eigval, wi_v, rl_val, order_id, ndim, klterm)
            uu[:] = zeta[:][0]
            solve_h_2d(parti_x, parti_para, wi_v, eigval, uu, x_trans, order_id, randf_svf[para_no],
                       randf_svp[para_no], range_max[1], range_max[1], parti_nonb_type, para_no, n_nonb, klterm)
            # solve H-2D for cohesion of soil
            # calculate wi_v
            rl_val[0] = randf_svp[para_no + 1][2]
            rl_val[1] = randf_svp[para_no + 1][3]
            solve_wii(wi_v, a_x, rl_val, ndim, klterm)
            # calculate ramda
            solve_ramdai(eigval, wi_v, rl_val, order_id, ndim, klterm)
            uu[:] = zeta[:][1]
            solve_h_2d(parti_x, parti_para, wi_v, eigval, uu, x_trans, order_id, randf_svf[para_no + 1],
                       randf_svp[para_no + 1], range_max[1], range_max[1], parti_nonb_type, para_no, n_nonb, klterm)
            # output to files
            data_to_imd(parti_x, parti_para, dr, ndim, n_nonb, proj_path, step_s, parti_nonb_type)
    else:
        for step_s in range(0, samples_num):
            # generate random variables of standard normal distribution
            uu_cfai = np.random.randn(klterm, 2)
            # calculate the correlated random variables of standard normal distribution
            zeta = uu_cfai @ covar_mat
            # solve H-2D for friction angle of soil
            # calculate wi_v
            rl_val[0] = randf_svp[para_no][2]
            rl_val[1] = randf_svp[para_no][3]
            rl_val[2] = randf_svp[para_no][4]
            solve_wii(wi_v, a_x, rl_val, ndim, klterm)
            # calculate ramda
            solve_ramdai(eigval, wi_v, rl_val, order_id, ndim, klterm)
            uu[:] = zeta[:][0]
            solve_h_3d(parti_x, parti_para, wi_v, eigval, uu, x_trans, order_id, randf_svf[para_no],
                       randf_svp[para_no], range_max[2], range_max[2], parti_nonb_type, para_no, n_nonb, klterm)
            # solve H-2D for cohesion of soil
            # calculate wi_v
            rl_val[0] = randf_svp[para_no + 1][2]
            rl_val[1] = randf_svp[para_no + 1][3]
            rl_val[2] = randf_svp[para_no + 1][4]
            solve_wii(wi_v, a_x, rl_val, ndim, klterm)
            # calculate ramda
            solve_ramdai(eigval, wi_v, rl_val, order_id, ndim, klterm)
            uu[:] = zeta[:][1]
            solve_h_3d(parti_x, parti_para, wi_v, eigval, uu, x_trans, order_id, randf_svf[para_no + 1],
                       randf_svp[para_no + 1], range_max[2], range_max[2], parti_nonb_type, para_no, n_nonb, klterm)
            # output to files
            data_to_imd(parti_x, parti_para, dr, ndim, n_nonb, proj_path, step_s, parti_nonb_type)


# ----------------------------------------------------------------------------------------------------------------------
# define the ordinary K-L expansion procedure
def kl_nocorr_cfai(parti_x, parti_para, a_x, x_trans, range_max, randf_flag, randf_svf, randf_svp, klterm, samples_num,
                   n_nonb, parti_nonb_type, ndim, dr, proj_path):
    rl_val = np.zeros(3, 'f4').reshape(3, 1)
    order_id = np.zeros(klterm * 3, 'i4').reshape(klterm, 3)
    wi_v = np.zeros(klterm * 3, 'f4').reshape(klterm, 3)
    eigval = np.zeros(klterm, 'f4').reshape(klterm, 1)
    if ndim == 2:
        for step_s in range(0, samples_num):
            for para_no in range(0, 6):
                if randf_flag[para_no] > 0:
                    # generate random variables of standard normal distribution
                    uu = np.random.randn(klterm, 1)
                    # solve H-2D for friction angle of soil
                    # calculate wi_v
                    rl_val[0] = randf_svp[para_no][2]
                    rl_val[1] = randf_svp[para_no][3]
                    solve_wii(wi_v, a_x, rl_val, ndim, klterm)
                    # calculate ramda
                    solve_ramdai(eigval, wi_v, rl_val, order_id, ndim, klterm)
                    solve_h_2d(parti_x, parti_para, wi_v, eigval, uu, x_trans, order_id, randf_svf[para_no],
                               randf_svp[para_no], range_max[1], range_max[1], parti_nonb_type, para_no, n_nonb, klterm)
            # output to files
            data_to_imd(parti_x, parti_para, dr, ndim, n_nonb, proj_path, step_s, parti_nonb_type)
    else:
        for step_s in range(0, samples_num):
            for para_no in range(0, 6):
                if randf_flag[para_no] > 0:
                    # generate random variables of standard normal distribution
                    uu = np.random.randn(klterm, 1)
                    # solve H-2D for friction angle of soil
                    # calculate wi_v
                    rl_val[0] = randf_svp[para_no][2]
                    rl_val[1] = randf_svp[para_no][3]
                    rl_val[2] = randf_svp[para_no][4]
                    solve_wii(wi_v, a_x, rl_val, ndim, klterm)
                    # calculate ramda
                    solve_ramdai(eigval, wi_v, rl_val, order_id, ndim, klterm)
                    solve_h_3d(parti_x, parti_para, wi_v, eigval, uu, x_trans, order_id, randf_svf[para_no],
                               randf_svp[para_no], range_max[2], range_max[2], parti_nonb_type, para_no, n_nonb, klterm)
            # output to files
            data_to_imd(parti_x, parti_para, dr, ndim, n_nonb, proj_path, step_s, parti_nonb_type)

