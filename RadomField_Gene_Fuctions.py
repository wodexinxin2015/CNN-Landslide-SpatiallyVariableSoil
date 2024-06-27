# ----------------------------------------------------------------------------------------------------------------------
# RadomField_Gene_Fuctions.py
# - functions used in the genrating process of landslide random fields
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp


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
            eigval_2d[order_id[i][0]][order_id[i][1]] = 0.0
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
            eigval_3d[order_id[i][0]][order_id[i][1]] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# define the function to solve h_2d
def solve_h_2d(part_x, part_para, wi_v, eigval, uu, x_trans, order_id, randf_svf, randf_svp, soil_max, water_max,
                parti_nonb_type, type_no, ntotal, klterm):
    x_a = np.zeros(2, 'f4').reshape((2, 1))
    fai = np.zeros(2, 'f4').reshape((2, 1))
    eigvect = np.zeros(klterm, 'f4').reshape((klterm, 1))
    for i in range(0, ntotal):
        # x value
        x_a[0] = part_x[0] - x_trans[0]
        x_a[1] = part_x[1] - x_trans[1]
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
                temp_mean = randf_svp[0] + (soil_max - part_x[1]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[1]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = randf_svp[1]
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j] * eigvect[j] * uu[j])
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[0] = temp_mean + temp_h * temp_std
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[1] = temp_mean + temp_h * temp_std
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[2] = temp_mean + temp_h * temp_std
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[3] = temp_mean + temp_h * temp_std
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[0] = temp_mean + temp_h * temp_std
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[1] = temp_mean + temp_h * temp_std
        else:  # log normal distribution
            # setting the mean value considering the increasing of parameters with depths and standard deviation
            if parti_nonb_type[i] == 2:
                temp_mean = randf_svp[0] + (soil_max - part_x[1]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[1]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = randf_svp[1]
            temp_std = np.sqrt(np.log(1.0 + (randf_svp[1] / temp_mean) * (randf_svp[1] / temp_mean)))
            temp_mean = np.log(temp_mean) - 0.5 * temp_std * temp_std
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j] * eigvect[j] * uu[j])
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[1] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[2] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[3] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[1] = np.exp(temp_mean + temp_h * temp_std)


# ----------------------------------------------------------------------------------------------------------------------
# define the function to solve h_3d
def solve_h_3d(part_x, part_para, wi_v, eigval, uu, x_trans, order_id, randf_svf, randf_svp, soil_max, water_max,
                parti_nonb_type, type_no, ntotal, klterm):
    x_a = np.zeros(3, 'f4').reshape((3, 1))
    fai = np.zeros(3, 'f4').reshape((3, 1))
    eigvect = np.zeros(klterm, 'f4').reshape((klterm, 1))
    for i in range(0, ntotal):
        # x value
        x_a[0] = part_x[0] - x_trans[0]
        x_a[1] = part_x[1] - x_trans[1]
        x_a[2] = part_x[2] - x_trans[2]
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
                temp_mean = randf_svp[0] + (soil_max - part_x[1]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[1]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = randf_svp[1]
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j] * eigvect[j] * uu[j])
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[0] = temp_mean + temp_h * temp_std
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[1] = temp_mean + temp_h * temp_std
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[2] = temp_mean + temp_h * temp_std
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[3] = temp_mean + temp_h * temp_std
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[0] = temp_mean + temp_h * temp_std
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[1] = temp_mean + temp_h * temp_std
        else:  # log normal distribution
            # setting the mean value considering the increasing of parameters with depths and standard deviation
            if parti_nonb_type[i] == 2:
                temp_mean = randf_svp[0] + (soil_max - part_x[1]) * cof_dep
            elif parti_nonb_type[i] == 1:
                temp_mean = randf_svp[0] + (water_max - part_x[1]) * cof_dep
            else:
                temp_mean = randf_svp[0]
            temp_std = randf_svp[1]
            temp_std = np.sqrt(np.log(1.0 + (randf_svp[1] / temp_mean) * (randf_svp[1] / temp_mean)))
            temp_mean = np.log(temp_mean) - 0.5 * temp_std * temp_std
            # solve H
            temp_h = 0.0
            for j in range(0, klterm):
                temp_h = temp_h + np.sqrt(eigval[j] * eigvect[j] * uu[j])
            # reflect the random field to particle information
            if type_no == 0 and parti_nonb_type[i] == 2:
                part_para[0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 1 and parti_nonb_type[i] == 2:
                part_para[1] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 2 and parti_nonb_type[i] == 2:
                part_para[2] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 3 and parti_nonb_type[i] == 2:
                part_para[3] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 4 and parti_nonb_type[i] == 1:
                part_para[0] = np.exp(temp_mean + temp_h * temp_std)
            elif type_no == 5 and parti_nonb_type[i] == 1:
                part_para[1] = np.exp(temp_mean + temp_h * temp_std)

