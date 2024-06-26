# ----------------------------------------------------------------------------------------------------------------------
# CNN_RadomField_Fuctions.py
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
        alphai = 1.0 / np.sqrt(np.sin(2.0 * wi * a) / 2.0 / wi + a)
        faii = alphai * np.cos(wi * x)
    else:
        alphai = 1.0 / np.sqrt(-np.sin(2.0 * wi * a) / 2.0 / wi + a)
        faii = alphai * np.sin(wi * x)
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
                x1 = (i + 0.5) * np.pi / a + precision
                x2 = (i + 1.0) * np.pi / a - precision
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
        eigval_2d = np.zeros(klterm * klterm, 'f4').reshape(klterm, klterm)
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
        eigval_3d = np.zeros(klterm * klterm * klterm, 'f4').reshape(klterm, klterm, klterm)
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


