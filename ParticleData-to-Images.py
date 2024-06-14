# ----------------------------------------------------------------------------------------------------------------------
# ParticleData-to-Images.py
# -to convert the particle data file with spatially variable strength parameters to image data (*.imd)
# -to pad free area with additional data information in a rectangle that covers the landslide model
# -Coded by Prof. Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
flag = True
while flag:
    # input the file folder
    proj_path = input("Input the parameter directory of SPH program: (where is the input.dat) \n")
    if os.sep == "/":  # linux platform
        file_path = proj_path + r'/Parameters.txt'
        file_path_set = proj_path + r'/input.dat'
    else:  # windows platform
        file_path = proj_path + r'\Parameters.txt'
        file_path_set = proj_path + r'\input.dat'
    if os.path.exists(file_path):
        particle_file = open(file_path, 'r')  # open the file
        input_file = open(file_path_set, 'r')  # open the file
    else:
        print("There is no input.dat file in working directory, thus exit.")
        os.sys.exit(1)
    # ------------------------------------------------------------------------------------------------------------------
    # input file -> array and select the strength parameters
    fldata = input_file.readlines()
    flline_0 = fldata[15].split()
    flline_1 = fldata[15].split()
    flline_2 = fldata[16].split()
    ndim = int(flline_0[0])    # setting the number of dimensions
    ntotal = int(flline_1[3])   # setting the total number of particles
    dr = float(flline_2[1])  # setting the particle spacing
    parti_type = np.zeros(ntotal, 'i4').reshape(ntotal, 1)  # particle type: main type, sub mat type

    n_nonb = 0
    for k in range(253, ntotal + 253):
        flline = fldata[k].split()
        if ndim == 2:
            parti_type[k - 253] = int(flline[4])
        else:
            parti_type[k - 253] = int(flline[6])
        if parti_type[k - 253] != 0:
            n_nonb = n_nonb + 1

    parti_x = np.zeros(ntotal * 3, 'f4').reshape(ntotal, 3)  # x, y, z of particle coordinates
    parti_str = np.zeros(ntotal * 2, 'f4').reshape(ntotal, 2)   # strength parameters: friction angle, cohesion
    parti_seepage = np.zeros(ntotal * 2, 'f4').reshape(ntotal, 2)  # seepage parameters: coefficient, soil particle size
    parti_type = np.zeros(ntotal * 2, 'i4').reshape(ntotal, 2)  # particle type: main type, sub mat type

    fldata_2 = particle_file.readlines()[1:n_nonb + 1]
    for k in range(0, n_nonb - 1):
        flline_k = fldata_2[k + 1].split()
        parti_x[k][0] = float(flline_k[1])
        parti_x[k][1] = float(flline_k[2])
        parti_x[k][2] = float(flline_k[3])
        parti_str[k][0] = float(flline_k[4])
        parti_str[k][1] = float(flline_k[5])
        parti_seepage[k][0] = float(flline_k[6])
        parti_seepage[k][1] = float(flline_k[7])
        parti_type[k][0] = float(flline_k[8])
        parti_type[k][1] = float(flline_k[9])
    # ------------------------------------------------------------------------------------------------------------------
    # calculate the grid in the rectangle that covers the landslide model and find particles in each grid
    range_min = (parti_x.min(axis=0)).T  # the left and lower corner of rectangle or box
    range_max = (parti_x.max(axis=0)).T # the right and upper corner of rectangle or box

    grid_dim = np.zeros(3, 'i4').reshape(3, 1)
    grid_dim[0] = int((range_max[0] - range_min[0]) / dr) + 1
    grid_dim[1] = int((range_max[1] - range_min[1]) / dr) + 1
    grid_dim[2] = int((range_max[2] - range_min[2]) / dr) + 1

    # ------------------------------------------------------------------------------------------------------------------
    # pad a default value on the free space of rectangle

    # ------------------------------------------------------------------------------------------------------------------
    # out put to results file with designated name

    flag = (input("Continue (y) or Exit (n):  \n") == 'y' or input("Continue (y) or Exit (n):  \n") == 'Y')
# ----------------------------------------------------------------------------------------------------------------------
else:
    exit(0)
