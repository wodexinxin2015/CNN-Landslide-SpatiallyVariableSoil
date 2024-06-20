# ----------------------------------------------------------------------------------------------------------------------
# ParticleData-to-Images.py
# -to convert the particle data file with spatially variable strength parameters to image data (*.imd)
# -to pad free area with additional data information in a rectangle that covers the landslide model
# -Coded by Prof. Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch

# ----------------------------------------------------------------------------------------------------------------------
# select the features or labels
type_con = int(input("Input the convert type: 1--Feature file; 2--Label file.\n"))
# ----------------------------------------------------------------------------------------------------------------------
# Convert the feature file: parameters.txt
if type_con == 1:
    # ------------------------------------------------------------------------------------------------------------------
    parti_type_0 = np.zeros(4 * 1, 'i4')  # particle type: main type
    parti_x_0 = np.zeros(4 * 3, 'f4')  # x, y, z of particle coordinates
    parti_para_0 = np.zeros(4 * 4, 'f4')  # strength parameters: friction angle, cohesion; cop, ds
    x_id = np.zeros(3, "i4").reshape((3, 1))
    grid_dim = np.zeros(3, 'i4')

    flag = True
    no_file = 1
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
            # ----------------------------------------------------------------------------------------------------------
            # input file -> array and select the strength parameters
            fldata = input_file.readlines()
            flline_0 = fldata[14].split()
            flline_1 = fldata[15].split()
            flline_2 = fldata[16].split()
            ndim = int(flline_0[0])  # setting the number of dimensions
            ntotal = int(flline_1[3])  # setting the total number of particles
            dr = float(flline_2[1])  # setting the particle spacing

            parti_type = np.resize(parti_type_0, (ntotal, 1))
            n_nonb = 0
            for k in range(253, ntotal + 253):
                flline = fldata[k].split()
                if ndim == 2:
                    parti_type[k - 253] = int(flline[4])
                else:
                    parti_type[k - 253] = int(flline[6])
                if parti_type[k - 253] != 0:
                    n_nonb = n_nonb + 1

            parti_x = np.resize(parti_x_0, (n_nonb, 3))
            parti_para = np.resize(parti_para_0, (n_nonb, 4))

            fldata_2 = particle_file.readlines()[0:n_nonb + 1]
            for k in range(0, n_nonb):
                flline_k = fldata_2[k + 1].split()
                parti_x[k][0] = float(flline_k[1])
                parti_x[k][1] = float(flline_k[2])
                parti_x[k][2] = float(flline_k[3])
                parti_para[k][0] = float(flline_k[4])
                parti_para[k][1] = float(flline_k[5]) / 1000.0
                parti_para[k][2] = float(flline_k[6])
                parti_para[k][3] = float(flline_k[7])
            # ----------------------------------------------------------------------------------------------------------
            # calculate the grid in the rectangle that covers the landslide model and find particles in each grid
            range_min = (parti_x.min(axis=0)).T  # the left and lower corner of rectangle or box
            range_max = (parti_x.max(axis=0)).T  # the right and upper corner of rectangle or box

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
                imd_dat = torch.from_numpy(grid_para.reshape((4, grid_dim[0], grid_dim[1])))
            else:
                imd_dat = torch.from_numpy(grid_para.reshape((4 * grid_dim[0], grid_dim[1], grid_dim[2])))
            if os.sep == "/":  # linux platform
                file_path_out = proj_path + '/features-' + f"{no_file:0>4}" + r'.imd'
            else:  # Windows platform
                file_path_out = proj_path + '\\faeture-' + f"{no_file:0>4}" + r'.imd'
            torch.save(imd_dat, file_path_out)

            chara = input("Continue (y) or Exit (n):  \n")
            flag = (chara == 'y' or chara == 'Y')
            no_file = no_file + 1
        else:
            print("There is no input.dat file in working directory. \n")
            chara = input("Continue (y) or Exit (n):  \n")
            flag = (chara == 'y' or chara == 'Y')

    # ------------------------------------------------------------------------------------------------------------------
    else:
        exit(0)
# ----------------------------------------------------------------------------------------------------------------------
# Convert the label file: label.txt
elif type_con == 2:
    label_dat = np.zeros(4 * 1, 'i4')  # label: results corresponding to .imd files.

    # read data from label file
    label_file = input("Input file name of label file: \n")
    if os.path.exists(label_file):
        label_file = open(label_file, 'r')  # open the label-data file
        label_dat = np.loadtxt(label_file)
        label_tensor = torch.from_numpy(label_dat)
    else:
        print("There is no label file in the directory, exit. \n")
        exit(0)

    # output data to label.imd
    proj_path = input("Input the directory of label output: \n")
    if os.sep == "/":  # linux platform
        file_path = proj_path + r'/label.imd'
    else:  # windows platform
        file_path = proj_path + r'\label.imd'
    torch.save(label_tensor, file_path)
else:
    exit(0)

