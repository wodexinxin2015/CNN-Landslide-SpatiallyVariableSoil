# ----------------------------------------------------------------------------------------------------------------------
# ParticleData_to_Images.py
# -to convert the particle data file with spatially variable parameters to image data (*.imd)
# -to pad free area with additional data information in a rectangle that covers the landslide model
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import torch


# ------------------------------------------------------------------------------------------------------------------
def particle_to_images():
    parti_type_0 = np.zeros(4 * 1, 'i4')  # particle type: main type
    parti_x_0 = np.zeros(4 * 3, 'f4')  # x, y, z of particle coordinates
    parti_para_0 = np.zeros(4 * 4, 'f4')  # strength parameters: friction angle, cohesion; cop, ds
    x_id = np.zeros(3, "i4").reshape((3, 1))
    grid_dim = np.zeros(3, 'i4')

    flag = True
    no_file = 1
    while flag:
        proj_path = input("Please input the working folder: \n")
        # input the file folder
        if os.sep == "/":  # linux platform
            file_path = proj_path + r'/Parameters.txt'
            file_path_set = proj_path + r'/input.dat'
            label_file = proj_path + r'/label.txt'
        else:  # windows platform
            file_path = proj_path + r'\Parameters.txt'
            file_path_set = proj_path + r'\input.dat'
            label_file = proj_path + r'\label.txt'

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

            fldata_2 = particle_file.readlines()
            k_1 = 0
            for k in range(len(fldata_2) - 2):
                flline_k = fldata_2[k + 1].split()
                type_par = int(flline_k[8])
                if type_par > 0:
                    parti_x[k_1][0] = float(flline_k[1])
                    parti_x[k_1][1] = float(flline_k[2])
                    parti_x[k_1][2] = float(flline_k[3])
                    parti_para[k_1][0] = float(flline_k[4])
                    parti_para[k_1][1] = float(flline_k[5]) / 1000.0
                    parti_para[k_1][2] = float(flline_k[6])
                    parti_para[k_1][3] = float(flline_k[7])
                    k_1 = k_1 + 1
            # ----------------------------------------------------------------------------------------------------------
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
                file_feature_out = proj_path + r'/features-' + f"{no_file:0>4}" + r'.imd'
            else:  # Windows platform
                file_feature_out = proj_path + r'\\features-' + f"{no_file:0>4}" + r'.imd'
            torch.save(feat_ten, file_feature_out)

            if os.path.exists(label_file):
                label_ten = torch.from_numpy(np.loadtxt(label_file))
                if os.sep == "/":  # linux platform
                    file_feature_out = proj_path + r'/label-' + f"{no_file:0>4}" + r'.imd'
                else:  # Windows platform
                    file_feature_out = proj_path + r'\\label-' + f"{no_file:0>4}" + r'.imd'
                torch.save(label_ten, file_feature_out)
                print(grid_dim)
            else:
                return 1
            # ----------------------------------------------------------------------------------------------------------
            # Continue (y) or Exit (n)
            chara = input("Continue (y) or Exit (n) for the particle data to images:  \n")
            flag = (chara == 'y' or chara == 'Y')
            no_file = no_file + 1
