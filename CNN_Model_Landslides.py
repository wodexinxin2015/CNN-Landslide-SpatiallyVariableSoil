# ----------------------------------------------------------------------------------------------------------------------
# CNN_Model_Landslides.py
# -main program of CNN prediction model for landslide run-out distance, coverage area and impact force
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import os
from ParticleData_to_Images import particle_to_images
from RandomSamples_Generation import random_samples_generating
from CNN_Model_Functions import cross_validation_function
from CNN_Model_Functions import train_test_cnn_function
from CNN_Model_Functions import prediction_cnn_function


# ----------------------------------------------------------------------------------------------------------------------
# define the CNN model with pytorch

# ----------------------------------------------------------------------------------------------------------------------
# the running mode: 1--converting particle data to *.imd files; 2--generating random samples of landslide; 3--cross
# validation of hyperparameters; 4--train and test of CNN model for landslides; 5--prediction using trained CNN model
info_line = r"Input the type of running mode: \n 1--converting particle data to *.imd files; \n 2--generating random " \
            r"samples of landslide; \n 3--cross validation of hyperparameters; \n 4--train and test of CNN model for " \
            r"landslides; \n 5--prediction using trained CNN model. \n"
type_mode = int(input(info_line))
# ----------------------------------------------------------------------------------------------------------------------
# while loop
check = True
while check:
    # input the working folder
    proj_path = input("Please input the working folder: \n")
    # ----------------------------------------------------------------------------------------------------------------------
    # running of each subroutine
    if type_mode == 1:
        particle_to_images(proj_path)   # 1--converting particle data to *.imd files;
    elif type_mode == 2:
        random_samples_generating(proj_path)
    elif type_mode == 3:
        cross_validation_function(proj_path)
    elif type_mode == 4:
        train_test_cnn_function(proj_path)
    elif type_mode == 5:
        prediction_cnn_function(proj_path)
    else:
        print("The type of function type is incorrect! \n")
    # check if continue
    chara = input("Continue (y) or Exit (n):  \n")
    check = (chara == 'y' or chara == 'Y')
