# ----------------------------------------------------------------------------------------------------------------------
# CNN_Model_Landslides.py
# -main program of CNN prediction model for landslide run-out distance, coverage area and impact force
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import os
from ParticleData_to_Images import particle_to_images
from RandomSamples_Generation import random_samples_generate_kl
from RandomSamples_Generation import random_samples_generate_midp
from CNN_Model_Functions import train_test_cnn_function
from CNN_Model_Functions import prediction_cnn_function
from CNN_Model_Functions import cross_validation_function


# ----------------------------------------------------------------------------------------------------------------------
# define the CNN model with pytorch


# ----------------------------------------------------------------------------------------------------------------------
# while loop
check = True
while check:
    # ------------------------------------------------------------------------------------------------------------------
    # the running mode: 1--converting particle data to *.imd files; 2--generating random samples of landslide; 3--cross
    # validation of hyperparameters;4--train and test of CNN model for landslides;5--prediction using trained CNN model
    print("Input the type of running mode:")
    print(" 1--converting particle data to *.imd files;")
    print(" 2--generating random samples of landslide;")
    print(" 3--cross validation of hyperparameters;")
    print(" 4--train and test of CNN model for landslides;")
    print(" 5--prediction using trained CNN model.")
    type_mode = int(input())
    # ------------------------------------------------------------------------------------------------------------------
    # running of each subroutine
    if type_mode == 1:
        particle_to_images()   # 1--converting particle data to *.imd files;
    elif type_mode == 2:
        # input the working folder
        proj_path = input("Please input the working folder:\n")
        # input the type of random field generating method
        gen_type = int(input("Input the type of random field generating method:1--KL expansion;2--Mid-point method \n"))
        if gen_type == 1:
            random_samples_generate_kl(proj_path)   # 2--generating random samples of landslide using KL expansion
        elif gen_type == 2:
            random_samples_generate_midp(proj_path)   # 2--generating random samples of landslide using KL expansion
        else:
            print("Incorrect type of random field generating method.")
    elif type_mode == 3:
        # input the working folder
        proj_path = input("Please input the working folder: \n")
        cross_validation_function(proj_path)
    elif type_mode == 4:
        # input the working folder
        proj_path = input("Please input the working folder: \n")
        train_test_cnn_function(proj_path)
    elif type_mode == 5:
        # input the working folder
        proj_path = input("Please input the working folder: \n")
        prediction_cnn_function(proj_path)
    else:
        print("The type of function type is incorrect! \n")
    # check if continue
    chara = input("Continue (y) or Exit (n) for main program:  \n")
    check = (chara == 'y' or chara == 'Y')
