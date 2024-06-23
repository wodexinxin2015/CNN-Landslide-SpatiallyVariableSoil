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

# ----------------------------------------------------------------------------------------------------------------------
# the running mode: 1--converting particle data to *.imd files; 2--generating random samples of landslide; 3--cross
# validation of hyperparameters; 4--train and test of CNN model for landslides; 5--prediction using trained CNN model
info_line = r"Input the type of running mode: \n 1--converting particle data to *.imd files; \n 2--generating random " \
            r"samples of landslide; \n 3--cross validation of hyperparameters; \n 4--train and test of CNN model for " \
            r"landslides; \n 5--prediction using trained CNN model. \n"
type_mode = int(input(info_line))
# ----------------------------------------------------------------------------------------------------------------------
# input the working folder


