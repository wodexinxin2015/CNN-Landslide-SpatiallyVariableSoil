# ----------------------------------------------------------------------------------------------------------------------
# CNN_Model_Landslides.py
# -main program of CNN prediction model for landslide run-out distance, coverage area and impact force
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import torch
from ParticleData_to_Images import particle_to_images
from RandomSamples_Generation import random_samples_generate_kl
from CNN_Model_Functions import train_test_cnn_function
from CNN_Model_Functions import prediction_cnn_function
from CNN_Model_Functions import cross_validation_function


# ----------------------------------------------------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
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
        print("Particles to images has finished.\n")
    elif type_mode == 2:
        # input the working folder
        proj_path = input("Please input the working folder:\n")
        # input the type of random field generating method
        random_samples_generate_kl(proj_path)   # 1--generating random samples of landslide using KL expansion
        print("Generation of random fields has finished.\n")
    elif type_mode == 3:
        # input the working folder
        proj_path_train = input("Please input the train folder: \n")
        proj_path_test = input("Please input the test folder: \n")
        cross_validation_function(proj_path_train, device)
        print("K-fold cross validation has finished.\n")
    elif type_mode == 4:
        # input the working folder
        proj_path_train = input("Please input the train folder: \n")
        proj_path_test = input("Please input the test folder: \n")
        train_test_cnn_function(proj_path_train, proj_path_test, device)
        print("Train and test of CNN model has finished.\n")
    elif type_mode == 5:
        # input the working folder
        proj_path = input("Please input the working folder for prediction: \n")
        prediction_cnn_function(proj_path, device)
        print("Prediction using the trained model has finished.\n")
    else:
        print("The type of function type is incorrect! \n")
    # check if continue
    chara = input("Continue (y) or Exit (n) for main program:  \n")
    check = (chara == 'y' or chara == 'Y')
