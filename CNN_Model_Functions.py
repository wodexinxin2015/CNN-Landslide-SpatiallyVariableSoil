# ----------------------------------------------------------------------------------------------------------------------
# CNN_Model_Functions.py
# -functions in the train, test and prediction of CNN model of landslide run-out distance,coverage area dn impact force
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import os
import scipy as sp


# ----------------------------------------------------------------------------------------------------------------------
# define the CNN model with pytorch
class landslide_cnn(nn.Module):
    def __init__(self, size_c,  kernel_size_1, pool_size_1, kernel_size_2, pool_size_2, kernel_size_3,
                 kernel_size_4, kernel_size_5):
        super(landslide_cnn, self).__init__()
        self.net = nn.Sequential(
            # Convolution layer 1 with (FN, C, FH, FW) = (size_c * 4, size_c, kernel_size_1, kernel_size_1)
            # o_ch = size_c * 4,
            # o_h = (size_y + 2 * 1 - kernel_size_1) / 1 + 1
            # o_w = (size_x + 2 * 1 - kernel_size_1) / 1 + 1
            nn.Conv2d(size_c, size_c * 4, kernel_size=kernel_size_1, padding=1), nn.Tanh(),
            # Average pooling layer
            # o_ch_1 = size_c * 4
            # o_h_1 = o_h / 2
            # o_w_1 = o_w / 2
            nn.AvgPool2d(kernel_size=pool_size_1, stride=2),
            # Convolution layer 2 with (FN, C, FH, FW) = (size_c * 4, size_c * 4, kernel_size_2, kernel_size_2)
            # o_ch_2 = size_c * 4
            # o_h_2 = (o_h_1 + 2 * 1 - kernel_size_2) / 1 + 1
            # o_w_2 = (o_w_1 + 2 * 1 - kernel_size_2) / 1 + 1
            nn.Conv2d(size_c * 4, size_c * 4, kernel_size=kernel_size_2, padding=1), nn.Tanh(),
            # Average pooling layer
            # o_ch_3 = size_c * 4
            # o_h_3 = o_h_2 / 2
            # o_w_3 = o_w_2 / 2
            nn.AvgPool2d(kernel_size=pool_size_2, stride=2),
            # Convolution layer 3 with (FN, C, FH, FW) = (1, size_c * 4, kernel_size_3, kernel_size_3)
            # o_ch_4 = size_c * 4
            # o_h_4 = 1
            # o_w_4 = 1
            nn.Conv2d(size_c * 4, 1, kernel_size=kernel_size_3, padding=1), nn.Tanh(),
            # convert the img to one-dimensional array
            nn.Flatten(),
            # Affine layer-1 with dropout of neurons
            nn.Linear(kernel_size_4, kernel_size_5), nn.Tanh(),
            nn.Dropout(0.5),
            # Affine layer-2 to get the result
            nn.Linear(kernel_size_5, 1), nn.Tanh(),
        )

    def forward(self, x):
        y = self.net(x)
        return y


# ----------------------------------------------------------------------------------------------------------------------
# cross validation function to determine hyperparameters
def cross_validation_function(proj_path):
    print("")


# ----------------------------------------------------------------------------------------------------------------------
# train and test function for CNN model
def train_test_cnn_function(proj_path):
    print("")


# ----------------------------------------------------------------------------------------------------------------------
# prediction function using the trained CNN model
def prediction_cnn_function(proj_path):
    print("")

