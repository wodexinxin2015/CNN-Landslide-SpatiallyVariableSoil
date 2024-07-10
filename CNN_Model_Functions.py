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
    def __init__(self, size_c,  filter_n_1, kernel_size_1, stride_1, padding_1, pool_size_1, pool_stride_1,
                 filter_n_2, kernel_size_2, stride_2, padding_2, pool_size_2, pool_stride_2,
                 filter_n_3, kernel_size_3, stride_3, padding_3, linear_size_4, linear_size_5):
        super(landslide_cnn, self).__init__()
        self.net = nn.Sequential(
            # Convolution layer 1
            nn.Conv2d(size_c, filter_n_1, kernel_size_1, stride_1, padding_1), nn.Tanh(),
            # Average pooling layer 1
            nn.AvgPool2d(pool_size_1, pool_stride_1),
            # Convolution layer 2
            nn.Conv2d(filter_n_1, filter_n_2, kernel_size_2, stride_2, padding_2), nn.Tanh(),
            # Average pooling layer 2
            nn.AvgPool2d(pool_size_2, pool_stride_2),
            # Convolution layer 3
            nn.Conv2d(filter_n_2, filter_n_3, kernel_size_3, stride_3, padding_3), nn.Tanh(),
            # convert the img to one-dimensional array
            nn.Flatten(),
            # Affine layer-1 with dropout of neurons
            nn.Linear(linear_size_4, linear_size_5), nn.Tanh(),
            nn.Dropout(0.5),
            # Affine layer-2 to get the result
            nn.Linear(linear_size_5, 1), nn.Tanh(),
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

