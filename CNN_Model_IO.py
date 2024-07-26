# ----------------------------------------------------------------------------------------------------------------------
# CNN_Model_IO.py
# -input feature and label files from *.imd files for train and test of CNN model
# -input feature files from *.imd files for the prediction of landslides
# -output predicted results from the prediction of landslides
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import torch


# ----------------------------------------------------------------------------------------------------------------------
# loading feature and label tensors from imd files for training or cross validation
def tensor_load_fromfile(proj_path):
    # scanning the file list and write them in files array
    files = []
    for filename in os.listdir(proj_path):
        filepath = os.path.join(proj_path, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    # total loop of files for *.imd
    num_loop = int(0.5 * files.__len__())
    # read the feature and label imd files
    feature_data_temp = []
    label_data_temp = []
    for id_file in range(0, num_loop):
        feature_data_temp.append(torch.load(files[id_file]))
        label_data_temp.append(torch.load(files[id_file + num_loop]))
    # return
    feature_data = (torch.stack(feature_data_temp)).float()
    label_data = (torch.stack(label_data_temp)).float()
    return feature_data, label_data


# ----------------------------------------------------------------------------------------------------------------------
# loading feature and label tensors from imd files for test
def predict_tensor_fromfile(proj_path):
    # scanning the file list and write them in files array
    files = []
    for filename in os.listdir(proj_path):
        filepath = os.path.join(proj_path, filename)
        if os.path.isfile(filepath):
            files.append(filepath)
    # total loop of files for *.imd
    num_loop = int(0.5 * files.__len__())
    # read the feature and label imd files
    feature_data_temp = []
    for id_file in range(0, num_loop):
        feature_data_temp.append(torch.load(files[id_file]))
    # return
    feature_data = (torch.stack(feature_data_temp)).float()
    return feature_data
