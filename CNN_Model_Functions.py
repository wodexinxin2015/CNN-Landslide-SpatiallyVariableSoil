# ----------------------------------------------------------------------------------------------------------------------
# CNN_Model_Functions.py
# -functions in the train, test and prediction of CNN model of landslide run-out distance,coverage area dn impact force
# -Coded by Prof. Weijie Zhang, GeoHohai, Hohai University, Nanjing, China
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from CNN_Model_IO import tensor_load_fromfile
from CNN_Model_IO import predict_tensor_fromfile


# ----------------------------------------------------------------------------------------------------------------------
# Hyper parameters
class hyper_para:
    def __init__(self):
        # -------------------------------------------------------------------------------
        self.optim_type = 1  # the type of optimizer: 1--SGD; 2--Adam; 3--RMSprop; 4--Adagrad
        self.type_loss = 1  # the type of loss function: 1--mean squared error function;
        # 2--cross entropy error function; 3--PoissonNLLLoss function
        # -------------------------------------------------------------------------------
        self.batch_s = 4  # batch size
        self.learn_rate = 0.040  # learning rate
        # -------------------------------------------------------------------------------
        self.size_c = 4  # channel number
        # -------------------------------------------------------------------------------
        self.filter_n_1 = 6  # number of Filter No.1
        self.kernel_size_1 = 5  # size of Filter No.1
        self.stride_1 = 2  # stride steps of Filter No.1
        self.padding_1 = 2  # padding size of Filter No.1
        self.pool_size_1 = 5  # kernel size of pooling layer No.1
        self.pool_stride_1 = 1  # stride size of pooling layer No.1
        # -------------------------------------------------------------------------------
        self.filter_n_2 = 16  # number of Filter No.2
        self.kernel_size_2 = 10  # size of Filter No.2
        self.stride_2 = 1  # stride steps of Filter No.2
        self.padding_2 = 0  # padding size of Filter No.2
        self.pool_size_2 = 5  # kernel size of pooling layer No.2
        self.pool_stride_2 = 2  # stride size of pooling layer No.2
        # -------------------------------------------------------------------------------
        self.filter_n_3 = 10  # number of Filter No.3
        self.kernel_size_3 = 20  # size of Filter No.3
        self.stride_3 = 4  # stride steps of Filter No.3
        self.padding_3 = 0  # padding size of Filter No.3
        # -------------------------------------------------------------------------------
        self.linear_size_4 = 6000  # size of affine layer No.4
        self.linear_size_5 = 84  # size of affine layer No.5
        self.output_size = 1  # number of labels
        # -------------------------------------------------------------------------------
        self.epoch = 1000  # training epoches


# ----------------------------------------------------------------------------------------------------------------------
# define customized dataset class
class Landslide_Dataset(Dataset):
    def __init__(self, features, labels):
        self.data = features
        self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.targets[item]
        return x, y


# ----------------------------------------------------------------------------------------------------------------------
# define the CNN model with pytorch
class landslide_cnn(nn.Module):
    def __init__(self, size_c, filter_n_1, kernel_size_1, stride_1, padding_1, pool_size_1, pool_stride_1,
                 filter_n_2, kernel_size_2, stride_2, padding_2, pool_size_2, pool_stride_2,
                 filter_n_3, kernel_size_3, stride_3, padding_3, linear_size_4, linear_size_5, output_size):
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
            nn.Linear(linear_size_5, output_size), nn.Tanh(),
        )

    def forward(self, x):
        y = self.net(x)
        return y


# ----------------------------------------------------------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation loss decrease.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(model, 'finish_model.pkl')
        self.val_loss_min = val_loss


# ----------------------------------------------------------------------------------------------------------------------
def get_k_fold_data(k, i, train_feat, train_label):
    assert k > 1
    fold_size = len(train_feat) / k

    feat_train = None
    feat_valid = None
    label_train = None
    label_valid = None
    for j in range(k):
        idx = slice(int(j * fold_size), int((j + 1) * fold_size))
        # idx are the validation group
        feat_part = torch.FloatTensor(train_feat[idx, :])
        label_part = torch.FloatTensor(train_label[idx])
        if j == i:
            feat_valid = feat_part
            label_valid = label_part
        elif feat_train is None:
            feat_train = feat_part
            label_train = label_part
        else:
            feat_train = torch.cat([feat_train, feat_part], dim=0)
            label_train = torch.cat([label_train, label_part], dim=0)
    return feat_train, feat_valid, label_train, label_valid


# ----------------------------------------------------------------------------------------------------------------------
def k_fold(device_t, hyper_inst, k_k, train_feat, train_label):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0
    e_s_temp = 0
    # CNN class instance
    cnn_land_inst = landslide_cnn(hyper_inst.size_c, hyper_inst.filter_n_1, hyper_inst.kernel_size_1,
                                  hyper_inst.stride_1, hyper_inst.padding_1, hyper_inst.pool_size_1,
                                  hyper_inst.pool_stride_1, hyper_inst.filter_n_2, hyper_inst.kernel_size_2,
                                  hyper_inst.stride_2, hyper_inst.padding_2, hyper_inst.pool_size_2,
                                  hyper_inst.pool_stride_2, hyper_inst.filter_n_3, hyper_inst.kernel_size_3,
                                  hyper_inst.stride_3, hyper_inst.padding_3, hyper_inst.linear_size_4,
                                  hyper_inst.linear_size_5, hyper_inst.output_size).to(device_t)
    for i in range(k_k):
        feat_train, feat_valid, label_train, label_valid = get_k_fold_data(k_k, i, train_feat, train_label)
        data_train = torch.utils.data.TensorDataset(feat_train, label_train)
        data_valid = torch.utils.data.TensorDataset(feat_valid, label_valid)
        # train the network
        if device_t == torch.device("cuda"):
            torch.cuda.empty_cache()  # clean the cache
        train_loss, valid_loss, train_acc, valid_acc, e_stop = train_process(cnn_land_inst, device_t, data_train,
                                                                             data_valid, hyper_inst)
        train_loss_sum += train_loss[-1]
        valid_loss_sum += valid_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += valid_acc[-1]
        e_s_temp += e_stop

    print('\n', '#' * 10, 'Result of k-fold cross validation', '#' * 10)
    print('average train loss:{:.4f}, average train accuracy:{}'.format(train_loss_sum / k_k, train_acc_sum / k_k))
    print('average valid loss:{:.4f}, average valid accuracy:{}'.format(valid_loss_sum / k_k, valid_acc_sum / k_k))
    return train_loss_sum / k_k, valid_loss_sum / k_k, train_acc_sum / k_k, valid_acc_sum / k_k, e_s_temp / k_k


# ----------------------------------------------------------------------------------------------------------------------
def train_process(net, device_t, data_train, data_valid, hyper_inst):
    train_acc, valid_acc = [], []
    train_loss, valid_loss = [], []

    # define the optimizer
    if hyper_inst.optim_type == 1:  # using the Stochastic Gradient Descent method
        optimizer = torch.optim.SGD(net.parameters(), hyper_inst.learn_rate)
    elif hyper_inst.optim_type == 2:  # using the Adam method
        optimizer = torch.optim.Adam(net.parameters(), hyper_inst.learn_rate)
    elif hyper_inst.optim_type == 3:  # using the RMSprop method
        optimizer = torch.optim.RMSprop(net.parameters(), hyper_inst.learn_rate)
    elif hyper_inst.optim_type == 4:  # using the Adagrad method
        optimizer = torch.optim.Adagrad(net.parameters(), hyper_inst.learn_rate)
    else:  # using the Adagrad method
        optimizer = torch.optim.Adagrad(net.parameters(), hyper_inst.learn_rate)

    # define the loss function
    if hyper_inst.type_loss == 1:
        criterion = nn.MSELoss(reduction='mean').to(device_t)  # mean squared error function
    elif hyper_inst.type_loss == 2:
        criterion = nn.CrossEntropyLoss().to(device_t)  # cross entropy error function
    elif hyper_inst.type_loss == 3:
        criterion = nn.PoissonNLLLoss().to(device_t)  # PoissonNLLLoss function
    else:
        criterion = nn.PoissonNLLLoss().to(device_t)  # PoissonNLLLoss function
    # define the early stopping function
    early_stopper = EarlyStopping(10)
    acc_fun = tm.R2Score(multioutput='raw_values').to(device_t)

    stop = 0
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=hyper_inst.batch_s, shuffle=True)
    for epoch in range(hyper_inst.epoch):
        # train process with mini-batch training
        for (data, label) in enumerate(train_loader):
            # to device
            data, label = data.to(device_t), label.to(device_t)
            # the result of forward process
            output = net.forward(data)
            # loss value
            loss = criterion(output, label)
            # back propagation: all the gradients are zero before back propagation.
            optimizer.zero_grad()
            loss.backward()
            # update parameters
            optimizer.step()
            # delete the tensor object and clean the cache
            if device_t == torch.device("cuda"):
                del data, label, output
                torch.cuda.empty_cache()  # clean the cache

        # the accuracy of train set
        train_loader_1 = torch.utils.data.DataLoader(data_train, batch_size=len(data_train), shuffle=False)
        score_per = torch.zeros(hyper_inst.output_size).reshape([1, hyper_inst.output_size])
        for (data_1, label_1) in enumerate(train_loader_1):
            # the result of forward process
            data_1, label_1 = data_1.to(device_t), label_1.to(device_t)
            output_1 = net.forward(data_1)
            train_loss.append(criterion(output_1, label_1).cpu().detach().numpy())
            if hyper_inst.output_size == 1:
                score_per += acc_fun(torch.squeeze(label_1), torch.squeeze(output_1)).cpu()
            else:
                score_per += acc_fun(label_1, output_1).cpu()
            # delete the tensor object and clean the cache
            if device_t == torch.device("cuda"):
                del data_1, label_1, output_1
                torch.cuda.empty_cache()  # clean the cache
        train_acc.append(score_per.detach().numpy() / len(train_loader_1))

        # the accuracy of valid set
        valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=len(data_valid), shuffle=False)
        score_per = torch.zeros(hyper_inst.output_size).reshape([1, hyper_inst.output_size])
        for (data_2, label_2) in enumerate(valid_loader):
            # the result of forward process
            data_2, label_2 = data_2.to(device_t), label_2.to(device_t)
            output_2 = net.forward(data_2)
            valid_loss.append(criterion(output_2, label_2).cpu().detach().numpy())
            if hyper_inst.output_size == 1:
                score_per += acc_fun(torch.squeeze(label_2), torch.squeeze(output_2)).cpu()
            else:
                score_per += acc_fun(label_2, output_2).cpu()
            # delete the tensor object and clean the cache
            if device_t == torch.device("cuda"):
                del data_2, label_2, output_2
                torch.cuda.empty_cache()  # clean the cache
        valid_acc.append(score_per.detach().numpy() / len(valid_loader))
        stop = epoch

        # early stopping
        early_stopper(valid_loss[-1], net)
        if early_stopper.early_stop:
            break

    return train_loss, valid_loss, train_acc, valid_acc, stop


# ----------------------------------------------------------------------------------------------------------------------
# cross validation function to determine hyperparameters
def cross_validation_function(proj_path_train, device_type):
    k_k = 10
    hyper_inst = hyper_para()
    # load tensor data from files
    feature_data_train, label_data_train = tensor_load_fromfile(proj_path_train)  # train features and labels from files
    t_loss_1, v_loss_1, t_acc_1, v_acc_1, e_s_1 = k_fold(device_type, hyper_inst, k_k, feature_data_train,
                                                         label_data_train)
    t_loss_2, v_loss_2, t_acc_2, v_acc_2, e_s_2 = k_fold(device_type, hyper_inst, k_k, feature_data_train,
                                                         label_data_train)
    t_loss_3, v_loss_3, t_acc_3, v_acc_3, e_s_3 = k_fold(device_type, hyper_inst, k_k, feature_data_train,
                                                         label_data_train)
    t_loss_4, v_loss_4, t_acc_4, v_acc_4, e_s_4 = k_fold(device_type, hyper_inst, k_k, feature_data_train,
                                                         label_data_train)
    v_acc = (v_acc_1 + v_acc_2 + v_acc_3 + v_acc_4) * 0.25
    t_loss = (t_loss_2 + t_loss_2 + t_loss_3 + t_loss_4) * 0.25
    v_loss = (v_loss_2 + v_loss_2 + v_loss_3 + v_loss_4) * 0.25
    t_acc = (t_acc_1 + t_acc_2 + t_acc_3 + t_acc_4) * 0.25
    e_s = (e_s_1 + e_s_2 + e_s_3 + e_s_4) * 0.25
    # print the result
    print("----------------------------------------")
    print("The best set of hyperparameters:")
    print('learn_rate:{:.4f}, batch_size:{:2d}'.format(hyper_inst.learn_rate, hyper_inst.batch_s))
    print('early stopping epoch:{:.1f}'.format(e_s))
    print('average train loss:{:.4f}, average train accuracy:{}'.format(t_loss, t_acc))
    print('average valid loss:{:.4f}, average valid accuracy:{}'.format(v_loss, v_acc))


# ----------------------------------------------------------------------------------------------------------------------
# train and test function for CNN model
def train_test_cnn_function(proj_path_train, proj_path_test, device_type):
    hyper_inst = hyper_para()
    # type of loss function
    if hyper_inst.type_loss == 1:  # mse loss function
        loss_func = nn.MSELoss(reduction='mean').to(device_type)
    elif hyper_inst.type_loss == 2:  # cross entropy loss function
        loss_func = nn.CrossEntropyLoss().to(device_type)
    else:  # other loss function
        loss_func = nn.GaussianNLLLoss().to(device_type)
    # -------------------------------------------------------------------------------
    # load tensor data from files
    feature_data_train, label_data_train = tensor_load_fromfile(proj_path_train)  # train features and labels from files
    feature_data_test, label_data_test = tensor_load_fromfile(proj_path_test)  # test features and labels from files
    # assemble datasets for train and test
    train_dataset = Landslide_Dataset(feature_data_train, label_data_train)
    test_dataset = Landslide_Dataset(feature_data_test, label_data_test)
    # CNN class instance
    cnn_land_inst = landslide_cnn(hyper_inst.size_c, hyper_inst.filter_n_1, hyper_inst.kernel_size_1,
                                  hyper_inst.stride_1, hyper_inst.padding_1, hyper_inst.pool_size_1,
                                  hyper_inst.pool_stride_1, hyper_inst.filter_n_2, hyper_inst.kernel_size_2,
                                  hyper_inst.stride_2, hyper_inst.padding_2, hyper_inst.pool_size_2,
                                  hyper_inst.pool_stride_2, hyper_inst.filter_n_3, hyper_inst.kernel_size_3,
                                  hyper_inst.stride_3, hyper_inst.padding_3, hyper_inst.linear_size_4,
                                  hyper_inst.linear_size_5, hyper_inst.output_size).to(device_type)
    # optimizer
    # define the optimizer
    if hyper_inst.optim_type == 1:  # using the Stochastic Gradient Descent method
        optimizer = torch.optim.SGD(cnn_land_inst.parameters(), hyper_inst.learn_rate)
    elif hyper_inst.optim_type == 2:  # using the Adam method
        optimizer = torch.optim.Adam(cnn_land_inst.parameters(), hyper_inst.learn_rate)
    elif hyper_inst.optim_type == 3:  # using the RMSprop method
        optimizer = torch.optim.RMSprop(cnn_land_inst.parameters(), hyper_inst.learn_rate)
    elif hyper_inst.optim_type == 4:  # using the Adagrad method
        optimizer = torch.optim.Adagrad(cnn_land_inst.parameters(), hyper_inst.learn_rate)
    else:
        optimizer = torch.optim.AdamW(cnn_land_inst.parameters(), hyper_inst.learn_rate)
    # error function
    accuracy_fun_1 = tm.PearsonCorrCoef().to(device_type)
    accuracy_fun_2 = tm.R2Score().to(device_type)
    # training process
    loss_holder_train = []
    loss_holder_test = []
    simu_score_train = []
    simu_score_test = []
    loss_value = np.inf
    # dataloaders of train and test for the calculation of Model accuracy
    train_loader_all = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader_all = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    # train loop
    for epoch in range(hyper_inst.epoch):
        # form the data loader
        data_load_train = DataLoader(train_dataset, hyper_inst.batch_s, shuffle=True)
        for (data, targets) in data_load_train:  # get x and y from mini batch
            x, y = data.to(device_type), targets.to(device_type)  # put minibatch on device
            pred = cnn_land_inst.forward(x)  # forward propagation
            loss = loss_func(pred, y)  # calculate the loss value
            optimizer.zero_grad()  # clear the previous gradients
            loss.backward()  # backward propagation
            optimizer.step()  # optimization of internal parameters
            # delete the tensor object and clean the cache
            if device_type == torch.device("cuda"):
                del x, y, pred
                torch.cuda.empty_cache()  # clean the cache

            # calculating the similarity score for the training data
            for (data_2, label_2) in enumerate(train_loader_all):
                # the result of forward process
                data_2, label_2 = data_2.to(device_type), label_2.to(device_type)
                output_2 = cnn_land_inst.forward(data_2)
                if hyper_inst.output_size == 1:
                    score_train_per = accuracy_fun_1(torch.squeeze(output_2), torch.squeeze(label_2)).cpu()
                    score_train_cos = accuracy_fun_2(torch.squeeze(output_2), torch.squeeze(label_2)).cpu()
                else:
                    score_train_per = accuracy_fun_1(output_2, label_2).cpu()
                    score_train_cos = accuracy_fun_2(output_2, label_2).cpu()
                loss_1 = loss_func(output_2, label_2)
                loss_holder_train.append([epoch, loss_func(output_2, label_2).cpu().detach().numpy()])
                if loss_1 < loss_value:
                    torch.save(cnn_land_inst.state_dict(), '0-model.pt')
                    loss_value = loss_1
                simu_score_train.append([epoch, score_train_per.detach().numpy(), score_train_cos.detach().numpy()])
                # delete the tensor object and clean the cache
                if device_type == torch.device("cuda"):
                    del data_2, label_2, output_2
                    torch.cuda.empty_cache()  # clean the cache

            # calculating the similarity score for the testing data
            for (data_3, label_3) in enumerate(test_loader_all):
                # the result of forward process
                data_3, label_3 = data_3.to(device_type), label_3.to(device_type)
                output_3 = cnn_land_inst.forward(data_3)
                loss_holder_test.append([epoch, loss_func(output_3, label_3).cpu().detach().numpy()])
                if hyper_inst.output_size == 1:
                    score_test_per = accuracy_fun_1(torch.squeeze(output_3), torch.squeeze(label_3)).cpu()
                    score_test_cos = accuracy_fun_2(torch.squeeze(output_3), torch.squeeze(label_3)).cpu()
                else:
                    score_test_per = accuracy_fun_1(output_3, label_3).cpu()
                    score_test_cos = accuracy_fun_2(output_3, label_3).cpu()
                simu_score_test.append([epoch, score_test_per.detach().numpy(), score_test_cos.detach().numpy()])
                # delete the tensor object and clean the cache
                if device_type == torch.device("cuda"):
                    del data_3, label_3, output_3
                    torch.cuda.empty_cache()  # clean the cache
    # plot the relationship between loss_value and iteration step
    loss_df_1 = pd.DataFrame(loss_holder_train, columns=['step', 'loss'])
    loss_df_2 = pd.DataFrame(loss_holder_test, columns=['step', 'loss'])
    plt.plot(loss_df_1['loss'].values, 'go', markersize=2)
    plt.plot(loss_df_2['loss'].values, 'mo', markersize=2)
    plt.xlabel('Iteration step')
    plt.ylabel('Loss')
    plt.show()
    # save training data in the log file
    score_train_df = pd.DataFrame(simu_score_train, columns=['step', 'score-per', 'score-r2'])
    score_test_df = pd.DataFrame(simu_score_test, columns=['step', 'score-per', 'score-r2'])
    loss_df_1.to_csv('1-loss-process_train.txt', sep='\t', index=False)
    loss_df_2.to_csv('1-loss-process_test.txt', sep='\t', index=False)
    score_train_df.to_csv('1-score-train-process.txt', sep='\t', index=False)
    score_test_df.to_csv('1-score-test-process.txt', sep='\t', index=False)


# ----------------------------------------------------------------------------------------------------------------------
# prediction function using the trained CNN model
def prediction_cnn_function(proj_path, device_type):
    feature_data = predict_tensor_fromfile(proj_path)  # load the feature tensor from file
    # initialize the hyperparameters
    hyper_inst = hyper_para()
    # CNN class instance
    cnn_land_inst = landslide_cnn(hyper_inst.size_c, hyper_inst.filter_n_1, hyper_inst.kernel_size_1,
                                  hyper_inst.stride_1, hyper_inst.padding_1, hyper_inst.pool_size_1,
                                  hyper_inst.pool_stride_1, hyper_inst.filter_n_2, hyper_inst.kernel_size_2,
                                  hyper_inst.stride_2, hyper_inst.padding_2, hyper_inst.pool_size_2,
                                  hyper_inst.pool_stride_2, hyper_inst.filter_n_3, hyper_inst.kernel_size_3,
                                  hyper_inst.stride_3, hyper_inst.padding_3, hyper_inst.linear_size_4,
                                  hyper_inst.linear_size_5, hyper_inst.output_size).to(device_type)
    # load the model state from file
    cnn_land_inst.load_state_dict(torch.load('0-model.pt'))
    # predict the labels using the forward process
    label_out = cnn_land_inst.forward(feature_data.to(device_type))
    # write results
    if os.sep == "/":  # linux platform
        file_label_out = proj_path + r'/Predicted-labels.txt'
    else:  # Windows platform
        file_label_out = proj_path + r'\\Predicted-labels.txt'
    np.savetxt(file_label_out, label_out.detach().numpy(), fmt='%.6f', delimiter=',')
