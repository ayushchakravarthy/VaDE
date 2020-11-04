import numpy as np
import gzip
import sys
import cPickle
import scipy.io as scio
import torch
from torch.nn.parameter import Parameter


def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max)+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/Y_pred.size, w


def load_data(dataset):
    path = 'dataset/' + dataset + '/'
    if dataset == 'mnist':
        path = path + 'mnist.pkl.gz'
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        # checking for python version related issues
        if sys.version_info < (3,):
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")

        f.close()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))

    if dataset == 'reuters10k':
        data = scio.loadmat(path+'reuters10k.mat')
        X = data['X']
        y = data['Y'].squeeze()

    if dataset == 'har':
        data = scio.loadmat(path+'HAR.mat')
        X = data['X']
        X = X.astype('float32')
        y = data['Y'] - 1
        X = X[:10200]
        y = y[:10200]

    return X, y


def config_init(dataset):
    if dataset == 'mnist':
        return 784, 3000, 10, 0.002, 0.002, 10, 0.9, 0.9, 1, 'sigmoid'
    if dataset == 'reuters10k':
        return 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1, 'linear'
    if dataset == 'har':
        return 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5, 'linear'
