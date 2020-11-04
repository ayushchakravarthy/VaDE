import sys
import numpy as np
import torch

from utils import load_data
from utils import config_init


if __name__ == '__main__':
    dataset = 'none'
    assert len(sys.argv) == 2
    arg = sys.argv[1]
    if arg in ['mnist', 'reuters10k', 'har']:
        dataset = arg
    else:
        print("Enter a valid dataset\n")
    ispretrain = False
    # copied over from the DEC paper
    batch_size = 100
    latent_dim = 10
    intermediate_dim = [500, 500, 2000]
    torch.set_default.dtype('float32')
    accuracy = []
    X, Y = load_data(dataset)
    input_dim, epoch, n_classes, lr_nn, lr_gmm, decay_n, decay_nn, decay_gmm, alpha, data_type = config_init(dataset)
    
