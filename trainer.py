import math
import torch
from torch import optim
import torch.nn.functional as F


from models import Autoencoder, VaDE

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Trainer:
    def __init__(self, args, device, dataloader):
        self.autoencoder = Autoencoder().to(device)
        self.VaDE = VaDE().to(device)
        self.dataloader = dataloader
        self.device = device
        self.args = args

    def pretrain(self):
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.002)
        self.autoencoder.apply(weights_init)
        self.autoencoder.train()
        print('Training autoencoder...')
        for epoch in range(30):
            total_loss = 0
            for x, _ in self.dataloader:
                optimizer.zero_grad()
                x = x.to(self.device)
                x_hat = self.autoencoder(x)
                loss = F.binary_cross_entropy(x_hat, x, reduction='mean')
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('Autoencoder...Epoch: {}, Loss: {}'.format(epoch, total_loss))
        self.train_GMM()
        self.save_weights_for_VaDE()

    def train_GMM(self):
        print('Fitting GMM')

