""" 
    An Implementation of Variational Deep Embedding
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class VaDE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, intermediate_dim):
        super(VaDE, self).__init__()
        
        # GMM Parameters Initialization
        self.pi_p = Parameter(torch.ones(num_classes)/num_classes)
        self.mu_p = Parameter(torch.zeros(num_classes, latent_dim))
        self.log_var_p = Parameter(torch.randn(latent_dim, num_classes))

        # Encoder
        self.fc1 = nn.Linear(input_dim, intermediate_dim[0])
        self.fc2 = nn.Linear(intermediate_dim[0], intermediate_dim[1])
        self.fc3 = nn.Linear(intermediate_dim[1], intermediate_dim[2])

        # Latent z_mean
        self.z_mean = nn.Linear(intermediate_dim[2], latent_dim)
        # Latent z_log_var
        self.z_log_var = nn.Linear(intermediate_dim[2], latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, intermediate_dim[-1])
        self.fc5 = nn.Linear(intermediate_dim[-1], intermediate_dim[-2])
        self.fc6 = nn.Linear(intermediate_dim[-2], intermediate_dim[-3])
        self.fc7 = nn.Linear(intermediate_dim[-3], input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.z_mean(h), self.z_log_var(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        # this has to change based on dataset - fix
        return F.sigmoid(self.fc7(h))
 
    def sampling(self, z_mean, z_log_var):
        std = torch.exp(z_log_var/2)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.sampling(z_mean, z_log_var)
        x_hat = self.decode(z)
        return x_hat, z_mean, z_log_var, z


class Autoencoder(torch.nn.module):
    def __init__(self, input_dim, latent_dim, intermediate_dim):
        super(Autoencoder, self).__self__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, intermediate_dim[0])
        self.fc2 = nn.Linear(intermediate_dim[0], intermediate_dim[1])
        self.fc3 = nn.Linear(intermediate_dim[1], intermediate_dim[2])

        # Latent
        self.mean = nn.Linear(intermediate_dim[2], latent_dim)
        
        # Decoder
        self.fc4 = nn.Linear(latent_dim, intermediate_dim[-1])
        self.fc5 = nn.Linear(intermediate_dim[-1], intermediate_dim[-2])
        self.fc6 = nn.Linear(intermediate_dim[-2], intermediate_dim[-3])
        self.fc7 = nn.Linear(intermediate_dim[-3], input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.mean(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        # same comment as before this should change based on dataset(why?)
        return F.sigmoid(self.fc7(h))

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
