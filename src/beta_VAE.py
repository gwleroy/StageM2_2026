import torch
import torch.nn as nn
import torch.nn.functional as F

class Beta_VAE(nn.Module) :
    def __init__(self, latent_dim = 2, beta = 0.005) :
        super(Beta_VAE, self), __init__() 
        self.latent_dim = latent_dim
        self.beta = beta

        # Encodeur
        self.encodeur = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1),
                nn.ELU(), # on peut également mettre nn.RELU, c'est ici que l'on choisi la fonction d'activation

                nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Flatten()
        )

        self.fc_mu = nn.linear(18432, latent_dim)
        self.fc_logvar = nn.linear(18432, latent_dim)






