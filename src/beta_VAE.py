import torch
import torch.nn as nn
import torch.nn.functional as F

class Beta_VAE(nn.Module) :
    def __init__(self, latent_dim = 2, beta = 0.005, input_shape=(2, 101, 101)) :
        super(Beta_VAE, self).__init__() 
        self.latent_dim = latent_dim
        self.beta = beta

        # Encodeur
        self.encoder_conv = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1),
                nn.ELU(), # on peut également mettre nn.RELU, c'est ici que l'on choisi la fonction d'activation

                nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ELU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
        )

        self.pre_flatten_shape, self.flatten_size = self._get_shapes(input_shape)

        self.flatten = nn.Flatten()
        
        # Couches latentes
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)


        self.decoder_input =nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
                nn.Unflatten(1, self.pre_flatten_shape),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConstantPad2d((0, 0, 0, -1), 0),
                nn.ELU(),

                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConstantPad2d((0, 0, 0, -1), 0),
                nn.ELU(),

                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConstantPad2d((0, -1, 0, 0), 0),
                nn.ELU(),

                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConstantPad2d((0, -1, 0, 0), 0),
                nn.ELU(),

                nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConstantPad2d((0, -1, 0, 0), 0),
                nn.ELU(),

                nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConstantPad2d((0, -1, 0, 0), 0),
                nn.ELU(),

        )

    def _get_shapes(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.encoder_conv(dummy)
            # Retourne la forme (C, H, W) ET la taille aplatie
            return x.shape[1:], x.numel()

    def forward(self, x) : 
        original_size = x.shape[2:]

        # encode
        h = self.encoder_conv(x)
        h = self.flatten(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # espace latent
        z = self.reparametrization(mu, logvar)

        # decode
        out = self.decoder_input(z)
        recon = self.decoder(out)

        recon = F.interpolate(recon, size=original_size, mode='bilinear', align_corners=False)

        return recon[:, :, :x.size(2), :x.size(3)], mu, logvar

    def reparametrization(self, mu, logvar) :
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def loss_function(self, recon_x, x, mu, logvar):
        
        # On utilise mean pour que la loss soit stable quel que soit le batch_size
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # Divergence KL
        total_loss = recon_loss + (self.beta * kld_loss) # Perte totale
        
        return total_loss, recon_loss, kld_loss
