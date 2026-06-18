import torch
from beta_VAE import Beta_VAE

# 1. Instanciation
model = Beta_VAE(latent_dim=16, beta=1.0, input_shape=(2, 101, 101))

# 2. Création d'une donnée factice (Batch de 4, 2 canaux, 101x101)
dummy_data = torch.randn(4, 2, 101, 101)

# 3. Test du forward
recon, mu, logvar = model(dummy_data)

# 4. Vérification
print(f"Entrée : {dummy_data.shape}")
print(f"Sortie : {recon.shape}")
print(f"Latent Mu : {mu.shape}")

assert recon.shape == dummy_data.shape, "Erreur : La sortie doit être identique à l'entrée !"
print("Succès : Le modèle fonctionne parfaitement !")
