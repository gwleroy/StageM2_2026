import torch
from beta_VAE import Beta_VAE

# Instanciation
model = Beta_VAE(latent_dim=16, beta=1.0, input_shape=(2, 101, 101))

# Création d'une donnée factice (Batch de 4, 2 canaux, 101x101)
dummy_data = torch.randn(4, 2, 101, 101)

# Test du forward
recon, mu, logvar = model(dummy_data)

# Vérification
print(f"Entrée : {dummy_data.shape}")
print(f"Sortie : {recon.shape}")
print(f"Latent Mu : {mu.shape}")

assert recon.shape == dummy_data.shape, "Erreur : La sortie doit être identique à l'entrée !"
print("Succès : Le modèle fonctionne parfaitement !")


from dataset import FluidDataset

# we build the three set
train = FluidDataset("../data/DataRe40.hdf5", "UV", split="train", mode="stride")
test = FluidDataset("../data/DataRe40.hdf5", "UV", split="test", mode="stride")
val = FluidDataset("../data/DataRe40.hdf5", "UV", split="val", mode="stride")

# Verifying if their is no intersection
set_train = set(train.indices)
set_test = set(test.indices)
intersection = set_train.intersection(set_test)

print(f"Intersection entre Train et Test : {len(intersection)} indices.")
