# Fonction d'entrainement de model

import torch
from torch.utils.data import DataLoader
from beta_VAE import Beta_VAE
from dataset import FluidDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Travaille lancer sur {device}")

batch_size = 32
epochs = 50
learning_rate = 1e-4

data_path = "../data/data.h5"
dataset_name = "UV"

dataset_train = FluidDataset(file_path = data_path, dataset_name = dataset_name, split = "train") # Changer le nom du dataset ici
dataset_val = FluidDataset(file_path = data_path, dataset_name = dataset_name, split = "val") # Changer le nom du dataset ici
dataset_test = FluidDataset(file_path = data_path, dataset_name = dataset_name, split = "test") # Changer le nom du dataset ici

train_dataloader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
val_dataloader = DataLoader(dataset_val, batch_size = batch_size, shuffle = False, num_workers = 4)
test_dataloader = DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 4)

model = Beta_VAE(latent_dim = 2, beta = 0.005, input_shape = (2, 88, 300)).to(device) # modifier input_shape en fonction des besoins
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs) :
    model.train()
    total_train_loss = 0
    total_train_mse = 0
    total_train_kld = 0

    for batch in train_dataloader :
        batch = batch.to(device)

        #forward
        recon, mu, logvar = model(batch)

        #loss
        loss, mse, kld = model.loss_function(recon, batch, mu, logvar)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_mse += mse.item()
        total_train_kld += kld.item()

    avg_loss = total_train_loss / len(train_dataloader)
    avg_mse = total_train_mse / len(train_dataloader)
    avg_kld = total_train_kld / len(train_dataloader)

    
    model.eval()
    total_val_loss = 0
    with torch.no_grad() :
        for batch in val_dataloader :
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, mse, kld = model.loss_function(recon, batch, mu, logvar)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f}")


print("Évaluation sur le set de test...")
total_test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        loss, _, _ = model.loss_function(recon, batch, mu, logvar)
        total_test_loss += loss.item()

print(f"Test Loss finale: {total_test_loss/len(test_loader):.4f}")
torch.save(model.state_dict(), "../checkpoints/beta_vae_final.pth")
