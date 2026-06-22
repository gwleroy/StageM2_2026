# Fonction d'entrainement de model
import os
import torch
from torch.utils.data import DataLoader
from src.beta_VAE import Beta_VAE
from src.dataset import FluidDataset

def run_training(config) :
    if not os.path.exists(config["data_path"]) : 
        raise FileNotFoundError(f"Data files not found at : {config['data_path']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training start on {device} with configuration : {config}")
    
    
    dataset_train = FluidDataset(file_path = config["data_path"], dataset_name = config["dataset_name"], split = "train", mode = "stride")
    dataset_val = FluidDataset(file_path = config["data_path"], dataset_name = config["dataset_name"], split = "val", mode = "stride") 
    dataset_test = FluidDataset(file_path = config["data_path"], dataset_name = config["dataset_name"], split = "test", mode = "stride")

    train_dataloader = DataLoader(dataset_train, batch_size = config["batch_size"], shuffle = True, num_workers = 4)
    val_dataloader = DataLoader(dataset_val, batch_size = config["batch_size"], shuffle = False, num_workers = 4)
    test_dataloader = DataLoader(dataset_test, batch_size = config["batch_size"], shuffle = False, num_workers = 4)

    model = Beta_VAE(latent_dim = config["latent_dim"], beta = config["beta"], input_shape = config["input_shape"]).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    for epoch in range(config["epochs"]) :
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

        print(f"Epoch {epoch+1}/{config['epochs']} | Training Loss: {avg_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | MSE: {avg_mse:.4f} | KLD: {avg_kld:.4f}")


    print("Évaluation sur le set de test...")
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = model.loss_function(recon, batch, mu, logvar)
            total_test_loss += loss.item()

    print(f"Test Loss finale: {total_test_loss/len(test_loader):.4f}")
    torch.save(model.state_dict(), "../checkpoints/beta_vae_final.pth")
