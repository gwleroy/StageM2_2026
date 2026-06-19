# Script qui permet de diagnostiquer la normalisation des données

import torch
import time
from torch.utils.data import DataLoader
from dataset import FluidDataset

def diagnostics(file_path, dataset_name) : 
    print(f"DÉBUT DU DIAGNOSTIC SUR : {dataset_name}")

    ds = FluidDataset(file_path, dataset_name, split='train')
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    # Test de format et statistiques
    sample_batch = next(iter(loader))
    print(f"Forme du batch (N, C, H, W) : {sample_batch.shape}")
    print(f"Stats - Moyenne: {sample_batch.mean():.4f}, Std: {sample_batch.std():.4f}")
    print(f"Valeurs extremes - Min: {sample_batch.min():.4f}, Max: {sample_batch.max():.4f}")
    
    # Test de performance (vitesse de lecture)
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 5: break # On test sur 5 batches
    end = time.time()

    print(f"Performance : {5 * 32 / (end - start):.2f} snapshots/seconde")

    # Validation physique (Check pour NaN/Inf)
    if torch.isnan(sample_batch).any():
        print("ALERTE : Des valeurs NaN ont été détectées !")
    else:
        print("Diagnostic réussi : Les données sont propres.")


"""
if __name__ == "__main__" :
    diagnostics("file","dataset_name")

"""

