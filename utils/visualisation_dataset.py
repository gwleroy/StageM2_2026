# Fonction permettant de visualiser la forme et l'evolution des champs contenu dans le dataset

import numpy as np
import matplotlib.pyplot as plt
import h5py

def visualisation_champs(file_name, dataset_name, index = 0, canal = "norm") :
    """ 
    input : dataset_name = quel partie du dataset (ie : U, P, V)
            index        = quel snapshot 
            canal        = dans le cas ou il y a 2 canal, lequel choisir (ie = UV)

    output : Affiche la carte de chaleur du champ sélectionné

    """

    with h5py.File(file_name, "r") as f :
        data = f[dataset_name][index]

    grid_data = np.squeeze(data) # Nettoyage des dimensions egales a 1

    if grid_data.ndim == 3 :
        if canal == "norm" :
            champs = np.sqrt(np.sum(grid_data**2, axis = -1))
            titre = f"{dataset_name} (Norme) - Snapshot {index}"

        elif canal == 0 :
            champs = grid_data[...,0]
            titre = f"{dataset_name} {canal} - Snapshot {index}"

        elif canal == 1 :
            champs = grid_data[...,1]
            titre = f"{dataset_name} {canal} - Snapshot {index}"

    else :
        champs = grid_data
        titre = f"{dataset_name} - Snapshot {index}"

    nom_image = (f"plot_{dataset_name}_{canal}_snap{index}.png".lower() )
    # Figure
    plt.figure(figsize = (10, 4))
    plt.imshow(champs, cmap = "jet", origin = "lower", aspect = "auto")
    plt.colorbar(label = "Valeur")
    plt.title(titre)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"figs/{nom_image}", dpi = 300)
    plt.close()



