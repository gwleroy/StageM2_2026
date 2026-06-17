# Fonction pour explorer les fichier de données

import numpy as np 
import h5py

def explorer_hdf5(file_name, max_items=20) : 
    with h5py.File(file_name, "r") as f:
        print(f"Structure du fichier : {file_name}")
        print(f"Clé racines : {list(f.keys())}")
    
        names = []
        f.visit(names.append)

        for name in names : 
            obj = f[name]
            indent = " " * (name.count("/")+1)

            if isinstance(obj, h5py.Group) :
                print(f"{indent} Groupe : /{name}")

            elif isinstance(obj, h5py.Dataset) :
                print(f"{indent} Dataset : /{name} shape = {obj.shape}, dtype = {obj.dtype}")
                data = obj[:max_items].flatten()
                print(f"{indent} Aperçu : {data[:max_items]}{'...' if data.size > max_items else ''}")
            
            if obj.attrs : 
                print(f"{indent} Attributs : { {k: v for k, v in obj.attrs.items()} }")


