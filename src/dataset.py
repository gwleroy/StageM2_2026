import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, random_split

class FluidDataset(Dataset) : 

    def __init__(self, file_path, dataset_name, split = "train", train_ratio = (0.8, 0.1, 0.1), seed = 42) : 
        self.file_path = file_path
        self.dataset_name = dataset_name

        with h5py.File(self.file_path, "r") as f : 
            total_len = f[self.dataset_name].shape[0]

            if 'mean' in f and 'std' in f:
                self.mean = f['mean'][()]
                self.std = f['std'][()]
            else:
                self.mean, self.std = self._compute_stats(f)

            rng = np.random.default_rng(seed)
            indices = np.arange(total_len)
            rng.shuffle(indices)

            train_size = int(train_ratio[0] * total_len)
            val_size = int(train_ratio[1] * total_len)


            if split == 'train':
                self.indices = np.sort(indices[:train_size])
            elif split == 'val':
                self.indices = np.sort(indices[train_size:train_size + val_size])
            else: # test
                self.indices = np.sort(indices[train_size + val_size:])

    def __len__(self) :
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.dataset_name][actual_idx].astype(np.float32)

        data = np.squeeze(data)

        m = np.squeeze(self.mean)
        s = np.squeeze(self.std)

        if m.ndim == 1:
            m = m.reshape(1, 1, -1)
            s = s.reshape(1, 1, -1)

        data = (data - m) / (s + 1e-8)

        # Format PyTorch (C, H, W)
        data = np.transpose(data, (2, 0, 1))

        return torch.from_numpy(data)

    def _compute_stats(self, f):
        print("Calcul des statistiques (moyenne et std) du dataset...")
        # On charge tout le dataset en RAM pour le calcul (attention si dataset gigantesque)
        data = f[self.dataset_name][:] 
        mean = np.mean(data, axis=(0, 1, 2)) # Moyenne par canal
        std = np.std(data, axis=(0, 1, 2))
        return mean, std
