# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os

class ENSODataset(Dataset):
    """
    PyTorch Dataset for loading pre-processed ENSO data.
    Assumes data is stored in NetCDF files, with variables and time.
    """
    def __init__(self, data_path, variables, input_time_steps, lead_time):
        super().__init__()
        self.data_path = data_path
        self.variables = variables
        self.input_time_steps = input_time_steps
        self.lead_time = lead_time
        
        # This is a simplified loader. A real implementation would handle
        # multiple files, lazy loading, and complex time indexing.
        self.data_arrays = self._load_data()
        self.n_samples = self.data_arrays['sst'].shape[0] - self.input_time_steps - self.lead_time

    def _load_data(self):
        """Loads and concatenates data from NetCDF files."""
        # This is a placeholder for a real data loading pipeline.
        # It assumes a single large NetCDF file for simplicity.
        print(f"Loading data from {self.data_path}...")
        try:
            # A real implementation would search for files and concatenate
            ds = xr.open_dataset(os.path.join(self.data_path, "sample_data.nc"))
            data_dict = {var: ds[var].values for var in self.variables}
            self.nino34 = ds['nino34'].values
            return data_dict
        except FileNotFoundError:
            print(f"Error: Data file not found. Please create a sample NetCDF file at {self.data_path}")
            # Create dummy data for demonstration if file not found
            shape = (500, 36, 72)
            data_dict = {var: np.random.randn(*shape).astype(np.float32) for var in self.variables}
            self.nino34 = np.random.randn(500).astype(np.float32)
            return data_dict

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.input_time_steps
        target_idx = end_idx + self.lead_time - 1
        
        # Stack variables to form input tensor
        input_data_list = []
        for var in self.variables:
            # Shape: (T, H, W)
            input_data_list.append(self.data_arrays[var][start_idx:end_idx, :, :])
        
        # Concatenate along a new channel dimension, then flatten
        # Final shape for model: (T*C, H, W)
        input_tensor = np.concatenate(input_data_list, axis=0)
        
        # Get target Niño3.4 index
        target_val = self.nino34[target_idx]
        
        return torch.from_numpy(input_tensor).float(), torch.tensor(target_val).float()

def create_dataloader(data_path, variables, input_time_steps, lead_time, batch_size, shuffle=True):
    dataset = ENSODataset(data_path, variables, input_time_steps, lead_time)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader
