import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import netCDF4 as nc
import numpy as np

class OceanCurrentDataset(Dataset):
    def __init__(self, data_path, input_steps=10, output_steps=10, 
                 start_time=None, end_time=None, mean=None, std=None, transform=None):
        """
        Ocean current dataset class
        :param data_path: Path to NetCDF file
        :param input_steps: Number of input time steps
        :param output_steps: Number of prediction time steps
        :param start_time: Start time (days since 1950-01-01)
        :param end_time: End time (days since 1950-01-01)
        :param mean: Precomputed mean
        :param std: Precomputed standard deviation
        :param transform: Data augmentation transform
        """
        self.data_path = data_path
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.transform = transform
        self.total_steps = input_steps + output_steps
        self.start_time = start_time
        self.end_time = end_time
        
        # Load and preprocess data
        self.data, self.time_values = self._load_and_process_data()
        
        # Set statistics
        if mean is not None and std is not None:
            self.mean = 0
            self.std = 1
        else:
            self.mean, self.std = 0, 1

    def _load_and_process_data(self):
        """Load and process NetCDF data"""
        with nc.Dataset(self.data_path, 'r') as ds:
            # Read time variable
            time_var = ds['time']
            time_values = time_var[:]
            
            # Filter indices within time range
            if self.start_time is not None or self.end_time is not None:
                time_mask = np.ones_like(time_values, dtype=bool)
                if self.start_time is not None:
                    time_mask &= (time_values >= self.start_time)
                if self.end_time is not None:
                    time_mask &= (time_values <= self.end_time)
                valid_indices = np.where(time_mask)[0]
            else:
                valid_indices = np.arange(len(time_values))
            
            # Handle missing values
            def process_var(var):
                arr = var[:][valid_indices]  # Slice by time range
                if '_FillValue' in var.ncattrs():
                    fill_value = var._FillValue
                    arr = np.ma.masked_values(arr, fill_value).filled(np.nan)
                return torch.nan_to_num(torch.FloatTensor(arr), nan=0.0)

            # Load and merge UV components
            ugos = process_var(ds['ugos'])  # (time, lat, lon)
            vgos = process_var(ds['vgos'])
            
            # Adjust dimension order [time, channels, lat, lon]
            data = torch.stack([ugos, vgos], dim=1)
            return data, time_values[valid_indices]

    def _compute_stats(self):
        """Calculate dataset statistics"""
        # Compute mean and std using entire dataset
        return torch.mean(self.data), torch.std(self.data)

    def __len__(self):
        return len(self.data) - self.total_steps + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.total_steps]  # [T_total, C, H, W]
        
        # Normalization
        window = (window - self.mean) / self.std
        
        # Split input and output
        input_seq = window[:self.input_steps]
        target_seq = window[self.input_steps:]
        
        if self.transform:
            input_seq = self.transform(input_seq)
            target_seq = self.transform(target_seq)
            
        # Spatial downsampling
        return input_seq, target_seq

def create_dataloaders(config):
    # Define time range (days since 1950-01-01)
    # 1993-01-01 ≈ 15706 (confirmed from first value of time variable in raw data)
    # 2020-12-31 ≈ 25931 (365*71 + 18 leap years - 1 day)
    # 2023-12-31 ≈ 27027 (end time of raw data)
    
    # Create training set (1993-2020)
    train_dataset = OceanCurrentDataset(
        data_path=config['data_path'],
        input_steps=config['input_steps'],
        output_steps=config['output_steps'],
        start_time=15706,  # 1993-01-01
        end_time=25931     # 2020-12-31
    )
    
    # Get training set statistics
    train_mean, train_std = train_dataset.mean, train_dataset.std
    
    # Create validation and test sets (2021-2024) not include 2024 year
    val_dataset = OceanCurrentDataset(
        data_path=config['data_path'],
        input_steps=config['input_steps'],
        output_steps=config['output_steps'],
        start_time=25932,  # 2021-01-01
        end_time=27027,    # 2023-12-31
        mean=train_mean,   
        std=train_std
    )
    

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    dataloader_test = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        sampler=test_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        shuffle=False,  
        drop_last=True
    )
    
    return dataloader_train, dataloader_val, dataloader_test, train_mean, train_std
