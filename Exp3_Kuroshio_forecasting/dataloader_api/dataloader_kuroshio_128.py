import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import netCDF4 as nc
import numpy as np

class OceanCurrentDataset(Dataset):
    def __init__(self, data_path, input_steps=10, output_steps=10, transform=None):
        self.data_path = data_path
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.transform = transform
        self.total_steps = input_steps + output_steps
        
        self.data = self._load_and_process_data()
        self.mean, self.std = 0, 1

    def _load_and_process_data(self):
        with nc.Dataset(self.data_path, 'r') as ds:
            def process_var(var):
                arr = var[:]
                if '_FillValue' in var.ncattrs():
                    fill_value = var._FillValue
                    arr = np.ma.masked_values(arr, fill_value).filled(np.nan)
                return torch.nan_to_num(torch.FloatTensor(arr), nan=0.0)

            ugos = process_var(ds['ugos']) 
            vgos = process_var(ds['vgos'])
            
            #  [time, channels, lat, lon]
            return torch.stack([ugos, vgos], dim=1)  

    def _compute_stats(self):
        return torch.mean(self.data[:10000]), torch.std(self.data[:10000])

    def __len__(self):
        return len(self.data) - self.total_steps + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.total_steps]  # [T_total, C, H, W]
        
        window = (window - self.mean) / self.std
        
        input_seq = window[:self.input_steps]
        target_seq = window[self.input_steps:]
        
        if self.transform:
            input_seq = self.transform(input_seq)
            target_seq = self.transform(target_seq)
            
        return input_seq[:,:,::2,::2], target_seq[:,:,::2,::2]

def create_dataloaders(config):
    full_dataset = OceanCurrentDataset(
        data_path=config['data_path'],
        input_steps=config['input_steps'],
        output_steps=config['output_steps']
    )
    
    train_size = 10000 - config['input_steps'] - config['output_steps'] + 1
    val_size = 500
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
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
        test_dataset,
        batch_size=config['val_batch_size'],
        sampler=test_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader_train, dataloader_val, dataloader_test, full_dataset.mean, full_dataset.std