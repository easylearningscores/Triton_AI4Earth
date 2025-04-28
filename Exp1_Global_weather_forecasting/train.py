import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import netCDF4 as nc
import torchvision.transforms as transforms
from dataloader_api.dataloader import *
from tqdm import tqdm
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from model.Triton_model import *
from model_baselines.pangu_model import *

# Setup logging
backbone = 'triton_weather_20250326_v1'
logging.basicConfig(filename=f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_0_Weather/logs/{backbone}_training_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(message)s')

# Set a specific seed
seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# =========================================================================== dist train ========================================================================================================================
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
num_gpus = torch.cuda.device_count()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# ============================================================= data load ===================================================
args = {
        'data_path': '/jizhicfs/easyluwu/scaling_law/ft_local/low_res',
        'ocean_lead_time': 1,
        'atmosphere_lead_time': 1,
        'shuffle': True,
        'variables_input': list(range(69)),
        'variables_output': list(range(69)),
        'lon_start': 0,
        'lat_start': 0,
        'lon_end': 1440,
        'lat_end': 720,
        'ds_factor': 1,
    }

train_dataset = train_Dataset(args)
test_dataset = test_Dataset(args)

train_sampler = data.distributed.DistributedSampler(train_dataset)
train_loader = data.DataLoader(train_dataset,
                                num_workers=0,
                                batch_size=1, 
                                sampler=train_sampler)

test_sampler = data.distributed.DistributedSampler(test_dataset)
test_loader = data.DataLoader(test_dataset,
                            num_workers=0,
                            batch_size=1,
                            sampler=test_sampler)

for inputs, targets in iter(train_loader):
    print(inputs.shape, targets.shape)
    break

# ================================================ model load ===========================================================
model = Triton(
    shape_in=(1, 69, 180, 360),
    spatial_hidden_dim=256,
    output_channels=69,
    temporal_hidden_dim=512,
    num_spatial_layers=4,
    num_temporal_layers=8)

model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

# ============================== criterion and optimizer ======================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

# ===========================train val and test ======================================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", disable=local_rank != 0):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)

def test(model, test_loader, criterion, device):
    path = '/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_0_Weather/results'
    model.eval()
    test_loss = 0.0
    
    all_inputs = []
    all_targets = []
    all_outputs = []
    i = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", disable=local_rank != 0):
            i += 1
            print(f"{i} : {inputs.shape}")
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            
            # Convert tensors to numpy arrays and append to lists
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    if local_rank == 0:
        np.save(f'{path}/{backbone}_inputs.npy', all_inputs)
        np.save(f'{path}/{backbone}_targets.npy', all_targets)
        np.save(f'{path}/{backbone}_outputs.npy', all_outputs)
    print(test_loss)
    print(len(test_loader.dataset))
    print(i)
    return test_loss / len(test_loader.dataset)

num_epochs = 1000
best_val_loss = float('inf')
best_model_path = f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_0_Weather/checkpoints/{backbone}_best_model.pth'

if local_rank == 0 and os.path.exists(best_model_path):
    try:
        logging.info('Loading best model from checkpoint.')
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        logging.error(f'Error loading model checkpoint: {e}')

for epoch in range(num_epochs):
    if local_rank == 0:
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, test_loader, criterion, device)

    if local_rank == 0:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        logging.info(f'Train Loss: {train_loss * num_gpus:.7f}, Val Loss: {val_loss * num_gpus:.7f}') 

if local_rank == 0:
    try:
        model.load_state_dict(torch.load(best_model_path))
        test_loss = test(model, test_loader, criterion, device)
        logging.info(f"Testing completed and best model saved. | test_loss:{test_loss * num_gpus:.7f}")
    except Exception as e:
        logging.error(f'Error loading model checkpoint during testing: {e}')

dist.destroy_process_group()