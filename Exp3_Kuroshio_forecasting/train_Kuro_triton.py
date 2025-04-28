import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torch.distributed as dist
import netCDF4 as nc
import logging
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from model.Triton_model import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import logging
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup logging
backbone = 'Kuro_Triton_exp1_20250224'
logging.basicConfig(filename=f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_2_Kuroshio/logs/{backbone}_training_log.log',
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

# ========================== Distributed Training Setup ==========================
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

# ============================== Data Loading ==============================
from dataloader_api.dataloader_kuroshio_256 import *

config = {
    'data_path': '/jizhicfs/easyluwu/ocean_project/kuro/KURO.nc',
    'input_steps': 10,
    'output_steps': 10,
    'batch_size': 2,
    'val_batch_size': 2,
    'num_workers': 4,
    'seed': 42
}

train_loader, val_loader, test_loader, data_mean, data_std = create_dataloaders(config)


for sample_input, sample_target in train_loader:
    print(sample_input.shape, sample_target.shape)
    print(f"Input data range: [{sample_input.min():.2f}, {sample_input.max():.2f}]")
    print(f"Existence of NaN values: {torch.isnan(sample_input).any().item()}")
    print(f"Existence of Inf values: {torch.isinf(sample_input).any().item()}")
    print("mean, std", data_mean, data_std)
    break
# ============================== Model Setup ==============================
model = Triton(
        shape_in=(10, 2, 256, 256),
        spatial_hidden_dim=256,
        output_channels=2,
        temporal_hidden_dim=512,
        num_spatial_layers=4,
        num_temporal_layers=8)

model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

# ============================== Criterion and Optimizer ==============================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 2000
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

# ============================== Training, Validation, and Testing Functions ==============================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
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
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)

def test(model, test_loader, criterion, device):
    path = '/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_2_Kuroshio/results'
    model.eval()
    test_loss = 0.0
    all_inputs = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", disable=local_rank != 0):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            # Collect results
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

    return test_loss / len(test_loader.dataset)
# ============================== Main Training Loop ==============================
best_val_loss = float('inf')
best_model_path = f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_2_Kuroshio/checkpoints/{backbone}_best_model.pth'

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

    scheduler.step()

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current Learning Rate: {current_lr:.10f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        logging.info(f'Train Loss: {train_loss * num_gpus:.7f}, Val Loss: {val_loss * num_gpus:.7f}')

if local_rank == 0:
    try:
        model.load_state_dict(torch.load(best_model_path))
        test_loss = test(model, test_loader, criterion, device)
        logging.info("Testing completed and best model saved.")
    except Exception as e:
        logging.error(f'Error loading model checkpoint during testing: {e}')

dist.destroy_process_group()