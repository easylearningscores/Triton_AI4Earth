import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
import logging
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data_utils
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

selected_model = config['selected_model']

model_config = config['models'][selected_model]
training_config = config['trainings'][selected_model]
data_config = config['datas'][selected_model]
logging_config = config['loggings'][selected_model]

backbone = logging_config['backbone']
log_dir = logging_config['log_dir']
checkpoint_dir = logging_config['checkpoint_dir']
result_dir = logging_config['result_dir']

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

logging.basicConfig(filename=f'{log_dir}/{backbone}_training_log.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

seed = training_config['seed']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(seed)

# ============== Distributed Training Settings ===============
parallel_method = training_config.get('parallel_method', 'DistributedDataParallel')

if parallel_method == 'DistributedDataParallel':
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    num_gpus = dist.get_world_size()

    def reduce_mean(tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    local_rank = 0  #  For DataParallel, can be setted as 0

    def reduce_mean(tensor, nprocs):
        return tensor

import torch
import torch.utils.data as data_utils

# ============== Data loader ==============
data_path = data_config['data_path']
data = torch.load(data_path)
ns_all = data['vorticity']  # Assuming your data dict has a key 'vorticity'

# Split the data into training, validation, and test sets
total_samples = ns_all.shape[0]  # Total number of samples (1280)
train_end = int(0.8 * total_samples)  # 80% for training
val_end = int(0.9 * total_samples)    # Next 10% for validation

train_data = ns_all[:train_end]               # Shape: [1024, 100, 128, 128] # 
val_data = ns_all[train_end:val_end]          # Shape: [128, 100, 128, 128]
test_data = ns_all[val_end:]                  # Shape: [128, 100, 128, 128]

args = {
    'input_length': data_config['input_length'],
    'target_length': data_config['target_length'],
    'variables_input': data_config.get('variables_input', [0]),  
    'variables_output': data_config.get('variables_output', [0]), 
    'downsample_factor': data_config['downsample_factor']
}

from dataloader_ns import train_Dataset, test_Dataset

train_dataset = train_Dataset(train_data, args)
val_dataset = test_Dataset(val_data, args)
test_dataset = test_Dataset(test_data, args)

if parallel_method == 'DistributedDataParallel':
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
else:
    train_sampler = None
    val_sampler = None
    test_sampler = None

train_loader = data_utils.DataLoader(
    train_dataset,
    num_workers=0,
    batch_size=training_config['batch_size'],
    sampler=train_sampler,
    shuffle=(train_sampler is None)
)

val_loader = data_utils.DataLoader(
    val_dataset,
    num_workers=0,
    batch_size=training_config['batch_size'],
    sampler=val_sampler,
    shuffle=False
)

test_loader = data_utils.DataLoader(
    test_dataset,
    num_workers=0,
    batch_size=training_config['batch_size'],
    sampler=test_sampler,
    shuffle=False
)


if local_rank == 0:
    for input_frames, output_frames in train_loader:
        print(f'Dataloader Input shape: {input_frames.shape}, Output shape: {output_frames.shape}')
        break

# ============== Model settings ==============
# Model registry
from model.triton_model import Triton
from model_baselines.fno import FNO2d 
from model_baselines.dit import Dit
from model_baselines.simvp import SimVP
from model.triton_model_v2 import Triton_v2
from model_baselines.cno import CNO
from model_baselines.mgno import MgNO
from model_baselines.lsm import LSM
from model_baselines.pastnet import PastNetModel
from model_baselines.resnet import ResNet
from model_baselines.unet import U_net


model_dict = {
    'Triton': Triton,
    'Triton_V2': Triton_v2,
    'FNO': FNO2d,
    'DiT': Dit,
    'SimVP': SimVP,
    'CNO': CNO,
    'MGNO': MgNO,
    'LSM': LSM,
    'PastNet': PastNetModel,
    'ResNet': ResNet,
    'U_net': U_net,
}

model_name = selected_model
print(f"{model_name} has been successful load !")
model_params = model_config['parameters']

# Check if the model is in the registry
if model_name in model_dict:
    ModelClass = model_dict[model_name]
    model = ModelClass(**model_params)
else:
    raise ValueError(f"Model {model_name} is not defined.")

model = model.to(device)

# Process according to the parallelization method
if parallel_method == 'DistributedDataParallel':
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
elif parallel_method == 'DataParallel':
    model = nn.DataParallel(model)
else:
    raise ValueError(f"Unknown parallel method: {parallel_method}")

# ============== Loss Function and Optimizer ==============
criterion = nn.MSELoss()
learning_rate = training_config['learning_rate']
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = training_config['num_epochs']
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

# ============== Training, validation, and testing functions ==============
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    if parallel_method == 'DistributedDataParallel' and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        inputs = inputs.to(device, non_blocking=True).float() # B 1 1 128 128
        targets = targets.to(device, non_blocking=True).float()
        optimizer.zero_grad()
        outputs = model(inputs) # B 1 1 128 128
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if parallel_method == 'DistributedDataParallel':
            loss_value = reduce_mean(loss, num_gpus).item()
        else:
            loss_value = loss.item()
        train_loss += loss_value * inputs.size(0)
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", disable=local_rank != 0):
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if parallel_method == 'DistributedDataParallel':
                loss_value = reduce_mean(loss, num_gpus).item()
            else:
                loss_value = loss.item()
            val_loss += loss_value * inputs.size(0)
    return val_loss / len(val_loader.dataset)

def test(model, test_loader, criterion, device):
    path = result_dir
    model.eval()
    test_loss = 0.0
    all_inputs = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", disable=local_rank != 0):
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            outputs = model(inputs)

            if local_rank == 0:
                all_inputs.append(inputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

            loss = criterion(outputs, targets)
            if parallel_method == 'DistributedDataParallel':
                loss_value = reduce_mean(loss, num_gpus).item()
            else:
                loss_value = loss.item()
            test_loss += loss_value * inputs.size(0)

    if local_rank == 0:
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        np.save(f'{path}/{backbone}_inputs.npy', all_inputs)
        np.save(f'{path}/{backbone}_targets.npy', all_targets)
        np.save(f'{path}/{backbone}_outputs.npy', all_outputs)

    return test_loss / len(test_loader.dataset)

# ============== Main training Loop ==============
best_val_loss = float('inf')
best_model_path = f'{checkpoint_dir}/{backbone}_best_model.pth'

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
    train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
    val_loss = validate(model, val_loader, criterion, device)

    scheduler.step()

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current Learning Rate: {current_lr:.10f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        logging.info(f'Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}')

if local_rank == 0:
    try:
        model.load_state_dict(torch.load(best_model_path))
        test_loss = test(model, test_loader, criterion, device)
        logging.info(f"Testing completed. Test Loss: {test_loss:.7f}")
    except Exception as e:
        logging.error(f'Error loading model checkpoint during testing: {e}')

if parallel_method == 'DistributedDataParallel':
    dist.destroy_process_group()