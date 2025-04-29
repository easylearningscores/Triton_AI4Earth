import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Configuration Loading ==================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model configuration
selected_model = config['selected_model']
model_config = config['models'][selected_model]
training_config = config['trainings'][selected_model]
data_config = config['datas'][selected_model]
logging_config = config['loggings'][selected_model]

# Path configuration
backbone = logging_config['backbone']
checkpoint_dir = logging_config['checkpoint_dir']
result_dir = logging_config['result_dir']
os.makedirs(result_dir, exist_ok=True)

# ================== Random Seed Setting ==================
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(training_config['seed'])

# ================== Data Loading ==================
print("\n========== Loading Data ==========")
data_path = data_config['data_path']
data = torch.load(data_path, map_location='cpu', weights_only=False)  # Safe loading
ns_all = data['vorticity']  # Original shape: [1280, 100, 128, 128]

# Dataset splitting
total_samples = ns_all.shape[0]
test_data = ns_all[total_samples//10 * 9:]  # Take last 10% as test set
test_data = test_data.unsqueeze(2)        # Add channel dimension [128, 100, 1, 128, 128]

# ================== Model Initialization ==================
print("\n========== Initializing Model ==========")
# Model registry (must include all models completely)
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

# Model instantiation
ModelClass = model_dict[selected_model]
model = ModelClass(**model_config['parameters']).to(device)
model.eval()
print(f"{selected_model} has beed loaded")
# ================== Loading Model Weights ==================
print("\n========== Loading Model Weights ==========")
best_model_path = os.path.join(checkpoint_dir, f"{backbone}_best_model.pth")
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # Handle possible DataParallel wrapper
    if all(k.startswith('module.') for k in checkpoint.keys()):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    print(f"Successfully loaded model weights: {best_model_path}")
else:
    raise FileNotFoundError(f"Model checkpoint not found: {best_model_path}")

# ================== Inference Configuration ==================
torch.set_grad_enabled(False)
rollout_steps = 99  # Total prediction steps
input_length = data_config['input_length']  # Input timesteps
variables_input = data_config.get('variables_input', [0])  # Input variable indices
variables_output = data_config.get('variables_output', [0])  # Output variable indices
downsample_factor = data_config['downsample_factor']  # Downsampling factor

# Dimension calculation
original_H, original_W = test_data.shape[-2], test_data.shape[-1]  # Original spatial dimensions
H = original_H // downsample_factor  # Downsampled height
W = original_W // downsample_factor  # Downsampled width

# ================== Result Containers ==================
num_samples = test_data.size(0)
all_inputs = np.zeros((num_samples, input_length, len(variables_input), H, W), dtype=np.float32)
all_outputs = np.zeros((num_samples, rollout_steps, len(variables_output), H, W), dtype=np.float32)
all_targets = np.zeros((num_samples, rollout_steps, len(variables_output), H, W), dtype=np.float32)

# ================== Visualization Settings ==================
viz_dir = os.path.join(result_dir, f"{backbone}_visualizations")
os.makedirs(viz_dir, exist_ok=True)

def visualize_comparison(pred, true, step, save_path):
    """Visualization comparison function"""
    plt.figure(figsize=(16, 6), dpi=150)
    
    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap='coolwarm', vmin=-3, vmax=3)
    plt.title(f'Prediction @ Step {step}', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 2, 2)
    plt.imshow(true, cmap='coolwarm', vmin=-3, vmax=3)
    plt.title(f'Ground Truth @ Step {step}', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ================== Main Inference Loop ==================
print("\n========== Starting Inference ==========")
total_steps = num_samples * rollout_steps
with tqdm(total=total_steps, desc="Overall Progress", unit="step", position=0) as pbar_total:
    for sample_idx in tqdm(range(num_samples), desc="Sample Processing", unit="sample", position=1):
        # Current sample data [100, 1, 128, 128]
        current_sample = test_data[sample_idx]
        
        # Create sample visualization directory
        sample_viz_dir = os.path.join(viz_dir, f"sample_{sample_idx:04d}")
        os.makedirs(sample_viz_dir, exist_ok=True)
        
        # ===== Initial Condition Processing =====
        initial_input = current_sample[:input_length, variables_input, ::downsample_factor, ::downsample_factor]
        all_inputs[sample_idx] = initial_input.cpu().numpy()
        
        # ===== Ground Truth Processing =====
        ground_truth = current_sample[input_length:input_length+rollout_steps, variables_output, ::downsample_factor, ::downsample_factor]
        all_targets[sample_idx] = ground_truth.cpu().numpy()
        
        # ===== Inference Initialization =====
        inputs = initial_input.clone().to(device)
        predictions = []
        
        # ===== Timestep Loop =====
        for step_idx in tqdm(range(rollout_steps), desc=f"Sample {sample_idx} Timesteps", leave=False, position=2):
            # Model input [1, T, C, H, W]
            model_input = inputs.unsqueeze(0).float()
            
            # Model inference
            with torch.cuda.amp.autocast(enabled=False):  # Mixed precision acceleration
                pred = model(model_input)
            
            # Extract last prediction step [1, 1, C, H, W]
            last_pred = pred[:, -1:]
            
            # Save prediction result
            pred_np = last_pred.squeeze(0).squeeze(0).cpu().numpy()  # [C, H, W]
            predictions.append(pred_np)
            
            # Update input sequence (rolling window)
            inputs = torch.cat([inputs[1:], last_pred.squeeze(0)], dim=0)
            
            # Save visualization (every 10 steps)
            if step_idx % 10 == 0:
                true_frame = ground_truth[step_idx].squeeze(0).numpy()
                viz_path = os.path.join(sample_viz_dir, f"step_{step_idx:04d}.png")
                visualize_comparison(pred_np[0], true_frame, step_idx, viz_path)
            
            # Update progress bar
            pbar_total.update(1)
        
        # Save current sample results
        all_outputs[sample_idx] = np.stack(predictions, axis=0)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================== Result Saving ==================
print("\n========== Saving Results ==========")
np.save(os.path.join(result_dir, f"{backbone}_initial_conditions.npy"), all_inputs)
np.save(os.path.join(result_dir, f"{backbone}_predictions.npy"), all_outputs)
np.save(os.path.join(result_dir, f"{backbone}_ground_truth.npy"), all_targets)

# ================== Final Validation ==================
print("\n========== Result Verification ==========")
print(f"Initial conditions shape: {all_inputs.shape}")
print(f"Predictions shape: {all_outputs.shape}")
print(f"Ground truth shape: {all_targets.shape}")
print(f"Max value validation - Initial: {all_inputs.max():.4f}, Prediction: {all_outputs.max():.4f}, Ground truth: {all_targets.max():.4f}")
print(f"Min value validation - Initial: {all_inputs.min():.4f}, Prediction: {all_outputs.min():.4f}, Ground truth: {all_targets.min():.4f}")

print("\n✅ Inference completed! All results saved to:", os.path.abspath(result_dir))