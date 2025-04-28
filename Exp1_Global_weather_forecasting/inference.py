import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import logging
import argparse
from tqdm import tqdm  
from collections import OrderedDict

years = 2018
# ==========================================
# dataset
# ==========================================
class TestInferenceDataset(Dataset):
    def __init__(self, args, target_year, target_date):
        super(TestInferenceDataset, self).__init__()
        self.args = args
        self.target_year = target_year
        self.target_date = target_date  

        self.data_path = os.path.join(self.args["data_path"], f'{self.target_year}_norm.nc')
        self.dataset = nc.Dataset(self.data_path)
        self.atm_vars = self.dataset.variables['atmosphere_variables']
        self.max_time_index = self.atm_vars.shape[0]

        self.initial_time = self.target_date
        if self.initial_time >= self.max_time_index:
            raise ValueError("Initial time index exceeds data range.")

        self.rollout_steps = args['rollout_steps']
        self.true_labels = []
        for step in range(self.rollout_steps):
            time_index = self.initial_time + step + 1
            if time_index >= self.max_time_index:
                break
            true_label = self.atm_vars[
                time_index,
                self.args['variables_output'],
                self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
                self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']
            ]
            true_label = torch.tensor(true_label, dtype=torch.float32)
            true_label = torch.nan_to_num(true_label, nan=0.0)
            self.true_labels.append(true_label)

        self.initial_input = self.atm_vars[
            self.initial_time,
            self.args['variables_input'],
            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']
        ]
        self.initial_input = torch.tensor(self.initial_input, dtype=torch.float32)
        self.initial_input = torch.nan_to_num(self.initial_input, nan=0.0)

        self.dataset.close()

    def __len__(self):
        return 1 

    def __getitem__(self, index):
        return self.initial_input, self.true_labels 

# ==========================================
# define model 
# ==========================================
from model.Triton_model import *
from model_baselines.fuxi_model import *
from model_baselines.pangu_model import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# delete "module." 
# ==========================================
def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # load new state_dict
    model.load_state_dict(new_state_dict)
    return model

# ==========================================
# Main inference function
# ==========================================
def main():
    # ==========================================
    # Parameter parsing.
    # ==========================================
    parser = argparse.ArgumentParser(description='Incremental Inference')
    parser.add_argument('--start_step', type=int, default=0, help='The initial prediction step starts from 0.')
    args_parsed = parser.parse_args()

    # ==========================================
    # Parameters seetting
    # ==========================================
    backbone = 'triton_weather_20250326_v1'
    #backbone = 'baselines_Pangu_exp_1101'
    #backbone = 'baselines_Fuxi_exp4_1105'
    args = {
        'data_path': '/jizhicfs/easyluwu/scaling_law/ft_local/low_res',
        'variables_input': list(range(69)),
        'variables_output': list(range(69)),
        'lon_start': 0,
        'lat_start': 0,
        'lon_end': 1440,
        'lat_end': 720,
        'ds_factor': 1,
        'model_path': f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_0_Weather/checkpoints/{backbone}_best_model.pth',
        'results_path': f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_0_Weather/results_{years}',
        'log_path': f'/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_0_Weather/logs//inference_log_{backbone}.log',
        'backbone': backbone,
        'start_step': args_parsed.start_step,  
        'rollout_steps': 364,  
    }

    # ==========================================
    # Set logs
    # ==========================================
    os.makedirs(os.path.dirname(args['log_path']), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args['log_path']),
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    logging.info(f"The inference script starts running, with the initial step: {args['start_step']}。")

    seed = 42
    set_seed(seed)
    logging.info(f"The random seed is set to {seed}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"The device used is: {device}")


    target_year = years
    target_date = 0

    dataset = TestInferenceDataset(args, target_year, target_date)
    initial_input, true_labels = dataset[0]  


    model = Triton(
            shape_in=(1, 69, 180, 360),
            spatial_hidden_dim=256,
            output_channels=69,
            temporal_hidden_dim=512,
            num_spatial_layers=4,
            num_temporal_layers=8)
    # model = Pangu(in_shape=(1, 69, 180, 360))
    #model = Fuxi(in_shape=(1, 69, 180, 360))


    model = model.to(device)
    if os.path.exists(args['model_path']):
        model = load_model(model, args['model_path'], device)
        logging.info(f"Successfully loaded the model weights: {args['model_path']}")
    else:
        logging.error(f"The model weight file does not exist: {args['model_path']}")
        return

    model.eval()


    os.makedirs(args['results_path'], exist_ok=True)

    # Save the initial input (only at step 0).
    if args['start_step'] == 0:
        input_data_np = initial_input.cpu().numpy()  # shape: [69, H, W]
        np.save(os.path.join(args['results_path'], f'{backbone}_initial_input.npy'), input_data_np)
        logging.info("Initial input has been saved!")
        current_input = initial_input.unsqueeze(0).unsqueeze(0).to(device)  # shape: [1, 1, 69, H, W]
    else:
        # Load the prediction result from the previous step as input.
        previous_step = args['start_step'] - 1
        prediction_path = os.path.join(args['results_path'], f"{args['backbone']}_prediction_step_{previous_step}.npy")
        if not os.path.exists(prediction_path):
            raise FileNotFoundError(f"The prediction result file does not exist: {prediction_path}")
        input_data = np.load(prediction_path)
        input_data = torch.from_numpy(input_data).float()
        current_input = input_data.unsqueeze(0).unsqueeze(0).to(device)  # shape: [1, 1, 69, H, W]

    # ==========================================
    # Predict the remaining steps.
    # ==========================================
    total_steps = args['rollout_steps']
    start_step = args['start_step']

    logging.info(f"Start multi-step prediction, from step {start_step} to step {total_steps - 1}.")

    for step in tqdm(range(start_step, total_steps), desc="Prediction progress."):
        with torch.no_grad():
            output = model(current_input)  #  [B, T, C, H, W]

            output_cpu = output.squeeze(0).squeeze(0).cpu().numpy()  # [69, H, W]
            np.save(os.path.join(args['results_path'], f'{backbone}_prediction_step_{step}.npy'), output_cpu)
            logging.info(f"The prediction result for step {step} has been saved.")

            if step < len(true_labels):
                true_label = true_labels[step]
                true_label_np = true_label.cpu().numpy()  # [69, H, W]
                np.save(os.path.join(args['results_path'], f'{backbone}_true_label_step_{step}.npy'), true_label_np)
                logging.info(f"The ground truth for step {step} has been saved.")

            current_input = output  # [B, T, C, H, W]

            del output, output_cpu
            torch.cuda.empty_cache()

    logging.info("The inference script has finished running!")

if __name__ == '__main__':
    main()