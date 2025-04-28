import numpy as np
import netCDF4 as nc
import torch
import torch.utils.data as data

class train_Dataset(data.Dataset):
    def __init__(self, args):
        super(train_Dataset, self).__init__()
        self.args = args
        self.years = range(1993, 2018)
        self.dates = range(12, 357, 3)
        self.indices = []

        for m in self.years:
            train_data = nc.Dataset(f'{self.args["data_path"]}/{m}_norm.nc')
            max_time_index = train_data.variables['atmosphere_variables'].shape[0] - 1  
            train_data.close()  

            for n in self.dates:
                input_start = n - self.args['atmosphere_lead_time'] + 1
                target_end = n + self.args['ocean_lead_time'] + 1

                if input_start >= 0 and target_end <= max_time_index:
                    self.indices.append((m, n))

    def __getitem__(self, index):
        year, date = self.indices[index]
        train_data = nc.Dataset(f'{self.args["data_path"]}/{year}_norm.nc')

        # Calculate indices
        input_start = date - self.args['atmosphere_lead_time'] + 1
        input_end = date + 1
        target_start = date + 1
        target_end = date + self.args['ocean_lead_time'] + 1

        # Load input data
        input = train_data.variables['atmosphere_variables'][
            input_start:input_end,
            self.args['variables_input'],
            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']
        ]

        # Load target data
        target = train_data.variables['atmosphere_variables'][
            target_start:target_end,
            self.args['variables_output'],
            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']
        ]

        train_data.close()  # Close the dataset after use

        # Convert to tensors and handle NaNs
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        input = torch.nan_to_num(input, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)

        # Ensure matching time dimensions
        min_time_steps = min(input.shape[0], target.shape[0])
        input = input[:min_time_steps]
        target = target[:min_time_steps]

        return input, target

    def __len__(self):
        return len(self.indices)

class test_Dataset(data.Dataset):
    def __init__(self, args):
        super(test_Dataset, self).__init__()
        self.args = args
        self.years = range(2018, 2022)
        self.dates = range(12, 357, 3)
        self.indices = []

        # Build valid indices to avoid out-of-bounds errors
        for m in self.years:
            test_data = nc.Dataset(f'{self.args["data_path"]}/{m}_norm.nc')
            max_time_index = test_data.variables['atmosphere_variables'].shape[0] - 1  # Adjust for zero-based indexing
            test_data.close()  # Close the dataset after use

            for n in self.dates:
                input_start = n - self.args['atmosphere_lead_time'] + 1
                target_end = n + self.args['ocean_lead_time'] + 1

                # Ensure indices are within bounds
                if input_start >= 0 and target_end <= max_time_index:
                    self.indices.append((m, n))

    def __getitem__(self, index):
        year, date = self.indices[index]
        test_data = nc.Dataset(f'{self.args["data_path"]}/{year}_norm.nc')

        # Calculate indices
        input_start = date - self.args['atmosphere_lead_time'] + 1
        input_end = date + 1
        target_start = date + 1
        target_end = date + self.args['ocean_lead_time'] + 1

        # Load input data
        input = test_data.variables['atmosphere_variables'][
            input_start:input_end,
            self.args['variables_input'],
            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']
        ]

        # Load target data
        target = test_data.variables['atmosphere_variables'][
            target_start:target_end,
            self.args['variables_output'],
            self.args['lat_start']:self.args['lat_end']:self.args['ds_factor'],
            self.args['lon_start']:self.args['lon_end']:self.args['ds_factor']
        ]

        test_data.close()  # Close the dataset after use

        # Convert to tensors and handle NaNs
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        input = torch.nan_to_num(input, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)

        # Ensure matching time dimensions
        min_time_steps = min(input.shape[0], target.shape[0])
        input = input[:min_time_steps]
        target = target[:min_time_steps]

        return input, target

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
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

    train_loader = data.DataLoader(train_dataset, batch_size=1)
    test_loader = data.DataLoader(test_dataset, batch_size=1)

    for inputs, targets in iter(train_loader):
        print(inputs.shape, targets.shape)
        break