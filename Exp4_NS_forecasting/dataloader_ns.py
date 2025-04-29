import torch
import torch.utils.data as data_utils

class train_Dataset(data_utils.Dataset):
    def __init__(self, data, args):
        super(train_Dataset, self).__init__()
        self.args = args
        self.input_length = args['input_length']
        self.target_length = args['target_length']
        self.downsample_factor = args['downsample_factor']

        # The data is expected to be a tensor of shape [num_samples, time_steps, H, W]
        # Add a variables dimension to make it [num_samples, time_steps, variables, H, W]
        # Since we have only one variable (vorticity), variables = 1
        self.data = data.unsqueeze(2)  # Shape: [num_samples, time_steps, 1, H, W]

        # Split the data into training set (first 80%)
        total_samples = self.data.shape[0]
        self.start_index = 0
        self.end_index = int(0.8 * total_samples)
        self.data = self.data[self.start_index:self.end_index]

        self.num_samples = self.data.shape[0]
        self.num_time_steps = self.data.shape[1]
        self.variables_input = args.get('variables_input', [0])
        self.variables_output = args.get('variables_output', [0])

        # Create indices for sampling
        self.sample_indices = []

        max_t = self.num_time_steps - self.input_length - self.target_length + 1
        for s in range(self.num_samples):
            for t in range(self.input_length, max_t + self.input_length):
                self.sample_indices.append((s, t))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        s, t = self.sample_indices[idx]

        # Extract input and target sequences
        input_seq = self.data[s, t - self.input_length:t, self.variables_input, :, :]  # [input_length, variables_input, H, W]
        target_seq = self.data[s, t:t + self.target_length, self.variables_output, :, :]  # [target_length, variables_output, H, W]

        # Apply downsampling if needed
        dsf = self.downsample_factor
        input_seq = input_seq[:, :, ::dsf, ::dsf]
        target_seq = target_seq[:, :, ::dsf, ::dsf]

        input_seq = input_seq.float()
        target_seq = target_seq.float()

        return input_seq, target_seq  # Shapes: [input_length, variables_input, H', W']

class test_Dataset(data_utils.Dataset):
    def __init__(self, data, args):
        super(test_Dataset, self).__init__()
        self.args = args
        self.input_length = args['input_length']
        self.target_length = args['target_length']
        self.downsample_factor = args['downsample_factor']

        # The data is expected to be a tensor of shape [num_samples, time_steps, H, W]
        # Add a variables dimension to make it [num_samples, time_steps, variables, H, W]
        self.data = data.unsqueeze(2)  # Shape: [num_samples, time_steps, 1, H, W]

        # Split the data into test set (last 10%)
        total_samples = self.data.shape[0]
        self.start_index = int(0.9 * total_samples)
        self.end_index = total_samples
        self.data = self.data[self.start_index:self.end_index]

        self.num_samples = self.data.shape[0]
        self.num_time_steps = self.data.shape[1]
        self.variables_input = args.get('variables_input', [0])
        self.variables_output = args.get('variables_output', [0])

        # Create indices for sampling
        self.sample_indices = []

        max_t = self.num_time_steps - self.input_length - self.target_length + 1
        for s in range(self.num_samples):
            for t in range(self.input_length, max_t + self.input_length):
                self.sample_indices.append((s, t))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        s, t = self.sample_indices[idx]

        # Extract input and target sequences
        input_seq = self.data[s, t - self.input_length:t, self.variables_input, :, :]  # [input_length, variables_input, H, W]
        target_seq = self.data[s, t:t + self.target_length, self.variables_output, :, :]  # [target_length, variables_output, H, W]

        # Apply downsampling if needed
        dsf = self.downsample_factor
        input_seq = input_seq[:, :, ::dsf, ::dsf]
        target_seq = target_seq[:, :, ::dsf, ::dsf]

        input_seq = input_seq.float()
        target_seq = target_seq.float()

        return input_seq, target_seq  # Shapes: [target_length, variables_output, H', W']