import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

torch.manual_seed(0)
np.random.seed(0)

################################################################
# Fourier Layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It performs FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # Initialize weights with complex numbers
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something)
        x_ft = torch.fft.rfft2(x, norm='forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='forward')
        return x

################################################################
# Fourier Neural Operator 2D
################################################################

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, C_in=1, C_out=1):
        super(FNO2d, self).__init__()

        """
        FNO2d model adapted to inputs with variable channels and spatial resolution.

        Expected input shape: (batch_size, time_steps=1, channels=C_in, height=H, width=W)
        Output shape: (batch_size, time_steps=1, channels=C_out, height=H, width=W)
        """

        self.modes1 = modes1  # Number of Fourier modes in x direction
        self.modes2 = modes2  # Number of Fourier modes in y direction
        self.width = width
        self.padding = 2  # Padding for non-periodic input, can be adjusted

        self.C_in = C_in
        self.C_out = C_out

        self.fc0 = nn.Linear(self.C_in + 2, self.width)  # Input channels + 2 coordinates (x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.C_out)  # Output channels

    def forward(self, x):
        # Expected input shape: (B, T, C_in, H, W)
        B, T, C_in, H, W = x.shape
        # Since T=1, we can squeeze the time dimension
        x = x.squeeze(1)  # Remove the time dimension, shape becomes (B, C_in, H, W)
        x = x.permute(0, 2, 3, 1)  # Permute to shape (B, H, W, C_in)

        # Get grid and concatenate with input
        grid = self.get_grid(x.shape, x.device)  # Shape: (B, H, W, 2)
        x = torch.cat((x, grid), dim=-1)  # Shape becomes (B, H, W, C_in+2)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # Shape: (B, width, H, W)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)  # Shape: (B, H, W, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # Shape: (B, H, W, C_out)

        # Adjust output shape to (B, T, C_out, H, W)
        x = x.permute(0, 3, 1, 2)  # Shape: (B, C_out, H, W)
        x = x.unsqueeze(1)  # Add time dimension T=1, shape becomes (B, 1, C_out, H, W)
        return x  # Final output shape: (B, T, C_out, H, W)

    def get_grid(self, shape, device):
        '''
        Returns a grid of shape (batchsize, H, W, 2)
        '''
        batchsize, size_x, size_y, _ = shape
        gridx = torch.linspace(0, 1, steps=size_x, device=device)
        gridx = gridx.view(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, steps=size_y, device=device)
        gridy = gridy.view(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)  # Shape: (B, H, W, 2)

################################################################
# Testing the Model with New Dimensions
################################################################

if __name__ == '__main__':

    modes1 = 16   # Adjusted modes, should be less than or equal to H/2
    modes2 = 16   # Adjusted modes, should be less than or equal to W/2
    width = 64    # Width of the neural network
    batch_size = 1  # You can adjust the batch size as needed

    T = 1         # Time steps, remains 1 as per your data
    C_in = 1      # Input channels, adjusted to 1
    C_out = 1     # Output channels, adjusted to 1
    H, W = 128, 128  # Height and Width of the input, adjusted to 128 x 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO2d(modes1, modes2, width, C_in=C_in, C_out=C_out).to(device)

    x = torch.randn(batch_size, T, C_in, H, W).to(device)
    print("Input shape:", x.shape)
    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)