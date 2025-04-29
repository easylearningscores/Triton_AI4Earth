import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout_rate, res=True):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        self.res = res

    def forward(self, x):
        out = self.layer1(x)
        if self.res:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_rate):
        super(ResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        layers = [Resblock(64, 64, dropout_rate) for _ in range(3)]
        layers += [Resblock(64, 128, dropout_rate, res=False)] + [Resblock(128, 128, dropout_rate) for _ in range(3)]
        layers += [Resblock(128, 256, dropout_rate, res=False)] + [Resblock(256, 256, dropout_rate) for _ in range(5)]
        layers += [Resblock(256, 512, dropout_rate, res=False)] + [Resblock(512, 512, dropout_rate) for _ in range(2)]
        self.middle_layer = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(512, output_channels, kernel_size=3, padding=1)

        # Initialize batch normalization layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x should be of shape (B, T, C, H, W)
        B, T, C, H, W = x.size()
        assert T == 1, "Expected T=1 since TCHW=1 2 64 448"

        x = x.squeeze(1)  # Remove the T dimension; now x is of shape (B, C, H, W)
        out = self.input_layer(x)
        out = self.middle_layer(out)
        out = self.output_layer(out)
        out = out.unsqueeze(1)  # Add the T dimension back; now out is of shape (B, T, C', H, W)
        return out

# # Instantiate the model with the specified input and output channels
# input_channels = 2    # As per your input dimension C=2
# output_channels = 2   # Assuming you want the output to have the same number of channels
# dropout_rate = 0.5    # You can adjust this value as needed

# model = ResNet(input_channels, output_channels, dropout_rate)

# # Example input tensor with dimensions BTCHW = (Batch, Time=1, Channels=2, Height=64, Width=448)
# B = 1   # Example batch size
# T = 1
# C = 2
# H = 64
# W = 448
# input_tensor = torch.randn(B, T, C, H, W)

# # Pass the input tensor through the model
# output = model(input_tensor)

# print(f"Input shape: {input_tensor.shape}")   # Should print torch.Size([B, 1, 2, 64, 448])
# print(f"Output shape: {output.shape}")        # Should print torch.Size([B, 1, 2, 64, 448])