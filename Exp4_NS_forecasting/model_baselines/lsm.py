import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import argparse

################################################################
# 定义必要的模型模块
################################################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值，使用标准卷积来减少通道数量
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算padding以匹配尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

################################################################
# Patchify 和 Neural Spectral Block
################################################################
class NeuralSpectralBlock2d(nn.Module):
    def __init__(self, width, num_basis, patch_size=[3, 3], num_token=4):
        super(NeuralSpectralBlock2d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # 基函数
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
                                                                  dtype=torch.float)
        # 将modes_list注册为缓冲区，将在模型移动到GPU时一起移动
        self.register_buffer('modes_list_buffer', self.modes_list)
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # 潜在表示
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv2d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv2d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W = x.shape
        L = H * W
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = self.encoder_attn(x).view(B, C * 2, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3).contiguous()
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C H W
        x_init = x
        B, C, H, W = x.shape
        L = H * W
        latent_token = latent_token.view(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2).contiguous()
        x_tmp = self.decoder_attn(x).view(B, C, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        # 使用存储在缓冲区中的modes_list
        x_sin = torch.sin(self.modes_list_buffer[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list_buffer[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape

        # 检查patch_size是否能整除H和W
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            raise ValueError(f"Input height and width must be divisible by patch_size. Got input size ({H}, {W}) and patch_size {self.patch_size}.")

        # patchify
        x = x.view(x.shape[0], x.shape[1],
                   x.shape[2] // self.patch_size[0], self.patch_size[0],
                   x.shape[3] // self.patch_size[1], self.patch_size[1]).contiguous() \
            .permute(0, 2, 4, 1, 3, 5).contiguous() \
            .view(x.shape[0] * (x.shape[2] // self.patch_size[0]) * (x.shape[3] // self.patch_size[1]), x.shape[1],
                  self.patch_size[0],
                  self.patch_size[1])
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = x.view(B, (H // self.patch_size[0]), (W // self.patch_size[1]), C, self.patch_size[0],
                   self.patch_size[1]).permute(0, 3, 1, 4, 2, 5).contiguous() \
            .view(B, C, H, W).contiguous()
        return x

################################################################
# 定义完整的模型
################################################################
class LSM(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, num_token, num_basis, patch_size, padding, bilinear=True):
        super(LSM, self).__init__()
        in_channels = in_dim
        out_channels = out_dim
        width = d_model
        num_token = num_token
        num_basis = num_basis
        patch_size = [int(x) for x in patch_size.split(',')]
        padding = [int(x) for x in padding.split(',')]
        # 多尺度模块
        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(width * 8, width * 16 // factor)
        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)
        # Patchified Neural Spectral Blocks
        self.process1 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock2d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock2d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock2d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock2d(width * 16 // factor, num_basis, patch_size, num_token)
        # 投影层
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x的输入形状：(B, T, C_in, H, W)，其中T=1
        x = x.squeeze(1)  # 去除时间维度，x的形状：(B, C_in, H, W)
        x = x.permute(0, 2, 3, 1)  # 转换为 (B, H, W, C_in)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # 转换为 (B, C, H, W)

        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[1], :-self.padding[0]]

        x = x.permute(0, 2, 3, 1)  # 转换回 (B, H, W, C)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # 输出调整
        x = x.permute(0, 3, 1, 2)  # 转换为 (B, C_out, H, W)
        x = x.unsqueeze(1)  # 添加时间维度，x的形状：(B, T, C_out, H, W)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, _ = shape
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)

################################################################
# 实例化和测试模型
################################################################
if __name__ == "__main__":
    import argparse

    # 定义args对象，包含模型初始化所需的参数
    in_dim = 1               # 输入维度，根据您的数据调整
    out_dim = 1              # 输出维度，根据您的任务调整
    d_model = 64             # 模型宽度，可根据需求调整
    num_token = 4            # Token数量，可根据需求调整
    num_basis = 16           # 基函数数量，可根据需求调整
    patch_size = '4,4'       # Patch大小，确保能整除下采样后的尺寸
    padding = '0,0'          # Padding大小，格式为字符串，例如'0,0'

    # 实例化模型
    model = LSM(in_dim=in_dim, out_dim=out_dim, d_model=d_model, num_token=num_token, num_basis=num_basis, patch_size=patch_size, padding=padding)

    # 创建一个示例输入数据，假设输入大小为(batch_size, T, in_channels, height, width)
    batch_size = 1
    T = 1
    in_channels = in_dim
    height = 128         # 输入高度
    width = 128         # 输入宽度
    x = torch.randn(batch_size, T, in_channels, height, width)

    # 将输入数据移动到模型所在线程的设备（CPU或GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x = x.to(device)

    # 进行一次前向传播
    output = model(x)

    # 输出结果的形状
    print("输出形状：", output.shape)