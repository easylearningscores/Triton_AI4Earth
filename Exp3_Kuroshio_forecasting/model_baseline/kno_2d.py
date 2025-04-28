# KNO model
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# The structure of Auto-Encoder
class encoder_mlp(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_mlp, self).__init__()
        self.layer = nn.Linear(t_len, op_size)
    def forward(self, x):
        x = self.layer(x)
        return x

class decoder_mlp(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_mlp, self).__init__()
        self.layer = nn.Linear(op_size, t_len)
    def forward(self, x):
        x = self.layer(x)
        return x

class encoder_conv1d(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(t_len, op_size,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class decoder_conv1d(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(op_size, t_len,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class encoder_conv2d(nn.Module):
    def __init__(self, t_len, op_size):
        super(encoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(t_len, op_size,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x

class decoder_conv2d(nn.Module):
    def __init__(self, t_len, op_size):
        super(decoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(op_size, t_len,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x


class Koopman_Operator2D(nn.Module):
    def __init__(self, op_size, modes_x, modes_y):
        super(Koopman_Operator2D, self).__init__()
        self.op_size = op_size
        self.scale = (1 / (op_size * op_size))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def time_marching(self, input, weights):
        return torch.einsum("btxy,tfxy->bfxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y], self.koopman_matrix)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class KNO2d(nn.Module):
    def __init__(self, encoder, decoder, op_size, modes_x=10, modes_y=10, decompose=6, linear_type=True, normalization=False, x_coeff=0.1, skip_coeff=1):
        super(KNO2d, self).__init__()
        self.op_size = op_size
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.x_coeff = x_coeff
        self.skip_coeff = skip_coeff
        
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = Koopman_Operator2D(self.op_size, self.modes_x, self.modes_y)
        self.w0 = nn.Conv2d(op_size, op_size, 1)
        self.linear_type = linear_type
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(op_size)

    def forward_(self, x):
        x_reconstruct = self.enc(x)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        
        x = self.enc(x)
        x = torch.tanh(x)
        x = x.permute(0, 3, 1, 2)
        x_w = x
        
        for i in range(self.decompose):
            x1 = self.koopman_layer(x)
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
                
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
            
        x = x.permute(0, 2, 3, 1)
        x = self.x_coeff * x
        x = self.dec(x)
        return x, x_reconstruct

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, H, W, T)
        x = x.reshape(B*C, H, W, T)  
        
        output, rec = self.forward_(x)
        
        output = output.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)  # (B, T, C, H, W)
        rec = rec.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)
        
        return output, rec

if __name__ == "__main__":    
    # hyper parameters
    t_len = 10
    o = 16
    f_x = 16
    f_y = 16
    r = 8
    encoder = encoder_mlp(t_len, op_size=o)
    decoder = decoder_mlp(t_len, op_size=o)
    model = KNO2d(encoder, decoder, op_size=o, modes_x=f_x, modes_y=f_y, decompose=r)
    
    # Test with new input shape (B, T, C, H, W)
    inputs = torch.rand(1, 10, 2, 256, 256)  # (B, T, C, H, W)
    output, rec = model(inputs)
    print(output.shape, rec.shape)  # Should be (torch.Size([1, 10, 2, 256, 256]), torch.Size([1, 10, 2, 256, 256]))