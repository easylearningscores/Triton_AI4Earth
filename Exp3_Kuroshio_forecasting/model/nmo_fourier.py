import torch
from torch import nn
from model.modules_api.modules import ConvSC, Inception
from model.modules_api.fouriermodules import *
from model.modules_api.evolution import Spatio_temporal_evolution
import math

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y




class Temporal_evo(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, h, w, incep_ker=[3, 5, 7, 11], groups=8):
        super(Temporal_evo, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)

        self.enc = nn.Sequential(*enc_layers)
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.h = h
        self.w = w
        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=channel_hid,
            mlp_ratio=4,
            drop=0.,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h = self.h,
            w = self.w)
            for i in range(12)
        ])
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        bias = x
        x = x.reshape(B, T * C, H, W)

        # downsampling
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        
        # Spectral Domain
        B, D, H, W = z.shape
        N = H * W
        z = z.permute(0, 2, 3, 1)
        z = z.view(B, N, D)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).permute(0, 2, 1)

        z = z.reshape(B, D, H, W)

        # upsampling
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y + bias


    
class NMOModel(nn.Module):
    def __init__(self, shape_in, model_type='uniformer', hid_S=64, output_dim = 4, hid_T=128, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=4, 
                 in_time_seq_length=10, out_time_seq_length=10):
        super(NMOModel, self).__init__()
        T, C, H, W = shape_in
        self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))
        self.out_dim = output_dim
        self.in_time_seq_length = in_time_seq_length
        self.out_time_seq_length = out_time_seq_length
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Temporal_evo(T*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups) #
        self.temporal_evolution = Spatio_temporal_evolution(T*hid_S, hid_T, N_T,
                                                            input_resolution=[self.H1, self.W1],
                                                            model_type = model_type,
                                                            mlp_ratio=4.,
                                                            drop_path=0.1)

        self.dec = Decoder(hid_S, self.out_dim, N_S)


    def _forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        bias = z
        bias_hid = self.temporal_evolution(bias)
        hid = bias_hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, -1, H, W)
        return Y
    
    def forward(self, xx):
        yy = self._forward(xx)
        in_time_seq_length, out_time_seq_length = self.in_time_seq_length, self.out_time_seq_length
        if out_time_seq_length == in_time_seq_length:
            y_pred = yy
        if out_time_seq_length < in_time_seq_length:
            y_pred = yy[:, :out_time_seq_length]
        elif out_time_seq_length > in_time_seq_length:
            y_pred = [yy]
            d = out_time_seq_length // in_time_seq_length
            m = out_time_seq_length % in_time_seq_length
            
            for _ in range(1, d):
                cur_seq = self._forward(y_pred[-1])
                y_pred.append(cur_seq)
            
            if m != 0:
                cur_seq = self._forward(y_pred[-1])
                y_pred.append(cur_seq[:, :m])
            
            y_pred = torch.cat(y_pred, dim=1)
        
        return y_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    inputs = torch.randn(1, 10, 7, 600, 600)
    model = NMOModel(shape_in=(10, 7, 600, 600), hid_S=64, output_dim = 1, hid_T=128)
    # print(model)
    output = model(inputs)

    print(output.shape)

    # Print the number of parameters
    #print(f'The model has {count_parameters(model):,} trainable parameters.')
