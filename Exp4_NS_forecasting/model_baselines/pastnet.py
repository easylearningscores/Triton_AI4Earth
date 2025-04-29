import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.backends.cudnn as cudnn
import numpy as np
import random
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint_sequential

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Splits the input image into patches and projects to an embedding space.
    """
    def __init__(self, img_size=None, patch_size=8, in_c=13, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # Compute the number of patches in height and width dimensions
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (h, w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # Projection layer: Conv2D to project patches to embedding dimension
        self.projection = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Optional normalization layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure input size matches the expected image size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        '''
        Input shape: [B, C, H, W]
        After Conv2D: [B, embed_dim, H', W'], where H' and W' are reduced by patch_size
        Flatten to shape: [B, embed_dim, H'*W']
        Transpose to shape: [B, H'*W', embed_dim]
        '''
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformers and other models.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Positional Encoding for Transformer-based models.
    """
    def __init__(self, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        super().__init__()
        self.M = M  # Input dimension
        self.F_dim = F_dim  # Fourier feature dimension
        self.H_dim = H_dim  # Hidden dimension in MLP
        self.D = D  # Output embedding dimension
        self.gamma = gamma

        # Linear projection to Fourier feature space
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP to map Fourier features to positional embeddings
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        # Initialize weights of Wr with normal distribution
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        # x shape: [B, N, M], where B=batch size, N=number of tokens, M=input dimension
        B, N, M = x.shape
        # Project input to Fourier space
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        # Concatenate sine and cosine components
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Apply MLP to get positional embeddings
        Y = self.mlp(F)
        PEx = Y.reshape((B, N, self.D))
        return PEx

class AdativeFourierNeuralOperator(nn.Module):
    """
    Adaptive Fourier Neural Operator layer for modeling long-range dependencies.
    """
    def __init__(self, dim, h=14, w=14, fno_blocks=4, fno_bias=False, fno_softshrink=0.0):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0, "Hidden size must be divisible by number of FNO blocks"

        self.scale = 0.02  # Initialization scale
        # Fourier weights and biases
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        if fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        # Element-wise multiplication used in Fourier space
        return torch.einsum('...bd, bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            # Apply bias if specified
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros_like(x)

        # Reshape to 2D spatial grid
        x = x.reshape(B, self.h, self.w, C)
        # Perform 2D Fourier transform
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # Reshape for block-wise operations
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        # Real and imaginary parts
        x_real = x.real
        x_imag = x.imag

        # First layer operations in Fourier space
        x_real_new = F.relu(
            self.multiply(x_real, self.w1[0]) - self.multiply(x_imag, self.w1[1]) + self.b1[0],
            inplace=True
        )
        x_imag_new = F.relu(
            self.multiply(x_real, self.w1[1]) + self.multiply(x_imag, self.w1[0]) + self.b1[1],
            inplace=True
        )
        # Second layer operations in Fourier space
        x_real = self.multiply(x_real_new, self.w2[0]) - self.multiply(x_imag_new, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real_new, self.w2[1]) + self.multiply(x_imag_new, self.w2[0]) + self.b2[1]

        # Stack real and imaginary parts
        x = torch.stack([x_real, x_imag], dim=-1)
        # Apply softshrink activation if specified
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x
        # Convert back to complex type
        x = torch.view_as_complex(x)
        # Reshape back to original dimension
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        # Inverse 2D Fourier transform
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias

class FourierNetBlock(nn.Module):
    """
    A Transformer block with Fourier Neural Operator for modeling global dependencies.
    """
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 h=14,
                 w=14,
                 fno_blocks=4,
                 fno_bias=False,
                 fno_softshrink=0.0,
                 double_skip=False):
        super(FourierNetBlock, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim, h=h, w=w, fno_blocks=fno_blocks,
                                                   fno_bias=fno_bias, fno_softshrink=fno_softshrink)

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.normlayer2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP block
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        # Apply Fourier Neural Operator layer
        x = x + self.drop_path(self.filter(self.normlayer1(x)))
        # Apply MLP layer
        x = x + self.drop_path(self.mlp(self.normlayer2(x)))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim  # D
        self._num_embeddings = num_embeddings  # K

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)  #
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from B, C, H, W -> B, H, W, C
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from B, H, W, C -> B, C, H, W
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    def lookup(self, x):
        embeddings = F.embedding(x, self._embedding)
        return embeddings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    def lookup(self, x):
        embeddings = F.embedding(x, self._embedding)
        return embeddings

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        # input shape: [B, C, H, W]
        x = self._conv_1(inputs)  # [B, hidden_units//2 , H//2, W//2]
        x = F.relu(x)

        x = self._conv_2(x)  # [B, hidden_units, H//4, W//4]
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker,
                          stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

class FPG(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=20,
                 out_channels=20,
                 input_frames=20,
                 embed_dim=768,
                 depth=12,
                 mlp_ratio=4.,
                 uniform_drop=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 dropcls=0.):
        super(FPG, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = input_frames
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_c=in_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # [1, N_patches, embed_dim]
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = self.patch_embed.grid_size[0]
        self.w = self.patch_embed.grid_size[1]
        '''
        stochastic depth decay rule
        '''
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h=self.h,
            w=self.w)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.linearprojection = nn.Sequential(OrderedDict([
            ('transposeconv1', nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=(2, 2), stride=(2, 2))),
            ('act1', nn.Tanh()),
            ('transposeconv2', nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=(2, 2), stride=(2, 2))),
            ('act2', nn.Tanh()),
            ('transposeconv3', nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=(4, 4), stride=(4, 4)))
        ]))

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x):
        '''
        patch_embed:
        [B, T, C, H, W] -> [B*T, num_patches, embed_dim]
        '''
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.linearprojection(x)
        x = x.reshape(B, T, -1, self.h * self.patch_embed.patch_size[0], self.w * self.patch_embed.patch_size[1])
        return x

class DST(nn.Module):
    def __init__(self,
                 in_channel=1,
                 num_hiddens=128,
                 res_layers=2,
                 res_units=32,
                 embedding_nums=512,  # K
                 embedding_dim=64,    # D
                 commitment_cost=0.25):
        super(DST, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = embedding_nums
        self._encoder = Encoder(in_channel, num_hiddens,
                                res_layers, res_units)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        # code book
        self._vq_vae = VectorQuantizerEMA(embedding_nums,
                                          embedding_dim,
                                          commitment_cost,
                                          decay=0.99)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                res_layers,
                                res_units,
                                in_channel)

    def forward(self, x):
        # input shape : [B, C, H, W]
        z = self._encoder(x)  # [B, hidden_units, H//4, W//4]
        # [B, embedding_dims, H//4, W//4] z -> encoding
        z = self._pre_vq_conv(z)
        # quantized -> embedding, quantized is similar to the encoder output in videoGPT
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

    def get_embedding(self, x):
        return self._pre_vq_conv(self._encoder(x))

    def get_quantization(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, quantized, _, _ = self._vq_vae(z)
        return quantized

    def reconstruct_img_by_embedding(self, embedding):
        loss, quantized, perplexity, _ = self._vq_vae(embedding)
        return self._decoder(quantized)

    def reconstruct_img(self, q):
        return self._decoder(q)

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

class DynamicPropagation(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(DynamicPropagation, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(
            channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(
                channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid,
                          channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(
            channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(
                2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//
                          2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.ModuleList(enc_layers)
        self.dec = nn.ModuleList(dec_layers)

    def forward(self, input_state):
        B, T, C, H, W = input_state.shape
        input_state = input_state.reshape(B, T*C, H, W)
        # encoder
        skips = []
        hidden_embed = input_state
        for i in range(self.N_T):
            hidden_embed = self.enc[i](hidden_embed)
            if i < self.N_T - 1:
                skips.append(hidden_embed)

        # decoder
        hidden_embed = self.dec[0](hidden_embed)
        for i in range(1, self.N_T):
            hidden_embed = self.dec[i](torch.cat([hidden_embed, skips[-i]], dim=1))

        output_state = hidden_embed.reshape(B, T, C, H, W)
        return output_state

class PastNetModel(nn.Module):
    def __init__(self,
                 shape_in,
                 hid_T=256,
                 N_T=8,
                 incep_ker=[3, 5, 7, 11],
                 groups=8,
                 res_units=32,
                 res_layers=2,
                 embedding_nums=512,
                 embedding_dim=64):
        super(PastNetModel, self).__init__()
        T, C, H, W = shape_in
        self.DST_module = DST(in_channel=C,
                              res_units=res_units,
                              res_layers=res_layers,
                              embedding_dim=embedding_dim,
                              embedding_nums=embedding_nums)

        self.FPG_module = FPG(img_size=(H, W),  
                              patch_size=16,     
                              in_channels=C,
                              out_channels=C,
                              embed_dim=128,
                              input_frames=T,
                              depth=1,
                              mlp_ratio=2.,
                              uniform_drop=False,
                              drop_rate=0.,
                              drop_path_rate=0.,
                              norm_layer=None,
                              dropcls=0.)

        self.DynamicPro = DynamicPropagation(T*embedding_dim, hid_T, N_T, incep_ker, groups)

    def forward(self, input_frames):
        B, T, C, H, W = input_frames.shape
        # PDE features from FPG module
        pde_features = self.FPG_module(input_frames)
        input_features = input_frames.view(B * T, C, H, W)
        encoder_embed = self.DST_module._encoder(input_features)
        z = self.DST_module._pre_vq_conv(encoder_embed)
        vq_loss, Latent_embed, _, _ = self.DST_module._vq_vae(z)

        _, C_, H_, W_ = Latent_embed.shape
        Latent_embed = Latent_embed.reshape(B, T, C_, H_, W_)

        # Latent_embed_flat = Latent_embed.view(B, T*C_, H_, W_)
        hidden_dim = self.DynamicPro(Latent_embed)
        B_, T_, C_out, H_, W_ = hidden_dim.shape

        # Reshape hidden_dim to feed into decoder
        hid = hidden_dim.view(B_ * T_, C_out, H_, W_)

        # Reconstruct features from decoder
        predicti_feature = self.DST_module._decoder(hid)
        predicti_feature = predicti_feature.view(B, T, C, H, W)

        # Combine with PDE features
        output = predicti_feature + pde_features

        return output

if __name__ == "__main__":
    set_seed(42)

    B = 1
    T = 1   
    C = 2    
    H = 64  
    W = 448 

    model = PastNetModel(
        shape_in=(T, C, H, W),
        hid_T=256,
        N_T=8,
        incep_ker=[3, 5, 7, 11],
        groups=8,
        res_units=32,
        res_layers=2,
        embedding_nums=512,
        embedding_dim=64
    )

    input_frames = torch.randn(B, T, C, H, W)

    output = model(input_frames)

    print("Input shape:", input_frames.shape)
    print("Output shape:", output.shape)