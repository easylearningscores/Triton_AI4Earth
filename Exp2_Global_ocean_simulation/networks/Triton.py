import torch
from torch import nn
import math
from timm.layers import DropPath, trunc_normal_

def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(ConvMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super(ConvBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = (
                m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x))))
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        init_value=1e-6,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super(SelfAttentionBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_1', 'gamma_2'}

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x

def UniformerSubBlock(
    embed_dims,
    mlp_ratio=4.,
    drop=0.,
    drop_path=0.,
    init_value=1e-6,
    block_type='Conv'
):
    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        # return ConvBlock(dim=embed_dims, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        return SelfAttentionBlock(
            dim=embed_dims,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value
        )
    else:
        return SelfAttentionBlock(
            dim=embed_dims,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value
        )

class SpatioTemporalEvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_resolution=None,
        mlp_ratio=8.,
        drop=0.0,
        drop_path=0.0,
        layer_i=0
    ):
        super(SpatioTemporalEvolutionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
        self.block = UniformerSubBlock(
            in_channels,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            block_type=block_type
        )

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        z = self.block(x)
        if self.in_channels != self.out_channels:
            z = self.reduction(z)
        return z

class SpatioTemporalEvolution(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_hid,
        N2,
        input_resolution=None,
        mlp_ratio=4.,
        drop=0.0,
        drop_path=0.1
    ):
        super(SpatioTemporalEvolution, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        evolution_layers = [SpatioTemporalEvolutionBlock(
            channel_in,
            channel_hid,
            input_resolution,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=dpr[0],
            layer_i=0
        )]

        for i in range(1, N2 - 1):
            evolution_layers.append(SpatioTemporalEvolutionBlock(
                channel_hid,
                channel_hid,
                input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=dpr[i],
                layer_i=i
            ))

        evolution_layers.append(SpatioTemporalEvolutionBlock(
            channel_hid,
            channel_in,
            input_resolution,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            layer_i=N2 - 1
        ))
        self.enc = nn.Sequential(*evolution_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        y = z.reshape(B, T, C, H, W)
        return y

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        transpose=False,
        act_norm=False
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvDynamicsLayer(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvDynamicsLayer, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class MultiGroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        act_norm=False
    ):
        super(MultiGroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class AtmosphericEncoder(nn.Module):
    def __init__(self, C_in, spatial_hidden_dim, num_spatial_layers):
        super(AtmosphericEncoder, self).__init__()
        strides = stride_generator(num_spatial_layers)
        self.enc = nn.Sequential(
            ConvDynamicsLayer(C_in, spatial_hidden_dim, stride=strides[0]),
            *[ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s) for s in strides[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

class AtmosphericDecoder(nn.Module):
    def __init__(self, spatial_hidden_dim, C_out, num_spatial_layers):
        super(AtmosphericDecoder, self).__init__()
        strides = stride_generator(num_spatial_layers, reverse=True)
        self.dec = nn.Sequential(
            *[ConvDynamicsLayer(spatial_hidden_dim, spatial_hidden_dim, stride=s, transpose=True) for s in strides[:-1]],
            ConvDynamicsLayer(2 * spatial_hidden_dim, spatial_hidden_dim, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(spatial_hidden_dim, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Triton(nn.Module):
    def __init__(
        self,
        params,
        shape_in=(97, 120, 240),
        spatial_hidden_dim=256,
        output_channels=93,
        temporal_hidden_dim=512,
        num_spatial_layers=4,
        num_temporal_layers=8,
        in_time_seq_length=10,
        out_time_seq_length=10
    ):
        super(Triton, self).__init__()
        C, H, W = shape_in
        T = 1
        self.H1 = int(H / 2 ** (num_spatial_layers / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (num_spatial_layers / 2))
        self.W1 = int(W / 2 ** (num_spatial_layers / 2))
        self.output_dim = output_channels
        self.input_time_seq_length = in_time_seq_length
        self.output_time_seq_length = out_time_seq_length
        
        self.atmospheric_encoder = AtmosphericEncoder(C, spatial_hidden_dim, num_spatial_layers)
        self.temporal_evolution = SpatioTemporalEvolution(
            T * spatial_hidden_dim,
            temporal_hidden_dim,
            num_temporal_layers,
            input_resolution=[self.H1, self.W1],
            mlp_ratio=4.0,
            drop_path=0.1
        )
        self.atmospheric_decoder = AtmosphericDecoder(spatial_hidden_dim, self.output_dim, num_spatial_layers)

    def forward(self, input_state):
        input_state = input_state.unsqueeze(1)
        batch_size, temporal_length, channels, height, width = input_state.shape
        reshaped_input = input_state.view(batch_size * temporal_length, channels, height, width)
        
        encoded_features, skip_connection = self.atmospheric_encoder(reshaped_input)
        _, encoded_channels, encoded_height, encoded_width = encoded_features.shape
        encoded_features = encoded_features.view(batch_size, temporal_length, encoded_channels, encoded_height, encoded_width)
        
        temporal_bias = encoded_features
        temporal_hidden = self.temporal_evolution(temporal_bias)
        reshaped_hidden = temporal_hidden.view(batch_size * temporal_length, encoded_channels, encoded_height, encoded_width)
    
        decoded_output = self.atmospheric_decoder(reshaped_hidden, skip_connection)
        final_output = decoded_output.view(batch_size, temporal_length, -1, height, width)
        
        final_output = final_output.squeeze(1) 
        
        return final_output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from thop import profile

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = Triton().to(device)

    input = torch.randn(1, 97, 120, 240).to(device)
    output = net(input)

    macs, params = profile(net, inputs=(input, ))
    
    print('macs: ', macs, 'params: ', params)
    print('macs: %.2f G, params: %.2f M' % (macs / 1000000000.0, params / 1000000.0))
    print(output.shape)