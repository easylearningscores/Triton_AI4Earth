import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.to(next(self.parameters()).device)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(next(self.parameters()).device)
        t_emb = self.mlp(t_freq)
        t_emb = t_emb.to(next(self.parameters()).device)
        return t_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=(32, 32),
        patch_size=(2, 2),
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=None,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(
            img_size=input_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # 将使用固定的 sin-cos 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        # 如果 num_classes 不为 None，才初始化 y_embedder
        if num_classes is not None:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
            self.y_embedder = None

    def initialize_weights(self):
        # 初始化 Transformer 层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 获取网格尺寸用于位置嵌入
        grid_size_h, grid_size_w = self.x_embedder.grid_size
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (grid_size_h, grid_size_w))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化 patch_embed，如同 nn.Linear（而不是 nn.Conv2d）
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化时间步嵌入 MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 将 DiT 块中的 adaLN 调制层初始化为零
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 将输出层初始化为零
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size[0]*patch_size[1]*C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p_h, p_w = self.x_embedder.patch_size  # 元组
        h_patches, w_patches = self.x_embedder.grid_size
        assert h_patches * w_patches == x.shape[1], "Mismatch in number of patches"

        x = x.reshape(shape=(x.shape[0], h_patches, w_patches, p_h, p_w, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h_patches * p_h, w_patches * p_w))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels or None
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)，其中 T = H * W / (patch_size[0] * patch_size[1])
        t = self.t_embedder(t)                   # (N, D)
        if self.y_embedder is not None and y is not None:
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t                                    # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size[0] * patch_size[1] * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: (grid_size_h, grid_size_w)
    return:
    pos_embed: [grid_size_h*grid_size_w, embed_dim] 或 [1+grid_size_h*grid_size_w, embed_dim]
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 这里 w 先行
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, grid_size[0] * grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # 使用一半的维度来编码 grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 每个位置的输出维度
    pos: 要编码的位置列表：大小 (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                               Other Components                                #
#################################################################################

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class ConvSC(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, transpose=False):
        super(ConvSC, self).__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                           padding=1, output_padding=stride-1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Inception(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, incep_ker=[3,5,7,11], groups=4):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=incep_ker[0], padding=incep_ker[0]//2, groups=groups)
        self.branch3 = nn.Conv2d(in_channels, hidden_channels, kernel_size=incep_ker[1], padding=incep_ker[1]//2, groups=groups)
        self.branch4 = nn.Conv2d(in_channels, hidden_channels, kernel_size=incep_ker[2], padding=incep_ker[2]//2, groups=groups)
        self.branch5 = nn.Conv2d(in_channels, hidden_channels, kernel_size=incep_ker[3], padding=incep_ker[3]//2, groups=groups)
        self.conv = nn.Conv2d(hidden_channels * 5, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv(x)
        x = self.act(self.norm(x))
        return x

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        layers = [ConvSC(C_in, C_hid, stride=strides[0])]
        for s in strides[1:]:
            layers.append(ConvSC(C_hid, C_hid, stride=s))
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
        return x, skips  # 返回所有的 skips

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        layers = []
        for s in strides[:-1]:
            layers.append(ConvSC(C_hid, C_hid, stride=s, transpose=True))
        layers.append(ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True))
        self.dec = nn.Sequential(*layers)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, skip):
        for i in range(len(self.dec)-1):
            hid = self.dec[i](hid)
        hid = self.dec[-1](torch.cat([hid, skip], dim=1))
        return self.readout(hid)

class Temporal_evo(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, h, w, incep_ker=[3, 5, 7, 11], groups=8):
        super(Temporal_evo, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for _ in range(1, N_T - 1):
            dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # Downsampling
        skips = []
        for i in range(self.N_T):
            x = self.enc[i](x)
            if i < self.N_T - 1:
                skips.append(x)

        # Upsampling
        x = self.dec[0](x)
        for i in range(1, self.N_T):
            x = self.dec[i](torch.cat([x, skips[-i]], dim=1))

        x = x.reshape(B, T, C, H, W)
        return x

class Dit(nn.Module):
    def __init__(self, shape_in, hid_S=32, hid_T=64, N_S=4, N_T=8, time_step=1000, incep_ker=[3,5,7,11], groups=4, 
                 in_time_seq_length=10, out_time_seq_length=10):
        super(Dit, self).__init__()
        B, T, C, H, W = shape_in

        strides = stride_generator(N_S)
        num_stride2_layers = strides[:N_S].count(2)
        self.downsample_factor = 2 ** num_stride2_layers
        self.H1 = H // self.downsample_factor
        self.W1 = W // self.downsample_factor

        self.in_time_seq_length = in_time_seq_length
        self.out_time_seq_length = out_time_seq_length
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Temporal_evo(T*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups)
        self.dit_block = DiT(
            input_size=(self.H1, self.W1),
            patch_size=(1, 1),  # Changed patch_size to (1, 1)
            in_channels=T*hid_S,
            hidden_size=256,
            depth=12,
            num_heads=2,
            mlp_ratio=4.0,
            class_dropout_prob=0.0,
            num_classes=None,
            learn_sigma=False,
        )

        self.dec = Decoder(hid_S, C, N_S)
        self.time_step = torch.randint(0, time_step, (B,))

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skips = self.enc(x)
        skip = skips[0] 
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        bias = z.reshape(B, T*C_, H_, W_)
        bias_hid = self.dit_block(bias, self.time_step)
        hid = bias_hid.reshape(B*T, C_, H_, W_)  # Now the dimensions should match

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, -1, H, W)
        return Y

if __name__ == '__main__':
    inputs = torch.randn(1, 10, 2, 64, 448)
    model = Dit(shape_in=(1, 10, 2, 64, 448))
    output = model(inputs)
    print('inputs shape:', inputs.shape)
    print('output shape:', output.shape)
