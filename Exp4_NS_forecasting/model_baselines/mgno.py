import torch
import torch.nn as nn

# Define the MgIte class (no changes needed)
class MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
        self.A = A  # Linear operator (e.g., Conv2d)
        self.S = S  # Smoother operator (e.g., Conv2d)

    def forward(self, out):
        u, f = out
        u = u + self.S(f - self.A(u))
        out = (u, f)
        return out

# Define the initial iteration class (no changes needed)
class MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S  # Initial smoother operator (e.g., Conv2d)

    def forward(self, f):
        u = self.S(f)
        return (u, f)

# Define the Restrict class (no changes needed)
class Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi  # Projection operator for u
        self.R = R    # Restriction operator for residual
        self.A = A    # Linear operator

    def forward(self, out):
        u, f = out
        f = self.R(f - self.A(u))
        u = self.Pi(u)
        out = (u, f)
        return out

# Define the MgConv class (no changes needed)
class MgConv(nn.Module):
    def __init__(self, num_iteration, out_channel, in_channel,
                 height=64, width=64, padding_mode='zeros', normalization=True):
        super().__init__()
        self.num_iteration = num_iteration
        self.out_channel = out_channel
        self.in_channel = in_channel

        # Calculate resolutions for downsampling levels
        self.resolutions = self.calculate_downsampling_levels(
            height, width, kernel_sizes=[3] * (len(num_iteration) - 1))
        self.upsample_kernels = self.calculate_adjusted_upsample_kernels_simple(
            self.resolutions[-1], self.resolutions)

        # Create normalization layers for each level (except the last one)
        if normalization:
            self.norm_layer_list = nn.ModuleList([
                nn.GroupNorm(1, out_channel, eps=1e-5, affine=True)
                for _ in range(len(num_iteration) - 1)
            ])
        else:
            self.norm_layer_list = nn.ModuleList([
                nn.Identity() for _ in range(len(num_iteration) - 1)
            ])

        # Create transposed convolution layers for upsampling
        self.rt_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                out_channel, out_channel,
                kernel_size=self.upsample_kernels[j], stride=2, padding=0, bias=False
            )
            for j in range(len(num_iteration) - 1)
        ])

        self.layers = nn.ModuleList()
        self.post_smooth_layers = nn.ModuleList()

        layer = []
        for l, num_iteration_l in enumerate(num_iteration):

            post_smooth_layers = []
            # Pre-smoothing iterations
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(
                    in_channel, out_channel, kernel_size=3, stride=1, padding=1,
                    padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layer.append(MgIte_init(S))  # Initial iteration at the first level
                else:
                    A = nn.Conv2d(
                        out_channel, in_channel, kernel_size=3, stride=1, padding=1,
                        padding_mode=padding_mode)
                    layer.append(MgIte(A, S))
            # Post-smoothing iterations
            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(
                        in_channel, out_channel, kernel_size=3, stride=1, padding=1,
                        padding_mode=padding_mode)
                    A = nn.Conv2d(
                        out_channel, in_channel, kernel_size=3, stride=1, padding=1,
                        padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            self.layers.append(nn.Sequential(*layer))
            self.post_smooth_layers.append(nn.Sequential(*post_smooth_layers))

            # If not at the last level, add Restrict layer for downsampling
            if l < len(num_iteration) - 1:
                A = nn.Conv2d(
                    out_channel, in_channel, kernel_size=3, stride=1, padding=1,
                    padding_mode=padding_mode)
                Pi = nn.Conv2d(
                    out_channel, out_channel, kernel_size=3, stride=2, padding=1,
                    bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(
                    in_channel, in_channel, kernel_size=3, stride=2, padding=1,
                    bias=False, padding_mode=padding_mode)

                layer = [Restrict(Pi, R, A)]  # Start a new layer list with the Restrict operation

    def calculate_downsampling_levels(self, H, W, kernel_sizes, stride=2, padding=1):
        """Calculate the height and width at each downsampling level."""
        height_sizes = [H]
        width_sizes = [W]
        for kernel_size in kernel_sizes:
            # Calculate output height
            H_out = (H + 2 * padding - kernel_size) // stride + 1
            height_sizes.append(H_out)
            H = H_out
            # Calculate output width
            W_out = (W + 2 * padding - kernel_size) // stride + 1
            width_sizes.append(W_out)
            W = W_out
        return list(zip(height_sizes, width_sizes))

    def calculate_adjusted_upsample_kernels_simple(self, final_size, downsampling_sizes, stride=2, padding=0):
        """Calculate the kernel sizes needed for transposed convolutions during upsampling."""
        adjusted_kernel_sizes = []
        for i in range(len(downsampling_sizes) - 1, 0, -1):
            H_in, W_in = downsampling_sizes[i]
            H_out, W_out = downsampling_sizes[i - 1]
            kernel_size_h = H_out - stride * (H_in - 1)
            kernel_size_w = W_out - stride * (W_in - 1)
            adjusted_kernel_sizes.append((kernel_size_h, kernel_size_w))
        return adjusted_kernel_sizes[::-1]

    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f

        # Downsampling path
        for l in range(len(self.num_iteration)):
            out = self.layers[l](out)
            out_list[l] = out

        # Upsampling path
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_upsampled = self.rt_layers[j](out_list[j + 1][0])
            u_post = self.norm_layer_list[j](u + u_upsampled)
            out = (u_post, f)
            out_list[j] = self.post_smooth_layers[j](out)

        return out_list[0][0]

# Define the MgNO class, modify forward method to handle T dimension
class MgNO(nn.Module):
    def __init__(self, num_layer, out_channel, in_channel, num_iteration,
                 output_dim=1, height=64, width=448,
                 normalizer=None, activation='gelu', init=False):
        super().__init__()
        self.num_layer = num_layer
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.num_iteration = num_iteration

        # Create a list of MgConv layers
        self.conv_list = nn.ModuleList([
            MgConv(
                num_iteration, out_channel, out_channel,
                height=height, width=width
            ) for _ in range(num_layer)
        ])

        self.last_layer = nn.Conv2d(out_channel, output_dim, kernel_size=1)
        self.normalizer = normalizer

        # Choose activation function
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise NameError('Invalid activation function specified')

    def forward(self, u):
        # u is of shape [B, T, C, H, W]
        B, T, C, H, W = u.size()
        # Reshape u to [B*T, C, H, W]
        u = u.view(B * T, C, H, W)

        # Process through the layers
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))

        u = self.last_layer(u)

        if self.normalizer:
            u = self.normalizer.decode(u)

        # Reshape back to [B, T, output_dim, H, W]
        u = u.view(B, T, -1, H, W)
        return u

# Testing the code with input shape BTCHW
if __name__ == "__main__":
    # Define input dimensions
    batch_size = 1
    T = 1  # Time dimension size
    height = 64
    width = 448
    in_channels = 2
    out_channels = 2
    num_iteration = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

    # Instantiate the MgNO model with output_dim=2
    model = MgNO(
        num_layer=6,
        out_channel=out_channels,
        in_channel=in_channels,
        num_iteration=num_iteration,
        output_dim=2,
        height=height,
        width=width
    ).to('cuda')

    # Create random input data with the specified dimensions
    x = torch.randn(batch_size, T, in_channels, height, width).to('cuda')

    # Measure execution time
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    tic.record()

    # Run the model without computing gradients (inference mode)
    with torch.no_grad():
        for _ in range(10):
            output = model(x)

    toc.record()
    torch.cuda.synchronize()
    print(f"Execution time: {tic.elapsed_time(toc)} milliseconds")
    print(f"Output size: {output.size()}")