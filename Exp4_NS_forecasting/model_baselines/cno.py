import torch
import torch.nn as nn
from torch.nn import LeakyReLU as LReLu

class CNOBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size_h,
                 in_size_w,
                 out_size_h,
                 out_size_w,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(CNOBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size_h = in_size_h
        self.in_size_w = in_size_w
        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.conv_kernel = conv_kernel
        self.batch_norm_flag = batch_norm
        
        #---------- Filter properties -----------
        self.critically_sampled = False # We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.critically_sampled = True
        self.in_cutoff_h  = self.in_size_h / cutoff_den
        self.in_cutoff_w  = self.in_size_w / cutoff_den
        self.out_cutoff_h = self.out_size_h / cutoff_den
        self.out_cutoff_w = self.out_size_w / cutoff_den

        self.in_halfwidth_h =  half_width_mult*self.in_size_h - self.in_size_h / cutoff_den
        self.in_halfwidth_w =  half_width_mult*self.in_size_w - self.in_size_w / cutoff_den
        self.out_halfwidth_h = half_width_mult*self.out_size_h - self.out_size_h / cutoff_den
        self.out_halfwidth_w = half_width_mult*self.out_size_w - self.out_size_w / cutoff_den



        pad = (self.conv_kernel - 1) // 2
        self.convolution = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                           kernel_size=self.conv_kernel,
                                           padding=pad)
    
        if self.batch_norm_flag:
            self.batch_norm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm = None
        self.activation = LReLu()  #
        
    def forward(self, x):
        x = self.convolution(x)
        if self.batch_norm_flag:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

class LiftProjectBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size_h,
                 in_size_w,
                 out_size_h,
                 out_size_w,
                 latent_dim = 64,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(LiftProjectBlock, self).__init__()
    
        self.inter_CNOBlock = CNOBlock(in_channels=in_channels,
                                       out_channels=latent_dim,
                                       in_size_h=in_size_h,
                                       in_size_w=in_size_w,
                                       out_size_h=out_size_h,
                                       out_size_w=out_size_w,
                                       cutoff_den=cutoff_den,
                                       conv_kernel=conv_kernel,
                                       filter_size=filter_size,
                                       lrelu_upsampling=lrelu_upsampling,
                                       half_width_mult=half_width_mult,
                                       radial=radial,
                                       batch_norm=batch_norm,
                                       activation=activation)
        
        pad = (conv_kernel - 1) // 2
        self.convolution = torch.nn.Conv2d(in_channels=latent_dim, out_channels=out_channels,
                                           kernel_size=conv_kernel, stride=1,
                                           padding=pad)
        
        self.batch_norm_flag = batch_norm
        if self.batch_norm_flag:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = None
        
    def forward(self, x):
        x = self.inter_CNOBlock(x)
        
        x = self.convolution(x)
        if self.batch_norm_flag:
            x = self.batch_norm(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 size_h,
                 size_w,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size_h = size_h
        self.size_w = size_w
        self.conv_kernel = conv_kernel
        self.batch_norm_flag = batch_norm

        #---------- Filter properties -----------
        self.critically_sampled = False # We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.critically_sampled = True
        self.cutoff_h  = self.size_h / cutoff_den        
        self.cutoff_w  = self.size_w / cutoff_den        
        self.halfwidth_h =  half_width_mult*self.size_h - self.size_h / cutoff_den
        self.halfwidth_w =  half_width_mult*self.size_w - self.size_w / cutoff_den

        #-----------------------------------------
        
        pad = (self.conv_kernel - 1) // 2
        self.convolution1 = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                            kernel_size=self.conv_kernel, stride=1,
                                            padding=pad)
        self.convolution2 = torch.nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                            kernel_size=self.conv_kernel, stride=1,
                                            padding=pad)
        
        if self.batch_norm_flag:
            self.batch_norm1 = nn.BatchNorm2d(self.channels)
            self.batch_norm2 = nn.BatchNorm2d(self.channels)
        else:
            self.batch_norm1 = self.batch_norm2 = None
        self.activation = LReLu()

    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm_flag:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm_flag:
            out = self.batch_norm2(out)
        
        return x + out

class CNO(nn.Module):
    def __init__(self,  
                 in_dim,                    
                 in_size_h,                 
                 in_size_w,               
                 N_layers,                  
                 N_res = 1,                 
                 N_res_neck = 6,            
                 channel_multiplier = 32,  
                 conv_kernel=3,             
                 cutoff_den = 2.0001,       
                 filter_size=6,            
                 lrelu_upsampling = 2,      
                 half_width_mult  = 0.8,    
                 radial = False,            
                 batch_norm = True,      
                 out_dim = 10,            
                 out_size_h = 1,        
                 out_size_w = 1,          
                 expand_input = False,     
                 latent_lift_proj_dim = 64, 
                 add_inv = True,            
                 activation = 'cno_lrelu'  
                ):
        
        super(CNO, self).__init__()


        self.N_layers = int(N_layers)
        
        self.lift_dim = channel_multiplier // 2         
        self.out_dim  = out_dim
        
        self.add_inv = add_inv
        
        self.channel_multiplier = channel_multiplier        
        
        if radial == 0:
            self.radial = False
        else:
            self.radial = True
        

        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i * self.channel_multiplier)
        
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]  

        self.inv_features = self.decoder_features_in.copy()
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])  

        
        if not expand_input:
            latent_size_h = in_size_h  
            latent_size_w = in_size_w  
        else:
            down_exponent = 2 ** N_layers
            latent_size_h = in_size_h - (in_size_h % down_exponent) + down_exponent
            latent_size_w = in_size_w - (in_size_w % down_exponent) + down_exponent
        
        if out_size_h == 1:
            latent_size_out_h = latent_size_h
        else:
            if not expand_input:
                latent_size_out_h = out_size_h 
            else:
                down_exponent = 2 ** N_layers
                latent_size_out_h = out_size_h - (out_size_h % down_exponent) + down_exponent

        if out_size_w == 1:
            latent_size_out_w = latent_size_w
        else:
            if not expand_input:
                latent_size_out_w = out_size_w 
            else:
                down_exponent = 2 ** N_layers
                latent_size_out_w = out_size_w - (out_size_w % down_exponent) + down_exponent
        
        self.encoder_sizes_h = []
        self.encoder_sizes_w = []
        self.decoder_sizes_h = []
        self.decoder_sizes_w = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes_h.append(latent_size_h // (2 ** i))
            self.encoder_sizes_w.append(latent_size_w // (2 ** i))
            self.decoder_sizes_h.append(latent_size_out_h // 2 ** (self.N_layers - i))
            self.decoder_sizes_w.append(latent_size_out_w // 2 ** (self.N_layers - i))
        
    
        self.lift = LiftProjectBlock(in_channels=in_dim,
                                     out_channels=self.encoder_features[0],
                                     in_size_h=in_size_h,
                                     in_size_w=in_size_w,
                                     out_size_h=self.encoder_sizes_h[0],
                                     out_size_w=self.encoder_sizes_w[0],
                                     latent_dim=latent_lift_proj_dim,
                                     cutoff_den=cutoff_den,
                                     conv_kernel=conv_kernel,
                                     filter_size=filter_size,
                                     lrelu_upsampling=lrelu_upsampling,
                                     half_width_mult=half_width_mult,
                                     radial=radial,
                                     batch_norm=False,
                                     activation=activation)
        _out_size_h = out_size_h
        _out_size_w = out_size_w
        if out_size_h == 1:
            _out_size_h = in_size_h
        if out_size_w == 1:
            _out_size_w = in_size_w
            
        self.project = LiftProjectBlock(in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels=out_dim,
                                        in_size_h=self.decoder_sizes_h[-1],
                                        in_size_w=self.decoder_sizes_w[-1],
                                        out_size_h=_out_size_h,
                                        out_size_w=_out_size_w,
                                        latent_dim=latent_lift_proj_dim,
                                        cutoff_den=cutoff_den,
                                        conv_kernel=conv_kernel,
                                        filter_size=filter_size,
                                        lrelu_upsampling=lrelu_upsampling,
                                        half_width_mult=half_width_mult,
                                        radial=radial,
                                        batch_norm=False,
                                        activation=activation)


        self.encoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i + 1],
                in_size_h=self.encoder_sizes_h[i],
                in_size_w=self.encoder_sizes_w[i],
                out_size_h=self.encoder_sizes_h[i + 1],
                out_size_w=self.encoder_sizes_w[i + 1],
                cutoff_den=cutoff_den,
                conv_kernel=conv_kernel,
                filter_size=filter_size,
                lrelu_upsampling=lrelu_upsampling,
                half_width_mult=half_width_mult,
                radial=radial,
                batch_norm=batch_norm,
                activation=activation
            )
            for i in range(self.N_layers)
        ])
        

        self.ED_expansion = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i],
                in_size_h=self.encoder_sizes_h[i],
                in_size_w=self.encoder_sizes_w[i],
                out_size_h=self.decoder_sizes_h[self.N_layers - i],
                out_size_w=self.decoder_sizes_w[self.N_layers - i],
                cutoff_den=cutoff_den,
                conv_kernel=conv_kernel,
                filter_size=filter_size,
                lrelu_upsampling=lrelu_upsampling,
                half_width_mult=half_width_mult,
                radial=radial,
                batch_norm=batch_norm,
                activation=activation
            )
            for i in range(self.N_layers + 1)
        ])
        
        self.decoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.decoder_features_in[i],
                out_channels=self.decoder_features_out[i],
                in_size_h=self.decoder_sizes_h[i],
                in_size_w=self.decoder_sizes_w[i],
                out_size_h=self.decoder_sizes_h[i + 1],
                out_size_w=self.decoder_sizes_w[i + 1],
                cutoff_den=cutoff_den,
                conv_kernel=conv_kernel,
                filter_size=filter_size,
                lrelu_upsampling=lrelu_upsampling,
                half_width_mult=half_width_mult,
                radial=radial,
                batch_norm=batch_norm,
                activation=activation
            )
            for i in range(self.N_layers)
        ])
        
        self.decoder_inv = nn.ModuleList([
            CNOBlock(
                in_channels=self.inv_features[i],
                out_channels=self.inv_features[i],
                in_size_h=self.decoder_sizes_h[i],
                in_size_w=self.decoder_sizes_w[i],
                out_size_h=self.decoder_sizes_h[i],
                out_size_w=self.decoder_sizes_w[i],
                cutoff_den=cutoff_den,
                conv_kernel=conv_kernel,
                filter_size=filter_size,
                lrelu_upsampling=lrelu_upsampling,
                half_width_mult=half_width_mult,
                radial=radial,
                batch_norm=batch_norm,
                activation=activation
            )
            for i in range(self.N_layers + 1)
        ])
        

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(
                    ResidualBlock(
                        channels=self.encoder_features[l],
                        size_h=self.encoder_sizes_h[l],
                        size_w=self.encoder_sizes_w[l],
                        cutoff_den=cutoff_den,
                        conv_kernel=conv_kernel,
                        filter_size=filter_size,
                        lrelu_upsampling=lrelu_upsampling,
                        half_width_mult=half_width_mult,
                        radial=radial,
                        batch_norm=batch_norm,
                        activation=activation
                    )
                )
        for i in range(self.N_res_neck):
            self.res_nets.append(
                ResidualBlock(
                    channels=self.encoder_features[self.N_layers],
                    size_h=self.encoder_sizes_h[self.N_layers],
                    size_w=self.encoder_sizes_w[self.N_layers],
                    cutoff_den=cutoff_den,
                    conv_kernel=conv_kernel,
                    filter_size=filter_size,
                    lrelu_upsampling=lrelu_upsampling,
                    half_width_mult=half_width_mult,
                    radial=radial,
                    batch_norm=batch_norm,
                    activation=activation
                )
            )
        
        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.lift(x)
        skip = []
        
        res_nets_idx = 0  
        for i in range(self.N_layers):
            
            y = x
            for j in range(self.N_res):
                y = self.res_nets[res_nets_idx](y)
                res_nets_idx += 1
            skip.append(y)
            
            x = self.encoder[i](x)   
        
        #----------------------------------------------------------------------
        
        for j in range(self.N_res_neck):
            x = self.res_nets[res_nets_idx](x)
            res_nets_idx += 1

        for i in range(self.N_layers):
            
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)  
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])), 1)
            
            if self.add_inv:
                x = self.decoder_inv[i](x)
            x = self.decoder[i](x)
        
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)
        x = x.reshape(b, t, -1, x.shape[-2], x.shape[-1])
        
        del skip
        del y
        
        return x

    def get_n_params(self):
        pp = 0
        
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'{nparams} (~{nbytes / 1e6:.2f} MB)')

        return nparams

if __name__ == '__main__':

    in_dim = 1
    in_size_h = 128   
    in_size_w = 128  
    N_layers = 4
    
    model = CNO(
        in_dim=in_dim,
        in_size_h=in_size_h,
        in_size_w=in_size_w,
        N_layers=N_layers,
        N_res=1,
        N_res_neck=6,
        channel_multiplier=32,
        conv_kernel=3,
        cutoff_den=2.0001,
        filter_size=6,
        lrelu_upsampling=2,
        half_width_mult=0.8,
        radial=False,
        batch_norm=True,
        out_dim=1,
        out_size_h=1,
        out_size_w=1,
        expand_input=False,
        latent_lift_proj_dim=64,
        add_inv=True,
        activation='cno_lrelu'
    )
    
    batch_size = 1
    time_steps = 1
    channels = in_dim
    height = in_size_h
    width = in_size_w
    
    x = torch.randn(batch_size, time_steps, channels, height, width)
    
    output = model(x)
    
    print(f"Output shape: {output.shape}")