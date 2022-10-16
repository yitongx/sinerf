from pyparsing import col
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from typing import List, Optional, Union

class OfficialNerf(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerf, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, 3)

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=3)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den


class Sine(nn.Module):
    def __init__(self, w=1.):
        super(Sine, self).__init__()
        self.w = w

    def forward(self, x):
        return torch.sin(self.w * x)

class SirenLinear(nn.Module):
    '''
    Based on Pi-GAN.
    https://github.com/lucidrains/pi-GAN-pytorch/blob/main/pi_gan_pytorch/pi_gan_pytorch.py
    '''
    def __init__(self, input_dim=256, output_dim=256, use_bias=True, w=1., is_first=False):
        super(SirenLinear, self).__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.use_bias = use_bias
        self.activation = Sine(w)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w = w
        self.c = 6.0
        self.init_params()

    def init_params(self):
        with torch.no_grad():
            dim = self.input_dim
            # w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w)            
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim))
            self.fc_layer.weight.uniform_(-w_std, w_std)
            if self.use_bias and self.fc_layer.bias is not None:
                self.fc_layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.fc_layer(x) 
        out = self.activation(out)
        return out


class SiNeRF(nn.Module):
    def __init__(self, pos_in_dims=3, 
                 dir_in_dims=3, 
                 D=256, 
                 sine_weights_layers0=[30, 1, 1, 1], 
                 sine_weights_layers1=[1, 1, 1, 1], 
                 rgb_mag=1.0, 
                 den_mag=25.0):
        """        
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions.

        """
        super(SiNeRF, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.w_layers0 = iter(sine_weights_layers0)
        self.w_layers1 = iter(sine_weights_layers1)
        self.rgb_mag = rgb_mag
        self.den_mag = den_mag
        
        self.layers0 = nn.Sequential(
            SirenLinear(pos_in_dims, D, True, next(self.w_layers0), is_first=True), 
            SirenLinear(D, D, True, next(self.w_layers0)), 
            SirenLinear(D, D, True, next(self.w_layers0)), 
            SirenLinear(D, D, True, next(self.w_layers0)), 
        )
        ### Cancel shortcut for SirenNeRF.
        self.layers1 = nn.Sequential(
            SirenLinear(D, D, True, next(self.w_layers1)), 
            SirenLinear(D, D, True, next(self.w_layers1)), 
            SirenLinear(D, D, True, next(self.w_layers1)), 
            SirenLinear(D, D, True, next(self.w_layers1)), 
        )

        _fc_density = [SirenLinear(D, D // 2, True, 1.), nn.Linear(D // 2, 1, True)] 
        self.fc_density = nn.Sequential(*_fc_density)
        self.fc_feature = nn.Linear(D, D)

        _rgb_layers = [SirenLinear(D + dir_in_dims, D // 2, True, 1.), nn.Linear(D // 2, 3, True)] 
        self.rgb_layers = nn.Sequential(*_rgb_layers)

    def forward(self, pos_enc, dir_enc):
        """       
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        z = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        
        rgb = self.rgb_layers(z)    # (H, W, N_sample, 3)
        
        ### magnifying color and density. No output regulating here.
        rgb = self.rgb_mag * rgb
        density = self.den_mag * density
    
        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den
