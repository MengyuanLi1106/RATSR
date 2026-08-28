import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

class LRingConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(LRingConv, self).__init__()
        assert kernel_size % 2 != 0, "Kernel size must be odd"
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        mask = torch.zeros(kernel_size, kernel_size)
        mask[0, :] = 1.0
        mask[-1, :] = 1.0
        mask[:, 0] = 1.0
        mask[:, -1] = 1.0

        self.center_idx = kernel_size // 2
        mask[self.center_idx, self.center_idx] = 0.0

        self.register_buffer('ring_mask', mask.unsqueeze(0).unsqueeze(0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def get_effective_weight(self):
        ring_weight = self.weight * self.ring_mask

        sum_over_ring = ring_weight.sum(dim=(2, 3), keepdim=True)

        center_mask = torch.zeros_like(ring_weight)
        center_mask[:, :, self.center_idx, self.center_idx] = 1.0

        effective_weight = ring_weight - (sum_over_ring * center_mask)
        return effective_weight

    def forward(self, x):
        eff_weight = self.get_effective_weight()
        return F.conv2d(x, eff_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class Mlp(nn.Module):
    def __init__(self, in_features, ratio, act_layer=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, in_features*ratio, 1)
        self.dwconv = nn.Conv2d(in_features*ratio, in_features*ratio, 3, 1, 1, bias=True, groups=in_features*ratio)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_features*ratio, in_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Ring(nn.Module):
    def __init__(self, in_features, act_layer=nn.ReLU):
        super().__init__()
        self.ringconv5x5 = LRingConv(in_features, in_features, groups=in_features)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(in_features, in_features, 1)
        self.dwconv = nn.Conv2d(in_features, in_features, 3, 1, 1, bias=True, groups=in_features)

    def forward(self, x):
        x = self.ringconv5x5(x)
        x = self.act(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        return x

class RingBlock(nn.Module):
    def __init__(
        self, n_feats, ratio):
        super(RingBlock, self).__init__()

        self.norm1 = nn.BatchNorm2d(n_feats)
        self.norm2 = nn.BatchNorm2d(n_feats)
        self.ringconv = Ring(n_feats, act_layer=nn.ReLU)
        self.mlp = Mlp(n_feats, ratio, act_layer=nn.ReLU)
        
    def forward(self, x):
        x = x + self.ringconv(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


class RingSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RingSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        ratio = args.ratio
        kernel_size = 3
        act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            RingBlock(
                n_feats, ratio
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats
       

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = res
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('ringsr-baseline')
def make_ringsr_baseline(n_resblocks=30, n_feats=64, ratio=4, rgb_range=1, no_upsampling=True):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.ratio = ratio
    args.rgb_range = rgb_range
    args.n_colors = 1
    return RingSR(args)

