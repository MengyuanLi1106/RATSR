import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from models import register
from utils import make_coord
from utils import show_feature_map

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_list:
            layers.append(nn.Conv2d(last, h,1,1))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Conv2d(last, out_dim,1,1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        B, C,H,W = x.shape
        # x = x.reshape(B * HW, C)
        x = self.layers(x)
        # x = x.reshape(B, HW, -1)
        return x


@register('ratsr')
class RATSR(nn.Module):

    def __init__(self, encoder_spec):
        super().__init__()
        self.encoder = models.make(encoder_spec)

        imnet_in_dim = 64 + 2 +3

        self.metric_head = MLP(64, 3, [64])
        self.imnet = MLP(imnet_in_dim, 1, [256, 256,256])

    def _calc_maha(self, q_feat, rel_coord):
        metric = self.metric_head(q_feat)
        l11 = F.softplus(metric[:, 0:1]) + 1e-6
        l21 = metric[:, 1:2]
        l22 = F.softplus(metric[:, 2:3]) + 1e-6

        m00 = l11 * l11
        m01 = l11 * l21
        m11 = l21 * l21 + l22 * l22

        dx = rel_coord[:, 0:1]
        dy = rel_coord[:, 1:2]
        d2 = dx * dx * m00 + 2.0 * dx * dy * m01 + dy * dy * m11
        return torch.sqrt(F.relu(d2) + 1e-9)
        
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat
        
    def query_rgb(self, coord, cell):      
        feat = self.feat
        B,H_out,W_out,C=coord.shape

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).\
            expand(feat.shape[0], 2, *feat.shape[-2:])

        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]
        rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H_out, W_out)

        rx = 1 / feat.shape[-2]
        ry = 1 / feat.shape[-1]
        eps_shift = 1e-6

        pred_sum = 0
        weight_sum = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + eps_shift, 1 - eps_shift)

                q_feat = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                feat_coord_up = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)

                rel_coord = coord.permute(0, 3, 1, 2) - feat_coord_up
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                maha = self._calc_maha(q_feat, rel_coord)
                x = torch.cat([q_feat, rel_coord, rel_cell, maha], dim=1)
                pred = self.imnet(x)

                weight = 1.0 / (maha.pow(2) + 1e-6)
                pred_sum = pred_sum + pred * weight
                weight_sum = weight_sum + weight

        out = pred_sum / (weight_sum + 1e-6)
        out = out + F.grid_sample(self.inp, coord.flip(-1), mode='bicubic', align_corners=False)

        return out


    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
