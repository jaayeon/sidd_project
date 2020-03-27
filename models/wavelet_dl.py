import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from models.convs import attention

from utils.wavelet import normalize_coeffs, unnormalize_coeffs, standarize_coeffs, unstandarize_coeffs

def make_model(opt):
    return WaveletDL(opt)

class WaveletDL(nn.Module):
    def __init__(self, opt):
        super(WaveletDL, self).__init__()

        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = opt.kernel_size
        self.in_skip = opt.in_skip

        self.n_channels = opt.swt_num_channels
        self.out_channels = opt.swt_num_channels
        bn = opt.bn
        bias = not bn

        self.ch_mean = opt.ch_mean
        self.ch_std = opt.ch_std
        
        act = nn.ReLU(True)
        conv = common.default_conv

        ResBlock = common.ResBlock
        if opt.attn:
            CBAM = attention.CBAM

        # define head module
        m_head = [conv(self.n_channels, n_feats, kernel_size)]

        # define body module
        if opt.attn:
            m_body = []
            for i in range(n_resblocks):
                m_body.append(
                    ResBlock(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale)
                )

                if (i + 1)% opt.each_attn == 0:
                    m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
                    m_body.append(
                        CBAM(n_feats, reduction_ratio=opt.reduction_ratio, pool_types=opt.pool_type, spatial_attn=opt.spatial_attn)
                    )
        else : 
            m_body = [
                ResBlock(
                    conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
                ) for _ in range(n_resblocks)
            ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
                conv(n_feats, self.out_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = standarize_coeffs(x, ch_mean=self.ch_mean, ch_std=self.ch_std)
        global_res = x

        x = self.head(x)
        res = self.body(x)
        if self.in_skip:
            res += x

        x = self.tail(res)
        x = x + global_res

        x = unstandarize_coeffs(x, ch_mean=self.ch_mean, ch_std=self.ch_std)

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

