import torch
import torch.nn as nn

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        def autopad(k, p=None, d=1):
            if d > 1: k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
            if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
            return p
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(c1 // r, 1) 
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden_dim, 1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, c1, 1, bias=False), 
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avgpool(x))

class HierarchicalFusionBlock(nn.Module):
    def __init__(self, c_out, c_deep, c_shallow, r_spatial=4, r_channel=16):
        super().__init__()
        self.c_deep = c_deep
        c_inter = c_out // 2
        self.conv_deep = Conv(c_deep, c_inter, 1, 1)
        self.conv_shallow = Conv(c_shallow, c_inter, 1, 1)
        spatial_hidden_dim = max(c_inter // r_spatial, 1)
        self.spatial_attn = nn.Sequential(
            Conv(c_inter, spatial_hidden_dim, 1, 1),
            Conv(spatial_hidden_dim, 1, 1, 1, act=nn.Sigmoid())
        )
        self.channel_attn = SELayer(c_inter, r=r_channel)
        self.conv_fuse = Conv(c_inter * 2, c_out, 3, 1, p=1)

    def forward(self, x):
        x_deep = x[:, :self.c_deep, :, :]
        x_shallow = x[:, self.c_deep:, :, :]
        x_deep_aligned = self.conv_deep(x_deep)
        x_shallow_aligned = self.conv_shallow(x_shallow)
        x_shallow_enhanced = x_shallow_aligned * self.spatial_attn(x_deep_aligned)
        x_deep_enhanced = x_deep_aligned * self.channel_attn(x_shallow_aligned)
        y = torch.cat([x_deep_enhanced, x_shallow_enhanced], dim=1)
        return self.conv_fuse(y)
