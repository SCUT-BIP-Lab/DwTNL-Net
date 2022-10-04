# Demo Code for Paper:
# [Title]  - "Depthwise Temporal Non-local Network for Faster and Better Dynamic Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/DwTNL-Net.git

from torch import nn
import torch
import torch.nn.functional as F

class DwTNL_module(nn.Module):
    def __init__(self,  in_channels: int, reduction=2):

        super(DwTNL_module, self).__init__()
        self.reduction = reduction
        self.in_channels = in_channels

        self.convQ = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.convK = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.convV = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)

        self.conv_reconstruct = nn.Sequential(
            nn.Conv3d(in_channels // self.reduction, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels)
        )

        nn.init.constant_(self.conv_reconstruct[1].weight, 0)
        nn.init.constant_(self.conv_reconstruct[1].bias, 0)

    def forward(self, x: torch.Tensor):

        b, c, t, h, w = x.size()
        cr = c // self.reduction
        assert c == self.in_channels, 'input channel not equal!'

        Q = self.convQ(x)  # (b, cr, t, h, w)
        K = self.convK(x)  # (b, cr, t, h, w)
        V = self.convV(x)  # (b, cr, t, h, w)

        # for channel-independent temporal self-attention
        Q = Q.view(b*cr, t, h * w) # (b*cr, t, h*w)
        K = K.view(b*cr, t, h * w) # (b*cr, t, h*w)
        V = V.view(b*cr, t, h * w) # (b*cr, t, h*w)

        # pattern matching response sorting (top49)
        Q_tk = torch.topk(Q, k=h * w // 4, dim=-1, largest=True, sorted=True)[0] # (b*cr, t, 49)
        K_tk = torch.topk(K, k=h * w // 4, dim=-1, largest=True, sorted=True)[0] # (b*cr, t, 49)

        # calculate affinity matrices with the strongest 49 response values for each pattern (channel)
        correlation = torch.bmm(Q_tk, K_tk.permute(0, 2, 1))  # (b*cr, t, t)
        correlation_attention = F.softmax(correlation, dim=-1)
        # global temporal information aggregated across TSClip features for each channel
        y = torch.matmul(correlation_attention, V) # (b*cr, t, h*w)
        y = y.view(b, cr, t, h, w) # (b, cr, t, h, w)
        y = self.conv_reconstruct(y) # (b, c, t, h, w)
        # local identity feature enhancement with global features
        z = y + x
        return z