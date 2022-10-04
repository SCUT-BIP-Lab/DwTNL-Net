# Demo Code for Paper:
# [Title]  - "Depthwise Temporal Non-local Network for Faster and Better Dynamic Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/DwTNL-Net.git

import torch
import torch.nn as nn
import torchvision
from module.DwTNL_module import DwTNL_module


class Model_DwTNLNet(torch.nn.Module):
    def __init__(self, frame_length, group_size, feature_dim, out_dim):
        super(Model_DwTNLNet, self).__init__()
        self.out_dim = out_dim  # the identity feature dim
        # load the pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)
        # change the last fc with the shape of 512×512
        self.model.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        # there are 64 frames in each dynamic hand gesture video
        self.frame_length = frame_length
        # the number of frames in each group
        self.group_size = group_size
        # the number of groups
        self.group_num = frame_length // group_size

        # build DwTNL_Module in the layer3
        self.dwtnl_module = DwTNL_module(in_channels=self.model.layer3[0].conv2.out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # calculate the TSClips (TS-Moudle)
    def getTemporalSharpenedClips(self, v):
        v = v.view((-1, self.group_size)+v.shape[-3:]) # batch*group_num, group_size, c, h, w
        v_g = torch.sum(v, 2) # b*gn, gs, h, w (gs=4 in this work)
        v_ts = v_g[:, :self.group_size-1] * 2 - v_g[:, 1:self.group_size] #alpha = 1 in eq.2
        return v_ts

    def forward(self, data, label=None):
        # get TSClip
        ts = self.getTemporalSharpenedClips(data) #b*gn, 3, h, w
        # local identity feature extraction and analysis
        x = self.model.conv1(ts)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        for i in range(2):
            layer_name = "layer" + str(i + 1)
            layer = getattr(self.model, layer_name)
            x = layer(x)
        x = self.model.layer3[0](x)

        # global identity feature aggregation
        bn, c, h, w = x.size()
        x = x.view(-1, self.group_num, c, h, w).transpose(1, 2).contiguous()
        x = self.dwtnl_module(x)
        x = x.transpose(1, 2).contiguous().view(bn, c, h, w)
        # global identity feature extraction and analysis
        x = self.model.layer3[1](x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        x = x.view(-1, self.group_num, self.out_dim)

        id_feature = torch.mean(x, dim=1, keepdim=False)
        id_feature = torch.div(id_feature, torch.norm(id_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization for AMSoftmax

        return id_feature


