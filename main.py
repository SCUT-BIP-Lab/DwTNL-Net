# Demo Code for Paper:
# [Title]  - "Depthwise Temporal Non-local Network for Faster and Better Dynamic Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang
# [Github] - https://github.com/SCUT-BIP-Lab/DwTNL-Net.git

import torch
from model.DwTNLNet import Model_DwTNLNet
# from loss.loss import AMSoftmax

def feedforward_demo(frame_length, group_size, feature_dim, out_dim):
    model = Model_DwTNLNet(frame_length=frame_length, group_size=group_size, feature_dim=feature_dim, out_dim=out_dim)
    # AMSoftmax loss function
    # criterian = AMSoftmax(in_feats=out_dim, n_classes=143)
    # there are 143 identities in the training set
    data = torch.randn(2, 64, 3, 224, 224) #batch, frame, channel, h, w
    data = data.view(-1, 3, 224, 224) #regard the frame as batch
    id_feature = model(data) # feedforward
    # Use the id_feature to calculate the EER when testing or to calculate the loss when training
    # when training
    # loss_backbone, _ = self.criterian(id_feature, label)

    return id_feature

if __name__ == '__main__':
    # there are 64 frames in each dynamic hand gesture video
    frame_length = 64
    # we set group size as 4 to cater for pretrained 2D CNN
    group_size = 4
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512

    # feedforward process
    id_feature = feedforward_demo(frame_length, group_size, feature_dim, out_dim)
    print("Demo is finished!")

