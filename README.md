# DwTNL-Net: Depthwise Temporal Non-local Network
Pytorch Implementation of paper:

> **Depthwise Temporal Non-local Network for Faster and Better Dynamic Hand Gesture Authentication**
>
> Wenwei Song, Wenxiong Kang\*.

## Main Contribution
Dynamic hand gesture is an emerging and promising biometric trait. It contains both physiological and behavioral characteristics, which on the one hand can theoretically make authentication systems more accurate and more secure, and on the other hand can increase the difficulty of model design because it is essentially a fine-grained video understanding task. For authentication systems, equal error rate (EER) and real-time performance are two vital metrics. Current video understanding-based hand gesture authentication methods mainly focus on lowering the EER while neglecting to reduce the computational cost. In this paper, we propose a 2D CNN-based depthwise temporal non-local network (DwTNL-Net) that can take into account both EER and running efficiency.  To enable the DwTNL-Net with spatiotemporal information processing capability, we design a temporal sharpening (TS) module and a DwTNL module for short- and long-term identity feature modeling, respectively. The TS module can assist the backbone in local behavioral characteristic understanding and can simultaneously remove redundant information and highlight behavioral cues while retaining sufficient physiological characteristics. In contrast, the DwTNL module focuses on summarizing global information and discovering stable patterns, which are finally used for local information enhancement. The complementary combination of our TS and DwTNL modules makes DwTNL-Net achieve substantial performance improvements. Extensive experiments on the SCUT-DHGA dataset and sufficient statistical analyses fully demonstrate the superiority and efficiency of our DwTNL-Net. 
 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/DwTNL-Net/master/img/DwTNL-Net.png" />
</p>
</div>

 The overall architecture of DwTNL-Net. The GSAP and GTAP denote global spatial average pooling and global temporal average pooling, respectively. In order not to adjust the structure of the pretrained 2D CNN, we group four frames into a clip.


## Comparisons with selected SOTAs
To prove the rationality and superiority of our DwTNL-Net, we conduct extensive experiments on the SCUT-DHGA dataset. The EERs shown in the figure are all average values over six test configurations on the cross session.

 <div align="center">
 <p align="center">
  <img src="https://raw.githubusercontent.com/SCUT-BIP-Lab/DwTNL-Net/master/img/Comprehensive_Comparison.png" />
 </p>
</div>

  Comparisons with some representative video understanding networks and 3D version attention modules (based on the TSClips) selected from the experiment section in terms of EER, GFLOPs/Video, and parameter number (#Params). The TSClips denotes the TSN equipped with our TS module. It is clear that compared with other excellent video understanding networks covering 3D CNNs, two-stream CNNs, and 2D CNNs, our TS module can reduce the computational burden to a great extent and can also significantly lower the EER. As illustrated in the enlarged view of the area around the TSClips, our DwTNL module  can also balance EER and FLOPs, and thus can further improve the performance of hand gesture authentication systems.

## Dependencies
Please make sure the following libraries are installed successfully:
- [PyTorch](https://pytorch.org/) >= 1.7.0

## How to use
This repository is a demo of DwTNL-Net. Through debugging ([main.py](/main.py)), you can quickly understand the 
configuration and building method of [DwTNL-Net](/model/DwTNLNet.py), including the TS and DwTNL module.

If you want to explore the entire dynamic hand gesture authentication framework, please refer to our pervious work [SCUT-DHGA](https://github.com/SCUT-BIP-Lab/SCUT-DHGA) 
or send an email to Prof. Kang (auwxkang@scut.edu.cn).
