# import numpy as np
import torch
import torch.nn as nn
# import torchvision
from torchvision import models

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}


class ResNetFc(nn.Module):
  def __init__(self, resnet_name, bottleneck_dim=256, class_num=1000,
               init_fc=0, NoRelu=0):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.NoRelu = NoRelu
    self.class_num = class_num

    if NoRelu == 1:
        self.bottleneck = nn.Sequential(
            nn.Linear(model_resnet.fc.in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim))
    else:
        self.bottleneck = nn.Sequential(
            nn.Linear(model_resnet.fc.in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU())
    self.fc = nn.Linear(bottleneck_dim, class_num)
    if init_fc == 1:
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
    self.__in_features = bottleneck_dim

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 0.1, 'decay_mult': 2}, \
                      {"params": self.bottleneck.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                      {"params": self.fc.parameters(), "lr_mult": 1, 'decay_mult': 2}]
    return parameter_list