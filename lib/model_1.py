#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import Resize,ToTensor,Compose,CenterCrop
import cv2
import numpy as np
class DeepLabLargeFOV(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super(DeepLabLargeFOV, self).__init__(*args, **kwargs)
        vgg16 = torchvision.models.vgg16()
        self.features = self.get_VGG16(in_dim)
        self.classifier = self.get_classifer(out_dim)
        self.MDC_features = self.get_VGG16(in_dim)
        self.MDC_DC_1 = self.get_DC(0) 
        self.MDC_DC_2 = self.get_DC(1)
        self.MDC_DC_3 = self.get_DC(2)
        self.MDC_DC_4 = self.get_DC(3)
        self.gap = self.get_GAP()
        self.linear1 = nn.Linear(1024,21,bias=False)
        self.linear2 = nn.Linear(1024,21,bias=False)
        self.linear3 = nn.Linear(1024,21,bias=False)
        self.linear4 = nn.Linear(1024,21,bias=False)
        self.init_weights()
    def get_VGG16(self,in_dim):

        layers = []
        layers.append(nn.Conv2d(in_dim, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))

        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))
        return nn.Sequential(*layers)
    def get_classifer(self,out_dim):
        classifier = []
        classifier.append(nn.AvgPool2d(3, stride = 1, padding = 1))
        classifier.append(nn.Conv2d(512,
            1024,
            kernel_size = 3,
            stride = 1,
            padding = 12,
            dilation = 12))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Conv2d(1024, out_dim, kernel_size=1))
        return nn.Sequential(*classifier)
    def get_DC(self,times):
        layers = []
        if times == 0:
            layers.append(nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1))
        else:
            layers.append(nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=3*times,dilation=times*3))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    def get_GAP(self):
        layers = []
        layers.append(nn.AvgPool2d(kernel_size=41,stride=1))
        layers.append(nn.Dropout(p=0.5))
        return nn.Sequential(*layers)
    def forward(self, img):
        N, C, H, W = img.size()

        x = self.features(img)
        #print(x.size())
        x = self.classifier(x)#{16,21,41,41}
        #print('shape of fov classifier output is {}'.format(x.size()))
        fov_out = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)#16,21,321,321
        #print(fov_out.size())
        x = self.MDC_features(img)

        x1 = x.clone()
        x2 = x.clone()
        x3 = x.clone()
        x4 = x.clone()
        
        
        x1 = self.MDC_DC_1(x1)
        x1 = self.gap(x1)
        x1 = self.linear1(x1.view(N,-1))
        x1 = torch.sigmoid(x1)
        

        x2 = self.MDC_DC_2(x2)
        x2 = self.gap(x2)
        x2 = self.linear2(x2.view(N,-1))
        x2 = torch.sigmoid(x2)
        

        x3 = self.MDC_DC_3(x3)
        x3 = self.gap(x3)
        x3 = self.linear3(x3.view(N,-1))
        x3 = torch.sigmoid(x3)


        x4 = self.MDC_DC_4(x4)
        x4 = self.gap(x4)
        x4 = self.linear3(x4.view(N,-1))
        x4 = torch.sigmoid(x4)
        return fov_out,x1,x2,x3,x4



    def init_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_vgg = vgg.features.state_dict()
        self.features.load_state_dict(state_vgg)

        for ly in self.classifier.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)



if __name__ == "__main__":
    net = DeepLabLargeFOV(3, 10)
    in_ten = torch.randn(1, 3, 224, 224)
    out = net(in_ten)
    print(out.size())

    in_ten = torch.randn(1, 3, 64, 64)
    mod = nn.Conv2d(3,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2)
    out = mod(in_ten)
    print(out.shape)

