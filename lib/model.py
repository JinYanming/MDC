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
        return nn.Sequential(*layers)
    def get_GAP(self):
        layers = []
        layers.append(nn.ReLU(inplace=True))
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
        feature_1 =  x1.clone()
        x1 = self.gap(x1)
        x1 = self.linear1(x1.view(N,-1))
        linear_weight_1 = list(self.parameters())[-1]
        x1 = torch.sigmoid(x1)
        

        x2 = self.MDC_DC_2(x2)
        feature_2 = x2.clone()
        x2 = self.gap(x2)
        x2 = self.linear2(x2.view(N,-1))
        linear_weight_2 = list(self.parameters())[-2]
        x2 = torch.sigmoid(x2)
        

        x3 = self.MDC_DC_3(x3)
        feature_3 = x3.clone()
        x3 = self.gap(x3)
        x3 = self.linear3(x3.view(N,-1))
        linear_weight_3 = list(self.parameters())[-3]
        x3 = torch.sigmoid(x3)


        x4 = self.MDC_DC_4(x4)
        feature_4 = x4.clone()
        x4 = self.gap(x4)
        x4 = self.linear3(x4.view(N,-1))
        linear_weight_4 = list(self.parameters())[-4]
        x4 = torch.sigmoid(x4)
        CAMs_1 = self.getCams(x1,feature_1,linear_weight_1)
        CAMs_2 = self.getCams(x2,feature_2,linear_weight_2)
        CAMs_3 = self.getCams(x3,feature_3,linear_weight_3)
        CAMs_4 = self.getCams(x4,feature_4,linear_weight_4)
        location_map = torch.argmax((CAMs_1+(CAMs_2+CAMs_3+CAMs_4)/3),dim=1)
        pred_mask = torch.argmax(fov_out,dim=1)
        return fov_out,x1,x2,x3,x4,CAMs_1,CAMs_2,CAMs_3,CAMs_4,location_map,pred_mask



    def init_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_vgg = vgg.features.state_dict()
        self.features.load_state_dict(state_vgg)

        for ly in self.classifier.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)

    def returnCAM(self,feature_conv, weight_softmax):
        # generate the class activation maps upsample to 256x256
        size_upsample = (321, 321)
        bz, nc, h, w = feature_conv.size()
        #print(feature_conv.shape)
        #print("class_idx :{}".format(len(class_idx)))
        #print('idx is {} weight_softmax{} shape is {}'.format(idx,idx,len(weight_softmax)))
        #print(nc,h,w)
        feature_conv = feature_conv.permute(1,0,2,3)
        feature_conv = feature_conv.reshape(1024,-1)
        cam = torch.mm(weight_softmax,feature_conv)#weightsoftmax (21,1024) feature_conv (16,1024,41,41)
        #cam (16,21,321,321)
        cam = cam.reshape(21,64,41,41)
        cam = cam.permute(1,0,2,3)
        min = torch.min(cam).clone()
        max = torch.max(cam).clone()
        cam = cam - min
        cam_img = cam / max
        cam_img = cam_img.cpu().detach().numpy()
        temp  = np.zeros((0,21,321,321))
        cam_img = np.uint8(255*cam_img)

        for item in cam_img:
            item = item.swapaxes(0,2)
            item = cv2.resize(item,size_upsample)
            item = item.swapaxes(0,2)
            item = np.expand_dims(item,axis=0)
            temp = np.r_[temp,item]
        #for i in range(16):
        #ci = ci.unsqueeze(0)
        #temp = torch.cat((temp,ci),0)
        #cam_img = cam_img.swapaxes(0,2)
        return torch.from_numpy(temp)
    def getCams(self,preds,feature,weight_softmax):
        # print('shape of preds:{}'.format(preds))
        bz, nc, h, w = feature.size()
        #print('bz is {}'.format(bz))
            #print(i)
            #print("shape of CAMs is {}".format(CAMs.shape))
            #print('shape of nextCams is{}'.format(returnCAM(feature[i], weight_softmax, [idxs[i,-1]]).shape))
        CAMs = self.returnCAM(feature, weight_softmax)
            #print(nextCAM)
        return CAMs
    def showCAM(self,CAMs,img):
        img = cv2.imread('test.jpg')
        height=497
        width=497
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite('CAM.jpg', result)


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

