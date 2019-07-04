#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import array
from torch.autograd import Variable
class GlobLoss(nn.Module):
    def __init__(self,thresh_ohem,n_min_ohem):
        super(GlobLoss, self).__init__()
        self.OhemCELoss = OhemCELoss(thresh_ohem,n_min_ohem)
        self.celoss = nn.BCELoss(reduce=True)
    def mdcLoss(logits,false_mask):
        for logit,mask in zip(logits,false_mask):
            print(2222)

    def forward(self, logits, locationmap,pred_mask, clabel,
            pred_c1,pred_c2,pred_c3,pred_c4,
            ):
        N, C, H, W = logits.size()
        clabel = Variable(clabel).cuda()
        #print('shape of clabel and pred_c1 is {} and{}'.format(array(clabel).shape,pred_c1.size()))
        locationmap = torch.tensor(locationmap,dtype=torch.long)
        locationmap = Variable(locationmap).cuda()
        #clabel = torch.tensor(clabel)
        pred_c1 = Variable(pred_c1,requires_grad=True)
        pred_c2 = Variable(pred_c2,requires_grad=True)
        pred_c3 = Variable(pred_c3,requires_grad=True)
        pred_c4 = Variable(pred_c4,requires_grad=True)
        #print((clabel),(pred_c1))
        #print(pred_c1.requires_grad,clabel.requires_grad)
        loss_mdc_1 = self.celoss(pred_c1,clabel)
        loss_mdc_2 = self.celoss(pred_c2,clabel)
        loss_mdc_3 = self.celoss(pred_c3,clabel)
        loss_mdc_4 = self.celoss(pred_c4,clabel)
        #print(loss_mdc_c.size())
        loss_lm = self.OhemCELoss(logits,locationmap)
        loss_pm = self.OhemCELoss(logits,pred_mask)
        loss = loss_lm+loss_pm
        loss += loss_mdc_1+loss_mdc_2+loss_mdc_3+loss_mdc_4
        return loss
class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss



if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    focal1 = FocalLoss(alpha=0.25, gamma=1)
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    #  loss1 = criteria1(logits1, lbs)
    loss1 = focal1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
