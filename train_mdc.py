#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import os.path as osp
import time
import sys
import logging
import numpy as np
import argparse
import importlib
import json
import cv2
from lib.model_1 import DeepLabLargeFOV
from lib.pascal_voc import PascalVoc
from lib.pascal_voc_aug import PascalVoc_Aug
from lib.transform import RandomCrop
from lib.optimizer import Optimizer
from lib.loss import *
from utils.logger import setup_logger
from evaluate import eval_model
from torchsummary import summary

batchsize = 8
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
            '--cfg',
            dest = 'cfg',
            type = str,
            default = 'config/pascal_voc_aug_multi_scale.py',
            help = 'config file for training'
            )
    return parser.parse_args()
def get_LocationMap(data,threshold = 0.7*255,max=255,min=0):
    change = lambda a,th:1 if a>th else 0
    fm = [change(j,threshold) for i in data for j in i]
    fm = np.array(fm).reshape(321,321)
    return fm

def returnCAM(feature_conv, weight_softmax):
    # generate the class activation maps upsample to 256x256
    size_upsample = (321, 321)
    nc, h, w = feature_conv.shape
    #print(feature_conv.shape)
    #print("class_idx :{}".format(len(class_idx)))
    #print('idx is {} weight_softmax{} shape is {}'.format(idx,idx,len(weight_softmax)))
    #print(nc,h,w)
    feature_conv = np.swapaxes(feature_conv,0,1)
    cam = np.dot(weight_softmax,feature_conv)#weightsoftmax (21,1024) feature_conv (1024,41,41)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cam_img.swapaxes(0,2)
    cam_img = cv2.resize(cam_img, size_upsample)
    cam_img = cam_img.swapaxes(0,2)
    return cam_img
def getCams(preds,feature,weight_softmax):
    # print('shape of preds:{}'.format(preds))
    bz, nc, h, w = feature.shape
    #print('bz is {}'.format(bz))
    CAMs = np.zeros(shape=(0,21,321,321))
    for i in range(bz):
        #print(i)
        #print("shape of CAMs is {}".format(CAMs.shape))
        #print('shape of nextCams is{}'.format(returnCAM(feature[i], weight_softmax, [idxs[i,-1]]).shape))
        nextCAM = returnCAM(feature[i], weight_softmax)
        #print(nextCAM)
        CAMs = np.r_[CAMs,np.expand_dims(nextCAM,axis=0)]
    return CAMs
def showCAM(CAMs,img):
    img = cv2.imread('test.jpg')
    height=321
    width=321
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)


def train(args):
    ## setup cfg and logger
    spec = importlib.util.spec_from_file_location('mod_cfg', args.cfg)
    mod_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_cfg)
    cfg = mod_cfg.cfg
    cfg_str = json.dumps(cfg, ensure_ascii=False, indent=2)
    if not osp.exists(cfg.res_pth): os.makedirs(cfg.res_pth)
    setup_logger(cfg.res_pth)
    logger = logging.getLogger(__name__)
    logger.info(cfg_str)
    device = torch.device('cuda:0')

    ## modules and losses
    logger.info('creating model and loss module')
    net = DeepLabLargeFOV(3, cfg.n_classes)
    #for item in net.parameters():
    #    print(item.size())
    net.train()
    net.cuda()
    if not torch.cuda.device_count() == 0: net = nn.DataParallel(net)
    n_min = (cfg.crop_size**2) * cfg.batchsize // 8
    criteria = OhemCELoss(0.7, n_min)
    globloss = GlobLoss(0.7,n_min)
    criteria.cuda()
    
    #hook the feature extractor
    features_blobs_1 = []
    features_blobs_2 = []
    features_blobs_3 = []
    features_blobs_4 = []
    
    def hook_feature(module, input, output):
        features_blobs_1.append(output.data.cpu().numpy())
        features_blobs_2.append(output.data.cpu().numpy())
        features_blobs_3.append(output.data.cpu().numpy())
        features_blobs_4.append(output.data.cpu().numpy())

    net._modules['module']._modules['MDC_DC_1']._modules.get('1').register_forward_hook(hook_feature)
    net._modules['module']._modules['MDC_DC_2']._modules.get('1').register_forward_hook(hook_feature)
    net._modules['module']._modules['MDC_DC_3']._modules.get('1').register_forward_hook(hook_feature)
    net._modules['module']._modules['MDC_DC_4']._modules.get('1').register_forward_hook(hook_feature)
    params = list(net.parameters())
    count =0
    #summary(net,(3,321,321))
    weight_softmax = np.squeeze(params[-3].data.cpu().numpy())
    
    ## dataset
    logger.info('creating dataset and dataloader')
    ds = eval(cfg.dataset)(cfg, mode='train')
    dl = DataLoader(ds,
            batch_size = cfg.batchsize,
            shuffle = True,
            num_workers = cfg.n_workers,
            drop_last = True)

    ## optimizer
    logger.info('creating optimizer')
    optimizer = Optimizer(
            params = net.parameters(),
            warmup_start_lr = cfg.warmup_start_lr,
            warmup_steps = cfg.warmup_iter,
            lr0 = cfg.start_lr,
            max_iter = cfg.iter_num,
            momentum = cfg.momentum,
            wd = cfg.weight_decay,
            power = cfg.power)

    ## train loop
    loss_avg = []
    st = time.time()
    diter = iter(dl)
    logger.info('start training')
    max = 0
    min = 0
    mean = 0
    for it in range(cfg.iter_num):
        if it/20 ==0:
            print('training {}/{}'.format(it,cfg.iter_num))
        try:
            im, lb ,clb = next(diter)
            if not im.size()[0] == cfg.batchsize: continue
        except StopIteration:
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()#16,1,321,321
        optimizer.zero_grad()
        out,pred_c1,pred_c2,pred_c3,pred_c4 = net(im)#out:16.21.321.321 pred:(16,21)
        #print(out.size())
        lb = torch.squeeze(lb)
        probs,idx = pred_c1.sort(1,True)


        CAMs_1 =getCams(pred_c1,features_blobs_1[0], weight_softmax)
        CAMs_2 =getCams(pred_c2,features_blobs_2[0], weight_softmax)
        CAMs_3 =getCams(pred_c3,features_blobs_3[0], weight_softmax)
        CAMs_4 =getCams(pred_c4,features_blobs_4[0], weight_softmax)
        #print(features_blobs_1[0].shape,len(features_blobs_1)) 
        location_map =np.argmax((CAMs_1 + (CAMs_2+CAMs_3+CAMs_4)/3),axis=1) #(16,21,321,321)
        pred_mask = torch.argmax(out,dim=1)

        #false_mask = torch.from_numpy(false_mask)
        loss = globloss(out,location_map,pred_mask,clb,pred_c1,pred_c2,pred_c3,pred_c4)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_avg.append(loss)
        ## log message
        if it%cfg.log_iter==0 and not it==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            ed = time.time()
            t_int = ed - st
            lr = optimizer.get_lr()
            msg = 'iter: {}/{}, loss: {:.4f}'.format(it, cfg.iter_num, loss_avg)
            msg = '{}, lr: {:4f}, time: {:.4f}'.format(msg, lr, t_int)
            logger.info(msg)
            st = ed
            loss_avg = []

    ## dump model
    model_pth = osp.join(cfg.res_pth, 'model_final.pkl')
    net.cpu()
    state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state_dict, model_pth)
    logger.info('training done, model saved to: {}'.format(model_pth))

    ## test after train
    if cfg.test_after_train:
        net.cuda()
        mIOU = eval_model(net, cfg)
        logger.info('iou in whole is: {}'.format(mIOU))


if __name__ == "__main__":
    args = get_args()
    train(args)
    test(args)
