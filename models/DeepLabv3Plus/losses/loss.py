import torch
import torch.nn as nn
import monai
import numpy as np

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py



class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def one_hot_encode(self,label):
        """
        Convert a segmentation image label array to one-hot format
        by replacing each pixel value with a vector of length num_classes
        # Arguments
            label: The 2D array segmentation image label
            label_values
            
        # Returns
            A 2D array with the same width and hieght as the input, but
            with a depth size of num_classes
        """
        semantic_map = []
        label_values=[(0),(1)]
        for colour in label_values:
            equality = np.equal(label, colour)
            class_map = np.all(equality)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map)
        semantic_map=semantic_map.reshape(2,1024,1022)
        print(semantic_map.shape)

        return semantic_map
        
    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == "dice":
            return self.DiceLoss
            # self.DiceLoss
        else:
            raise NotImplementedError

    def DiceLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') 
        #  monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()
        # target = torch.argmax(target, dim=2)
        target_dice=target.cpu().numpy()

        b, h, w = target_dice.shape
        gt_dice = []
        label_values=[(0),(1)]
        
        for num_image in range(b):
            mask_list=[] 
            for label, color in enumerate(label_values):
                mask=np.where(target_dice[num_image]!=label, 0, 1)
                mask_list.append(mask)
            mask_list = np.stack(mask_list)
            # print(mask_list.shape,'np.mask_list')
            gt_dice.append(mask_list)
        gt_dice=np.stack(gt_dice)
        target=torch.from_numpy(gt_dice)
   
        logit, target = logit.cuda(), target.cuda()
        # print(gt_dice.shape,'np.gt_dice')
        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

