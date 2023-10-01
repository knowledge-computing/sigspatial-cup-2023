from __future__ import print_function, division
import os
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
from preprocessing import custom_transforms as tr
import random
import cv2

class DeepFashionSegmentation(Dataset):
    def __init__(self,
                 config,
                data_dir_path,
                 split='train',
                 ):
        super().__init__()
        self.split = split

        self.train_pos_list=[]
        self.train_neg_list=[]
        self.test_list=[]

        # if self.split =='test':
        self._image_dir = os.path.join(data_dir_path, 'train_images')
        self._cat_dir = os.path.join(data_dir_path, 'train_mask')           
        # elif self.split =='train_pos'or self.split =='train_neg':
        #     self._image_dir = os.path.join(data_dir_path, 'train_images')
        #     self._cat_dir = os.path.join(data_dir_path, 'train_mask')
        # elif self.split =='train_pos_aug':
        #     self._image_dir = os.path.join(data_dir_path, 'train_images_aug')
        #     self._cat_dir = os.path.join(data_dir_path, 'train_mask_aug')
     
        self.config = config

        self.region_folder_path=self.config['dataset']['region_txt_base_path']

        self.train_pos_list=[]
        self.train_neg_list=[]
        self.test_list=[]

        pos_file=open(os.path.join(self.region_folder_path,'train_pos.txt'))
        tmp_train_pos_lines = pos_file.readlines()
        self.train_pos_list=[line.rstrip('\n') for line in tmp_train_pos_lines]

        neg_file=open(os.path.join(self.region_folder_path,'train_neg.txt'))    
        tmp_train_neg_lines = neg_file.readlines()
        self.train_neg_list=[line.rstrip('\n') for line in tmp_train_neg_lines]

        test_file=open(os.path.join(self.region_folder_path,'test.txt'))  
        tmp_test_lines = test_file.readlines()
        self.test_list=[line.rstrip('\n') for line in tmp_test_lines]

        self.images = []
        self.categories = []
        self.num_classes = self.config['network']['num_classes']

        self.shuffle_dataset()

        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def shuffle_dataset(self):
        #reset lists
        self.images.clear()
        self.categories.clear()
        each_image_list=sorted(os.listdir(self._image_dir))
        each_mask_list=sorted(os.listdir(self._cat_dir))

        target_list=[]
        if self.split =='train_pos':
            target_list=self.train_pos_list
        elif self.split =='train_neg':
            target_list=self.train_neg_list
        elif self.split =='test':
            target_list=self.test_list

        for num_img in range(len(each_image_list)):
            if each_image_list[num_img] in target_list:
                self.images.append(os.path.join(self._image_dir, each_image_list[num_img]))
                self.categories.append(os.path.join(self._cat_dir, each_mask_list[num_img]))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target}
        if self.split == "train_pos" or self.split == "train_pos_aug":
            aug_sample=self.transform_tr(sample)
            return aug_sample    
        elif self.split == "train_neg":
            return self.transform_tr(sample)
        elif self.split == "test":
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        seg=Image.open(self.categories[index])
        # Grayscale
        seg = seg.convert('L')
        # Threshold
        _target = seg.point( lambda p: 1 if p > 127 else 0 )


        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.config['image']['base_size'], crop_size=self.config['image']['crop_size']),
            tr.RandomGaussianBlur(),
            tr.RandomRotate(90),
            tr.RandomRotate(270),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def transform_tr_neg(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
#            tr.FixScaleCrop(crop_size=crop_size),
            tr.FixScaleCrop(crop_size=self.config['image']['crop_size']),
#            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    @staticmethod
    def preprocess(sample, crop_size=513):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'DeepFashion2(split=' + str(self.split) + ')'