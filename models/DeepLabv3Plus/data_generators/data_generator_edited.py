#from data_generators.datasets import cityscapes, coco, combine_dbs, pascal, sbd, deepfashion
from torch.utils.data import DataLoader,Dataset
from data_generators.deepfashion import DeepFashionSegmentation
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
from torch.utils.data import ConcatDataset,random_split 
# def get_augmentation():
#     transform=transforms.Compose([
#                     transforms.RandomRotation(90),transforms.RandomRotation(180),transforms.RandomRotation(270),transforms.ToTensor()])
#     return transform

class get_data_pair(Dataset):
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=None 
            # augmentation=None
    ):
        
        # self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        # self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.image_paths=[]
        self.mask_paths=[]
        for idx in range(len(images_dir)):
            tmp_mask_info=[]
            each_image_list=sorted(os.listdir(images_dir[idx]))
            each_mask_list=sorted(os.listdir(masks_dir[idx]))

            for num_img in range(len(each_image_list)):
                self.image_paths.append(os.path.join(images_dir[idx], each_image_list[num_img]))
                self.mask_paths.append(os.path.join(masks_dir[idx], each_mask_list[num_img]))

        self.class_rgb_values = class_rgb_values   
        # self.augmentation=augmentation

    def __getitem__(self, i):
        
        # read images and masks

        image = cv2.imread(self.image_paths[i])

        # if self.mask_paths[i]!='empty':
        # mask = cv2.imread(self.mask_paths[i])
        seg = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)  
        # mask = cv2.threshold(seg, 127, 1, cv2.THRESH_BINARY)[1]
        mask = cv2.threshold(seg, 127, 1, cv2.THRESH_BINARY)[1]               
        return {'image':image,'label': mask}
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)
        

def initialize_data_loader(config):
    # color_class_list=[0, 1]
    color_class_list=[0,1]
    
    train_val_set = get_data_pair(
        config['dataset']['train_img_dir'], config['dataset']['train_mask_dir'], 
        class_rgb_values=color_class_list
    )

    train_size=(int)(len(train_val_set)*0.96)
    valid_size=len(train_val_set)-train_size
    train_set,val_set = random_split(train_val_set,
                                               [train_size, valid_size])

    test_set= get_data_pair(
        config['dataset']['test_img_dir'], config['dataset']['test_mask_dir'], 
        class_rgb_values=color_class_list
    )

    num_classes = config['network']['num_classes']
    print('train,val and test')
    print(len(train_set))
    print(len(val_set))
    print(len(test_set))
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['workers'], pin_memory=True)
    val_loader = DataLoader(val_set,batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['workers'],pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['workers'],pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes

