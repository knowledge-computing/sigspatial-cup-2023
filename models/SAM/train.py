

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pdb
import cv2

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from transformers import SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from scipy.ndimage import zoom
import random 

from transformers import SamModel 
import argparse

import torch
torch.manual_seed(0)




INPUT_PATCH_SIZE = 1024
num_epochs = 10

# image_dir = '/home/yaoyi/shared/sigspatial/data_crop1024_shift512/train_images'
# mask_dir = '/home/yaoyi/shared/sigspatial/data_crop1024_shift512/train_mask'
# positive_file = '/home/yaoyi/namgu007/segment-anything/data-preparation/positive_train_samples.txt'
# negative_file = '/home/yaoyi/namgu007/segment-anything/data-preparation/negative_train_samples.txt'
# hard_negative_file = '/home/yaoyi/namgu007/segment-anything/data-preparation/hard_negative_samples.txt'


# image_dir= "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_images"
# mask_dir = "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_mask"
# positive_file = "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_Greenland26X_22W_Sentinel2_2019-06-03_05_r2/train_pos.txt"
# negative_file = "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_Greenland26X_22W_Sentinel2_2019-06-03_05_r2/train_neg.txt"
# hard_negative_file = "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_Greenland26X_22W_Sentinel2_2019-06-03_05_r2/train_hard_neg.txt"
# checkpoint_dir = 'checkpoints_lr5e-6_hardneg_Greenland26X_22W_Sentinel2_2019-06-03_05_r2/'


def main(args):
    
    num_epochs = args.epoch
    REGION = args.region
    
    print(f'{REGION} start training! ')
    
    image_dir= "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_images"
    mask_dir = "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_mask"
    positive_file = f"/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_{REGION}/train_pos.txt"
    negative_file = f"/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_{REGION}/train_neg.txt"
    hard_negative_file = f"/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_{REGION}/train_hard_neg.txt"
    checkpoint_dir = f'checkpoints_{REGION}_rerunV1setting/'
    
    print(f'saving folder is: {checkpoint_dir}')
    
    
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    
    with open(positive_file, 'r') as f:
        positive_list = f.readlines()

    with open(negative_file, 'r') as f:
        negative_list = f.readlines()

    with open(hard_negative_file, 'r') as f:
        hard_negative_list = f.readlines()

    class SAMDataset(Dataset):
        def __init__(self, img_dir, mask_dir, processor, transform = None):
            self.processor = processor

            # get mask file path list

            self.img_dir = img_dir
            self.mask_dir = mask_dir

            self.mask_path_list = os.listdir(mask_dir)
            self.transform = transform

            self.positive_list = positive_list 
            self.negative_list = negative_list
            self.hard_negative_list = hard_negative_list


        def get_bounding_box(self, ground_truth_map):
            # get bounding box from mask
            y_indices, x_indices = np.where(ground_truth_map > 0)
            if len(x_indices) == 0:
                return [0,0,INPUT_PATCH_SIZE,INPUT_PATCH_SIZE]

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = [x_min, y_min, x_max, y_max]
            return bbox

        def __len__(self):
            return len(self.positive_list) * 2

        def __getitem__(self, idx):


            if random.random() > 0.5: 
                # select postive 
                cur_filename = random.choice(self.positive_list).strip()
            else:
                if random.random() > 0.5: 
                    # select random negative
                    cur_filename = random.choice(self.negative_list).strip()
                else:
                    # select hard negative
                    cur_filename = random.choice(self.hard_negative_list).strip()


            mask_path = os.path.join(self.mask_dir,cur_filename)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1].astype(bool)

            img_path = os.path.join(self.img_dir, cur_filename)
            image = Image.open(img_path)


            if self.transform:
                image, mask = self.transform(image, mask)

            mask = zoom(mask, 256./INPUT_PATCH_SIZE, order=1)  # order=1 for bilinear interpolation
            
            # Train by inputting either bbox prompt or an entire image range
            # prompt = self.get_bounding_box(mask)
            prompt = [0,0,INPUT_PATCH_SIZE,INPUT_PATCH_SIZE]

            # prepare image and prompt for the model
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

            # pdb.set_trace()
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            # add ground truth segmentation
            inputs["ground_truth_mask"] = mask

            return inputs


    # Define transformations for both images and masks using torchvision.transforms.v2 and RandAugment
    transform = T.Compose([
        T.RandomResizedCrop((1024, 1024), scale=(0.8, 1.2)),  # Random resized crop
        T.RandomHorizontalFlip(),  # Random horizontal flipping
        T.RandomVerticalFlip(),    # Random vertical flipping
        T.RandomRotation(degrees=(-45, 45)),  # Random rotation
        T.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random shifts (adjust translate values as needed)
    ])

    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    train_dataset = SAMDataset(img_dir= image_dir, mask_dir= mask_dir, processor=processor, transform = None)

    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)



    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # Gradient update only during decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

            
    optimizer = Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        loop = tqdm(train_dataloader) 
        for idx, batch in enumerate(loop):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                          input_boxes=batch["input_boxes"].to(device),
                          multimask_output=False)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            epoch_losses.append(loss.item())
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())


        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

        torch.save(model.state_dict(), os.path.join(checkpoint_dir,'ep'+str(epoch)+'.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='Greenland26X_22W_Sentinel2_2019-07-31_25_r6')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=int, default=5e-6)
    parser.add_argument('--weight_decay', type=int, default=0)
    args = parser.parse_args()
    
    
    main(args)