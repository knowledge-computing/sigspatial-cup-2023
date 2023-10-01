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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import Sampler

from torch.optim import Adam
import monai
from scipy.ndimage import zoom
import random 

from transformers import SamModel 
import argparse

import torch
torch.manual_seed(1)


INPUT_PATCH_SIZE = 1024
num_epochs = 10


def main(args):
    
    num_epochs = args.epoch
    
    print(f'Train start!')
    
    image_dir= args.img_dir
    mask_dir = args.mask_dir
    
    positive_file = args.positive_file
    hard_negative_file = args.hard_negative_file
    
    checkpoint_dir = f'checkpoints/'
    
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
            self.hard_negative_list = hard_negative_list
            
            self.input_all = self.positive_list + self.hard_negative_list
            self.input_all = [x.strip() for x in self.input_all]

        def __len__(self):
            return len(self.input_all)

        def __getitem__(self, idx):
            cur_filename = self.input_all[idx]
            
            mask_path = os.path.join(self.mask_dir,cur_filename)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1].astype(bool)

            img_path = os.path.join(self.img_dir, cur_filename)
            image = Image.open(img_path)
            
            if self.transform:
                image, mask = self.transform(image, mask)

            mask = zoom(mask, 256./INPUT_PATCH_SIZE, order=1)  # order=1 for bilinear interpolation

            prompt = [0,0,INPUT_PATCH_SIZE,INPUT_PATCH_SIZE]

            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

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
    
    # Weighted batch approach
    pos_sample=1./(float(len(positive_list)))
    neg_sample=1./(float(len(hard_negative_list)))
    
    weight_pos = [pos_sample] * len(positive_list)
    weight_neg = [neg_sample] * len(hard_negative_list)
    
    samples_weight = weight_pos + weight_neg
    samples_weight = np.array(samples_weight)  
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, sampler=sampler, drop_last=True)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        loop = tqdm(train_dataloader) 
        for idx, batch in enumerate(loop):
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
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--img_dir', type=str, default="/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_images")
    parser.add_argument('--mask_dir', type=str, default="/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_mask")
    parser.add_argument('--positive_file', type=str, default="/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_pos.txt")
    parser.add_argument('--hard_negative_file', type=str, default="/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_hard_neg.txt")
    
    
    
    args = parser.parse_args()
    
    
    main(args)
