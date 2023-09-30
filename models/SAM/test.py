import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from transformers import SamProcessor
from transformers import SamModel 
import torch.nn.functional as F
import torch
import os
import argparse


def main(args):
    ep = args.epoch
    REGION = args.region
    
    test_file = f"/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_{REGION}/test.txt"

    with open(test_file, 'r') as f:
        test_list = f.readlines()
    print(f'total: {len(test_list)}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.to(device)

    model.load_state_dict(torch.load(f'checkpoints_{REGION}/ep{ep}.pth'))

    model.eval()
    INPUT_PATCH_SIZE=1024
    def test_model(img_path, model, processor):

        image = Image.open(img_path)

        prompt = [[[0,0,1024,1024]]]

        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # apply sigmoid
        lake_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        lake_seg_prob = F.interpolate(lake_seg_prob, (INPUT_PATCH_SIZE, INPUT_PATCH_SIZE), mode="bilinear", align_corners=False) 
        
        # convert soft mask to hard mask
        lake_seg_prob = lake_seg_prob.cpu().numpy().squeeze()
        lake_seg = (lake_seg_prob > 0.5).astype(np.uint8)

        return lake_seg, lake_seg_prob


    output_dir = f'{REGION}_predmask_ep{ep}_APP2/'
    output_dir_ary = f'{REGION}_predary_ep{ep}_APP2/'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir_ary):
        os.makedirs(output_dir_ary)

    target_size = 1024
    test_dir = "/home/yaoyi/shared/sigspatial/train_crop1024_shift512/train_images"

    # for img_name in img_list[14:]:
    for img_name in test_list:

        img_path = os.path.join(test_dir, img_name.strip())

        lake_seg, lake_seg_prob = test_model(img_path, model, processor)
        upscaled_mask = lake_seg 
        upscaled_image = Image.fromarray(upscaled_mask * 255)
        upscaled_image.save(os.path.join(output_dir, img_name.strip()))
        np.save(os.path.join(output_dir_ary, img_name.strip()), lake_seg_prob)

        print(f'done: {img_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='Greenland26X_22W_Sentinel2_2019-07-31_25_r6')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    
    
    main(args)