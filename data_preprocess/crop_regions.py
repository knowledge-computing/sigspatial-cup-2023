import os
import argparse
import numpy as np
import json
from PIL import Image

from utils import *

# these images are for target regions
target_regions = [
    'Greenland26X_22W_Sentinel2_2019-06-03_05_r1',
    'Greenland26X_22W_Sentinel2_2019-06-03_05_r3',
    'Greenland26X_22W_Sentinel2_2019-06-03_05_r5',
    'Greenland26X_22W_Sentinel2_2019-06-19_20_r2',
    'Greenland26X_22W_Sentinel2_2019-06-19_20_r4',
    'Greenland26X_22W_Sentinel2_2019-06-19_20_r6',
    'Greenland26X_22W_Sentinel2_2019-07-31_25_r1',
    'Greenland26X_22W_Sentinel2_2019-07-31_25_r3',
    'Greenland26X_22W_Sentinel2_2019-07-31_25_r5',
    'Greenland26X_22W_Sentinel2_2019-08-25_29_r2',
    'Greenland26X_22W_Sentinel2_2019-08-25_29_r4',
    'Greenland26X_22W_Sentinel2_2019-08-25_29_r6']

# these images are for test regions
target_regions_2 = [
    'Greenland26X_22W_Sentinel2_2019-06-03_05_r2',
    'Greenland26X_22W_Sentinel2_2019-06-03_05_r4',
    'Greenland26X_22W_Sentinel2_2019-06-03_05_r6',
    'Greenland26X_22W_Sentinel2_2019-06-19_20_r1',
    'Greenland26X_22W_Sentinel2_2019-06-19_20_r3',
    'Greenland26X_22W_Sentinel2_2019-06-19_20_r5',
    'Greenland26X_22W_Sentinel2_2019-07-31_25_r2',
    'Greenland26X_22W_Sentinel2_2019-07-31_25_r4',
    'Greenland26X_22W_Sentinel2_2019-07-31_25_r6',
    'Greenland26X_22W_Sentinel2_2019-08-25_29_r1',
    'Greenland26X_22W_Sentinel2_2019-08-25_29_r3',
    'Greenland26X_22W_Sentinel2_2019-08-25_29_r5']

def main():

    region_images_2 = [os.path.join(args.data_root, f'region_images/{i}.png') for i in target_regions_2]
    data_path = os.path.join(args.data_root, f'train_crop{args.crop_size}_shift{args.shift_size}')    
    output_img_path = os.path.join(data_path, 'train_images')
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    
    invalid_images = crop_region_images(region_images_2, output_img_path, args.crop_size, args.shift_size, 'jpg')
    with open(os.path.join(data_path, 'invalid_images.txt'), 'w') as f:
        for img in invalid_images:
            f.writelines(img + '\n')
    
    region_images = [os.path.join(args.data_root, f'region_images/{i}.png') for i in target_regions]
    data_path = os.path.join(args.data_root, f'target_crop{args.crop_size}_shift{args.shift_size}')    
    output_img_path = os.path.join(data_path, 'test_images')
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    
    invalid_images = crop_region_images(region_images, output_img_path, args.crop_size, args.shift_size, 'jpg')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--shift_size', type=int, default=256)
    args = parser.parse_args()

    main()



