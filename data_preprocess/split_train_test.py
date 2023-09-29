import os
import argparse
import json

target_regions = [
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
    
    data_path = os.path.join(args.data_root, f'train_crop{args.crop_size}_shift{args.shift_size}')
    train_json = os.path.join(data_path, 'train_poly.json')
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    invalid_images = {}
    with open(os.path.join(data_path, f'invalid_images.txt')) as f:
        for line in f.readlines():
            invalid_images[line.strip()] = 1
    
    with open(os.path.join(data_path, f'hard_neg_images.json')) as f:
        hard_neg_images = json.load(f)
    
    for region in target_regions:
    
        region_folder = os.path.join(data_path, 'train_' + region)
        if not os.path.exists(region_folder):
            os.makedirs(region_folder)
            
        train_pos_images, train_neg_images, test_images, train_hard_neg_images = [], [], [], []
        for image in data['images']:
            image_name = image['file_name']
            if invalid_images.get(image_name) is None and region not in image_name:
                if data['img2anno'].get(image_name):
                    train_pos_images.append(image_name)
                else:
                    train_neg_images.append(image_name)
                if hard_neg_images.get(image_name) and hard_neg_images[image_name] > 3.8:
                    train_hard_neg_images.append(image_name)
            elif region in image_name:
                test_images.append(image_name)
        
        with open(os.path.join(region_folder, f'train_pos.txt'), 'w') as f:
            for image in train_pos_images:
                f.writelines(image + '\n')
        
        with open(os.path.join(region_folder, f'train_neg.txt'), 'w') as f:
            for image in train_neg_images:
                f.writelines(image + '\n')
    
        with open(os.path.join(region_folder, f'train_hard_neg.txt'), 'w') as f:
            for image in train_hard_neg_images:
                f.writelines(image + '\n')
    
        with open(os.path.join(region_folder, f'test.txt'), 'w') as f:
            for image in test_images:
                f.writelines(image + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--shift_size', type=int, default=256)
    args = parser.parse_args()

    main()

