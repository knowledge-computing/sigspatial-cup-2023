import os
import glob
import argparse
import numpy as np
import cv2
import json

def main():
    output_path = os.path.join(args.data_root, f'train_crop{crop_size}_shift{shift_size}')

    output_mask_path = os.path.join(output_path, 'train_mask/')
    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
    
    with open(os.path.join(output_path, 'train_poly.json'), 'r') as f:
        data = json.load(f)
    
    for image in data['images']:
        image_file_name = image['file_name']
        canvas = np.zeros((args.crop_size, args.crop_size), np.uint8)    
        
        if data['img2anno'].get(image_file_name) is not None:
            anno_ids = data['img2anno'][image_file_name]
            for anno_id in anno_ids:
                anno = data['annotations'][anno_id]
                poly = np.array(anno['poly']).reshape(-1, 2).astype(int)
                canvas = cv2.drawContours(canvas, [poly], 0, color=255, thickness=-1)
    
        mask_file = os.path.join(output_mask_path, image_file_name)
        cv2.imwrite(mask_file, canvas)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--shift_size', type=int, default=256)
    args = parser.parse_args()

    main()