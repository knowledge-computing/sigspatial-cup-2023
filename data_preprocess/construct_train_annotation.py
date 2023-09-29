import os
import glob
import argparse
import numpy as np
import json
from PIL import Image

from utils import *

CAT = {'supercategory':'beverage', 'id':1, 'keypoints':['mean','xmin','x2','x3','xmax','ymin','y2','y3','ymax','cross'], 'name':'lake'}
IMG_OUTPUT_FORMAT = 'jpg'

def construct_annotations(crop_img_path, crop_size, shift_size):
    index_mapping = {}
    outputs = {'licenses': [], 'info': {}, 'images': [], 'annotations': [], 'categories': [CAT], 
               'img2anno': {}, 'index_mapping': {}}

    region_names = []
    images = glob.glob(os.path.join(crop_img_path, f'*.{IMG_OUTPUT_FORMAT}'))
    images = sorted(images)
    for image_path in images:
        file_name = os.path.basename(image_path)
        image_id = len(outputs['images'])
        outputs['index_mapping'][file_name] = image_id
        outputs['images'].append({
            'id': image_id,
            'file_name': file_name,
            'height': crop_size,
            'width': crop_size,
            'license': 0})
        region_names.append(file_name.split('__')[0])

    region_names = sorted(list(set(region_names)))
    outputs['train_regions'] = region_names
    for anno in annotations:
        region_image_file = anno['region_image_file']
        region_image_name = region_image_file.split('.')[0]
        if region_image_name not in region_names:
            continue
            
        poly = np.array(anno['poly']).reshape(-1, 2)
        min_num_tiles_w = max(0, int(np.min(poly[:, 0]) / shift_size) - crop_size // shift_size + 1)
        max_num_tiles_w = int(np.max(poly[:, 0]) / shift_size) + 1
        min_num_tiles_h = max(0, int(np.min(poly[:, 1]) / shift_size) - crop_size // shift_size + 1)
        max_num_tiles_h = int(np.max(poly[:, 1]) / shift_size) + 1
    
        for idx in range(min_num_tiles_h, max_num_tiles_h):
            for jdx in range(min_num_tiles_w, max_num_tiles_w):
                bbox_x_min = jdx * shift_size
                bbox_y_min = idx * shift_size
    
                new_poly = 1. * poly.copy()
                new_poly[:, 0] -= bbox_x_min
                new_poly[:, 1] -= bbox_y_min

                if (new_poly < 0).any() or (new_poly >= crop_size).any():
                    image_box = np.array([[0, 0], [0, crop_size], [crop_size, crop_size], [crop_size, 0]])
                    intersection = find_intersection(new_poly, image_box)
                    new_poly = fine_poly_from_image(intersection)  
                    
                if new_poly is None:
                    continue

                # # this is not a good way to do
                # new_poly[new_poly < 0] = 0.
                # new_poly[new_poly >= CROP_SIZE] = 1. * CROP_SIZE - 1

                image_file_name = f"{region_image_name}__h{idx}_w{jdx}.{IMG_OUTPUT_FORMAT}"
                if not os.path.exists(os.path.join(crop_img_path, image_file_name)):
                    print(f'Warning!!! The image {image_file_name} does not exist!')
                    continue

                image_id = outputs['index_mapping'][image_file_name]
                X, Y, w, h = cv2.boundingRect(new_poly.astype(int))
                bbox = [X, Y, w, h]
                area = cv2.contourArea(new_poly.astype(int))
                anno_id = len(outputs['annotations'])
                outputs['annotations'].append({
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': 1,
                    'area': area,
                    'id': anno_id,
                    'poly': list(1. * new_poly.reshape(-1))})

                if outputs['img2anno'].get(image_file_name) is None:
                    outputs['img2anno'][image_file_name] = []
                outputs['img2anno'][image_file_name].append(anno_id)

    print(f"Generated {len(outputs['annotations'])} annotations")
    return outputs


def main():    
    outputs = construct_annotations(args.crop_img_path, args.crop_size, args.shift_size)
    with open(args.output_json, 'w') as f:
        json.dump(outputs, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset')
    parser.add_argument('--output_dir', type=str, default='../dataset')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--shift_size', type=int, default=256)
    args = parser.parse_args()

    args.anno_file = os.path.join(args.data_root, 'annotations.json')
    args.output_dir = os.path.join(args.output_dir, f'train_crop{args.crop_size}_shift{args.shift_size}')
    args.crop_img_path = os.path.join(args.output_dir, 'train_images')
    args.output_json = os.path.join(args.output_dir, 'train_poly.json')
    with open(args.anno_file, 'r') as f:
        annotations = json.load(f)
    
    outputs = main()
    


    

