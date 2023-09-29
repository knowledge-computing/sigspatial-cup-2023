import os
import argparse
import numpy as np
import json
from PIL import Image
from imutils import build_montages
from imutils import paths
import imutils
import cv2


def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def main():

    data_path = os.path.join(args.data_root, f'train_crop{args.crop_size}_shift{args.shift_size}/')
    train_json = os.path.join(data_path, 'train_poly.json')
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    # initialize the results list
    print("[INFO] computing colorfulness metric for dataset...")
    results = {}
    # loop over the image paths
    for image in data['images']:
        file_name = image['file_name']
        if data['img2anno'].get(file_name) is not None:
            continue
        img = cv2.imread(os.path.join(data_path, 'train_images', file_name))
        C = image_colorfulness(img)
        results[file_name] = C
    
    with open(os.path.join(data_path, 'hard_neg_images.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--shift_size', type=int, default=256)
    args = parser.parse_args()

    main()










