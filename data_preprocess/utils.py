import os
import glob
import argparse
import numpy as np
import time
from PIL import Image
import cv2

Image.MAX_IMAGE_PIXELS=None # allow reading huge images

def crop_img(img_path, crop_size, shift_size, output_dir, output_format='png'):
    img_name = os.path.basename(img_path).split('.')[0]
    img = Image.open(img_path) 
    width, height = img.size 
    num_tiles_w = int(np.ceil(1. * width / shift_size))
    num_tiles_h = int(np.ceil(1. * height / shift_size))
    enlarged_width = int(shift_size * num_tiles_w)
    enlarged_height = int(shift_size * num_tiles_h)

    enlarged_map = Image.new(mode="RGB", size=(enlarged_width, enlarged_height))
    enlarged_map.paste(img) 

    invalid_images = []
    for idx in range(0, num_tiles_h):
        for jdx in range(0, num_tiles_w):
            img_clip = enlarged_map.crop((jdx * shift_size, idx * shift_size,
                                          jdx * shift_size + crop_size, idx * shift_size + crop_size,))
            pixel_value = np.sum(np.array(img_clip), axis=-1)
            out_path = os.path.join(output_dir, f"{img_name}__h{idx}_w{jdx}.{output_format}")
            
            # filter black images
            if np.sum(pixel_value == 0.) > crop_size ** 2 * 0.99:
                invalid_images.append(os.path.basename(out_path))
                
            img_clip.save(out_path)
    return invalid_images

def crop_region_images(region_images, output_dir, crop_size, shift_size, output_format):
    start_time = time.time()
    all_invalid_images = []
    for region_image in region_images:
        invalid_images = crop_img(region_image, crop_size, shift_size, output_dir, output_format)
        all_invalid_images += invalid_images
        print(f"----- {region_image} Done -----")
    print(f"Total time usage = {(time.time() - start_time)} sec.")
    return all_invalid_images
    

def find_intersection(contour1, contour2, image_size=(1024, 1024)):
    contour1 = contour1.astype(int)
    contour2 = contour2.astype(int)    
    contours = [contour1, contour2]
    blank = np.zeros(image_size, np.uint8)
    image1 = cv2.drawContours(blank.copy(), contours, 0, color=255, thickness=-1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, color=255, thickness=-1)
    intersection = np.logical_and(image1, image2)
    return intersection


def fine_poly_from_image(poly, image_size=(1024, 1024)):
    blank = np.zeros(image_size, np.uint8)
    blank[poly] = 255
    contours, _ = cv2.findContours(image=blank, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        poly = None
    elif len(contours) == 1:
        poly = contours[0].reshape(-1, 2)
    else:
        areas = [cv2.contourArea(c) for c in contours]
        poly = contours[areas.index(max(areas))].reshape(-1, 2)
    return poly

