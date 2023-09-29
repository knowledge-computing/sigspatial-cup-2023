import os
import argparse
from PIL import Image
import cv2
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import shape
from shapely import wkt
from shapely.geometry import Polygon

Image.MAX_IMAGE_PIXELS=None # allow reading huge images


def transform_image_coord(geometry, transform):
    poly = np.array(list(geometry.exterior.coords))
    transformed_pts = np.apply_along_axis(lambda x: rasterio.transform.rowcol(transform, x[0], x[1]), axis=1, arr=poly)
    transformed_pts = transformed_pts[:, [1, 0]]
    return np.array(transformed_pts)


def extract(ori_im, ori_df, rid, left, top, right, bottom):
    # (left, top, right, bottom)
    sub_img = ori_im.crop((1024*left, 1024*top, 1024*right, 1024*bottom,))
    print(np.min(sub_img))
    poly = ori_df.where(ori_df.region_num == rid).dropna()['image_coord'].values[0]
    poly[:, 0] -= 1024*left
    poly[:, 1] -= 1024*top
    w, h = sub_img.size
    mask = np.zeros((h, w, 3), np.uint8)
    mask = cv2.drawContours(mask, [poly], 0, color=(1,1,1), thickness=-1)
    return Image.fromarray(sub_img * mask)
    

def main():

    with open('transform.pgw') as file: # load transformation
        a, d, b, e, c, f = np.loadtxt(file)
        transform = rasterio.Affine(a, b, c, d, e, f)

    df = gpd.read_file(os.path.join(args.data_root, 'lakes_regions.gpkg'))
    df['image_coord'] = df['geometry'].apply(lambda x: transform_image_coord(x, transform))

    tif_file = os.path.join(args.data_root, args.tif_file)
    file_name = args.tif_file.split('.')[0]
    print(f'Processing the file: {tif_file}...')
    im = Image.open(tif_file)
    im_w, im_h = im.size
    
    sub_img_r1 = extract(im, df, 1, 2, 122, 16, 131)
    sub_img_r1.save(os.path.join(args.output_dir, f'{file_name}_r1.jpg'))
    sub_img_r2 = extract(im, df, 2, 0, 130, 15, 139)
    sub_img_r2.save(os.path.join(args.output_dir, f'{file_name}_r2.jpg'))
    sub_img_r3 = extract(im, df, 3, 0, 138, 15, 148)
    sub_img_r3.save(os.path.join(args.output_dir, f'{file_name}_r3.jpg'))
    sub_img_r4 = extract(im, df, 4, 0, 145, 14, 157)
    sub_img_r4.save(os.path.join(args.output_dir, f'{file_name}_r4.jpg'))
    sub_img_r5 = extract(im, df, 5, 54, 0, 85, 24)
    sub_img_r5.save(os.path.join(args.output_dir, f'{file_name}_r5.jpg'))
    sub_img_r6 = extract(im, df, 6, 55, 19, 83, 42)
    sub_img_r6.save(os.path.join(args.output_dir, f'{file_name}_r6.jpg'))
    im.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/2023_SIGSPATIAL_Cup_data_files')
    parser.add_argument('--output_dir', type=str, default='../dataset/region_images')
    parser.add_argument('--tif_file', type=str, default='Greenland26X_22W_Sentinel2_2019-08-25_29.tif')
    args = parser.parse_args()

    main()

# file_names = [#'Greenland26X_22W_Sentinel2_2019-06-03_05',
#               #'Greenland26X_22W_Sentinel2_2019-06-19_20',
#               #'Greenland26X_22W_Sentinel2_2019-07-31_25',
#               # 'Greenland26X_22W_Sentinel2_2019-08-25_29',]

