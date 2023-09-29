import os
import argparse
import geopandas as gpd

regions = [
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

    data_root = os.path.join(args.data_root, '../dataset/2023_SIGSPATIAL_Cup_data_files')
    df = gpd.read_file(os.path.join(data_root, 'lake_polygons_training.gpkg'))

    for region_name in regions:
        region = int(region_name[-1])
        image_name = region_name[:-3]
        out_df = df.copy()
        out_df = out_df[out_df['region_num'] == region]
        out_df = out_df[out_df['image'] == image_name + '.tif']
        out_df.to_file(os.path.join(args.data_root, f'region_gpkgs_geocoords/{region_name}.gpkg'), driver='GPKG')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/')
    args = parser.parse_args()

    main()
