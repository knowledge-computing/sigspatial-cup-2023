import argparse

import geopandas as gpd
import pandas as pd
import numpy as np
import json

from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from shapely.geometry import Polygon, shape

from postproc import remove_duplicates, split_poly_by_narrow_streams


def convert_polys_to_gpkg(test_region, polys, out_gpkg_file):
    output_dict = {'image': [], 'region_num': [], 'geometry': []}

    def convert_points_to_polygon(pts):
        poly_output = Polygon(pts)
        return str(poly_output)
        
    for poly in polys:
        region = int(test_region[-1])
        image = test_region[:-3]
        output_dict['image'].append(image + '.tif')
        output_dict['region_num'].append(region)
        polygon = convert_points_to_polygon(poly)
        output_dict['geometry'].append(polygon)

    def augment(x):
        try:
            return shapely.wkt.loads(x)
        except Exception as e:
            print(e)
            return None
    
    df = pd.DataFrame(output_dict)
    df['geometry'] = df['geometry'].apply(augment)
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs='EPSG:3857')
    gdf.to_file(out_gpkg_file, driver='GPKG')
    

def convert_img_to_geocoord(test_region, polys, out_gpkg_file):
    region_num = int(test_region[-1])
    region_image_file = test_region[:-3] + '.tif'
    
    region_to_min_x_y = {
        1: (2, 122),
        2: (0, 130),
        3: (0, 138),
        4: (0, 145),
        5: (54, 0),
        6: (55, 19),
    }

    out_img = []
    out_region = []
    out_geom = []

    control_points = [
        # order - row, col, x, y
        GroundControlPoint(0, 0, -5684468.9195119608193636, 15654303.3928040582686663),
        GroundControlPoint(0, 87040, -2357929.4485410945490003, 15654303.3928040582686663),
        GroundControlPoint(160000, 87040, -2357929.4485410945490003, 9539341.1299899667501450),
        GroundControlPoint(160000, 0, -5684468.9195119608193636, 9539341.1299899667501450)
    ]
    transform_matrix = from_gcps(control_points)

    # the write is bad at this step
    filtered_polys = []
    for poly in polys:
        pts = np.array(poly)
        pts[:, 0] += region_to_min_x_y[region_num][0] * 1024
        pts[:, 1] += region_to_min_x_y[region_num][1] * 1024
        transformed_pts = np.apply_along_axis(lambda x: transform_matrix * x, axis=1, arr=pts)
        geom = Polygon(transformed_pts)
        if shape(geom).area < 100000: continue;
        filtered_polys += split_poly_by_narrow_streams(poly)

    filtered_polys = remove_duplicates(filtered_polys)
    for poly in filtered_polys:
        polys = split_poly_by_narrow_streams(poly)
        pts = np.array(poly)
        pts[:, 0] += region_to_min_x_y[region_num][0] * 1024
        pts[:, 1] += region_to_min_x_y[region_num][1] * 1024
        transformed_pts = np.apply_along_axis(lambda x: transform_matrix * x, axis=1, arr=pts)
        geom = Polygon(transformed_pts)
        if shape(geom).area < 100000: continue;
        out_img.append(region_image_file)
        out_region.append(region_num)
        out_geom.append(geom)

    print('# valid geometries =', len(out_geom))
    out_df = pd.DataFrame({'image': out_img, 'region_num': out_region, 'geometry': out_geom})
    out_geo_df = gpd.GeoDataFrame(out_df, geometry=out_df['geometry'], crs='EPSG:3857')
    out_geo_df.to_file(out_gpkg_file, driver='GPKG')
    return out_geo_df

def convert_img_to_geocoord_ori():
    region_to_min_x_y = {
        1: (2, 122),
        2: (0, 130),
        3: (0, 138),
        4: (0, 145),
        5: (54, 0),
        6: (55, 19),
    }

    out_img = []
    out_region = []
    out_geom = []

    control_points = [
        # order - row, col, x, y
        GroundControlPoint(0, 0, -5684468.9195119608193636, 15654303.3928040582686663),
        GroundControlPoint(0, 87040, -2357929.4485410945490003, 15654303.3928040582686663),
        GroundControlPoint(160000, 87040, -2357929.4485410945490003, 9539341.1299899667501450),
        GroundControlPoint(160000, 0, -5684468.9195119608193636, 9539341.1299899667501450)
    ]
    transform_matrix = from_gcps(control_points)

    with open(args.in_json) as f:
        annotations = json.load(f) # columns - region_image_file, region_num, poly
        for annotation in annotations:
            pts = np.array(annotation['poly'])
            pts[:, 0] += region_to_min_x_y[annotation['region_num']][0] * 1024
            pts[:, 1] += region_to_min_x_y[annotation['region_num']][1] * 1024

            transformed_pts = np.apply_along_axis(lambda x: transform_matrix * x, axis=1, arr=pts)

            out_img.append(annotation['region_image_file'])
            out_region.append(annotation['region_num'])
            out_geom.append(Polygon(transformed_pts.tolist()))

        out_df = pd.DataFrame({'image': out_img, 'region_num': out_region, 'geometry': out_geom})
        out_geo_df = gpd.GeoDataFrame(out_df, geometry=out_df['geometry'], crs='EPSG:3857')
        out_geo_df.to_file(args.out_gpkg, driver='GPKG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_json', type=str, default='annotations.json',
                        help='input annotation json')
    parser.add_argument('--out_gpkg', type=str, default='converted_annotations.gpkg',
                        help='output json with geocoordinates')

    args = parser.parse_args()
    convert_img_to_geocoord()

