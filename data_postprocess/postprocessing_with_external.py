from osgeo import ogr, gdal, osr

import os
import shutil

import cv2
import math
import csv
import numpy as np
from scipy import ndimage

import geopandas as gpd
import pandas as pd
#pd.set_option('mode.chained_assignment', None)

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.plot import show

import tifffile
from datetime import datetime

from giscup_evaluation_f1 import giscup_evaluation_f1_per_map

training_stamp = [[False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False], [False, True, False, True], [True, False, True, False]]


def initialization_dirs():
    if not os.path.exists('Intermediate'):
        os.makedirs('Intermediate')
    if not os.path.exists('Intermediate/Extraction'):
        os.makedirs('Intermediate/Extraction')
    if not os.path.exists('Intermediate/Postprocessing_Output'):
        os.makedirs('Intermediate/Postprocessing_Output')
    if not os.path.exists('Intermediate/Output'):
        os.makedirs('Intermediate/Output')
    if not os.path.exists('Intermediate/External'):
        os.makedirs('Intermediate/External')


# Split the original satellite image into different regions
# Input: authority dataset
# Output: write to 'Intermediate'
def preprocessing_delineation(path_to_source):
    image_shape = []
    vector_fn = path_to_source+'lakes_regions.gpkg'
    image_stamp = path_to_source+'Greenland26X_22W_Sentinel2_'

    for fid in range(1, 7):
        image_shape.append([])
    timestamp = ['2019-06-03_05.tif', '2019-06-19_20.tif', '2019-07-31_25-001.tif', '2019-08-25_29-002.tif']

    for fid in range(1, 7):
        Vector = gpd.read_file(vector_fn)
        Vector = Vector[Vector['region_num'] == fid]
        #print(Vector)

        for tid in range(1, 5):
            target_stamp = image_stamp + timestamp[tid-1]
            raster_fn = 'Intermediate/delineated_region_'+str(fid)+'_'+str(tid)+'.tif'

            with rasterio.open(target_stamp) as src:
                out_image, out_transform = mask(src, Vector.geometry, crop=True)
                out_meta = src.meta.copy() # copy the metadata of the source DEM
                
            out_meta.update({
                "driver":"Gtiff",
                "height":out_image.shape[1],
                "width":out_image.shape[2],
                "transform":out_transform
            })
                        
            with rasterio.open(raster_fn,'w',**out_meta) as dst:
                dst.write(out_image)
            
            if tid == 1:
                image_shape[fid-1] = out_image.shape

    return image_shape
    # 5m 22.2s


# Get a basemap representing the region to support further floodfill
# Input: authority dataset, auxiliary info
# Output: write to 'Intermediate'
def preprocessing_boundary(path_to_source, image_shape):
    region_extent = []
    region_projection = []
    vector_fn = path_to_source+'lakes_regions.gpkg'

    for fid in range(1, 7):
        pixel_size = 38
        NoData_value = 0

        raster_fn = 'Intermediate/rasterized_boundary_'+str(fid)+'.tif'

        source_ds = ogr.Open(vector_fn)
        source_layer = source_ds.GetLayer()

        source_feature = source_layer.GetFeature(fid)
        spatialRef = source_feature.GetGeometryRef()

        x_min, x_max, y_min, y_max = spatialRef.GetEnvelope()
        region_extent.append(spatialRef.GetEnvelope())
        region_projection.append(spatialRef.ExportToWkt())

        x_res = image_shape[fid-1][2]
        y_res = image_shape[fid-1][1]
        x_pixel_size = int((x_max - x_min) / x_res)
        y_pixel_size = int((y_max - y_min) / y_res)
        target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)

        target_ds.SetGeoTransform((x_min, x_pixel_size, 0, y_max, 0, -y_pixel_size))
        projection = spatialRef.ExportToWkt()
        target_ds.SetProjection(projection)
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(NoData_value)

        # Rasterize
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1] ) #source_layer
        source_ds = None
        target_ds = None
    
    for fid in range(1, 7):
        boundary_image = cv2.imread('Intermediate/rasterized_boundary_'+str(fid)+'.tif')
        boundary_image[boundary_image > 0] = 255
        cv2.imwrite('Intermediate/rasterized_boundary_visualization_'+str(fid)+'.tif', boundary_image)
    
    return region_extent, region_projection
    # 25.5s


# Get the ideal geo-related information to support generating output gpkg
# Input: authority dataset, auxiliary info
# Output: write to 'Intermediate'
def preprocessing_visualization(path_to_source, image_shape, region_extent, region_projection):
    image_stamp = 'Greenland26X_22W_Sentinel2_'
    timestamp = ['2019-06-03_05.tif', '2019-06-19_20.tif', '2019-07-31_25.tif', '2019-08-25_29.tif']

    for fid in range(1, 7):
        for tid in range(1, 5):
            if training_stamp[fid-1][tid-1] == False:
                continue
            
            target_stamp = image_stamp + timestamp[tid-1]

            pixel_size = 38
            NoData_value = 0

            vector_fn = str(path_to_source) + 'lake_polygons_training.gpkg'
            raster_fn = 'Intermediate/rasterized_groundtruth_'+str(fid)+'_'+str(tid)+'.tif'

            source_ds = ogr.Open(vector_fn)
            source_layer = source_ds.GetLayer()
            source_layer.SetAttributeFilter('region_num = "'+str(fid)+'" and image = "'+str(target_stamp)+'"')

            x_min, x_max, y_min, y_max = region_extent[fid-1]

            x_res = image_shape[fid-1][2]
            y_res = image_shape[fid-1][1]
            x_pixel_size = ((x_max - x_min) / x_res)
            y_pixel_size = ((y_max - y_min) / y_res)
            target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)

            target_ds.SetGeoTransform((x_min, x_pixel_size, 0, y_max, 0, -y_pixel_size))
            projection = region_projection[fid-1]
            target_ds.SetProjection(projection)
            band = target_ds.GetRasterBand(1)
            band.SetNoDataValue(NoData_value)

            # Rasterize
            gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1] ) #source_layer
            source_ds = None
            target_ds = None

            try:
                cv_img = cv2.imread(raster_fn)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                cv_img = cv_img*255
                cv2.imwrite(raster_fn.replace('rasterized_groundtruth_', 'rasterized_groundtruth_visualization_'), cv_img)
            except:
                print('CV_IO_MAX_IMAGE_PIXELS...')
    # 30.4s


# Extraction polygons based on color thresholding
# Input: intermediate images from 'Intermediate'
# Output: write to 'Intermediate'
def color_based_extraction():
    for fid in range(1, 7):
        for tid in range(1, 5):
            solution_dir = 'Intermediate/Extraction/Fig_'+str(fid)+'_'+str(tid)+'/'
            if not os.path.exists(solution_dir):
                os.makedirs(solution_dir)
            
            groundtruth_image = cv2.imread('Intermediate/delineated_region_'+str(fid)+'_'+str(tid)+'.tif')
            groundtruth_rgb = cv2.cvtColor(groundtruth_image, cv2.COLOR_BGR2RGB)

            lower_white = np.array([180,180,180])
            upper_white = np.array([255,255,255])
            groundtruth_rgb_masked = cv2.inRange(groundtruth_rgb, lower_white, upper_white)
            groundtruth_rgb_masked = cv2.cvtColor(groundtruth_rgb_masked, cv2.COLOR_RGB2BGR)
            #cv2.imwrite('Intermediate/masked_region_'+str(fid)+'_'+str(tid)+'.tif', groundtruth_rgb_masked)

            lower_black = np.array([0,0,0])
            upper_black = np.array([1,1,1])
            area_of_interest = cv2.inRange(groundtruth_rgb, lower_black, upper_black)
            area_of_interest = 255 - area_of_interest

            groundtruth_rgb_masked = 255 - groundtruth_rgb_masked
            groundtruth_rgb_masked = cv2.bitwise_and(groundtruth_rgb_masked, groundtruth_rgb_masked, mask = area_of_interest)
            #cv2.imwrite('Intermediate/unmasked_region_'+str(fid)+'_'+str(tid)+'.tif', groundtruth_rgb_masked)

            lower_blue = np.array([0,0,90])
            upper_blue = np.array([30,255,255])
            area_of_blue = cv2.inRange(groundtruth_rgb, lower_blue, upper_blue)
            area_of_blue_masked_v1 = cv2.bitwise_and(groundtruth_rgb, groundtruth_rgb, mask = area_of_blue)
            area_of_blue_masked_v1 = cv2.cvtColor(area_of_blue_masked_v1, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(solution_dir + 'blue_region_(1)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_v1)

            lower_blue = np.array([0,0,105])
            upper_blue = np.array([60,255,255])
            area_of_blue = cv2.inRange(groundtruth_rgb, lower_blue, upper_blue)
            area_of_blue_masked_v2 = cv2.bitwise_and(groundtruth_rgb, groundtruth_rgb, mask = area_of_blue)
            area_of_blue_masked_v2 = cv2.cvtColor(area_of_blue_masked_v2, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(solution_dir + 'blue_region_(2)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_v2)

            lower_blue = np.array([0,0,120])
            upper_blue = np.array([90,255,255])
            area_of_blue = cv2.inRange(groundtruth_rgb, lower_blue, upper_blue)
            area_of_blue_masked_v3 = cv2.bitwise_and(groundtruth_rgb, groundtruth_rgb, mask = area_of_blue)
            area_of_blue_masked_v3 = cv2.cvtColor(area_of_blue_masked_v3, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(solution_dir + 'blue_region_(3)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_v3)

            lower_blue = np.array([0,0,150])
            upper_blue = np.array([120,255,255])
            area_of_blue = cv2.inRange(groundtruth_rgb, lower_blue, upper_blue)
            area_of_blue_masked_v4 = cv2.bitwise_and(groundtruth_rgb, groundtruth_rgb, mask = area_of_blue)
            area_of_blue_masked_v4 = cv2.cvtColor(area_of_blue_masked_v4, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(solution_dir + 'blue_region_(4)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_v4)

            area_of_blue_masked_merged = cv2.bitwise_or(area_of_blue_masked_v1, area_of_blue_masked_v2)
            area_of_blue_masked_merged = cv2.bitwise_or(area_of_blue_masked_merged, area_of_blue_masked_v3)
            area_of_blue_masked_merged = cv2.bitwise_or(area_of_blue_masked_merged, area_of_blue_masked_v4)
            cv2.imwrite(solution_dir + 'blue_region_(5)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_merged)
            area_of_blue_masked_merged = cv2.cvtColor(area_of_blue_masked_merged, cv2.COLOR_BGR2RGB)

            ### buffer to support floodfill
            area_of_blue_masked_merged = cv2.imread(solution_dir + 'blue_region_(5)_'+str(fid)+'_'+str(tid)+'.tif')
            area_of_blue_masked_merged = cv2.cvtColor(area_of_blue_masked_merged, cv2.COLOR_BGR2GRAY)
            area_of_blue_masked_merged[area_of_blue_masked_merged > 0] = 255

            blur_radius = 5.0
            threshold_blur = 255*0.0
            gaussian_buffer = ndimage.gaussian_filter(area_of_blue_masked_merged, blur_radius)
            gaussian_buffer[gaussian_buffer > threshold_blur] = 255
            gaussian_buffer[gaussian_buffer <= threshold_blur] = 0
            #cv2.imwrite(solution_dir + 'blue_region_(6)_'+str(fid)+'_'+str(tid)+'.tif', gaussian_buffer)

            ### floodfill
            img_bound = cv2.imread('Intermediate/rasterized_boundary_visualization_'+str(fid)+'.tif')
            img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)
            floodfill_candidate = np.copy(img_bound)

            # flood fill background to find inner holes
            holes = np.copy(gaussian_buffer)
            cv2.floodFill(holes, None, (0, 0), 255)

            # invert holes mask, bitwise or with img fill in holes
            holes = cv2.bitwise_not(holes)
            valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
            #filled_holes = cv2.bitwise_or(polygon_candidate_covered, valid_holes)

            blur_radius = 7.0
            threshold_blur = 255*0.0
            gaussian_buffer_2 = ndimage.gaussian_filter(valid_holes, blur_radius)
            gaussian_buffer_2[gaussian_buffer_2 > threshold_blur] = 255
            gaussian_buffer_2[gaussian_buffer_2 <= threshold_blur] = 0

            area_of_blue_masked_merged = cv2.bitwise_or(area_of_blue_masked_merged, gaussian_buffer_2)
            #cv2.imwrite(solution_dir + 'blue_region_(7)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_merged)

            ### floodfill
            # flood fill background to find inner holes
            holes = np.copy(area_of_blue_masked_merged)
            cv2.floodFill(holes, None, (0, 0), 255)

            # invert holes mask, bitwise or with img fill in holes
            holes = cv2.bitwise_not(holes)
            valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
            #filled_holes = cv2.bitwise_or(polygon_candidate_covered, valid_holes)

            area_of_blue_masked_merged = cv2.bitwise_or(area_of_blue_masked_merged, valid_holes)
            cv2.imwrite(solution_dir + 'blue_region_(8)_'+str(fid)+'_'+str(tid)+'.tif', area_of_blue_masked_merged)
            # 12m 50.7s


            basic = np.copy(area_of_blue_masked_merged)

            # remove noisy white pixel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            opening = cv2.morphologyEx(basic, cv2.MORPH_OPEN, kernel, iterations=1)
            basic = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
            opening = cv2.morphologyEx(basic, cv2.MORPH_OPEN, kernel, iterations=1)
            basic = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
            opening = cv2.morphologyEx(basic, cv2.MORPH_OPEN, kernel, iterations=1)
            basic = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

            ### floodfill
            img_bound = cv2.imread('Intermediate/rasterized_boundary_visualization_'+str(fid)+'.tif')
            img_bound = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)
            floodfill_candidate = np.copy(img_bound)

            # flood fill background to find inner holes
            holes = np.copy(basic)
            cv2.floodFill(holes, None, (0, 0), 255)

            # invert holes mask, bitwise or with img fill in holes
            holes = cv2.bitwise_not(holes)
            valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
            postprocessed_combined = cv2.bitwise_or(basic, valid_holes)

            cv2.imwrite('Intermediate/Postprocessing_Output/color_based_'+str(fid)+'_'+str(tid)+'.png', postprocessed_combined)


# Trun color-based polygons to gpkg
# Input: intermediate images from 'Intermediate'
# Output: write to 'Intermediate/Postprocessing_Output/color_based_polygon.gpkg'
def png_to_gpkg():
    if os.path.isfile('Intermediate/Postprocessing_Output/color_based_polygon.gpkg'):
        os.remove('Intermediate/Postprocessing_Output/color_based_polygon.gpkg')

    time_stamp = ['2019-06-03_05', '2019-06-19_20', '2019-07-31_25', '2019-08-25_29']

    for fid in range(1, 7):
        for tid in range(1, 5):
            targeted_testing_file = str(fid)+'_'+str(tid)
            this_region_id = fid
            this_time_id = tid-1

            img = rasterio.open('Intermediate/Postprocessing_Output/color_based_'+str(targeted_testing_file)+'.png')
            img = img.read([1])
            img = img.astype('uint16')

            cand_file = 'Intermediate/rasterized_groundtruth_'+str(targeted_testing_file)+'.tif'
            if os.path.isfile(cand_file) == False:
                if ('_1.tif' in cand_file) or ('_3.tif' in cand_file):
                    cand_file = cand_file.replace('_1.tif', '_2.tif')
                    cand_file = cand_file.replace('_3.tif', '_2.tif')
                else:
                    cand_file = cand_file.replace('_2.tif', '_1.tif')
                    cand_file = cand_file.replace('_4.tif', '_1.tif')


            with rasterio.open(cand_file) as naip:
                with rasterio.open(
                    'Intermediate/Postprocessing_Output/color_based_'+str(targeted_testing_file)+'_v2.tif',
                    'w',
                    driver='GTiff',
                    count=img.shape[0],
                    height=img.shape[1],
                    width=img.shape[2],
                    dtype=img.dtype,
                    crs=naip.crs,
                    transform=naip.transform,
                    ) as dst:
                        dst.write(img)


            with rasterio.open('Intermediate/Postprocessing_Output/color_based_'+str(targeted_testing_file)+'_v2.tif') as limg:
                    limg = limg.read(out_shape=(1,naip.shape[0],naip.shape[1]),
                                    resampling=Resampling.nearest)
                    with rasterio.open('resampled.tif','w', 
                                driver='GTiff',
                                count=limg.shape[0],
                                height=limg.shape[1],
                                width=limg.shape[2],
                                dtype=limg.dtype,
                                crs=naip.crs,
                                transform=naip.transform,
                                ) as dst:
                        dst.write(limg)


            in_path = 'Intermediate/Postprocessing_Output/color_based_'+str(targeted_testing_file)+'_v2.tif'
            out_path = 'Intermediate/Postprocessing_Output/color_based_'+str(targeted_testing_file)+'.gpkg'

            src_ds = gdal.Open( in_path )
            srcband = src_ds.GetRasterBand(1)
            dst_layername = 'polygon'
            drv = ogr.GetDriverByName("GPKG")
            dst_ds = drv.CreateDataSource( out_path )

            sp_ref = osr.SpatialReference()
            sp_ref.SetFromUserInput('EPSG:3857')

            dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )

            image_stamp = 'Greenland26X_22W_Sentinel2_'
            output_time_stamp = image_stamp + time_stamp[this_time_id] + '.tif'
            output_region_stamp = this_region_id

            gdal.Polygonize( srcband, None, dst_layer, 0, [], callback=None )

            del src_ds
            del dst_ds

            original_gpd = gpd.read_file(out_path)
            this_gpd = original_gpd.copy()
            this_gpd.drop(this_gpd.tail(1).index,inplace=True)
            this_gpd.loc[:, 'image'] = output_time_stamp
            this_gpd.loc[:, 'region_num'] = output_region_stamp

            if not os.path.isfile('Intermediate/Postprocessing_Output/color_based_polygon.gpkg'):
                this_gpd.to_file('Intermediate/Postprocessing_Output/color_based_polygon.gpkg', driver="GPKG")
            else:
                original_submission = gpd.read_file('Intermediate/Postprocessing_Output/color_based_polygon.gpkg')
                submission = original_submission.copy()
                submission = gpd.GeoDataFrame(pd.concat([submission, this_gpd], ignore_index=True))
                submission.to_file('Intermediate/Postprocessing_Output/color_based_polygon.gpkg', driver="GPKG")
    # 2m 31.9s


# Trun topo-based polygons to gpkg
# Input: authority dataset, topo dataset
# Output: write to 'Intermediate/External/topo_based_polygon.gpkg'
def topo_to_gpkg(path_to_source, path_to_topo):
    base_img = tifffile.imread(path_to_topo)
    base_img[base_img < 0] = 255
    base_img[base_img == None] = 0

    tifffile.imwrite('Intermediate/External/topographic_sink_reversed.tif', base_img)

    with rasterio.open('topographic_sink.tif') as naip:
        with rasterio.open(
            'Intermediate/External/topographic_sink_reversed_v2.tif',
            'w',
            driver='GTiff',
            count=1,
            height=base_img.shape[0],
            width=base_img.shape[1],
            dtype=base_img.dtype,
            crs=naip.crs,
            transform=naip.transform,
            ) as dst:
                dst.write(base_img, indexes = 1)
    # 6.6s


    src_ds = gdal.Open('Intermediate/External/topographic_sink_reversed_v2.tif')

    srcband = src_ds.GetRasterBand(1)
    dst_layername = 'polygon'
    drv = ogr.GetDriverByName("GPKG")
    dst_ds = drv.CreateDataSource('Intermediate/External/topographic_sink_reversed_v2.gpkg')

    sp_ref = osr.SpatialReference()
    sp_ref.SetFromUserInput('EPSG:3413')

    dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )
    gdal.Polygonize( srcband, None, dst_layer, 0, [], callback=None )

    del src_ds
    del dst_ds
    # 24.6s


    topo_gpkg = 'Intermediate/External/topographic_sink_reversed_v2.gpkg'
    region_gpkg = path_to_source+'lakes_regions.gpkg'

    layer1 = gpd.read_file(region_gpkg)
    layer2 = gpd.read_file(topo_gpkg)

    # Ovalap polygons from two gpkg
    overlay_polygon = gpd.overlay(layer1, layer2.to_crs(epsg=3857), how='intersection', keep_geom_type=False)
    overlay_polygon['overlap_area'] = overlay_polygon['geometry'].area/10**6

    overlay_polygon.drop(overlay_polygon[overlay_polygon['overlap_area'] >= 10000].index, inplace = True)
    overlay_polygon.to_file('Intermediate/External/topo_based_polygon.gpkg', layer='polygon', driver='GPKG')
    # 13.0s


def postprocessing(path_to_model_sam, path_to_model_lab, output_gpkg):
    import warnings
    warnings.filterwarnings("ignore")
    
    gpkg_timestamp = ['2019-06-03_05', '2019-06-19_20', '2019-07-31_25', '2019-08-25_29']
    if not os.path.exists('Intermediate/Output'):
        os.makedirs('Intermediate/Output')

    for fid in range(1, 7):
        for tid in range(1, 5):

            targeted_region = fid
            targeted_time = gpkg_timestamp[tid-1]
            image_stamp = 'Greenland26X_22W_Sentinel2_'
            target_stamp = image_stamp + targeted_time + '.tif'

            model_gpkg = path_to_model_sam + image_stamp + targeted_time +'_r'+ str(fid) +'_pred_'+ image_stamp + targeted_time +'_r'+ str(fid) +'__out.gpkg' # need to follow the naming
            model_gpkg_2 = path_to_model_lab + 'crop1024_shift512_dlv3p_eval_overlap_leeje_0928_' + image_stamp + targeted_time +'_r'+ str(fid) +'__out.gpkg' # need to follow the naming

            topo_gpkg = 'Intermediate/External/topo_based_polygon.gpkg'
            color_gpkg = 'Intermediate/Postprocessing_Output/color_based_polygon.gpkg'

            if os.path.isfile(model_gpkg) == False:
                print(model_gpkg + '... File not exist...')
                continue
            if os.path.isfile(model_gpkg_2) == False:
                print(model_gpkg_2 + '... File not exist...')
                continue

            print(image_stamp + targeted_time +'_r'+ str(fid))

            layer1 = gpd.read_file(model_gpkg)
            layer2 = gpd.read_file(topo_gpkg)
            layer3 = gpd.read_file(color_gpkg)
            layer4 = gpd.read_file(model_gpkg_2)

            model_polygon = layer1[(layer1['image']==target_stamp) & (layer1['region_num']==targeted_region)]
            model_polygon['m_id'] = range(1, model_polygon.shape[0]+1)
            model_polygon['m_area'] = model_polygon['geometry'].area/10**6

            topo_polygon = layer2[(layer2['region_num']==targeted_region)]
            topo_polygon['t_id'] = range(1, topo_polygon.shape[0]+1)
            topo_polygon['t_area'] = topo_polygon['geometry'].area/10**6

            color_polygon = layer3[(layer3['region_num']==targeted_region)]
            color_polygon['c_id'] = range(1, color_polygon.shape[0]+1)
            color_polygon['c_area'] = color_polygon['geometry'].area/10**6

            model_polygon2 = layer4[(layer4['image']==target_stamp) & (layer4['region_num']==targeted_region)]
            model_polygon2['m2_id'] = range(1, model_polygon2.shape[0]+1)
            model_polygon2['m2_area'] = model_polygon2['geometry'].area/10**6


            overlay_polygon = gpd.overlay(model_polygon, topo_polygon, how='intersection', keep_geom_type=False)
            overlay_polygon['overlap_area'] = overlay_polygon['geometry'].area/10**6

            overlay_polygon2 = gpd.overlay(model_polygon, color_polygon, how='intersection', keep_geom_type=False)
            overlay_polygon2['overlap_area'] = overlay_polygon2['geometry'].area/10**6

            #print(overlay_polygon)
            #overlay_polygon.to_file('Intermediate/External/model_topo_polygon_for_'+str(image_stamp)+str(targeted_time)+'_r'+str(targeted_region)+'.gpkg', layer='polygon', driver='GPKG')

            #print(overlay_polygon2)
            #overlay_polygon2.to_file('Intermediate/External/model_color_polygon_for_'+str(image_stamp)+str(targeted_time)+'_r'+str(targeted_region)+'.gpkg', layer='polygon', driver='GPKG')
            

            processed_polygon = gpd.GeoDataFrame.copy(model_polygon)
            dropped_polygon = gpd.GeoDataFrame(columns=['image', 'region_num', 'geometry', 'm_id', 'm_area',], crs=processed_polygon.crs) #geometry='feature',

            ### check overlap on topo_polygon to merge multiple model_polygons into one topo_polygon
            for tid in range(1, topo_polygon.shape[0]+1):
                this_tid = overlay_polygon[(overlay_polygon['t_id']==tid)]

                ### For each topo-based polygon, only keep the model-based polygon with the largest overlapping area with color-based polygons
                if this_tid.shape[0] > 1:
                    overlapped_mid = this_tid['m_id'].values.tolist()
                    
                    max_value = -1
                    mid_to_keep = -1
                    for selected_mid in overlapped_mid:
                        overlapping_area = sum(overlay_polygon2[overlay_polygon2['m_id'] == selected_mid]['overlap_area'].values.tolist())
                        if overlapping_area > max_value:
                            max_value = overlapping_area
                            mid_to_keep = selected_mid
                    #print(max_value, mid_to_keep)
                    
                    for selected_mid in overlapped_mid:
                        if selected_mid != mid_to_keep:
                            ### If the dropped polygon has an overlap with color-based polygons, keep it for further revival
                            dropped_record = gpd.GeoDataFrame([{'image':target_stamp, 'region_num':targeted_region, 'geometry':overlay_polygon[(overlay_polygon['t_id']==tid) & (overlay_polygon['m_id']==selected_mid)]['geometry'].values[0], 'm_id':selected_mid, 'm_area':model_polygon[(model_polygon['m_id']==selected_mid)]['m_area'].values[0]}])
                            dropped_polygon = gpd.GeoDataFrame( pd.concat( [dropped_polygon, dropped_record], ignore_index=True), crs=processed_polygon.crs )

                            ### Still drop this model-based polygon from the preliminary result
                            processed_polygon.drop(processed_polygon[processed_polygon['m_id'] == selected_mid].index, inplace = True)


            ### Remove model-based polygons with no overlap to either color-based or topo-based polygons
            counting1 = overlay_polygon['m_id'].values.tolist()
            counting2 = overlay_polygon2['m_id'].values.tolist()

            counting1 = np.array(counting1)
            counting2 = np.array(counting2)
            counting3 = np.concatenate((counting1, counting2), axis=0)

            for selected_mid in range(1, model_polygon.shape[0]+1):
                if selected_mid not in counting3:
                    processed_polygon.drop(processed_polygon[processed_polygon['m_id'] == selected_mid].index, inplace = True)


            ### Revive dropped model-based polygons based on its ratio of overlapping area with color-based polygons
            candidate_dropped_cid = []
            dropped_mid = dropped_polygon['m_id'].values.tolist()
            threshold_area = 0.2

            for selected_mid in dropped_mid:
                candidate_dropped_cid.append(overlay_polygon2[overlay_polygon2['m_id'] == selected_mid]['c_id'].values.tolist())
                #print(overlapped_color_polygon)

                overlapped_color_polygon = overlay_polygon2[(overlay_polygon2['m_id'] == selected_mid) & (overlay_polygon2['m_area'] > (0.5+threshold_area)*overlay_polygon2['c_area']) & (overlay_polygon2['m_area'] < (1.6-threshold_area)*overlay_polygon2['c_area'])]
                #print(overlapped_color_polygon)
                
                if overlapped_color_polygon.shape[0] > 0:
                    ### Revive this dropped model-based polygon
                    revived_record = gpd.GeoDataFrame([{'image':target_stamp, 'region_num':targeted_region, 'geometry':dropped_polygon[(dropped_polygon['m_id']==selected_mid)]['geometry'].values[0], 'm_id':selected_mid, 'm_area':model_polygon[(model_polygon['m_id']==selected_mid)]['m_area'].values[0]}])
                    processed_polygon = gpd.GeoDataFrame( pd.concat( [processed_polygon, revived_record], ignore_index=True), crs=processed_polygon.crs )

            #processed_polygon.to_file('Intermediate/Output/postprocessed_polygons_'+str(image_stamp)+str(targeted_time)+'_r'+str(targeted_region)+'(6).gpkg', layer='polygon', driver='GPKG')


            ### Apply deeplab-based polygon with relative-area criteria
            overlay_polygon3 = gpd.overlay(model_polygon2, topo_polygon, how='intersection', keep_geom_type=False)
            overlay_polygon3['overlap_area'] = overlay_polygon3['geometry'].area/10**6

            overlay_polygon4 = gpd.overlay(model_polygon2, color_polygon, how='intersection', keep_geom_type=False)
            overlay_polygon4['overlap_area'] = overlay_polygon4['geometry'].area/10**6

            overlay_polygon5 = gpd.overlay(model_polygon2, model_polygon, how='intersection', keep_geom_type=False)
            overlay_polygon5['overlap_area'] = overlay_polygon5['geometry'].area/10**6

            threshold_area = 0.2

            for selected_mid in range(1, model_polygon2.shape[0]+1):
                overlapped_topo_polygon = overlay_polygon3[(overlay_polygon3['m2_id'] == selected_mid) & (overlay_polygon3['m2_area'] > (0.5+threshold_area)*overlay_polygon3['t_area']) & (overlay_polygon3['m2_area'] < (1.6-threshold_area)*overlay_polygon3['t_area'])]
                overlapped_color_polygon = overlay_polygon4[(overlay_polygon4['m2_id'] == selected_mid) & (overlay_polygon4['m2_area'] > (0.5+threshold_area)*overlay_polygon4['c_area']) & (overlay_polygon4['m2_area'] < (1.6-threshold_area)*overlay_polygon4['c_area'])]

                overlapped_model_polygon = overlay_polygon5[(overlay_polygon5['m2_id'] == selected_mid)]

                if overlapped_model_polygon.shape[0] == 0 and (overlapped_topo_polygon.shape[0] > 0 or overlapped_color_polygon.shape[0] > 0):
                    ### Add this model-based polygon
                    revived_record = gpd.GeoDataFrame([{'image':target_stamp, 'region_num':targeted_region, 'geometry':model_polygon2[(model_polygon2['m2_id']==selected_mid)]['geometry'].values[0], 'm2_id':selected_mid, 'm2_area':model_polygon2[(model_polygon2['m2_id']==selected_mid)]['m2_area'].values[0]}])
                    processed_polygon = gpd.GeoDataFrame( pd.concat( [processed_polygon, revived_record], ignore_index=True), crs=processed_polygon.crs )


            processed_polygon.to_file('Intermediate/Output/postprocessed_polygons_'+str(image_stamp)+str(targeted_time)+'_r'+str(targeted_region)+'.gpkg', layer='polygon', driver='GPKG')

            print(giscup_evaluation_f1_per_map(targeted_time, targeted_region, 'Source/Authority/lake_polygons_training.gpkg' , model_gpkg_2))
            print(giscup_evaluation_f1_per_map(targeted_time, targeted_region, 'Source/Authority/lake_polygons_training.gpkg' , model_gpkg))
            print(giscup_evaluation_f1_per_map(targeted_time, targeted_region, 'Source/Authority/lake_polygons_training.gpkg' , 'Intermediate/Output/postprocessed_polygons_'+str(image_stamp)+str(targeted_time)+'_r'+str(targeted_region)+'.gpkg'))


            processed_polygon = processed_polygon.drop('m_id', axis=1)
            processed_polygon = processed_polygon.drop('m_area', axis=1)

            if not os.path.isfile(output_gpkg):
                processed_polygon.to_file(output_gpkg, driver="GPKG")
            else:
                original_submission = gpd.read_file(output_gpkg)
                submission = original_submission.copy()
                submission = gpd.GeoDataFrame(pd.concat([submission, processed_polygon], ignore_index=True))
                submission.to_file(output_gpkg, driver="GPKG")


def sweeper():
    dir_name = 'Intermediate'
    for intermediate_file in os.listdir(dir_name):
        if intermediate_file.endswith('.tif'):
            print(intermediate_file)
            os.remove(os.path.join(dir_name, intermediate_file))

    dir_name = 'Intermediate/Postprocessing_Output'
    for intermediate_file in os.listdir(dir_name):
        if intermediate_file.endswith('.tif'):
            print(intermediate_file)
            os.remove(os.path.join(dir_name, intermediate_file))
        if intermediate_file.endswith('.png'):
            print(intermediate_file)
            os.remove(os.path.join(dir_name, intermediate_file))

    dir_name = 'Intermediate/External'
    for intermediate_file in os.listdir(dir_name):
        if intermediate_file.endswith('.tif'):
            print(intermediate_file)
            os.remove(os.path.join(dir_name, intermediate_file))
    
    if os.path.exists('Intermediate/Extraction'):
        shutil.rmtree('Intermediate/Extraction')



def preparing_for_postprocessing(path_to_source, path_to_topo):
    runningtime_start = datetime.now()
    # some preprocessing for color-based extraction
    initialization_dirs()
    image_shape = preprocessing_delineation(path_to_source)
    region_extent, region_projection = preprocessing_boundary(path_to_source, image_shape)
    preprocessing_visualization(path_to_source, image_shape, region_extent, region_projection)
    print('Finish preprocessing for color-based extraction......', datetime.now()-runningtime_start)

    # gpkg for color-based polygon
    color_based_extraction()
    print('Finish conducting color-based extraction......', datetime.now()-runningtime_start)
    png_to_gpkg()

    # gpkg for topo-based polygon
    topo_to_gpkg(path_to_source, path_to_topo)
    print('Finish preparing for postprocessing......', datetime.now()-runningtime_start)

    sweeper()



def postprocessing_with_external(path_to_source, path_to_topo, path_to_model_sam, path_to_model_lab, output_gpkg):

    preparing_for_postprocessing(path_to_source, path_to_topo)
    sweeper()
    postprocessing(path_to_model_sam, path_to_model_lab, output_gpkg)


