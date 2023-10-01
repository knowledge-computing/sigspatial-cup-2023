import os
import glob
import argparse
import pandas as pd
import numpy as np
import json
from PIL import Image
import sys
import cv2

from shapely import Polygon, make_valid, is_valid
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

Image.MAX_IMAGE_PIXELS=None # allow reading huge images

def plot_N(images):
    fig = plt.figure(figsize=(32, 10))
    gs = GridSpec(nrows=1, ncols=len(images))
    gs.update( hspace = 0.5, wspace = 0.05)

    for i in range(len(images)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(images[i], vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()  

def extract_hw_index(file_name):
    h, w = os.path.basename(file_name).split('__')[1][:-4].split('_')
    hdx, wdx = int(h[1:]), int(w[1:])
    return hdx, wdx

def count_num_pixels(poly, size):
    blank = np.zeros(size, np.uint8)
    blank = cv2.drawContours(blank, [poly.astype(int)], -1, 255, thickness=-1)
    return np.count_nonzero(blank)

def seg2poly(seg):
    out_contours = []
    mask = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0,i,3] == -1:
            area = count_num_pixels(contours[i], seg.shape)
            # cv2.contourArea(contours[i].astype(int))
            if area < 68.5: continue;
            out_contours.append(contours[i])
    
    def retrieve_polygons(polygon):        
        try:
            pdet = Polygon(polygon)
            if not pdet.is_valid:
                pdet = pdet.buffer(0)
                if pdet.geom_type == 'MultiPolygon':
                    pdet = list(pdet.geoms)
                else:
                    pdet = [pdet]
                return [np.array(p.exterior.coords[:]).astype(int) for p in pdet]
            else:
                return [polygon]
        except Exception as e:
            print(e)
            return None

    valid_contours = []
    for contour in out_contours:
        contour = contour.reshape(-1, 2).astype(int)
        contour = retrieve_polygons(contour)
        if contour is not None:
            for c in contour:
                area = count_num_pixels(c, seg.shape)
                if area < 68.5: continue;
                valid_contours.append(c)    
    return valid_contours


def is_narrow_stream(poly):
    (x, y), (w, h), angle_of_rotation = cv2.minAreaRect(poly.astype(int))
    if min(w, h) < max(w, h) * 0.1: return True;
    else: return False;


def filter_narrow_streams(polys):
    out_polys = []
    for poly in polys:
        if is_narrow_stream(poly): continue;
        else: out_polys.append(poly);
    return out_polys


def split_poly_by_narrow_streams(poly, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    min_x, min_y = int(np.min(poly[:, 0])), int(np.min(poly[:, 1]))
    max_x, max_y = int(np.max(poly[:, 0])), int(np.max(poly[:, 1]))
    image_size = (max_y-min_y, max_x-min_x)
    poly_tmp = poly.copy()
    poly_tmp[:, 0] -= min_x
    poly_tmp[:, 1] -= min_y
    poly_tmp = poly_tmp.astype(int)
    canvas = np.zeros(image_size, np.uint8)
    canvas = cv2.drawContours(canvas, [poly_tmp], -1, 255, thickness=-1)
    canvas_2 = canvas.copy()
    canvas_2 = cv2.erode(canvas_2, kernel, iterations=1)
    canvas_2 = cv2.dilate(canvas_2, kernel, iterations=1)
    poly_tmp_2 = seg2poly(canvas_2)
    if len(poly_tmp_2) <= 1:
        return [poly]
    else:
        remove_polys = []
        delta_polys = seg2poly(canvas - canvas_2)
        for p in delta_polys:
            if is_narrow_stream(p):
                remove_polys.append(p)

        canvas = np.zeros(image_size, np.uint8)
        canvas = cv2.drawContours(canvas, [poly_tmp], -1, 255, thickness=-1)
        canvas = cv2.drawContours(canvas, remove_polys, -1, 0, thickness=-1)          
        polys_tmp = seg2poly(canvas)
        out_polys = []
        for p in polys_tmp:
            p[:, 0] += min_x
            p[:, 1] += min_y
            out_polys.append(p)
        return out_polys


def split_polys_by_narrow_streams(polys, kernel_size=3, image_size=(1024, 1024)):
    remove_polys = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for poly in polys:
        canvas = np.zeros(image_size, np.uint8)
        canvas = cv2.drawContours(canvas, [poly], -1, 255, thickness=-1)
        canvas_2 = canvas.copy()
        canvas_2 = cv2.erode(canvas_2, kernel, iterations=1)
        canvas_2 = cv2.dilate(canvas_2, kernel, iterations=1)
        poly_2 = seg2poly(canvas_2)
        if len(poly_2) == 1:
            continue
        else:
            delta_polys = seg2poly(canvas - canvas_2)
            for p in delta_polys:
                if is_narrow_stream(p):
                    remove_polys.append(p)
            
    blank = np.zeros(image_size, np.uint8)
    blank = cv2.drawContours(blank, polys, -1, 255, thickness=-1)          
    blank = cv2.drawContours(blank, remove_polys, -1, 0, thickness=-1)          
    return seg2poly(blank)

def extract_polys(seg, image_size):
    polys = seg2poly(seg)
    if len(polys) == 0: return [];
    filtered_polys = split_polys_by_narrow_streams(polys, image_size=image_size)
    if len(filtered_polys) == 0: return [];
    return polys

def convert_coordinates(polys, hdx, wdx, shift_size):
    # convert patch space to larger image space
    output_polys = []
    for poly in polys:
        bbox_x_min = wdx * shift_size
        bbox_y_min = hdx * shift_size
        new_poly = 1. * poly.copy()
        new_poly[:, 0] += bbox_x_min
        new_poly[:, 1] += bbox_y_min
        output_polys.append(new_poly)
    return output_polys     

    
def remove_duplicates(polys):
    area_sizes = dict()
    for i, poly in enumerate(polys):
        area_sizes[i] = cv2.contourArea(poly.astype(int))
    area_sizes = sorted(area_sizes.items(), key=lambda x: -x[1])
    all_poly_ids = [i[0] for i in area_sizes]
    
    out_polys = []
    used_ids = []
    for curr_id in all_poly_ids:
        if curr_id in used_ids: continue;
        curr_poly = polys[curr_id]
        unused_ids = [i for i in all_poly_ids if i not in used_ids and i != curr_id]
        for i in unused_ids:
            p1 = Polygon(curr_poly)
            p2 = Polygon(polys[i])
            if p1.intersects(p2):
                try:
                    p = p1.union(p2)
                    xx, yy = p.exterior.coords.xy
                    curr_poly = np.stack([xx, yy], axis=1)
                except Exception as e:
                    # min_w = int(min(np.min(curr_poly[:,0]), np.min(polys[i][:,0]))) - 1
                    # min_h = int(min(np.min(curr_poly[:,1]), np.min(polys[i][:,1]))) - 1
                    # max_w = int(max(np.max(curr_poly[:,0]), np.min(polys[i][:,0]))) + 1
                    # max_h = int(max(np.max(curr_poly[:,1]), np.max(polys[i][:,1]))) + 1
                    # blank = np.zeros((max_h - min_h, max_w - min_w), np.uint8)
                    # tmp_p1 = curr_poly.copy()
                    # tmp_p1[:, 0] -= min_w
                    # tmp_p1[:, 1] -= min_h
                    # tmp_p2 = polys[i].copy()
                    # tmp_p2[:, 0] -= min_w
                    # tmp_p2[:, 1] -= min_h
                    # # print(blank.shape)
                    # p = [i.astype(int) for i in [tmp_p1, tmp_p2]]
                    # blank = cv2.drawContours(blank, p, -1, 255, thickness=-1)
                    # contours = seg2poly(blank)[0]
                    pass
                    
                used_ids.append(i)
            else:    
                continue
    
        used_ids.append(curr_id)
        out_polys.append(curr_poly)
    return out_polys     
        

def extract_polys_main(seg_files, crop_size, shift_size, rock_mask_path=None):
    polys_dict = dict()
    for seg_file in seg_files:
        seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        seg = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)[1]
        if rock_mask_path is not None:
            mask = cv2.imread(os.path.join(rock_mask_path, os.path.basename(seg_file)), 
                              cv2.IMREAD_GRAYSCALE)
            mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]
            seg = seg * mask
        hdx, wdx = extract_hw_index(seg_file)
        polys = extract_polys(seg, image_size=(crop_size, crop_size))
        polys = convert_coordinates(polys, hdx, wdx, shift_size)
        polys_dict[seg_file] = polys
    print("Done - Extract polys from segmentations.")
    return polys_dict


def merge(test_region, data_path, crop_size, shift_size, rock_mask_path=None):
    out_polys, overlap_polys = [], []
    seg_files = glob.glob(os.path.join(data_path, f'{test_region}__*.jpg'))
    polys_dict = extract_polys_main(seg_files, crop_size, shift_size, rock_mask_path=rock_mask_path)

    count = 0
    for k, v in polys_dict.items():
        count += len(v)
    print('# polygons before merging', count)

    if count < 500:
        for k, v in polys_dict.items():
            out_polys += v
        out_polys = remove_duplicates(out_polys)

    else:
        for seg_file in seg_files:  

            hdx, wdx = extract_hw_index(seg_file)
            if hdx % 2 > 0 or wdx % 2 > 0: continue;
    
            cur_polys = polys_dict[seg_file]
            # if len(cur_polys) < 1: continue;

            bbox = np.array([[0,0],[0,crop_size],[crop_size,crop_size],[crop_size,0]])
            bbox = convert_coordinates([bbox], hdx, wdx, shift_size)[0]
            bbox_polygon = Polygon(bbox)
    
            # find the neighboring polygons that are overlapped with current bbox
            neighbor_polys = []
            for i in [hdx - 1, hdx, hdx + 1]:
                for j in [wdx - 1, wdx, wdx + 1]:
                    path = os.path.join(data_path, f'{test_region}__h{i}_w{j}.jpg')
                    if os.path.exists(path) and path != seg_file:
                        for poly in polys_dict[path]:
                            polygon = Polygon(poly)
                            if bbox_polygon.intersects(polygon):
                                neighbor_polys.append(poly)
            neighbor_polys = remove_duplicates(neighbor_polys)
            
            # find the neighboring polygons that are overlapped with current polys
            cur_polys_dup_ids, neighbor_polys_dup_ids = [], []
            for i, p1 in enumerate(cur_polys):
                for j, p2 in enumerate(neighbor_polys):
                    p1, p2 = Polygon(p1), Polygon(p2)
                    if p1.intersects(p2):
                        cur_polys_dup_ids.append(i)
                        neighbor_polys_dup_ids.append(j)
    
            cur_polys_dup_ids = sorted(list(set(cur_polys_dup_ids)))
            neighbor_polys_dup_ids = sorted(list(set(neighbor_polys_dup_ids)))
            dup_polys = [cur_polys[ix] for ix in cur_polys_dup_ids]
            dup_polys += [neighbor_polys[ix] for ix in neighbor_polys_dup_ids]
            nondup_polys = [cur_polys[ix] for ix in range(len(cur_polys)) if ix not in cur_polys_dup_ids]
            nondup_polys += [neighbor_polys[ix] for ix in range(len(neighbor_polys)) if ix not in neighbor_polys_dup_ids]
            
            # find the polygons at the edge that potentially merge with other patches
            for poly in remove_duplicates(dup_polys) + nondup_polys:
                min_x, min_y = np.min(poly[:, 0]), np.min(poly[:, 1])
                max_x, max_y = np.max(poly[:, 0]), np.max(poly[:, 1])
                if min_x <= bbox[0, 0] or min_y <= bbox[0, 1] or max_x >= bbox[2, 0] or max_y >= bbox[2, 1]:
                    overlap_polys.append(poly)
                else:
                    out_polys.append(poly)
    
        out_polys += remove_duplicates(overlap_polys)

    print('# polygons after merging', len(out_polys))
    print('Done - Merge polys.')
    return out_polys



def load_manual_mask(mask_path):
    if not os.path.exists(mask_path):
        return None
        
    data = pd.read_csv(mask_path)
    def parser(anc):
        pts = anc.split('points=')[1][1:-11]
        pts = [(float(pt.split(',')[0]), float(pt.split(',')[1])) for pt in pts.split(' ')]
        pts = np.array(pts).reshape(-1, 2)
        return pts
    
    masks = {}
    for i, row in data.iterrows():
        file, anchor = row['FILE'][:-4], row['ANCHOR']
        region_num = file[-1]
        if masks.get(region_num) is None:
            masks[region_num] = []
        anchor = parser(anchor)
        masks[region_num].append(anchor)

    return masks


def filter_by_manual_mask(polys, masks):
    out_polys = []
    for poly in polys:
        flag = True
        for mask in masks:
            p1, p2 = Polygon(poly), Polygon(mask)
            if p1.intersects(p2): flag = False;
        if flag:
            out_polys.append(poly)
    return out_polys

    
    

    
