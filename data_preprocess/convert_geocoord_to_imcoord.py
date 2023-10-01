from PIL import Image
import os
import numpy as np
import json

import geopandas as gpd
import rasterio
from shapely.geometry import shape
from shapely import wkt
from shapely.geometry import Polygon

Image.MAX_IMAGE_PIXELS=None # allow reading huge images

# settings = iface.mapCanvas().mapSettings()
# layer = iface.activeLayer()
# ext = layer.extent()
# settings.setOutputSize(QSize(87040, 160000))
# settings.setExtent(ext)
# pgw_content = QgsMapSettingsUtils.worldFileContent(settings)
# with open(pgw_output_file, "w") as f:
#     f.write(pgw_content)

with open('transform.pgw') as file: # load transformation
    a, d, b, e, c, f = np.loadtxt(file)
    transform = rasterio.Affine(a, b, c, d, e, f)

def transform_image_coord(geometry):
    poly = np.array(list(geometry.exterior.coords))
    transformed_pts = np.apply_along_axis(lambda x: rasterio.transform.rowcol(transform, x[0], x[1]), axis=1, arr=poly)
    transformed_pts = transformed_pts[:, [1, 0]]
    return np.array(transformed_pts)

data_root = '../dataset/2023_SIGSPATIAL_Cup_data_files'
df = gpd.read_file(os.path.join(data_root, 'lakes_regions.gpkg'))
df['image_coord'] = df['geometry'].apply(lambda x: transform_image_coord(x))
df['min_x'] = df['image_coord'].apply(lambda x: int(np.min(x[:, 0]) / 1024))
df['min_y'] = df['image_coord'].apply(lambda x: int(np.min(x[:, 1]) / 1024))
df['max_x'] = df['image_coord'].apply(lambda x: int(np.max(x[:, 0]) / 1024) + 1)
df['max_y'] = df['image_coord'].apply(lambda x: int(np.max(x[:, 1]) / 1024) + 1)
df = df[['region_num', 'min_x', 'min_y', 'max_x', 'max_y']]

poly_df = gpd.read_file(os.path.join(data_root, 'lake_polygons_training.gpkg'))
poly_df['image_coord'] = poly_df['geometry'].apply(lambda x: transform_image_coord(x))
poly_df = poly_df.join(df.set_index('region_num'), on='region_num')
# poly_df

out_json = []
for i, row in poly_df.iterrows():
    image = f"{row['image'][:-4]}_r{row['region_num']}.png"
    image_coord = row['image_coord'].copy()
    image_coord[:, 0] -= row['min_x'] * 1024
    image_coord[:, 1] -= row['min_y'] * 1024
    image_coord[image_coord < 0] = 0
    annotation = {
        "region_image_file": image,
        "region_num": row['region_num'],
        "poly": image_coord.tolist()}
    out_json.append(annotation)

with open("../dataset/annotations.json", 'w') as f:
    json.dump(out_json, f)



# the following script is for visualization
# import math
# import cv2
# import numpy as np
# from PIL import Image
# from matplotlib.gridspec import GridSpec
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# def plot_N(images):
#     fig = plt.figure(figsize=(32, 10))
#     gs = GridSpec(nrows=1, ncols=len(images))
#     gs.update( hspace = 0.5, wspace = 0.05)

#     for i in range(len(images)):
#         ax = fig.add_subplot(gs[0, i])
#         im = ax.imshow(images[i], vmin=0, vmax=255)
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()  


# image = "dataset/Greenland26X_22W_Sentinel2_2019-08-25_29_r1.jpg"
# img = cv2.imread(image)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# for item in items:
#     item = np.array(item).reshape(-1, 2).astype(int)
#     cv2.polylines(img, [item], True, color=(255, 0, 0), thickness=2)   
# plot_N([img])

