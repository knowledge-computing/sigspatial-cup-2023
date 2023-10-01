# sigspatial-cup-2023
The code and result for 12th SIGSPATIAL Cup competition (GISCUP 2023) [https://sigspatial2023.sigspatial.org/giscup/index.html](https://sigspatial2023.sigspatial.org/giscup/index.html)

The main task is to identify supraglacial lakes on the Greenland ice sheet from satellite imagery. 
Our team proposes an approach to detecting surface lakes by leveraging potential spatial data (e.g., topographic sinks, soil) and automatically generating synthetic training sets. 

## Data Preprocessing
**Directory** `./data_preprocess/`

**How to run**

- To produce the images for each region (i.e., crop the original tif into 6 regions), run
`python extract_regions.py --data_root [DATA_ROOT] --output_dir [OUTPUT_DIR] --tif_file [TIF_FILE]`

- To crop the region images to image patches for training machine learning models, run
`python crop_regions --data_root [DATA_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`

- To convert the ground truth polygons in the gpkg files to image patch level, run
`python construct_train_annotation.py --data_root [DATA_ROOT] --output_dir [OUTPUT_DIR] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`

- Then convert the segmentation masks for training, run
`python construct_segmentation_masks.py --data_root [DATA_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`

- Split the image patches into training and testing sets, run
`python split_train_test.py --data_root [DATA_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`

## Image Segmentation Models 
### Segment Anything Model
**Directory** `./models/SAM`

**Description** 
We fine-tuned Facebook's [Segment Anything Model (SAM)](https://segment-anything.com/) on the glacier training data provided to us. We formulate the problem as a pixel-level binary-class classification problem, where supraglacial lakes should be predicted as 1 and backgrounds should be predicted as 0. Since the ratio of lake and non-lake regions in the training data are skewed, we consider the positive (lake) and negative (non-lake) samples' ratio in the training samples (Weighted Random Sampler).

**Environment Setup** 
The model is trained with python 3.11 and CUDA 11.4. To install the environment,

`pip install -r environment.yml`

**How to run**

- To train model, run `train.py`
- To test model, run `test.py`
  
- Please refer to different training strategies (e.g., validation, 50% ratio positive/negative sampling) on [https://github.com/zekun-li/supraglacial_lake](https://github.com/zekun-li/supraglacial_lake)

### DeepLabv3Plus
**Directory** `./models/DeepLabv3Plus`

**Environment Setup**
<br>
The model is trained with python 3.8 and CUDA 11.3. To install the environment,

`pip install -r requirements.txt `

**Description** <br>
We fine-tuned [DeepLabv3+](https://github.com/giovanniguidi/deeplabV3-PyTorch) on the glacier training data. The model applied Weighted Random Sampler to address the data imbalance between pixels with Lake and pixels with Non-Lake. The input mask consists of 0 (Non-Lake) and 1(Lake). 

## External Data Resources
### Topographic Sink
**Description**
We generate the topographic sinks as one of the external data resources from the ArcticDEM. Supraglacial lakes are formed in surface sinks. Therefore, the topographic sinks are potential locations for supraglacial lakes. The process of generating the topographic sinks from ArcticDEM has two steps. 
First, we employ the open-source WhiteboxToolsTM library to fill the depressions in the ArcticDEM and eliminate flat areas. Second, we generate topographic sinks by subtracting the output of the first step from the original ArcticDEM. Locations, where the subtraction results yield values smaller than zero, represent the topographic sinks.

### Soil
**Description**
We use the NCSCDv2_Greenland_WGS84_nonsoil_pct_0012deg.tif from [Northern Circumpolar Soil Carbon Database version 2 (NCSCDv2)](https://apgc.awi.de/dataset/ncscdv2-greenland-geotiff-netcdf). 
NCSCDv2 is a geospatial database that records the amount of organic carbon storage in soils of the northern circumpolar permafrost region down to a depth of 300 cm. Since the dataset delimited the areas that are covered by glaciers for most times in a year, we use this dataset to identify the glacier area and exclude lakes that are not located on glaciers.


## Data Post-Processing
**Directory** `./data_postprocess/`

### Using External Data Resources
**Description**
First, we treat the union of topographic sink, color thresholding, and the inverse of soil allocation as lake candidate, and remove all the model-based polygons that are not in the lake candidate. 
We then compare the SAM-based results and topographic sink on a vector-wise basis. For each lake candidate, we only keep the SAM-based polygon that has the largest overlapping area with the color-thresholding polygon. 
Finally, we add model-based polygons (including DeepLab-based ones and SAM-based ones that were removed) that reach the relative-area criteria of the lake candidate.

**How to run**
- To generate a single GPKG file that integrates results from different models, using topo-based and color-based extraction as a reference to postprocess, run `python postprocessing_with_external.py --data_root [DATA_ROOT] --result_name [RESULT_NAME] --data_topo [DATA_TOPO] --data_soil [DATA_SOIL] --sam_dir [SAM_DIR] --dpl_dir [DPL_DIR]`
    - DATA_ROOT: path to the directory where you store the provided dataset (e.g., lakes_regions.gpkg, lake_polygons_training.gpkg, and all the tif files)
    - RESULT_NAME: name of the output gpkg file.
    - DATA_TOPO: path to topographic_sink.tif
    - DATA_SOIL: path to NCSCDv2_Greenland_WGS84_nonsoil_pct_0012deg.tif
    - SAM_DIR: path to the directory where you store the results from the SAM model.
    - DPL_DIR: path to the directory where you store the results from the DeepLab model
    
### Without Using External Data Resources
**How to run**
- To generate output GPKG from the segmentation results, and do evaluation if the ground truth file exists, run `python run.py --data_root [DATA_ROOT] --result_root [RESULT_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE] --result_name [RESULT_NAME]`
