# sigspatial-cup-2023
The code and result for 12th SIGSPATIAL Cup competition (GISCUP 2023) [https://sigspatial2023.sigspatial.org/giscup/index.html](https://sigspatial2023.sigspatial.org/giscup/index.html)

The main task is to identify supraglacial lakes on the Greenland ice sheet from satellite imagery. 
Our team proposes an approach to detecting surface lakes by leveraging potential spatial data (e.g., topographic sinks, soil) and automatically generating synthetic training sets. 

## Data Preprocessing
**Directory** `./data_preprocess/`

**How to run**

- To produce the images for each region (i.e., crop the original tif into 6 regions), run
`python extract_regions.py --data_root [DATA_ROOT] --output_dir [OUTPUT_DIR] --tif_file [TIF_FILE]`

    - DATA_ROOT: data directory path containing the competition data, e.g., ../dataset/2023_SIGSPATIAL_Cup_data_files
    - OUTPUT_DIR: output directory path, default ../dataset/region_images
    - TIF_FILE: the TIF file name you want to process, e.g., Greenland26X_22W_Sentinel2_2019-08-25_29.tif

- To crop the region images to image patches for training machine learning models, run
`python crop_regions.py --data_root [DATA_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`

    - DATA_ROOT: data directory path, e.g., ../dataset/
    - CROP_SIZE: cropped image size, default 1024
    - SHIFT_SIZE: shift size, default 512

- To convert the ground truth polygons in the gpkg files to image patch level polygons, run
`python construct_train_annotation.py --data_root [DATA_ROOT] --output_path [OUTPUT_PATH] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`
    
    - DATA_ROOT: data directory path, e.g., ../dataset/
    - OUTPUT_PATH: output directory path, e.g., ../dataset/train_crop1024_shift512
    - CROP_SIZE: cropped image size, default 1024
    - SHIFT_SIZE: shift size, default 1024 
    - You need [annotation.json](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link) under DATA_ROOT

- To generate the segmentation masks for training, run
`python construct_segmentation_masks.py --data_path [DATA_ROOT] --crop_size [CROP_SIZE]
    - DATA_PATH: data directory path, e.g., ../dataset/train_crop1024_shift512
    - CROP_SIZE: cropped image size, default 1024

- Generate training samples, run
`python generate_train_set.py --data_root [DATA_ROOT]
  
    - DATA_PATH: data directory path, e.g., ../dataset/train_crop1024_shift512
    - You need [hard_neg_images.json](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link), [hard_neg_images.txt](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link), and [invalid_image.txt](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link) under DATA_PATH

- Generate training and testing samples using K-fold, run
`python generate_train_test_set.py --data_root [DATA_ROOT]
  
    - DATA_PATH: data directory path, e.g., ../dataset/train_crop1024_shift512
    - You need [hard_neg_images.json](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link), [hard_neg_images.txt](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link), and [invalid_image.txt](https://drive.google.com/file/d/188ThxYEoLCgZ8kZb6QMVfCkY-VHPR9x1/view?usp=drive_link) under DATA_PATH

  
## Image Segmentation Models 
### Segment Anything Model
**Directory** `./models/SAM/`

**Description** 
We fine-tuned Facebook's [Segment Anything Model (SAM)](https://segment-anything.com/) on the glacier training data provided to us. We formulate the problem as a pixel-level binary-class classification problem, where supraglacial lakes should be predicted as 1 and backgrounds should be predicted as 0. Since the ratio of lake and non-lake regions in the training data are skewed, we consider the positive (lake) and negative (non-lake) samples' ratio in the training samples (Weighted Random Sampler).

**Environment Setup** 

The model is trained with python 3.11 and CUDA 11.4.
- To install the environment, `pip install -r environment.yml`

**How to run**

- To train model, run `train.py --epoch [NUM_EPOCH] --batch [NUM_BATCH]`
    - NUM_EPOCH: The number of epoch to train the model
    - NUM_BATCH: The number of batch to train the model
- To test model, run `test.py --region [REGION_NAME] --epoch [BEST_EPOCH]`
    - REGION_NAME: The region name where to generate the test prediction mask
    - BEST_EPOCH: The model from the (best) epoch from the training
  
- Please refer to different training strategies (e.g., validation, 50% ratio positive/negative sampling) on [https://github.com/zekun-li/supraglacial_lake](https://github.com/zekun-li/supraglacial_lake)

### DeepLabv3Plus<br><br>
**Directory** `./models/DeepLabv3Plus/`<br><br>
**Description** <br>
We fine-tuned [DeepLabv3+](https://github.com/giovanniguidi/deeplabV3-PyTorch) on the glacier training data. The model applied Weighted Random Sampler to address the data imbalance between pixels with Lake and pixels with Non-Lake. The input mask consists of 0 (Non-Lake) and 1(Lake). <br><br>

**Environment Setup** <br>
The model is trained with python 3.8 and CUDA 11.3. We recommend to run in a conda environment. <br>
`conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge` <br><br>

**How to run** <br>
- Configuration detail (./configs/config.yml) <br>
    - ` config['dataset']['base_path'] ` : the directory path of entire images and masks (train and test). Please make sure that the images are under  `train_images ` and the masks are under  `train_mask`  <br>
    - `config['dataset']['region_txt_base_path']` : the directory path of txt files that contains the list of Positive train set, Negative train set and Test set. Please make sure the name of each txt file : `train_pos.txt`,`train_neg.txt`,`test.txt` <br>
    - `config['dataset']['save_res_path']`: the directory path to save the prediction results <br><br>

- To install the environment, `pip install -r requirements.txt ` <br>
- To train the model, `python main.py -c configs/config.yml --train`<br>
- To test the model with the test data, `python main.py -c configs/config.yml --predict_on_test`<br>
- To predict the model with images `python main.py -c configs/config.yml --predict --filefolder IMAGES_DIR_PATH`<br>

## External Data Resources
**Directory** `./external_datasets/`

### Topographic Sink
**Description**
We generate the topographic sinks as one of the external data resources from the ArcticDEM. Supraglacial lakes are formed in surface sinks. Therefore, the topographic sinks are potential locations for supraglacial lakes. The process of generating the topographic sinks from ArcticDEM has two steps. 
First, we employ the open-source WhiteboxToolsTM library to fill the depressions in the ArcticDEM and eliminate flat areas. Second, we generate topographic sinks by subtracting the output of the first step from the original ArcticDEM. Locations, where the subtraction results yield values smaller than zero, represent the topographic sinks.

**How to use**
- TBA

### Soil
**Description**
We use [Northern Circumpolar Soil Carbon Database version 2 (NCSCDv2)](https://apgc.awi.de/dataset/ncscdv2-greenland-geotiff-netcdf), a geospatial database that records the amount of organic carbon storage in soils of the northern circumpolar permafrost region down to a depth of 300 cm. Since the dataset delimited the areas that are covered by glaciers for most times in a year, we use this dataset to identify the glacier area and exclude lakes that are not located on glaciers.

**How to use**
- Please download NCSCDv2_Greenland_WGS84_nonsoil_pct_0012deg.tif from [NCSCDv2](https://apgc.awi.de/dataset/ncscdv2-greenland-geotiff-netcdf) and place it in `./external_datasets/`. 

## Data Post-Processing
**Directory** `./data_postprocess/`

### Using External Data Resources
**Description**
First, we treat the union of topographic sink, color thresholding, and the inverse of soil allocation as lake candidate, and remove all the model-based polygons that are not in the lake candidate. 
We then compare the SAM-based results and topographic sink on a vector-wise basis. For each lake candidate, we only keep the SAM-based polygon that has the largest overlapping area with the color-thresholding polygon. 
Finally, we add model-based polygons (including DeepLab-based ones and SAM-based ones that were removed) that reach the relative-area criteria of the lake candidate.

**How to run**
- To generate a single GPKG file that integrates results from different models, using topo-based and color-based extraction as a reference to postprocess, run `python postprocessing_with_external.py --data_root [DATA_ROOT] --result_name [RESULT_NAME] --data_topo [DATA_TOPO] --data_soil [DATA_SOIL] --sam_dir [SAM_DIR] --dpl_dir [DPL_DIR]`
    - DATA_ROOT: data directory path for the provided dataset
    - RESULT_NAME: name of the output GPKG file
    - DATA_TOPO: directory path to topographic_sink.tif
    - DATA_SOIL: directory path to NCSCDv2_Greenland_WGS84_nonsoil_pct_0012deg.tif
    - SAM_DIR: directory path of the results from the SAM model
    - DPL_DIR: directory path of the results from the DeepLab model
    
[comment]: <> (### Without Using External Data Resources)
[comment]: <> (**How to run**)
[comment]: <> (- To generate output GPKG from the segmentation results, and do evaluation if the ground truth file exists, run `python run.py --data_root [DATA_ROOT] --result_root [RESULT_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE] --result_name [RESULT_NAME]`)
