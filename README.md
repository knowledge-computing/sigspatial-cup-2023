# ACM SIGSPATIAL Cup 2023
The code and result for 12th SIGSPATIAL Cup (GISCUP 2023) [https://sigspatial2023.sigspatial.org/giscup/index.html](https://sigspatial2023.sigspatial.org/giscup/index.html)

The goal is to detect supraglacial lakes on the Greenland ice sheet from satellite imagery. Our team proposes an ensembled approach that leverages two machine learning models, computer vision techniques, and external data (e.g., topographic sinks, soil) to automatically detect surface lakes.

## Results for Submission
Link to the submission GPKG: [https://github.com/knowledge-computing/sigspatial-cup-2023/tree/main/GPKG/lake_poygons_test.gpkg](https://github.com/knowledge-computing/sigspatial-cup-2023/tree/main/GPKG/lake_poygons_test.gpkg)

The following content is organized as follows:
1. <a href="#1-data-preprocessing">Data Preprocessing</a>
2. <a href="#2-image-segmentation-models">Image Segmentation Models</a>
    * 2.1. <a href="#21-segment-anything-model">Segment Anything Model</a>
    * 2.2. <a href="#22-deeplabv3">DeepLabv3+</a>
3. <a href="#3-external-data-resources">External Data Resources</a>
    * 3.1. <a href="#31-topographic-sink">Topographic Sink</a>
    * 3.2. <a href="#32-soil">Soil Data</a>
4. <a href="#4-data-postprocessing">Data Postprocessing</a>

---

## 1. Data Preprocessing
**Directory** `./data_preprocess/`

The data preprocssing creates a training dataset from the provided competition data for machine learnning models. The generated datasets include [region_images](https://drive.google.com/drive/folders/1tP-hur0vZkY_7WkilohA0zXQkk0qpHum?usp=drive_link), [training set](https://drive.google.com/drive/folders/1-3Ibl_DvAdwxMAIj3uwjMYr0JePSdWfF?usp=drive_link), [inference set](https://drive.google.com/file/d/1-MwE2wfwRkB7JJ3chsJdlghAcJ0VbiXC/view?usp=drive_link). You can download these files to avoid the following steps and directly go to [Image Segmentation Models](#2-image-segmentation-models).

**How to run**

- To produce the images for each region (i.e., crop the original tif into 6 regions), run
`python extract_regions.py --data_root [DATA_ROOT] --output_path [OUTPUT_PATH] --tif_file [TIF_FILE]`

    - DATA_ROOT: data directory of the competition data
    - OUTPUT_DIR: output directory, default: ../dataset/region_images
    - TIF_FILE: the TIF file name you want to process, e.g., Greenland26X_22W_Sentinel2_2019-08-25_29.tif

- To crop the region images to image patches, run
`python crop_regions.py --data_root [DATA_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`

    - DATA_ROOT: data root of the region images, e.g., ../dataset/
    - CROP_SIZE: cropped image size, default 1024
    - SHIFT_SIZE: shift size, default 512

- To convert the ground truth polygons in the gpkg file to image patch segmentation masks, run
`python construct_train_annotation.py --data_root [DATA_ROOT] --output_path [OUTPUT_PATH] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE]`
    
    - DATA_ROOT: data directory of the competition data
    - OUTPUT_PATH: output path, default: e.g., ../dataset/train_crop1024_shift512
    - CROP_SIZE: cropped image size, default 1024
    - SHIFT_SIZE: shift size, default 1024 

- To generate training samples, run
`python generate_train_set.py --data_root [DATA_ROOT]
  
    - DATA_PATH: data directory path, e.g., ../dataset/train_crop1024_shift512
    - You need [hard_neg_images.json](https://drive.google.com/file/d/13UDQGBR-KEjZ6sIOxSZwqKog6cjwuSGd/view?usp=drive_link), [hard_neg_images.txt](https://drive.google.com/file/d/1emtgoMzDUkeFf2RAOzmHjeSO7uX5zZ9e/view?usp=drive_link), and [invalid_image.txt](https://drive.google.com/file/d/19XkGd3ExaY5r9olrQakA311QMauGah-8/view?usp=drive_link) under DATA_PATH

- To generate K-fold training and testing samples, run
`python generate_train_test_set.py --data_root [DATA_ROOT]
  
    - DATA_PATH: data directory path, e.g., ../dataset/train_crop1024_shift512
    - You need [hard_neg_images.json](https://drive.google.com/file/d/13UDQGBR-KEjZ6sIOxSZwqKog6cjwuSGd/view?usp=drive_link), [hard_neg_images.txt](https://drive.google.com/file/d/1emtgoMzDUkeFf2RAOzmHjeSO7uX5zZ9e/view?usp=drive_link), and [invalid_image.txt](https://drive.google.com/file/d/19XkGd3ExaY5r9olrQakA311QMauGah-8/view?usp=drive_link) under DATA_PATH <br>


---
  
## 2. Image Segmentation Models 

### Description
We formulate the problem as a pixel-level binary-class classification problem, where supraglacial lakes should be predicted as 1 and backgrounds as 0. We use the provided lake labels to construct a training set of positive samples (containing lakes), negative samples, and hard negative samples (no lake but having high variability in image color or some detection failures). Due to imbalanced positive and negative samples, we use the weighted random sampler to select balanced positive and negative samples in each training batch. We fine-tuned two machine learning models for the task: Facebook's [Segment Anything Model (SAM)](https://segment-anything.com/) and [DeepLabv3+](https://github.com/giovanniguidi/deeplabV3-PyTorch).

### 2.1. Segment Anything Model
**Directory** `./models/SAM/`
<!-- - We fine-tuned Facebook's [Segment Anything Model (SAM)](https://segment-anything.com/) on the glacier training data provided to us.  -->

**Model Weights** You can download the model weights from [HERE](https://drive.google.com/file/d/1r6O1gCmeIz54xD7BZWTrrDWzo7Vqjedd/view).

**Environment Setup** 
- The model is trained with python 3.11 and CUDA 11.3
- To install the environment, `pip install -r environment.yml`

**How to run**

- To train, run `train.py --epoch [EPOCH] --batch [BATCH] --lr [LR] --img_dir [IMG_DIR] --mask_dir [MASK_DIR] --postive_file [POSITIVE_FILE] --hard_negative_file [HARD_NEGATIVE_FILE]`
    - EPOCH: The maximum number of training epoches 
    - BATCH: Batch size
    - IMG_DIR: the directory of training images
    - MASK_DIR: the directory of corresponding masks
    - POSITIVE_FILE: the txt file containing positive samples
    - HARD_NEGATIVE_FILE: the txt file containing hard negative samples
- To test/inference, run `test.py --region [REGION] --model [MODEL_WEIGHT] --img_dir [IMG_DIR] --test_file [TEST_FILE]`
    - REGION: The region name to generate segmentation masks
    - MODEL_WEIGHT: model weight (model checkpoint (.pth file))
    - IMG_DIR: the directory of testing images
    - TEST_FILE: the txt file containing test samples

<!-- - Please refer to different training strategies (e.g., validation, 50% ratio positive/negative sampling) on [https://github.com/zekun-li/supraglacial_lake](https://github.com/zekun-li/supraglacial_lake) -->

### 2.2. DeepLabv3+
**Directory** `./models/DeepLabv3Plus/`

**Model Weights** You can download the model weights from [HERE](https://drive.google.com/file/d/10CxDd_ZUCrLriYQ0bNU16XfphITb043q/view?usp=sharing). Please make sure to place the weights under `weight/` directory.

**Environment Setup** <br>
- The model is trained with python 3.8 and CUDA 11.3
- To install the environment in conda, run `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge`, then `pip install -r requirements.txt ` <br>

**How to run** <br>
- To modify configuration file ./configs/config.yml <br>
    - `base_path'` : the root directory of image and mask folder <br> 
    - `region_txt_base_path` : the directory of txt files indicating positive and negative samples, named as `train_pos.txt`, `train_neg.txt`, `test.txt` (if exists) <br>
    - `save_res_path`: the directory to the segmentation results <br><br>
- To train, run `python main.py -c configs/config.yml --train`<br>
- To test, run `python main.py -c configs/config.yml --predict_on_test`<br>
- To inference given images, run `python main.py -c configs/config.yml --predict --filefolder IMAGES_DIR_PATH`<br>

---


## 3. External Data Resources
### 3.1. Topographic Sink
Since lakes are defined by open water of some depth, we use the [ArcticDEM](https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/latest/100m/) to identify the potential locations for lakes. We generate topographic sinks to postprocess the model output. You can download the file [HERE](https://drive.google.com/file/d/1ZFti7I1OaInSv7wicis-acln5-y__e6p/view?usp=drive_link). 

The process of generating topographic sinks from ArcticDEM has two steps. First, we employ the open-source WhiteboxToolsTM library to fill the depressions in the ArcticDEM and eliminate flat areas. Second, we generate topographic sinks by subtracting the output of the first step from the original ArcticDEM. Locations, where the subtraction results yield values smaller than zero, represent the topographic sinks.


### 3.2. Soil Data
We use the soil information to further illiminate the detected lakes that are not located in the glacier area. We use [Northern Circumpolar Soil Carbon Database version 2 (NCSCDv2)](https://apgc.awi.de/dataset/ncscdv2-greenland-geotiff-netcdf), a geospatial database that records the amount of organic carbon storage in soils of the northern circumpolar permafrost region down to a depth of 300 cm. Since the dataset delimited the areas that are covered by glaciers for most times in a year, we use this dataset to identify the glacier area and exclude lakes that are not located on glaciers. You can download the file [HERE](https://drive.google.com/file/d/15RIpVXElhw882SXyvKBltMBRn3lMtsNr/view?usp=drive_link). 

---

## 4. Data Postprocessing
**Directory** `./data_postprocess/`

After generating the segmentation results from the two models (you can find the results [HERE](https://drive.google.com/drive/u/0/folders/1JAKijqh7vLPZ2_EofWuNO-lb1A-_Bd5f)), our approach first merges and extracts polygons from the segmentation masks respectively, and does some priliminary preprocesses on the polygons. Second, our approach treats the union of topographic sink, color thresholding, and the inverse of soil allocation as lake candidates, and removes all the model-based polygons that are not in the lake candidate. Then our approach compares the SAM-based results and topographic sink on a vector-wise basis. For each lake candidate, our approach only keeps the SAM-based polygon that has the largest overlapping area with the color-thresholding polygon. Lastly, our approach adds model-based polygons (from SAM and DeepLab that were removed) that reach the relative-area criteria of the lake candidates.

**How to run**
- To merge segmentation results from image patches, run `python postprocessing_merge.py --data_root [DATA_ROOT] --result_path [RESULT_PATH] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE] --out_gpkg_path [OUT_GPKG_PATH]`
    - DATA_ROOT: data directory of the competition data
    - RESULT_PATH: the segmentation result path, e.g., ../results/deeplabv3p_update/2019-08-25_29_r5
    - CROP_SIZE: cropped image size, default 1024
    - SHIFT_SIZE: shift size, default 1024 
    - OUT_GPKG_PATH: the directory of output gpkg files for each region
    
- To generate a single GPKG file that integrates results from different models, using topo-based and color-based extraction as a reference to postprocess, run `python postprocessing_with_external.py --data_root [DATA_ROOT] --result_name [RESULT_NAME] --data_topo [DATA_TOPO] --data_soil [DATA_SOIL] --sam_dir [SAM_DIR] --dpl_dir [DPL_DIR]`
    - DATA_ROOT: data directory path for the provided dataset
    - RESULT_NAME: name of the output GPKG file
    - DATA_TOPO: directory path to topographic_sink.tif
    - DATA_SOIL: directory path to NCSCDv2_Greenland_WGS84_nonsoil_pct_0012deg.tif
    - SAM_DIR: directory path of the results from the SAM model
    - DPL_DIR: directory path of the results from the DeepLab model
