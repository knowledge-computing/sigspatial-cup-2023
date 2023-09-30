# sigspatial-cup-2023
The code and result for 12th SIGSPATIAL Cup competition (GISCUP 2023) [https://sigspatial2023.sigspatial.org/giscup/index.html](https://sigspatial2023.sigspatial.org/giscup/index.html)

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
We fine-tuned Facebook's [Segment Anything Model (SAM)](https://segment-anything.com/) on the glacier training data provided to us. We formulate the problem as a pixel-level binary-class classification problem, where supraglacial lakes should be predicted as 1 and backgrounds should be predicted as 0. Since the ratio of lake and non-lake regions in the training data are skewed, we further conducted the positive (lake) and negative (non-lake) balancing where each input image patch has 50% chance to contain lake. 

**Environment Setup**
`pip install -r environment.yml`

**How to run**

- To train model, run `train.py`
- To test model, see `test.py`
- 
- Please refer more detail on (GitHub repository [https://github.com/zekun-li/supraglacial_lake](https://github.com/zekun-li/supraglacial_lake))

### DeepLabv3Plus
**Directory** `./models/DeepLabv3Plus`

**Environment Setup**

**Description** 
We fine-tuned [DeepLabv3+](https://github.com/giovanniguidi/deeplabV3-PyTorch) on the glacier training data. The model applied Weighted Random Sampler to address the data imbalance between pixels with Lake and pixels with Non-Lake. The input mask consists of 0 (Non-Lake) and 1(Lake). 

## Data Post-Processing
**Directory** `./data_postprocess/`

**How to run**

- To generate output gpkg from the segmentation results, and do evaluation if the ground truth file exists, run `python run.py --data_root [DATA_ROOT] --result_root [RESULT_ROOT] --crop_size [CROP_SIZE] --shift_size [SHIFT_SIZE] --result_name [RESULT_NAME]`

### Using External Data Resources
**Description**
We generate the topographic sinks as the external data from the ArcticDEM. Supraglacial lakes are formed in surface sinks. Therefore, the topographic sinks are potential locations for supraglacial lakes. The process of generating the topographic sinks from ArcticDEM has two steps. First, we employ the open-source WhiteboxToolsTM library to fill the depressions in the ArcticDEM and eliminate flat areas. Second, we generate topographic sinks by subtracting the output of the first step from the original ArcticDEM. Locations, where the subtraction results yield values smaller than zero, represent the topographic sinks.

**How to run**

- To generate output gpkg from the segmentation results using topo-based and color-based extraction as a reference, run `python postprocess_external.py`
