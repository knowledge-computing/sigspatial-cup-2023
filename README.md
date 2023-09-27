# sigspatial-cup-2023
The code and result for sigspatial competition 2023

## Models 
### Segment Anything Model
**Description** 
We fine-tuned Facebook's [Segment Anything Model (SAM)](https://segment-anything.com/) on the glacier training data provided to us. We formulate the problem as a pixel-level binary-class classification problem, where supraglacial lakes should be predicted as 1 and backgrounds should be predicted as 0. Since the ratio of lake and non-lake regions in the training data are skewed, we further conducted the positive (lake) and negative (non-lake) balancing where each input image patch has 50% chance to contain lake. 
**How to run**
- Training script: `train.py`
- Testing script: `test.py` 
