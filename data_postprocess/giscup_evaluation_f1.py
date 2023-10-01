import geopandas as gpd
import numpy as np
import pandas as pd

'''
Each polygon will be assessed as 
“true positive” (TP)
“false positive” (FP - identifying a lake that does not appear in the test dataset)
“false negative” (FN - not identifying a lake that is contained in the test dataset)

An F1-score will be produced as such: F1=2TP/(2TP + FP + FN).
'''

def giscup_evaluation_f1_per_map(targeted_time, targeted_region, groundtruth_gpkg, submitted_gpkg):
    '''
        INPUT
            targeted_time: either of ['2019-06-03_05', '2019-06-19_20', '2019-07-31_25', '2019-08-25_29']
            targeted_region: either of [1, 2, 3, 4, 5, 6]
            groundtruth_gpkg: path to the groundtruth gpkg file
            submitted_gpkg: path to the submitted gpkg file
        OUTPUT
            performance: TP, FP, FN
    '''

    image_stamp = 'Greenland26X_22W_Sentinel2_'
    target_stamp = image_stamp + targeted_time + '.tif'
    pd.set_option('mode.chained_assignment', None)

    # Load groundtruth gpkg
    layer1 = gpd.read_file(groundtruth_gpkg)
    groundtruth_polygon = layer1[(layer1['image']==target_stamp) & (layer1['region_num']==targeted_region)]
    groundtruth_polygon = groundtruth_polygon.drop('image', axis=1) # keep table simple
    groundtruth_polygon = groundtruth_polygon.drop('region_num', axis=1) # keep table simple
    groundtruth_polygon['g_id'] = range(1, groundtruth_polygon.shape[0]+1)
    groundtruth_polygon['g_area'] = groundtruth_polygon['geometry'].area/10**6

    # Load submitted gpkg
    layer2 = gpd.read_file(submitted_gpkg)
    submitted_polygon = layer2[(layer2['image']==target_stamp) & (layer2['region_num']==targeted_region)]
    submitted_polygon['s_id'] = range(1, submitted_polygon.shape[0]+1)
    submitted_polygon['s_area'] = submitted_polygon['geometry'].area/10**6

    # Ovalap polygons from two gpkg
    overlay_polygon = gpd.overlay(submitted_polygon, groundtruth_polygon, how='intersection')
    overlay_polygon['overlap_area'] = overlay_polygon['geometry'].area/10**6
    overlay_polygon = overlay_polygon.drop('geometry', axis=1)


    TP = 0
    FP = 0
    FN = 0
    groundtruth_flag = np.ones((groundtruth_polygon.shape[0]+1), dtype=bool) # An np array initilized all True
    for sid in range(1, submitted_polygon.shape[0]+1):
        this_sid = overlay_polygon[(overlay_polygon['s_id']==sid)]
        ### STEP 1: The polygon does not overlap any other polygon corresponding to the same image and region => FALSE POSITIVE
        ### STEP 2: The polygon partially or fully overlaps at least one polygon identified in the “test” dataset. If yes, go to Step 3. If no => FALSE POSITIVE
        if this_sid.shape[0] == 0:
            FP = FP+1
        else:
            ### STEP 2: If it overlaps more than one test polygon, it will be associated with the test polygon for which it has the greatest overlap (by area)
            selected_gid_4_this_sid = overlay_polygon.iloc[this_sid['overlap_area'].idxmax()]

            ### STEP 3: The submitted polygon area is no less than 50% of the area of the overlapping test polygon identified in Step 2.
            ### STEP 4: The submitted polygon area is no more than 160% of the area of the overlapping test polygon identified in Step 2.
            this_sid_area = selected_gid_4_this_sid['s_area'] # area of the submitted polygon
            this_gid_area = selected_gid_4_this_sid['g_area'] # area of the groundtruth polygon
            overlap_area = selected_gid_4_this_sid['overlap_area'] # area of the area overapped by submitted and groundtruth polygons
            
            if (this_sid_area >=  this_gid_area*0.5) and (this_sid_area <= this_gid_area*1.6):
                if groundtruth_flag[int(selected_gid_4_this_sid['g_id'])] == True:
                    TP = TP+1
                groundtruth_flag[int(selected_gid_4_this_sid['g_id'])] = False
            else:
                FP = FP+1
    ### STEP 5: After assessing each submitted lake, each remaining test polygon that does not have an associated submitted lake, will be considered a “False Negative” in computing the F1 score.
    FN = groundtruth_flag.sum()-1
    
    F1 = 2*TP/(2*TP + FP + FN)
    return TP, FP, FN, F1



'''
we decide test set: 2019-06-03 R6, 2019-06-19 R1, 2019-07-31 R2, 2019-08-25 R3, the rest are in train set
'''
def giscup_evaluation_f1(groundtruth_gpkg, submitted_gpkg):
    '''
        INPUT
            groundtruth_gpkg: path to the groundtruth gpkg file
            submitted_gpkg: path to the submitted gpkg file
        OUTPUT
            performance: F1
    '''

    our_testing_set = [
        ['2019-06-03_05', 2],
        ['2019-06-03_05', 4],
        ['2019-06-03_05', 6],
        ['2019-06-19_20', 1],
        ['2019-06-19_20', 3],
        ['2019-06-19_20', 5],        
        ['2019-07-31_25', 4],
        ['2019-08-25_29', 1],
        ['2019-08-25_29', 3],
        ['2019-08-25_29', 5],
        ['2019-07-31_25', 2],
        ['2019-07-31_25', 6]

    ]

    TP = 0
    FP = 0
    FN = 0
    for traversal in range(0, len(our_testing_set)):
        tp, fp, fn, _ = giscup_evaluation_f1_per_map(our_testing_set[traversal][0], our_testing_set[traversal][1], groundtruth_gpkg, submitted_gpkg)
        TP += tp
        FP += fp
        FN += fn
    
    F1 = 0.0
    if (TP+FP+FN) > 0:
        F1 = 2*TP/(2*TP + FP + FN)
    print('TP', TP)
    print('FP', FP)
    print('FN', FN)
    print('F1', F1)
    return F1