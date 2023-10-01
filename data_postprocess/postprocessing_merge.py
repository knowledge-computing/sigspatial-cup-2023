import os
import glob
import argparse

from utils import *

import warnings
warnings.filterwarnings("ignore")


def main():

    manual_mask_file = './Intermediate/mountain_mask.csv'
    if os.path.exists(manual_mask_file):
        manual_mask_dict = load_manual_mask(manual_mask_file)
    else:
        manual_mask_dict = None
    
    pred_files = glob.glob(os.path.join(args.result_root, '*.jpg'))
    test_regions = list(set([os.path.basename(p).split('__')[0] for p in pred_files]))
    if not os.path.exists(args.out_gpkg_path):
        os.makedirs(args.out_gpkg_path)
    
    for test_region in sorted(test_regions):
        print(test_region)
        region_num = test_region[-1]
        polys = merge(test_region, args.result_root, args.crop_size, args.shift_size, rock_mask_path=None)
        polys = filter_narrow_streams(polys)
        if manual_mask_dict is not None and manual_mask_dict.get(region_num) is not None:
            polys = filter_by_manual_mask(polys, manual_mask_dict[region_num])

        out_gpkg_file = os.path.join(args.out_gpkg_path, f'{test_region}__out.gpkg')
        convert_img_to_geocoord(test_region, polys, out_gpkg_file)
    
        gt_gpkg_file = os.path.join(args.data_root, 'region_gpkgs_geocoords', f'{test_region}.gpkg')
        if os.path.exists(gt_gpkg_file):
            print(out_gpkg_file)
            f1 = giscup_evaluation_f1(gt_gpkg_file, out_gpkg_file)
            print('F1=', f1)
            
        print()
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/')
    parser.add_argument('--result_root', type=str, default='../results/deeplabv3p_update/2019-08-25_29_r5')
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--shift_size', type=int, default=512)
    parser.add_argument('--out_gpkg_path', type=str, default='../results/deeplabv3p_update/gpkg/')
    args = parser.parse_args()

    print(f"Processing segmentation output: {args.result_root}")
    main()





