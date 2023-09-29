import os
import glob
import argparse
from postproc import *
from giscup_evaluation_f1 import *
from output_gpkg import convert_img_to_geocoord

import warnings
warnings.filterwarnings("ignore")


def main():
    
    # rock_mask_path = os.path.join(root, f'dataset/data_crop{args.crop_size}_shift{args.shift_size}/rock_mask_overlap')
    manual_mask_file = os.path.join(args.data_root, f'mountain_mask.csv')
    manual_mask_dict = load_manual_mask(manual_mask_file)
    
    pred_path = os.path.join(args.result_root, args.result_name)
    pred_files = glob.glob(os.path.join(pred_path, '*.jpg'))
    test_regions = list(set([os.path.basename(p).split('__')[0] for p in pred_files]))
    
    for test_region in sorted(test_regions):
        print(test_region)
        region_num = test_region[-1]
        polys = merge(test_region, pred_path, args.crop_size, args.shift_size, rock_mask_path=None)
        polys = filter_narrow_streams(polys)
        if manual_mask_dict.get(region_num) is not None:
            polys = filter_by_manual_mask(polys, manual_mask_dict[region_num])
    
        out_gpkg_file = os.path.join(args.result_root, f'{args.result_name}_{test_region}__out.gpkg')
        convert_img_to_geocoord(test_region, polys, out_gpkg_file)
    
        gt_gpkg_file = os.path.join(args.data_root, 'region_gpkgs_geocoords', f'{test_region}.gpkg')
        if gt_gpkg_file is not None:
            f1 = giscup_evaluation_f1(gt_gpkg_file, out_gpkg_file)
            print('F1=', f1)
            
        # region_image = f'dataset/region_images/{test_region}.png'
        # img = Image.open(region_image)
        # img = np.array(img)
        
        # polys = [i.astype(int) for i in polys]
        # blank = cv2.drawContours(img, polys, -1, (255, 0, 0), thickness=3)
        # blank = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(root, f'results/{result_name}_{test_region}.jpg'), blank)
        print()
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/Users/yijunlin/Work/Research/2023_SIGSPATIAL_Cup/dataset/')
    parser.add_argument('--result_root', type=str, default='/Users/yijunlin/Work/Research/2023_SIGSPATIAL_Cup/results/')
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--shift_size', type=int, default=512)
    parser.add_argument('--result_name', type=str, default='crop1024_shift512_SAM_eval_overlap_zekun_0926')
    args = parser.parse_args()

    print(f"Processing segmentation output: {args.result_name}")
    main()





