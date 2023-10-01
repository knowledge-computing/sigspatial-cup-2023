import argparse
import os
import numpy as np

import torch
import yaml
import cv2
from trainers.trainer import Trainer
from predictors.predictor import Predictor

def train(args):
    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()
    config['checkname'] = 'deeplab-'+str(config['network']['backbone'])

#    torch.manual_seed(config['seed'])
    trainer = Trainer(config)
        
    print('Starting Epoch:', trainer.config['training']['start_epoch'])
    print('Total Epoches:', trainer.config['training']['epochs'])
    
    for epoch in range(trainer.config['training']['start_epoch'], trainer.config['training']['epochs']):
        trainer.training(epoch)
        if not trainer.config['training']['no_val'] and epoch % config['training']['val_interval'] == (config['training']['val_interval'] - 1):
            trainer.validation(epoch)

    trainer.writer.close()

def predict_on_test_set(args):
#    print("predict on test")

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()
    check_path=config["model"]['weight']
    predictor = Predictor(config, checkpoint_path=check_path)

    predictor.inference_on_test_set()
    # predictor.segment_image()

def predict(args):
#    print("predict")

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    filefolder = args.filefolder
    config['network']['use_cuda'] = config['network']['use_cuda'] and torch.cuda.is_available()
    check_path=config["model"]['weight']
    predictor = Predictor(config, checkpoint_path=check_path)
    image_list=[] 
    prediction_list=[]
    prob_list=[]
    #txt file contains test image names
    # will be linked with test image path
    file_list=[]
    if(filefolder=='test'):     
        test_file=open(os.path.join(config['dataset']['region_txt_base_path'],'test.txt'))  
        tmp_test_lines = test_file.readlines()
        test_list=[line.rstrip('\n') for line in tmp_test_lines]           
        test_path=os.path.join(config['dataset']["test_base_path"], 'train_images') 
        file_list=test_list
    #is driectory
    else:
        test_path=filefolder
        file_list=os.listdir(filefolder)

    if os.path.exists(config['dataset']['save_res_path']) == False:
        os.mkdir(config['dataset']['save_res_path'])

    for filename in file_list:
        filename=os.path.join(test_path,filename)
        image, prediction,prob = predictor.segment_image(filename)

        result_file_name=filename.split('/')[-1].split('.')[0]+'.jpg'

        result_path=os.path.join(config['dataset']['save_res_path'],result_file_name)
        prediction=prediction*255
        # print(np.unique(prediction))
        prob=prob*100  
        # print(np.unique(prob))   
        prob=np.int8(prob)
        # print(np.unique(prob))
        cv2.imwrite(result_path,prediction)

        image_list.append(image)
        prob_list.append(prob)
        prediction_list.append(prediction)
        


    return image_list, prediction_list
#    print(np.max(prediction))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Seq2seq')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')    
    group.add_argument('--predict_on_test_set', action='store_true', help='Predict on test set')
    group.add_argument('--predict', action='store_true', help='Predict on single file')

    parser.add_argument('--filefolder', help='path to file')
    
    args = parser.parse_args()


    if args.predict_on_test_set:
        predict_on_test_set(args)      

    elif args.predict:
        if args.filefolder is None:
            raise Exception('missing --filefolder filefolder')
        else:
            predict(args)

    elif args.train:
        print('Starting training')
        train(args)   
    else:
        raise Exception('Unknown args') 
