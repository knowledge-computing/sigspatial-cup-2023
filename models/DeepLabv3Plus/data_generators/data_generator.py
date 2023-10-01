#from data_generators.datasets import cityscapes, coco, combine_dbs, pascal, sbd, deepfashion
from torch.utils.data import DataLoader,TensorDataset, ConcatDataset, WeightedRandomSampler
from data_generators.deepfashion import DeepFashionSegmentation
import os
import numpy as np
import torch
def initialize_data_loader(config):
    if config['dataset']['dataset_name'] == "lake_crop" :
        pos_data_dir=config['dataset']['base_path']
        neg_data_dir=config['dataset']['base_path']

        train_set_pos = DeepFashionSegmentation(config,pos_data_dir,split='train_pos')
        train_set_neg= DeepFashionSegmentation(config,neg_data_dir, split='train_neg')

        test_data_dir=config['dataset']['base_path']
        test_set = DeepFashionSegmentation(config, test_data_dir,split='test')
    else:
        raise Exception('dataset not implemented yet!')

    train_set = ConcatDataset([train_set_pos, train_set_neg])
    num_classes = config['network']['num_classes']
    # image,target = train_set
    # print 'target train 0/1: {}/{}'.format(
    # len(np.where(target == 0)[0]), len(np.where(target == 1)[0]))
    # print(len(train_set),'train_set')    
    
    pos_sample=1./(float(len(train_set_pos)))
    neg_sample=1./(float(len(train_set_neg)))
    weight_pos = [pos_sample] * len(train_set_pos)
    weight_neg = [neg_sample] * len(train_set_neg)
    samples_weight=weight_pos+weight_neg
    samples_weight=np.array(samples_weight)  
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    pos_test_sample=0
    neg_test_sample=0
    for idx,data in enumerate(test_set):
        if len(np.unique(data['label'].numpy()))==1:
            neg_test_sample+=1
        else: 
            pos_test_sample+=1
    print(pos_test_sample,'pos_test_sample')
    print(neg_test_sample,'neg_test_sample')
    test_sampler=[0]*len(test_set)

    for idx,data in enumerate(test_set):
        if len(np.unique(data['label'].numpy()))==1:
            test_sampler[idx]=1./(float)(neg_test_sample)
        else: 
            test_sampler[idx]=1./(float)(pos_test_sample)

    test_sampler=np.array(test_sampler)  
    test_sampler = torch.from_numpy(test_sampler)
    test_sampler = test_sampler.double()
    test_sampler = WeightedRandomSampler(test_sampler, len(test_sampler))

    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=True, sampler=sampler,drop_last=True)   
    val_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=True, sampler=test_sampler,drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], num_workers=config['training']['workers'], pin_memory=True, sampler=test_sampler,drop_last=True)

    return train_loader, val_loader, test_loader, num_classes
