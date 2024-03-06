'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
from models.resnet import ModelRes_ft
from models.mavl import MAVL_ft

from dataset.dataset import Chestxray14_Dataset, MURA_Dataset, Chexpert_Dataset, SIIM_ACR_Dataset, RSNA2018_Dataset, LERA_Dataset, CovidCXR2_Dataset



def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def test(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    chexray14_cls = [ 'atelectasis', 'cardiomegaly', 'effusion', 'infiltrate', 'mass', 'nodule', 'pneumonia',
                    'pneumothorax', 'consolidation', 'edema', 'emphysema', 'tail_abnorm_obs', 'thicken', 'hernia']  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  
    mura_cls = lera_cls = ['abnormality']
    if config['dataset'] == 'chexpert':
        chexpert_subset = config['chexpert_subset']

        if not chexpert_subset:
            chexpert_cls = ['normal', 'enlarge', 'cardiomegaly',
                'opacity', 'lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis',
                'pneumothorax', 'effusion', "abnormality", 'fracture', 'device']
        else:
            chexpert_cls = ['cardiomegaly','edema', 'consolidation', 'atelectasis','effusion']

    siim_cls = ['pneumothorax']
    rsna_cls = ['pneumonia']
    covid_cls = ['covid19']

    if config['dataset'] == 'chexray':
        dataset_cls = chexray14_cls
        dataset_cls_name =  Chestxray14_Dataset
    elif config['dataset'] == 'mura':
        dataset_cls = mura_cls
        dataset_cls_name =  MURA_Dataset
    elif config['dataset'] == 'lera':
        dataset_cls = lera_cls
        dataset_cls_name =  LERA_Dataset
    elif config['dataset'] == 'chexpert':
        dataset_cls = chexpert_cls
        dataset_cls_name = Chexpert_Dataset
    elif config['dataset'] == 'siim':
        dataset_cls = siim_cls
        dataset_cls_name = SIIM_ACR_Dataset
    elif config['dataset'] == 'rsna':
        dataset_cls = rsna_cls
        dataset_cls_name = RSNA2018_Dataset
    elif config['dataset'] == 'covid-cxr2':
        dataset_cls = covid_cls
        dataset_cls_name = CovidCXR2_Dataset
    config['num_classes'] = len(dataset_cls)
    print("Number of classes ", config['num_classes'])
    classes = dataset_cls

    #### Dataset #### 
    print("Creating dataset")
    test_dataset = dataset_cls_name(config['test_file'], root= config['root'], is_train=False) 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )              
    
    if config['model'] == 'medklip':
        model = ModelRes_ft(res_base_model='resnet50', out_size=config['num_classes'])
    elif config['model'] == 'mavl':
        model = MAVL_ft(base_model=config['base_model'], out_size=config['num_classes'])
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device) 

    print('Load model from checkpoint:', config['model_path'])
    checkpoint = torch.load(config['model_path'], map_location='cpu') 
    state_dict = checkpoint['model']          
    model.load_state_dict(state_dict)    

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    print("Start testing")
    model.eval()
    for i, sample in enumerate(test_dataloader):
        image = sample['image']
        label = sample['label'].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image)
            pred_class = F.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class), 0)
    
    AUROCs = compute_AUCs(gt, pred,config['num_classes'])
    AUROC_avg = np.array(AUROCs).mean()
    gt_np = gt[:, 0].cpu().numpy()
    pred_np = pred[:, 0].cpu().numpy()            
    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    acc = accuracy_score(gt_np, pred_np>max_f1_thresh)
    print('The max f1 is',max_f1)
    print('The accuracy is', acc) 
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(config['num_classes']):
        print(f'The AUROC of {classes[i]} is {AUROCs[i]}')
    return AUROC_avg, max_f1, acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Path/To/Res_train.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='Path/To/best_valid.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    test(args, config)
