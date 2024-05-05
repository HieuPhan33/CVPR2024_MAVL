from accelerate import Accelerator
import torchshow as ts
global accelerator
import argparse
import os
import yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from thop import profile
from thop import clever_format

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import csv

from models.model_MedKLIP import MedKLIP
from models.model_MedSLIP import MedSLIP2
from models import model_MedSLIP_v2
from models.model_MedSLIP_multi import MedSLIP_multi
from models.model_MedSLIP_three import MedSLIP_three
from models.model_MedSLIP_global import MedSLIP_global
from models.model_MedSLIP_combine import MedSLIP_combine
from dataset.dataset import Chestxray14_Dataset, MURA_Dataset, Chexpert_Dataset, SIIM_ACR_Dataset, RSNA2018_Dataset, LERA_Dataset, Padchest_Dataset, \
    CovidCXR2_Dataset, Covid19_Dataset
from models.tokenization_bert import BertTokenizer
from tabulate import tabulate


from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch.autograd.profiler as profiler
from ptflops import get_model_complexity_info



def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer


def test(config):
    device = accelerator.device
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type('torch.FloatTensor')

    ## Setup disease name
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
    original_class = [
                'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
                'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                'tail_abnorm_obs', 'excluded_obs'
            ]
    # padchest_seen_class = [
    #     'normal', 'pleural effusion', 'atelectasis', 'pneumonia', 'consolidation', 'fracture', 'emphysema', 'cardiomegaly', 'mass', 'nodule', 'edema',
    #     'pacemaker', 'catheter', 'pneumothorax', 'tracheal shift', 'vertebral compression', 'pulmonary fibrosis', 'mediastinal mass'
    # ]
    # padchest_seen_class = [
    #     'normal', 'pleural effusion', 'atelectasis', 'pneumonia', 'consolidation', 'fracture', 'emphysema', 'cardiomegaly', 'mass', 'nodule', 'edema',
    #     'pacemaker', 'catheter', 'pneumothorax', 'tracheal shift', 'vertebral compression', 'pulmonary fibrosis', 'mediastinal mass'
    # ]
    padchest_seen_class = [
        'normal', 'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'pleural effusion', 'pneumonia', 'pneumothorax'
    ]

    padchest_rare = ['suture material', 'sternotomy', 'supra aortic elongation', 'metal', 'abnormal foreign body', 'central venous catheter via jugular vein', 'vertebral anterior compression', 'diaphragmatic eventration', 'consolidation', 'calcified densities', 'volume loss', 'single chamber device', 'vertebral compression', 'bullas', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'dual chamber device', 'mediastinic lipomatosis',
                     'esophagic dilatation', 'azygoesophageal recess shift', 'breast mass', 'round atelectasis', 'surgery humeral', 'aortic aneurysm', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary artery hypertension', 'pleural mass', 'empyema', 'external foreign body', 'respiratory distress', 'total atelectasis', 'ventriculoperitoneal drain tube', 'right sided aortic arch', 'aortic endoprosthesis', 'cyst', 'pulmonary venous hypertension', 'double J stent']
    
    padchest_unseen_class = [
        'hypoexpansion basal', 'non axial articular degenerative changes', 'central venous catheter via jugular vein', 'multiple nodules', 
        'COPD signs', 'calcified densities', 'mediastinal shift', 'hiatal hernia', 
        'volume loss', 'mediastinic lipomatosis', 'central venous catheter', 
        'ground glass pattern', 'surgery lung', 'miliary opacities', 'sclerotic bone lesion', 'pleural plaques', 'osteosynthesis material', 
        'calcified mediastinal adenopathy', 'apical pleural thickening', 'aortic elongation', 'major fissure thickening', 'callus rib fracture', 
        'pulmonary venous hypertension', 'cervical rib', 'loculated pleural effusion', 
        'flattened diaphragm' 
    ]

    padchest_unseen_class = list(set(padchest_unseen_class + padchest_rare))
    if config['dataset'] == 'chexray':
        dataset_cls = chexray14_cls
        test_dataset =  Chestxray14_Dataset(config['test_file'], is_train=False, root=config['root']) 
    elif config['dataset'] == 'mura':
        dataset_cls = mura_cls
        test_dataset =  MURA_Dataset(config['test_file'], is_train=False, root=config['root']) 
    elif config['dataset'] == 'lera':
        dataset_cls = lera_cls
        test_dataset =  LERA_Dataset(config['test_file'], is_train=False, root=config['root']) 
    elif config['dataset'] == 'chexpert':
        dataset_cls = chexpert_cls
        test_dataset = Chexpert_Dataset(config['test_file'], is_train=False, root=config['root'],
                                         subset=chexpert_subset)
    elif config['dataset'] == 'siim':
        dataset_cls = siim_cls
        test_dataset = SIIM_ACR_Dataset(config['test_file'], is_train=False, root=config['root'])
    elif config['dataset'] == 'rsna':
        dataset_cls = rsna_cls
        test_dataset = RSNA2018_Dataset(config['test_file'], root=config['root'])
    elif config['dataset'] == 'covid-cxr2':
        dataset_cls = covid_cls
        original_class.append('covid19')
        test_dataset = CovidCXR2_Dataset(config['test_file'], root=config['root'])
    elif config['dataset'] == 'covid-r':
        dataset_cls = covid_cls
        original_class.append('covid19')
        test_dataset = Covid19_Dataset(config['test_file'], root=config['root'])
    elif config['dataset'] == 'padchest':
        # dataset_cls = padchest_seen_class + padchest_unseen_class
        # original_class += padchest_unseen_class
        #dataset_cls = padchest_seen_class
        dataset_cls = padchest_rare
        original_class.extend(item for item in padchest_rare if item not in original_class)
        test_dataset = Padchest_Dataset(config['test_file'], root=config['root'], classes=dataset_cls)
        if 'pleural effusion' in dataset_cls:
            dataset_cls[dataset_cls.index('pleural effusion')] = 'effusion'
    # original_class = dataset_cls
    mapping = []
    for disease in dataset_cls:
        if disease in original_class:
            print(disease)
            mapping.append(original_class.index(disease))
        else:
            mapping.append(-1)
    MIMIC_mapping = [ _ for i,_ in enumerate(mapping) if _ != -1] # valid MIMIC class index
    dataset_mapping = [ i for i,_ in enumerate(mapping) if _ != -1] # valid (exist in MIMIC) chexray class index
    target_class = [dataset_cls[i] for i in dataset_mapping ] # Filter out non-existing class
    print(MIMIC_mapping)

                 
    
    print("Creating book")
    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in original_class]
    ana_book = ['It is located at ' + i for i in
                ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                 'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                 'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
                 'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung',
                 'left_upper_lung',
                 'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung',
                 'right_apical_lung',
                 'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic',
                 'costophrenic_unspec',
                 'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium',
                 'right_ventricle', 'aorta', 'svc',
                 'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes',
                 'unspecified', 'other']]
    if config['none_location']:
        ana_book.append('It is not present')
    print("Number of anatomies:", len(ana_book))

    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)
    ana_book_tokenizer = get_tokenizer(tokenizer, ana_book).to(device)
    select_concepts = config.get('select_concepts', None)
    if 'concept_book' in config:
        concepts = json.load(open(config['concept_book'], 'r'))
        concepts = {i: concepts[i] for i in original_class}
        for i in original_class:
            if len(concepts[i]) != 8: print(i)
        if select_concepts is None:
            concepts_book = sum(concepts.values(), [])
        else:
            concepts_book = []
            for disease, concepts_ in concepts.items():
                concepts_book += [concepts_[i] for i in select_concepts]
        concepts_book_tokenizer = get_tokenizer(tokenizer, concepts_book).to(device)

    concepts = ['global description', 'fuzzy border', 'fluid accumulation', 'location', 'more opacity', 'other - assymetric chest, cloudiness, dark lines', 'cloudy patterns', 'irregular, patchy shapes', 'fluffy or grainy textures']
    
    print("Creating model")
    if config['model'] == 'medklip':
        model = MedKLIP(config, disease_book_tokenizer)
    elif config['model'] == 'medslip2':
        model = MedSLIP2(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    elif config['model'] == 'medslip_v2':
        model = model_MedSLIP_v2.MedSLIP2(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    elif config['model'] == 'medslip_multi':
        model = MedSLIP_multi(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    elif config['model'] == 'medslip_global':
        model = MedSLIP_global(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    elif config['model'] == 'medslip_combine':
        model = MedSLIP_combine(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    elif config['model'] == 'medslip_3level':
        model = MedSLIP_three(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    # model, test_dataloader = accelerator.prepare(model, test_dataloader)
    # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)  

    model.eval()
    input_image = torch.rand((1, 3, 224, 224)).to(device)
    for _ in range(5):
        _ = model(input_image)

    torch.cuda.synchronize()
    with profiler.profile(use_cuda=True) as prof:
        with profiler.record_function("inference"):
            
            with torch.no_grad():
                if config['model'] in ['medslip_global', 'medslip_combine']:
                    _ = model(input_image) #batch_size,num_class,dim
                    # preds_global = []
                    # for normal_idx in [0,1]:
                    #     normal = pred_global[:, :, [normal_idx]].repeat(1,1, len(original_class))
                    #     pred_global_ = torch.stack([normal, pred_global], dim=-1) # B, N_concepts, N_disease
                    #     pred_global_ = F.softmax(pred_global_, dim=-1)
                    #     preds_global.append(pred_global_)
                    # pred_global = torch.stack(preds_global, dim=0).mean(dim=0)
                    # pred_global = pred_global[:, :, MIMIC_mapping, 1]
    print(prof.key_averages())


    # Measure GFLOPs using the ptflops library
    macs, params = profile(model, inputs=(input_image,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/MedKLIP_config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    accelerator = Accelerator()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != '-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    test(config)

