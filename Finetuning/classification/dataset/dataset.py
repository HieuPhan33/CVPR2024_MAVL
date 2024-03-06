from cmath import nan
import csv
import json
import logging
import os
import sys
import pydicom

from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from skimage import exposure
import torch
from torchvision.transforms import InterpolationMode
from dataset.randaugment import RandomAugment
import glob
from albumentations import ShiftScaleRotate, Normalize, Resize, Compose, RandomResizedCrop, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
import cv2
np.random.seed(98)

def get_transforms(is_train, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    list_transforms = []
    if is_train:
        list_transforms.extend(
            [
                RandomResizedCrop(
                    224, 224,
                    scale=(0.2, 1.0), 
                    interpolation=cv2.INTER_CUBIC),
                ShiftScaleRotate(
                    shift_limit=0.05,  # no resizing
                    scale_limit=0.0,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                HorizontalFlip(),
                RandomBrightnessContrast()
            ]
        )
    else:
        list_transforms.extend(
            [
                Resize(224,224),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms

class SIIM_ACR_Dataset(Dataset):
    def __init__(self, csv_path, root, is_train=True, pct=0.01):
        data_info = pd.read_csv(csv_path)
        if is_train:
            print("Fine-tune on ", 100*pct,"%")
            total_len = int(pct*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = np.asarray(data_info.iloc[:,0])[choice_list]
        else:
            self.img_path_list = np.asarray(data_info.iloc[:,0])
            
        self.img_root = f'{root}/processed_images/'   
        self.seg_root = f'{root}/processed_masks/' # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        
        
        #normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #self.transform = get_transforms(is_train, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7, isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])     
        
        self.seg_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224],interpolation=InterpolationMode.NEAREST),
        ])

    def __getitem__(self, index):
        img_path = self.img_root + self.img_path_list[index] + '.png'
        seg_path = self.seg_root + self.img_path_list[index] + '.png'    # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)

        seg_map = PIL.Image.open(seg_path)
        seg_map = self.seg_transfrom(seg_map)
        #augmented = self.transform(image=np.asarray(img), mask=np.asarray(seg_map))
        seg_map = (seg_map > 0).type(torch.int)
        class_label = np.array([int(torch.sum(seg_map)>0)])
        return {
            "image": image,
            "label": class_label
            }

    def __len__(self):
        return len(self.img_path_list)
    

class CovidCXR2_Dataset(Dataset):
    def __init__(self, csv_path, root='/data/data/covidx-cxr2', is_train = False, pct=0.01):
        # Read the CSV file without a header and specify the column names
        data_info = pd.read_csv(csv_path, header=None, names=["id", "image_path", "class", "source"], delimiter=' ')

        self.root = root
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,3:])

        
        self.img_path_list = np.asarray(data_info['image_path'])
        self.class_list = np.asarray(data_info['class'] == 'positive').astype(np.int8)
        if is_train:
            print("Fine-tune on ", 100*pct,"%")
            total_len = int(pct*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = self.img_path_list[choice_list]
            self.class_list = self.class_list[choice_list]
            

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])   

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index]
        img_path = f'{self.root}/{img_path}'
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)

        return {
            "image": image,
            "label": np.array([class_label]),
            #"img_path": '/'.join(img_path.split('/')[-2:])
            }
    
    def __len__(self):
        return len(self.img_path_list)
    

class Chexpert_Dataset(Dataset):
    def __init__(self, csv_path, root='/data/VLM/data/CheXpert', is_train = False, subset=False, pct=0.01):
        data_info = pd.read_csv(csv_path)
        self.root = root
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,3:])
        # if 'test' not in csv_path:
        #     offset = 4
        #     # data_info = data_info[data_info['Frontal/Lateral'] == 'Frontal']
        # else:
        #     offset = 0
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        if not subset:
            self.class_list = np.asarray(data_info.iloc[:,1+offset:])
        else:
            self.class_list = np.asarray(data_info.iloc[:,[3+ offset, 6 + offset, 7 + offset, 9 + offset, 11 + offset]])
        if is_train:
            n_total = self.img_path_list.shape[0]
            total_len = int(pct*n_total)
            choice_list = np.random.choice(range(n_total), size = total_len, replace= False)
            self.img_path_list = self.img_path_list[choice_list]
            self.class_list = self.class_list[choice_list]
        

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])   

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index]
        img_path = f'{self.root}/{img_path}'
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label,
            #"img_path": '/'.join(img_path.split('/')[-2:])
            }
    
    def __len__(self):
        return len(self.img_path_list)
    

class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path, root='data_sample/cxr', is_train = False, pct=0.01):
        data_info = pd.read_csv(csv_path)
        self.root = root
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,3:])
        if is_train:
            print("Fine-tune on ", 100*pct,"%")
            total_len = int(pct*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = np.asarray(data_info.iloc[:,0])[choice_list]
            self.class_list = np.asarray(data_info.iloc[:,3:])[choice_list]
        else:
            self.img_path_list = np.asarray(data_info.iloc[:,0])
            self.class_list = np.asarray(data_info.iloc[:,3:])
            
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])   

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index]
        #img_path = f'{self.root}/{img_path}'
        img_path = img_path.replace('/DATA/Chestxray/ChestXray8', self.root)
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)

        return {
            "image": image,
            "label": class_label,
            #"img_path": img_path.split('/')[-1]
            }
    
    def __len__(self):
        return len(self.img_path_list)
    

    
class RSNA2018_Dataset(Dataset):
    def __init__(self, csv_path, root='../data', is_train = False, pct=0.01):
        data_info = pd.read_csv(csv_path)
        self.root = root
        self.img_path_list = np.asarray(data_info.iloc[:,1])
        self.class_list = np.asarray(data_info.iloc[:,3])
        if is_train:
            print("Fine-tune on ", 100*pct,"%")
            n_total = self.img_path_list.shape[0]
            total_len = int(pct*n_total)
            choice_list = np.random.choice(range(n_total), size = total_len, replace= False)
            self.img_path_list = self.img_path_list[choice_list]
            self.class_list = self.class_list[choice_list]
        #self.transform = get_transforms(is_train, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])
            self.seg_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ]) 
            

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = self.root + img_path
        class_label = np.array([self.class_list[index]]) # (14,)

        img = self.read_dcm(img_path)
        #image = self.transform(np.asarray(img))
        image = self.transform(img)
        
        return {
            "image": image,
            "label": class_label,
            "img_path": img_path.split('/')[-2:]
            }
    
    def read_dcm(self,dcm_path):
        dcm_data = pydicom.read_file(dcm_path)
        img = dcm_data.pixel_array.astype(float) / 255.
        img = exposure.equalize_hist(img)
        
        img = (255 * img).astype(np.uint8)
        img = PIL.Image.fromarray(img).convert('RGB')   
        return img


    def __len__(self):
        return len(self.img_path_list)

class MURA_Dataset(Dataset):
    def __init__(self, csv_path, root='/data/VLM/data', is_train=True ):
        data_info = pd.read_csv(csv_path, header=None)
        self.root = root
        self.img_path_list = np.asarray(data_info.iloc[:, 0])
        

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = 1 if 'positive' in img_path else 0
        # class_label = np.zeros(2)
        # class_label[label] = 1
        class_label = np.array([label])
        img_path = f'{self.root}/{img_path}'
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        return {
            "image": image,
            "label": class_label,
            "img_path": img_path.split('/')[-1]
        }

    def __len__(self):
        return len(self.img_path_list)

class LERA_Dataset(Dataset):
    def __init__(self, csv_path, root, is_train=True):
        df = pd.read_csv(csv_path, header=None)
        self.img_id2label= dict(zip(df[0], df[2]))
        img_path = glob.glob(f'{root}/*/*/*png')
        if is_train==True:
            total_len = int(pct*len(img_path))
            choice_list = np.random.choice(range(len(img_path)), size = total_len,replace= False)
            self.img_path_list = np.asarray(img_path)[choice_list]
        else:
            self.img_path_list = np.asarray(img_path)
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7, isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])     
        

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        img_id = img_path.split('/')[-3]

        class_label = self.img_id2label[int(img_id)]
        class_label = np.array([class_label])
        return {
            "image": image,
            "label": class_label,
            "img_path": img_id
        }

    def __len__(self):
        return len(self.img_path_list)
    

def create_loader_RSNA(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
