from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import pydicom
from skimage import exposure

class SIIM_ACR_Dataset(Dataset):
    def __init__(self, csv_path, root, is_train=True, pct=0.01):
        data_info = pd.read_csv(csv_path)
        if is_train:
            print("Fine-tune on ", pct*100, "%")
            total_len = int(pct*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = np.asarray(data_info.iloc[:,0])[choice_list]
        else:
            self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.img_root = f'{root}/processed_images/'   
        self.seg_root = f'{root}/processed_masks/' # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        
        if is_train:
            self.aug = A.Compose([
            A.RandomResizedCrop(width=224, height=224, scale=(0.2, 1.0), always_apply = True, interpolation=Image.BICUBIC),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225],always_apply = True),
            ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
            A.Resize(width=224, height=224, always_apply = True),
            A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225],always_apply = True),
            ToTensorV2()
            ])

    def __getitem__(self, index):
        img_path = self.img_root + self.img_path_list[index] + '.png'
        seg_path = self.seg_root + self.img_path_list[index] + '.png' # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        img = np.array(PIL.Image.open(img_path).convert('RGB') ) 
        seg_map = np.array(PIL.Image.open(seg_path))[:,:,np.newaxis]
        
        augmented = self.aug(image=img, mask=seg_map)
        img, seg_map = augmented['image'], augmented['mask']
        seg_map = seg_map.permute(2, 0, 1)
        
        class_label = np.array([int(torch.sum(seg_map)>0)])
        return {
            "image": img,
            "seg_map": seg_map,
            "label": class_label
            }


    def __len__(self):
        return len(self.img_path_list)

class RSNA2018_Dataset(Dataset):
    def __init__(self, csv_path, root='../data', is_train=True, pct=0.01):
        data_info = pd.read_csv(csv_path)
        self.root = root
        self.img_path_list = np.asarray(data_info.iloc[:,1])
        self.class_list = np.asarray(data_info.iloc[:,3])
        self.bbox = np.asarray(data_info.iloc[:,2])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if is_train:
            print("Fine-tune on ", 100*pct,"%")
            n_total = self.img_path_list.shape[0]
            total_len = int(pct*n_total)
            choice_list = np.random.choice(range(n_total), size = total_len, replace= False)
            self.img_path_list = self.img_path_list[choice_list]
            self.class_list = self.class_list[choice_list]
            self.bbox = self.bbox[choice_list]

        if is_train:
            self.aug = A.Compose([
                A.RandomResizedCrop(width=224, height=224, scale=(0.2, 1.0), always_apply = True, interpolation=Image.BICUBIC),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225], always_apply = True),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
            A.Resize(width=224, height=224, always_apply = True),
            A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225], always_apply = True),
            ToTensorV2()
            ])

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = f'{self.root}/{img_path}'
        class_label = np.array([self.class_list[index]]) # (14,)

        img = self.read_dcm(img_path)
        img = np.array(img) 
        #image = self.transform(img)
        
        bbox = self.bbox[index]
        seg_map = np.zeros((1024,1024))
        if class_label ==1:
            boxes = bbox.split('|')
            for box in boxes:
                cc = box.split(';')
                seg_map[int(float(cc[1])):(int(float(cc[1]))+int(float(cc[3]))),int(float(cc[0])):(int(float(cc[0]))+int(float(cc[2])))]=1
        seg_map = seg_map[:,:,np.newaxis]
        augmented = self.aug(image=img, mask=seg_map)
        image, seg_map = augmented['image'], augmented['mask']
        seg_map = seg_map.permute(2, 0, 1)
        return {
            "image": image,
            "label": class_label,
            "image_path": img_path,
            "seg_map":seg_map
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
    
    

class Covid19_Dataset(Dataset):
    def __init__(self, csv_path, root='../data', is_train=True, pct=0.01):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        if is_train:
            print("Fine-tune on ", pct*100, "%")
            total_len = int(pct*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = np.asarray(data_info.iloc[:,0])[choice_list]
        self.img_root = f'{root}/jpgs/'   
        self.seg_root = f'{root}/pngs_masks/' # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        
        
        if is_train:
            self.aug = A.Compose([
                A.RandomResizedCrop(width=224, height=224, scale=(0.2, 1.0), always_apply = True, interpolation=Image.BICUBIC),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225],always_apply = True),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
            A.Resize(width=224, height=224, always_apply = True),
            A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225],always_apply = True),
            ToTensorV2()
            ])

    def __getitem__(self, index):
        img_path = self.img_root +  self.img_path_list[index] + '.jpg'
        seg_path = self.seg_root + self.img_path_list[index] + '.png'    # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        img = np.array(PIL.Image.open(img_path).convert('RGB'))
        seg_map = np.array(PIL.Image.open(seg_path))[:,:,np.newaxis]
        augmented = self.aug(image=img, mask=seg_map)
        image, seg_map = augmented['image'], augmented['mask']
        seg_map = seg_map.permute(2, 0, 1)
        seg_map = (seg_map > 0).type(torch.int)
        class_label = np.array([int(torch.sum(seg_map)>0)])
        return {
            "image": image,
            "label": class_label,
            "image_path": img_path,
            "seg_map":seg_map
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
