# MedKLIP:Medical Knowledge Enhanced Language-Image Pre-Training

## Introduction: 

The official implementation  code for "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training".

[**Paper Web**](https://chaoyi-wu.github.io/MedKLIP/) 

[**Arxiv Version**](https://arxiv.org/abs/2301.02228)

## Download necessary files
Run ```bash download.sh```

## Pre-train:
Our pre-train code is given in ```Pretrain```. 
* Run download.sh to download necessary files to PreTrain_MedKLIP
* Modify the path in config file configs/MedSLIP2_ViT.yaml, and ```python train_MedSLIP.py``` to pre-train.
* Run `accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --num_cpu_threads_per_process=8 train_MedSLIP.py --root /data/2.0.0 --config configs/MedSLIP2_resnet_reg.yaml --bs 160 --num_workers 8 --output output_r50_reg_disc`

## Quick Start:
Check this [link](https://github.com/MediaBrain-SJTU/MedKLIP/tree/main/checkpoints) to download MedKLIP model. It can be used for all zero-shot && finetuning tasks 

* **Zero-Shot Classification:**
    
    We give examples in ```Sample_Zero-Shot_Classification```. Modify the path, and test our model by ```python test.py --config configs/dataset_name_medklip.yaml```
* **Zero-Shot Grounding:**
    
    We give examples in ```Sample_Zero-Shot_Grounding```. Modify the path, and test our model by ```python test.py```
* **Finetuning:**
    
    We give segmentation and classification finetune code on in ```Sample_Finetuning_SIIMACR```. Modify the path, and finetune our model by ```python I1_classification/train_res_ft.py --config configs/dataset_name_medklip.yaml``` or ```python I2_segementation/train_res_ft.py --config configs/dataset_name_medklip.yaml```
