# CVPR2024 - Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Matching Framework

## Introduction
Welcome to the official implementation code for "Decomposing Disease Descriptions for Enhanced Pathology Detection: A Multi-Aspect Vision-Language Matching Framework", accepted at CVPR2024 🎉!

Leveraging LLM to decompose disease descriptions into a set of visual aspects, our Multi-Aspect VL pre-trained model, dubbed MAVL, achieves the state-of-the-art performance across 7 datasets for zero-shot and low-shot fine-tuning settings for disease classification and segmentation.

<!-- [**Paper Web**](https://chaoyi-wu.github.io/MedKLIP/) 

[**Arxiv Version**](https://arxiv.org/abs/2301.02228) -->

## Download Necessary Files
To get started, install the gdown library:
```bash
pip install -U --no-cache-dir gdown --pre
```

Then, run ```bash download.sh```

The MIMIC-CXR2 needs to be downloaded from physionet.

## LLM Disease's Visual Concept Generation

Explore the script to generate diseases' visual aspects using LLM - GPT [here](Pretrain/concept_gen/concept_init.ipynb).

## Pre-train:

Our pre-train code is given in ```Pretrain```. 
* Run download.sh to download necessary files
* Modify the path in config file configs/MAVL_resnet.yaml, and ```python train_mavl.py``` to pre-train.

* Run `accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --num_cpu_threads_per_process=8 train_MAVL.py --root /data/2019.MIMIC-CXR-JPG/2.0.0 --config configs/MAVL_resnet.yaml --bs 124 --num_workers 8`

Note: The reported results in our paper are obtained by pre-training on 4 x A100 for 60 epochs. We provided the checkpoint [here](Pretrain/data_file/DATA_Prepare.md).

We also conducted a lighter pre-training schedule with 2 x A100 for 40 epochs using mixed precision training, achieving similar zero-shot classification results. Checkpoints for this setup are also available [here](Pretrain/data_file/DATA_Prepare.md).

```
accelerate launch --multi_gpu --num_processes=2 --num_machines=1 --num_cpu_threads_per_process=8 --mixed_precision=fp16 train_MAVL.py --root /data/2019.MIMIC-CXR-JPG/2.0.0 --config configs/MAVL_short.yaml --bs 124 --num_workers 8
```

## Quick Start:
Check this [link](Pretrain/data_file/DATA_Prepare.md) to download MAVL checkpoints. It can be used for all zero-shot && finetuning tasks 

* **Zero-Shot Classification:**
    
    We give examples in ```Sample_Zero-Shot_Classification```. Modify the path, and test our model by ```python test.py --config configs/dataset_name_mavl.yaml```
* **Zero-Shot Grounding:**
    
    We give examples in ```Sample_Zero-Shot_Grounding```. Modify the path, and test our model by ```python test.py```
* **Finetuning:**
    
    We give segmentation and classification finetune code on in ```Sample_Finetuning_SIIMACR```. Modify the path, and finetune our model by ```python I1_classification/train_res_ft.py --config configs/dataset_name_mavl.yaml``` or ```python I2_segementation/train_res_ft.py --config configs/dataset_name_mavl.yaml```


## Acknowledgement
Our code is built upon https://github.com/MediaBrain-SJTU/MedKLIP. We thank the authors for open-sourcing their code.

Feel free to reach out if you have any questions or need further assistance!