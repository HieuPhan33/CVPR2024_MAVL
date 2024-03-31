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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
import csv

from models.model_MedKLIP import MedKLIP
from models.model_MAVL import MAVL
from dataset.dataset import Chestxray14_Dataset, MURA_Dataset, Chexpert_Dataset, SIIM_ACR_Dataset, RSNA2018_Dataset, LERA_Dataset, Padchest_Dataset, \
    CovidCXR2_Dataset, Covid19_Dataset
from models.tokenization_bert import BertTokenizer
from tabulate import tabulate


from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


def combine_predictions(pred_class, pred_global):
    # Calculate entropy for both tensors
    entropy_class = binary_entropy(pred_class)
    entropy_global = binary_entropy(pred_global)
    
    # Determine which tensor has lower entropy for each element
    lower_entropy_mask = entropy_class < entropy_global
    
    # Combine predictions based on lower entropy
    combined_predictions = torch.where(lower_entropy_mask[..., None], pred_class, pred_global)
    
    return combined_predictions


def binary_entropy(predictions):
    # Ensure the input tensor has the correct shape
    assert predictions.size(-1) == 2, "Input tensor must have a shape (batch_size, N_class, 2)"


    # Calculate the log probabilities
    log_probs = torch.log(predictions)

    # Calculate the entropy for each prediction
    entropy = -torch.sum(predictions * log_probs, dim=-1)

    return entropy


def log_to_csv(filename, data, firstrow=None):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write header if file doesn't exist
            writer.writerow(firstrow)  # Example header
        writer.writerow(data)
        
        
def save_images_with_annotations(batch, predictions, classes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ignore_idx = [0, 3]
    for i in range(batch.size(0)):
        unnormalized_img = batch[i].cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
        img = to_pil(unnormalized_img).resize((512, 512))

        # Save the original image without annotations
        #img.save(os.path.join(output_dir, f"image_{i}.png"))

        # Create a new image for annotations
        annotated_img = Image.new("RGB", (img.width, img.height + 120))  # Increase height for annotations

        # Paste the original image onto the new image
        annotated_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(annotated_img)

        prediction_score = predictions[i]
        annotation_text = {
            j: f"{classes[j]}: {score.abs():.3f}" for j, score in enumerate(prediction_score) if j not in ignore_idx
            }

        font = ImageFont.load_default()
        text_color = (255, 255, 255)  # White color

        # Adjust the placement of the text underneath the image
        text_x = 10
        text_y = img.height + 10  # Place text below the image

        for j, text in annotation_text.items():
            if prediction_score[j].abs() > 1/len(prediction_score):
                text_color = (0, 255, 0)  # Green color for scores > 0.5

            text_size = draw.textsize(text, font)
            text_background_position = (text_x, text_y, text_x + text_size[0], text_y + text_size[1])
            draw.rectangle(text_background_position, fill=(0, 0, 0))

            draw.text((text_x, text_y), text, fill=text_color, font=font)
            text_y += text_size[1] + 5

        # Save the annotated image
        annotated_img.save(os.path.join(output_dir, f"annotated_image_{i}.png"))


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
        try:
            score = roc_auc_score(gt_np[:, i], pred_np[:, i])
            AUROCs.append(score)
        except ValueError:
            pass
    print("Eval AUC ", len(AUROCs), " / ", n_class)
    return AUROCs

def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def log_to_csv(filename, data, firstrow=None):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write header if file doesn't exist
            writer.writerow(firstrow)  # Example header
        writer.writerow(data)
        
        
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
    # padchest_seen_class = [
    #     'normal', 'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'pleural effusion', 'pneumonia', 'pneumothorax'
    # ]
    padchest_seen_class = ['normal', 'pleural effusion', 'pacemaker', 'atelectasis', 'pneumonia', 'consolidation', 'cardiomegaly', 'emphysema', 
                           'nodule', 'edema', 'pneumothorax', 'fracture', 'mass', 'catheter']
    # padchest_seen_class =  ['cardiomegaly',
    #            'edema',
    #            'consolidation',
    #            'pneumonia',
    #            'atelectasis',
    #            'pneumothorax',
    #            'pleural effusion',
    #            'fracture',
    #            'normal',
    #            'callus rib fracture', 'vertebral fracture','clavicle fracture', 'humeral fracture', 'rib fracture',
    #            'dual chamber device', 'electrical device', 'single chamber device',
    #            'sclerotic bone lesion', 'blastic bone lesion','lytic bone lesion']

    padchest_rare = ['suture material', 'sternotomy', 'supra aortic elongation', 'metal', 'abnormal foreign body', 'central venous catheter via jugular vein', 'vertebral anterior compression', 'diaphragmatic eventration', #'consolidation', 
    'calcified densities', 'volume loss', 'single chamber device', 'vertebral compression', 'bullas', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'dual chamber device', 'mediastinic lipomatosis',
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
        if config['class'] == 'unseen':
            original_class += padchest_unseen_class
            dataset_cls = padchest_unseen_class
        elif config['class'] == 'rare':
            dataset_cls = padchest_rare
        else:
            dataset_cls = padchest_seen_class
        test_dataset = Padchest_Dataset(config['test_file'], root=config['root'], classes=dataset_cls)
        if 'pleural effusion' in dataset_cls:
            dataset_cls[dataset_cls.index('pleural effusion')] = 'effusion'
        original_class.extend(item for item in dataset_cls if item not in original_class)
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

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
        ) 
                 
    
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


    print("Creating model")
    if config['model'] == 'medklip':
        model = MedKLIP(config, disease_book_tokenizer)
    elif config['model'] == 'mavl':
        model = MAVL(config, ana_book_tokenizer, disease_book_tokenizer, concepts_book_tokenizer)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    # model = model.to(device)  

    print('Load model from checkpoint:', config['model_path'])
    checkpoint = torch.load(config['model_path'], map_location='cpu') 
    state_dict = checkpoint['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if config.get('decoder', '') == 'slot':
        state_dict = {k: v for k, v in state_dict.items() if 'regularizer_clf' not in k}
    state_dict = {k: v for k, v in state_dict.items() if 'temp' not in k}
    model.load_state_dict(state_dict) 

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    location_pred = torch.FloatTensor().to(device)
    print("Start testing")
    model.eval()
    mode = config.get('mode', 'feature')
    print("Testing mode ", mode)
    for i, sample in enumerate(test_dataloader):
        image = sample['image']
        if config['dataset'] == 'chexray':
            label = sample['label'][:, dataset_mapping].float().to(device)
        else:
            label = sample['label'].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            if config['model'] == 'mavl':
                pred_class, location, concept_features, pred_global, ensemble = model(input_image) #batch_size,num_class,dim
                preds_global = []
                if not config.get('same_feature', False):
                    for normal_idx in [0,1]:
                        normal = pred_global[:, :, [normal_idx]].repeat(1,1, len(original_class))
                        pred_global_ = torch.stack([normal, pred_global], dim=-1) # B, N_concepts, N_disease
                        pred_global_ = F.softmax(pred_global_, dim=-1)
                        preds_global.append(pred_global_)
                    pred_global = torch.stack(preds_global, dim=0).mean(dim=0)
                # pred_global = pred_global[:, :, MIMIC_mapping, 1]
            else:
                pred_class, location, concept_features = model(input_image) #batch_size,num_class,dim
                pred_global = None

            if config['model'] != 'mavl' or mode == 'feature':
                pred_class = F.softmax(pred_class.reshape(-1,2)).reshape(-1,len(original_class),2)
                pred_class = pred_class[:,MIMIC_mapping,1]
            elif mode == 'text':
                pred_class = pred_global[:, :, MIMIC_mapping, 1].mean(dim=1)
            pred = torch.cat((pred, pred_class), 0) 
    # location is 
    AUROCs = compute_AUCs(gt, pred, len(target_class))
    AUROC_avg = np.array(AUROCs).mean()
    # print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
    # for i in range(len(target_class)):
    #     print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))
    max_f1s = []
    accs = []
    recalls = []
    precisions = []
        # location_np = location_pred[i].cpu().numpy().tolist()
        # location_names = [ana_book[int(l)] for l in location_np]
        # print(location_names)

        
    for i in range(len(target_class)):
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].cpu().numpy()
        # location_np = location_pred[:, i].cpu().numpy().tolist()
        # location_names = [ana_book[int(i)] for i in location_np]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        rec, pre = recall[np.argmax(f1_scores)], precision[np.argmax(f1_scores)]

        recalls.append(rec)
        precisions.append(pre)
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))

    f1_avg = np.nanmean(np.array(max_f1s))   
    acc_avg = np.array(accs).mean()
    precision_avg = np.mean(precisions)
    recall_avg = np.mean(recalls)
    # Create a list of lists to represent the rotated table
    # table_data = [["Class Name"] + target_class,
    #             ["Accuracy"] + accs + [acc_avg],
    #             ["Max F1"] + max_f1s + [f1_avg],
    #             ["AUC ROC"] + AUROCs + [AUROC_avg],
    #             ["Average", acc_avg, f1_avg, AUROC_avg]]

    # # Define the table headers
    # headers = table_data[0]

    # # Create and print the rotated table
    # table = tabulate(table_data[1:], headers, tablefmt="grid")
    # print(table)

    # Create a list of lists to represent the normal table
    table_data = [[class_name, accuracy, max_f1, auc_roc, precision, recall]
                for class_name, accuracy, max_f1, auc_roc, precision, recall in zip(
                    target_class, accs, max_f1s, AUROCs, precisions, recalls)]

    # Add a row for average values
    average_row = ["Average", acc_avg, f1_avg, AUROC_avg, precision_avg, recall_avg]
    table_data.append(average_row)

    # Define the table headers
    headers = ["Class Name", "Accuracy", "Max F1", "AUC ROC", "Precision", "Recall"]

    # Create and print the table
    table = tabulate(table_data, headers, tablefmt="grid")
    with open(f"result_{config['model']}_{config['dataset']}_{mode}.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the headers
        csv_writer.writerow(headers)
        
        # Write the data rows
        csv_writer.writerows(table_data)

    print(table)
    log_csv = True
    if log_csv:
        model_name = '_'.join(config['model_path'].split('/')[-2:])
        csv_filename = f'results/{model_name}.csv'
        os.makedirs('results', exist_ok=True)
        dataset_name = f"{config['dataset']}_{config['mode']}"
        if config['dataset'] == 'padchest':
            dataset_name = f"{dataset_name}_{config['class']}"
        data = [dataset_name, acc_avg, f1_avg, AUROC_avg, precision_avg, recall_avg]
        header = ['Dataset',  "Accuracy", "Max F1", "AUC ROC", "Precision", "Recall"]
        log_to_csv(csv_filename, data, header)
        

    # print('The average f1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
    # print('The average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))
    # for i in range(len(target_class)):
    #     print('F1 of {} is {}'.format(target_class[i], max_f1s[i]))
    #     print('ACC of {} is {}'.format(target_class[i], accs[i]))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/MedKLIP_config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    parser.add_argument('--model_path', type=str, default='', help='model path')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    accelerator = Accelerator()
    if args.model_path:
        config['model_path'] = args.model_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != '-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    test(config)