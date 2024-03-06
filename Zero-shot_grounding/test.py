
from accelerate import Accelerator
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import RSNA2018_Dataset, SIIM_ACR_Dataset, Covid19_Dataset
from models.model_MedKLIP import MedKLIP
from models.model_MAVL import MAVL
from models.tokenization_bert import BertTokenizer
import torchshow as ts
from sklearn.metrics import precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

global accelerator

original_class = [
            'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]
def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def score_cal(labels,seg_map,pred_map, th=0.008):
    '''
    labels B * 1
    seg_map B *H * W
    pred_map B * H * W
    '''
    device = labels.device
    total_num = torch.sum(labels)
    mask = (labels==1).squeeze()
    seg_map = seg_map[mask,:,:].reshape(total_num,-1)
    pred_map = pred_map[mask,:,:].reshape(total_num,-1)
    one_hot_map = (pred_map > th)
    dot_product = (seg_map *one_hot_map).reshape(total_num,-1)
    
    # Calculate Recall and Precision
    recall = torch.sum(dot_product, dim=-1) / torch.sum(seg_map, dim=-1)
    precision = torch.sum(dot_product, dim=-1) / torch.sum(one_hot_map, dim=-1)
    accuracy = torch.sum(seg_map == one_hot_map, dim=-1) / dot_product.shape[-1]


    max_number = torch.max(pred_map,dim=-1)[0]
    point_score = 0
    for i,number in enumerate(max_number):
        temp_pred = (pred_map[i] == number).type(torch.int)
        flag = int((torch.sum(temp_pred * seg_map[i]))>0)
        point_score = point_score + flag
    iou_score = torch.sum(dot_product,dim = -1)/((torch.sum(seg_map,dim=-1)+torch.sum(one_hot_map,dim=-1))-torch.sum(dot_product,dim = -1))
    dice_score = 2*(torch.sum(dot_product,dim=-1))/(torch.sum(seg_map,dim=-1)+torch.sum(one_hot_map,dim=-1))
    f1_score = 2*(recall*precision) / (recall + precision)
    return total_num, point_score, iou_score.to(device), dice_score.to(device), recall, precision, accuracy


def compute_f1(labels, seg_maps, pred_maps):
    total_num = torch.sum(labels)
    mask = (labels==1).squeeze()
    seg_maps = seg_maps[mask,:,:].reshape(total_num,-1).cpu().numpy()
    pred_maps = pred_maps[mask,:,:].reshape(total_num,-1).cpu().numpy()
    # location_np = location_pred[:, i].cpu().numpy().tolist()
    # location_names = [ana_book[int(i)] for i in location_np]
    max_f1s, accs, pres, recs = [], [], [], []
    for i in range(total_num):
        seg_map, pred_map = seg_maps[i], pred_maps[i]
        precision, recall, thresholds = precision_recall_curve(seg_map, pred_map)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        rec, pre = recall[np.argmax(f1_scores)], precision[np.argmax(f1_scores)]
        acc = accuracy_score(seg_map, pred_map>max_f1_thresh)
        max_f1s.append(max_f1)
        accs.append(acc)
        pres.append(pre)
        recs.append(rec)
    return max_f1s, accs, pres, recs

def main(args, config):
    device = accelerator.device
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    #### Dataset #### 
    print("Creating dataset")
    if config['dataset'] == 'rsna':
        test_dataset =  RSNA2018_Dataset(config['test_file'], config['root'])
    elif config['dataset'] == 'siim':
        test_dataset = SIIM_ACR_Dataset(config['test_file'], config['root'])
    elif config['dataset'] == 'covid-r':
        test_dataset = Covid19_Dataset(config['test_file'], config['root'])
        original_class.append('covid19')
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=0,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )              
    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in original_class]
    print("Total diseases ", len(disease_book))


    ana_book = [ 'It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
            'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
            'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
            'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
            'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
            'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
            'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
            'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    # ana_book_tokenizer = get_tokenizer(tokenizer, ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer, disease_book).to(device)
    
    if 'concept_book' in config:
        concepts = json.load(open(config['concept_book'], 'r'))
        concepts = {i: concepts[i] for i in original_class}
        concepts_book = sum(concepts.values(), [])
        concepts_book_tokenizer = get_tokenizer(tokenizer, concepts_book).to(device)
    
    
    target_cls_idx = original_class.index(config['target'])
    print("Target class ", original_class[target_cls_idx])

    print("Creating model")
    if config['model'] == 'medklip':
        model = MedKLIP(config, disease_book_tokenizer)
    elif config['model'] == 'mavl':
        model = MAVL(config, disease_book_tokenizer, concepts_book_tokenizer)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)    
    # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    # model = model.to(device)  
    

    checkpoint = torch.load(config['model_path'], map_location='cpu') 
    state_dict = checkpoint['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if config['model'] == 'mavl':
        state_dict = {k: v for k, v in state_dict.items() if 'temp' not in k}
    # if config['decoder'] == 'slot':
    #     state_dict = {k: v for k, v in state_dict.items() if 'regularizer_clf' not in k}
    model.load_state_dict(state_dict)  
    print('load checkpoint from %s'%config['model_path'])
    print("Start testing")
    model.eval()
    n_disease, n_concepts = len(original_class), 9
    avg = True
    ths = np.arange(0.0001, 0.1, 0.00001)
    #ths = [0.008]
    #print("========= Eval for threshold = ", th, " ================")
    dice_score_A = [torch.FloatTensor().to(device)]*len(ths)
    iou_score_A = [torch.FloatTensor().to(device)]*len(ths)
    recall_A = [torch.FloatTensor().to(device)]*len(ths)
    precision_A = [torch.FloatTensor().to(device)]*len(ths)
    acc_A = [torch.FloatTensor().to(device)]*len(ths)
    total_num_A = [0] * len(ths)
    point_num_A = [0] * len(ths)

    max_f1s = []
    max_precisions = []
    max_recalls = []
    max_accs = []
    mode = config.get('mode', 'avg')
    # th = 0.008
    for i, sample in enumerate(test_dataloader):
        images = sample['image'].to(device)
        image_path = sample['image_path']
        batch_size = images.shape[0]
        labels = sample['label'].to(device)
        seg_map = sample['seg_map'][:,0,:,:].to(device) #B C H W

        with torch.no_grad():
            _ , ws = model(images)
            if avg:
                ws = (ws[-4] +ws[-3]+ws[-2]+ws[-1])/4
            else:
                ws = ws[-1]
                
            sz = int(np.sqrt(ws[0].shape[-1]))

            B = ws.shape[0]
            if config['model'] == 'mavl':
                ws = ws.view(B, n_disease, n_concepts, -1)[:, :, [0, 6, 7,8]]
                if mode == 'max':
                    ws = ws.max(2)[0]
                elif mode == 'avg':
                    ws = ws.mean(2)
                elif mode == 'global':
                    ws = ws[:, :, 0,:]

            ws = ws.reshape(batch_size,ws.shape[1], sz, sz)
            pred_map = ws[:, target_cls_idx,:,:]
            
            #pred_map = torch.from_numpy(pred_map.repeat(16, axis=1).repeat(16, axis=2)).to(device) #Final Grounding Heatmap
            pred_map = torch.nn.functional.interpolate(pred_map.unsqueeze(1), (images.shape[-2], images.shape[-1]), mode='bilinear', align_corners=True).squeeze(1)
            max_f1, acc, pre, rec = compute_f1(labels, seg_map, pred_map)
            max_f1s += max_f1
            max_accs += acc
            max_precisions += pre
            max_recalls += rec
            for idx, th in enumerate(ths):
                total_num, point_num, iou_score, dice_score, recall, precision, acc  = score_cal(labels,seg_map,pred_map, th=th) 
                total_num_A[idx] = total_num_A[idx] + total_num
                point_num_A[idx] = point_num_A[idx] + point_num
                dice_score_A[idx] = torch.cat((dice_score_A[idx],dice_score.float()),dim=0)
                iou_score_A[idx] = torch.cat((iou_score_A[idx],iou_score.float()),dim=0)
                recall_A[idx] = torch.cat((recall_A[idx], recall.float()),dim=0)
                precision_A[idx] = torch.cat((precision_A[idx], precision.float()),dim=0)
                acc_A[idx] = torch.cat((acc_A[idx], acc.float()),dim=0)
    
    f1_avg = np.array(max_f1s).mean()    
    acc_avg = np.array(max_accs).mean()
    precision_avg = np.mean(max_precisions)
    recall_avg = np.mean(max_recalls)

    dice_score_avg = [torch.mean(dice_score_A[idx]).item() for idx in range(len(ths))]
    iou_score_avg = [torch.mean(iou_score_A[idx]).item() for idx in range(len(ths))]
    recall_avg = [torch.mean(recall_A[idx]).item() for idx in range(len(ths))]
    precision_avg = [torch.mean(precision_A[idx]).item() for idx in range(len(ths))]
    acc_avg = [torch.mean(acc_A[idx]).item() for idx in range(len(ths))]
    idx = np.argmax(dice_score_avg)
        #if best
    print("Best threshold ", ths[idx])
    print('The average dice_score is {dice_score_avg:.5f}'.format(dice_score_avg=dice_score_avg[idx]))
    print('The average iou_score is {iou_score_avg:.5f}'.format(iou_score_avg=iou_score_avg[idx]))
    print('The average recall is {recall_avg:.5f}'.format(recall_avg = recall_avg[idx]))
    print('The average precision is {precision_avg:.5f}'.format(precision_avg=precision_avg[idx]))
    point_score = point_num_A[idx]/total_num_A[idx]
    print('The average point_score is {point_score:.5f}'.format(point_score=point_score))  
    print('The average acc is {acc_avg:.5f}'.format(acc_avg=acc_avg[idx]))                      
                    
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/Path/To/MedKLIP_config.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='1', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    accelerator = Accelerator()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)