import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from test_res_ft import test
from tensorboardX import SummaryWriter
import utils
from models.resnet import ModelRes_ft
from models.mavl import MAVL_ft
from test_res_ft import test
from dataset.dataset import Chestxray14_Dataset, MURA_Dataset, Chexpert_Dataset, SIIM_ACR_Dataset, RSNA2018_Dataset, LERA_Dataset, CovidCXR2_Dataset
from scheduler import create_scheduler
from optim import create_optimizer
from sklearn.metrics import roc_auc_score


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

def train(model, data_loader, optimizer, criterion, epoch, warmup_steps, device, scheduler, args,config,writer):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1000   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)

    
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = sample['image']
        label = sample['label'].float().to(device) #batch_size,num_class
        input_image = image.to(device,non_blocking=True)  

        optimizer.zero_grad()
        pred_class = model(input_image) #batch_size,num_class

        loss = criterion(pred_class,label)
        loss.backward()
        optimizer.step()    
        writer.add_scalar('loss/loss', loss, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()


def valid(model, data_loader, criterion,epoch,device,config,writer):
    model.eval()
    val_scalar_step = epoch*len(data_loader)
    val_losses = []
    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    for i, sample in enumerate(data_loader):
        image = sample['image']
        label = sample['label'].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image)
            val_loss = criterion(pred_class,label)
            pred_class = F.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class), 0)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    avg_val_loss = np.array(val_losses).mean()
    AUROCs = compute_AUCs(gt, pred,config['num_classes'])
    AUROC_avg = np.array(AUROCs).mean()
    return AUROC_avg


def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

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
    #### Dataset #### 
    print("Creating dataset")
    train_dataset = dataset_cls_name(config['train_file'], root= config['root'], is_train=True, pct=config['data_pct']) 
    print("Number of samples ", len(train_dataset))
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            #drop_last=True,
        )     
    
    val_dataset = dataset_cls_name(config['valid_file'], root=config['root'], is_train = False) 
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    print("Length dataloader ", len(train_dataloader))
    if config['model'] == 'medklip':
        model = ModelRes_ft(res_base_model = 'resnet50', out_size = len(dataset_cls))
    elif config['model'] == 'mavl':
        model = MAVL_ft(base_model = config['base_model'],out_size = len(dataset_cls))
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device) 

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    criterion = nn.BCEWithLogitsLoss()

    if config['pretrain']:
        checkpoint = torch.load(config['pretrain'], map_location='cpu')
        state_dict = checkpoint['model']
        model_dict = model.state_dict()
        model_checkpoint = {k:v for k,v in state_dict.items() if k in model_dict}
        model_dict.update(model_checkpoint)
        model.load_state_dict(model_dict)
        print(f"load pretrain_path from {config['pretrain']}")
        
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                      
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1    
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)


    print("Start training")
    start_time = time.time()

    best_val_acc = 0.0
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        train_stats = train(model, train_dataloader, optimizer, criterion,epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        for k, v in train_stats.items():
            train_loss_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        val_acc = valid(model, val_dataloader, criterion,epoch,device,config,writer)
        writer.add_scalar('loss/val_acc_epoch', val_acc, epoch)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_acc': val_acc
                        }                     
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if val_acc > best_val_acc:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'best_valid.pth'))  
            best_val_acc = val_acc
            args.model_path = os.path.join(args.output_dir, 'best_valid.pth')
            config['model_path'] = args.model_path
            test_auc, test_f1, test_acc = test(args,config)
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=test_auc) + "\n")
                f.write('The average F1 is {f1:.4f}'.format(f1=test_f1) + "\n")
                f.write('The average accuracy is {acc:.4f}'.format(acc=test_acc) + "\n")
        
        if epoch % 40 == 1 and epoch>1:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))         
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Path/To/Res_train.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='') 
    parser.add_argument('--pretrain_path', default='Path/To/checkpoint.pth')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data_pct', type=float)
    parser.add_argument('--gpu', type=str,default='1', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if config['pretrain']:
        args.output_dir = f"output_{config['dataset']}_{config['model']}_{int(config['data_pct']*100)}"
    else:
        args.output_dir = f"output_{config['dataset']}_scratch_{int(config['data_pct']*100)}"
    print(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args, config)