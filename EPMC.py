'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
# export NCCL_P2P_DISABLE=1
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 5, 6"
import ruamel_yaml as yaml
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
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_EPMC import EPMC
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from sklearn.metrics import accuracy_score, f1_score

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, output_dir, flag=0):
    # evaluate
    model.eval()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    header = 'Evaluation:'
    print_freq = 50    

    output_temp = output_dir + '/acc.json'

    num_counter = [0, 0, 0]
    acc_counter = [0, 0, 0]
    text_str = ''
    pre_counter = [0, 0, 0]

    for image, text, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device,non_blocking=True) 
        target = target.to(device,non_blocking=True) 
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)  
        prediction = model(image, text_input, target=target, train=False)

        _, pred_class = prediction.max(1)

        for i in range(target.size(0)):
            num_counter[target[i]] += 1
            pre_counter[pred_class[i]] += 1
            acc_counter[target[i]] += int(target[i]==pred_class[i])
            if not target[i]==pred_class[i]:
                text_str += '{}\n{}\t{}\n'.format(text[i], str(target[i].item()), str(pred_class[i].item())) 

        accuracy = (target==pred_class).sum() / target.size(0)

        pred_class_f1 = pred_class.cpu().numpy()
        target_f1 = target.cpu().numpy()
        f1 = f1_score(target_f1, pred_class_f1, average='macro')
        
        metric_logger.meters['f1'].update(f1.item(), n=image.size(0))

        metric_logger.meters['acc'].update(accuracy.item(), n=image.size(0))         
        
    json_str = 'positive:{:.2f}\n'.format(acc_counter[0]/num_counter[0])
    json_str += 'natural:{:.2f}\n'.format(acc_counter[1]/num_counter[1])
    json_str += 'negative:{:.2f}\n'.format(acc_counter[2]/num_counter[2])
    json_str += 'pre:{}\n'.format(pre_counter)

    json_str += text_str

    if flag:
        with open(output_temp, 'w', encoding='UTF-8') as f:
            f.write(json_str)
        f.close()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    
def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss = model(images, text_inputs, target=targets, train=True) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

    
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    print("Creating dataset")
    datasets = create_dataset('msd', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
        
    [train_loader, val_loader, test_loader] = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size']]*3,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    print("Creating model")
    model = EPMC(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, fusion_encoder=args.fusion_encoder)
   
    model = model.to(device)   


    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']               
        model.load_state_dict(state_dict, strict=False)    

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    if args.eval:
        test_stats = evaluate(model, test_loader, tokenizer, device, config, args.output_dir, 1)
        print("eval ok")
        return
        

    print("Start training")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        val_stats = evaluate(model, val_loader, tokenizer, device, config, args.output_dir)
        test_stats = evaluate(model, test_loader, tokenizer, device, config, args.output_dir)

        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_stats['acc'])>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(val_stats['acc'])
                    best_epoch = epoch
        
        if args.evaluate:
            break
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/SC.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='output/SC')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--fusion_encoder', default='gaunernst/bert-mini-uncased')
    parser.add_argument('--evaluate', action='store_true')  
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
