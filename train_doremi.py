'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
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

from models.blip2_vqa import blip2_vqa as blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result


class Doremi():
    def __init__(self, args):
        self.args = args
        with open(self.args.domain_config_path, 'r') as f:
            self.domain_config = json.load(f)

        self.train_domain_weights_dict = self.domain_config['train_domain_weights']
        self.eval_domain_weights_dict = self.domain_config['eval_domain_weights']

        self.domain_list = list(sorted(self.train_domain_weights_dict.keys()))
        self.sampling_weights = torch.tensor([self.train_domain_weights_dict[domain] for domain in self.domain_list])

        self.pertoken_scores = []
        self.token_masks = []
        self.domain_ids = []

        # we will take care of skipping in dataloader
        self.args.ignore_data_skip = True

    def write_weights(self, weights):
        self.model.update_counter += 1
        self.model.train_domain_weights[:] = weights
        self.model.avg_domain_weights[:] = (self.model.avg_domain_weights * (self.model.update_counter  - 1) + weights) / self.model.update_counter

    def read_weights(self):
        return self.model.train_domain_weights.clone()

    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update_domain_weights(self, scores, scores_mask, domain_ids):
        train_domain_weights = self.read_weights()

        scores = scores.detach()
        domain_ids = domain_ids.detach()

	if self.args.doremi_optimizer == 'doremiv1':
	    perdomain_scores = []
	    for domain_id in range(len(train_domain_weights)):
		domain_mask = (domain_ids == domain_id)
		perdomain_scores_mask = scores_mask[domain_mask]
		if domain_mask.sum() > 0:
		    curr_domain_scores = torch.clip(scores[domain_mask][perdomain_scores_mask], min=0).mean()
		else:
		    curr_domain_scores = self.model.perdomain_scores[domain_id]
		perdomain_scores.append(curr_domain_scores)
	    self.model.perdomain_scores[:] = torch.tensor(perdomain_scores)
	    log_new_train_domain_weights = torch.log(train_domain_weights) + self.args.reweight_eta * self.model.perdomain_scores
	    new_train_domain_weights = nn.functional.softmax(log_new_train_domain_weights, dim=0)
	    train_domain_weights = (1-self.args.reweight_eps) * new_train_domain_weights + self.args.reweight_eps / len(new_train_domain_weights)
	    self.write_weights(train_domain_weights)
	else:
            raise ValueError(f"DoReMi optimizer {self.args.doremi_optimizer} not supported")


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, question, answer, weights, n, domain_ids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      

        # doremi requires pertoken loss
        loss, pertoken_loss, reference_pertoken_loss, token_mask = model(image, question, answer, train=True, n=n, weights=weights, domain_ids=domain_ids, return_pertoken_losses=True)        
        excess_loss = pertoken_loss - reference_pertoken_loss

        doremi.pertoken_scores.append(excess_loss.detach())
        doremi.token_masks.append(token_mask.detach())
        doremi.domain_ids.append(domain_id.detach())

        if len(self.pertoken_scores) == args.gradient_accumulation_steps:
            pertoken_scores = torch.cat(doremi.pertoken_scores, dim=0)
            token_masks = torch.cat(doremi.token_masks, dim=0).bool()
            domain_ids = torch.cat(doremi.domain_ids, dim=0)

            # update domain weights
            doremi.update_domain_weights(pertoken_scores, token_masks, domain_ids)
            
            doremi.pertoken_scores = []
            doremi.token_masks = []
            doremi.domain_ids = []
	
        # compute the rescaled loss, divide by domain weights
	train_domain_weights = model.train_domain_weights().clone().to(pertoken_loss.device)
	# if doing non-uniform sampling, normalize by inverse sampling weight
	train_domain_weights = train_domain_weights / doremi.sampling_weights.to(train_domain_weights.device)
	train_domain_weights = train_domain_weights / train_domain_weights.sum()
	curr_domain_weights = train_domain_weights[inputs['domain_ids']].unsqueeze(-1).expand_as(pertoken_loss).detach()
	curr_domain_weights = curr_domain_weights * token_mask
	normalizer = curr_domain_weights.sum()

	token_mask = token_mask.detach().type(pertoken_loss.dtype)
	curr_domain_weights = curr_domain_weights / normalizer
	loss = (pertoken_loss * curr_domain_weights.detach()).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            print(question)
            answers = model(image, question, train=False, inference='generate') 
            
            #answers = model.generate("image": image, "prompt":question)
            
            for answer, ques_id in zip(answers, question_id):
                # ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})             
            
        elif config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})   

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']], # Rulin TODO bachify test
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    #### Model #### 
    print("Creating model")
    model = blip_vqa(args) 
    model = model.to(device)       
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    # define your reference model here
    if args.doremi_train:
        reference_model = blip_vqa(args)
        model.reference_model = reference_model

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0 
       
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 

        else:         
            break        
        
        if utils.is_main_process():     
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                    
            if epoch == 2 or epoch == 9:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()         

    #TODO: how to do validation in instruct tuning
    #vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
    #result_file = save_result(vqa_result, args.result_dir, 'vqa_result')  
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml') 
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', default=None, type=str, help='path to the saved ckpt')
    parser.add_argument('--model_type', default="blip2_vicuna", type=str, choices=["blip2_vicuna", "blip2_t5"])
    parser.add_argument('--train_llm', action='store_true', help='set the llm trainable during training')
    parser.add_argument('--train_qformer', action='store_true', help='set the qformer trainable during training')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
