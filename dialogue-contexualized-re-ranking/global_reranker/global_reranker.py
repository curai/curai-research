import re
import os
import json
import argparse
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval

import time
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool

import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoConfig,AutoModelForTokenClassification,NystromformerForTokenClassification,LongformerForTokenClassification
from transformers import get_cosine_schedule_with_warmup
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import label_ranking_average_precision_score,ndcg_score
from transformers.utils import logging
from sklearn.utils import shuffle
logging.set_verbosity(40)

from tqdm import tqdm
import wandb

from data_utils import *

import sys
sys.path.append('./allRank/')
from allrank.models.losses import approxNDCG, neuralNDCG, binary_listNet, lambdaLoss, listMLE, rankNet





def train(encoder,train_loader, dev_loader, args):
   

    encoder.to(device) 

    optimizer = AdamW(encoder.parameters(), 
                    lr=args.lr,weight_decay=args.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=1000,num_training_steps=len(train_loader)*args.epoch)
    

    best_perf = 0
    iterations = 0
    grad_acc = args.grad_acc
    
    if args.loss == 'neuralNDCG':
        criterion = neuralNDCG
    elif args.loss == 'approxNDCG':
        criterion = approxNDCG.approxNDCGLoss
    elif args.loss == 'listNet':
        criterion = binary_listNet
    elif args.loss == 'lambdaLoss':
        criterion = lambdaLoss
    elif args.loss == 'listMLE':
        criterion = listMLE
    elif args.loss == 'rankNet':
        criterion = rankNet
    elif args.loss == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()

    
    wandb.watch(encoder)
    running_loss = 0
    for e in range(args.epoch):
        for i, (batch, labels, relevance) in tqdm(enumerate(train_loader)):
            encoder.train()

          
            batch = {k:v.to(device) for k,v in batch.items()}
            print(batch['input_ids'].shape)
            logits = encoder(**batch).logits.squeeze()
            
            if args.loss == 'neuralNDCG':
                scores = torch.tanh(logits.unsqueeze(0))
            else:
                scores = logits.unsqueeze(0)

            mask = labels!=-100
            scores = torch.masked_select(scores,mask.to(device))
            relevance = torch.tensor(relevance).float()
            print(scores)
            print(relevance)
            
            
            loss = criterion(scores.unsqueeze(0),relevance.to(device))
            loss = loss/grad_acc
            running_loss += loss.item()
            
            loss.backward()

            if i%args.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if i!=0:
                    writer.add_scalar("charts/loss", running_loss, iterations)
                iterations += 1
                running_loss = 0
                            
            if iterations%args.saving_step == 0:
                ndcg,mAP,eval_loss = evaluate(encoder,dev_loader,args)
                writer.add_scalar("charts/nDCG", ndcg, iterations)
                writer.add_scalar("charts/mAP", mAP, iterations)
                writer.add_scalar("charts/eval_loss", eval_loss, iterations)
                
                iterations += 1

                # only save the best model
                if best_perf <= mAP:
                    best_perf = mAP

                    encoder.save_pretrained(args.out_dir)
                    

                    if args.track:
                        artifact = wandb.Artifact(args.exp_name, type="model")
                        artifact.add_dir(args.out_dir)
                        run.log_artifact(artifact)


     
def evaluate(encoder,eval_dataloader,args):

    if args.loss == 'neuralNDCG':
        criterion = neuralNDCG
    elif args.loss == 'approxNDCG':
        criterion = approxNDCG.approxNDCGLoss
    elif args.loss == 'listNet':
        criterion = binary_listNet
    elif args.loss == 'lambdaLoss':
        criterion = lambdaLoss
    elif args.loss == 'listMLE':
        criterion = listMLE
    elif args.loss == 'rankNet':
        criterion = rankNet
    elif args.loss == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()

    encoder.eval()
    with open('errors','w') as out:
        out.write('Errors: \n')
    ndcg = []
    mAP = []
    losses = []
    with torch.no_grad():
        for i, (batch, labels, relevance) in tqdm(enumerate(eval_dataloader)):
            try:
                batch = {k:v.to(device) for k,v in batch.items()}
                logits = encoder(**batch).logits.squeeze()
                
                
                # extracting logit scores from the mask tokens
                scores = logits
                preds = []
                for pred,pointer in zip(scores,labels.squeeze()): 
                    if pointer!=-100:
                        preds.append(pred.cpu().detach().numpy())
                #print(relevance)


                # calculate ndcg and map
                preds = np.array([preds])
                print(preds)
                relevance = np.array(relevance) 
                print(relevance)
                if len(relevance.shape)==1:
                    relevance = [relevance]
                losses.append(criterion(torch.tensor(preds).float(),torch.tensor(relevance).float()))
                ndcg.append(ndcg_score(relevance,preds))
                mAP.append(label_ranking_average_precision_score(relevance,preds))
            except Exception as e:
                with open('errors','a') as out:
                    out.write(f"{e}"+'\n')
                continue


    return np.mean(ndcg),np.mean(mAP), np.mean(losses)



def parse_args():

    parser = argparse.ArgumentParser(parents=[expm.base_parser()])
    parser.add_argument('--dataset_artifact', default=None, type=str,help='the url to the dataset artifact')
    parser.add_argument('--pretrained_model_artifact',default=None,type=str)
    parser.add_argument('--encoder',default='uw-madison/nystromformer-4096',type=str)
    parser.add_argument('--pretrained_weights',default=None,type=str)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1,type=int, help='For now, this must be 1')
    parser.add_argument('--grad_acc', default=32, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=200,type=int)
    parser.add_argument('--weight_decay',default=1e-6,type=float)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--n_turns',default=0,type=int)
    parser.add_argument('--max_length',default=4096,type=int)
    parser.add_argument('--out_dir',default=None,type=str)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--attention_type',default=None,type=str)
    parser.add_argument("--type_id",default=True,type=bool)
    parser.add_argument("--position_id_restart",default=False,type=bool)
    parser.add_argument('--random_initialization',action='store_true')
    parser.add_argument("--loss",default="BCE",type=str)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.track:
        run = wandb.init()
        writer = SummaryWriter(f"runs/{args.exp_name}_{time.time()}")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

        
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)


    
    if args.track:
        data_artifact = run.use_artifact(args.dataset_artifact, type='dataset')
        data_artifact_dir = data_artifact.download()
        dataset_path = f"{data_artifact_dir}"



    if args.pretrained_weights is not None:
        encoder = AutoModelForTokenClassification.from_pretrained(args.pretrained_weights,num_labels=1)
        print('Pretrained weights loaded.')
    else:
        if args.attention_type == 'full_nystromformer':
            if args.random_initialization:
                config = AutoConfig.from_pretrained(args.encoder)
                config.num_labels=1
                encoder = NystromformerForTokenClassification(config)
            else:
                print("Loading full Nystromformer...Initializing from checkppint: %s"%args.encoder)
                encoder = NystromformerForTokenClassification.from_pretrained(args.encoder,num_labels=1)

        elif args.attention_type == 'nystromformer':
            if args.random_initialization:
                config = AutoConfig.from_pretrained(args.encoder)
                config.num_labels=1
                config.segment_means_seq_len=args.max_length
                config.max_position_embeddings=args.max_length
                encoder = NystromformerForTokenClassification(config)
            else:
                print("Loading Nystromformer...Initializing from checkppint: %s"%args.encoder)
                encoder = NystromformerForTokenClassification.from_pretrained(args.encoder,
                                                              num_labels=1,
                                                              segment_means_seq_len=args.max_length,
                                                              max_position_embeddings=args.max_length)



    # load datasets
    train_data = pd.read_json(os.path.join(dataset_path,'train_.json'),lines=True)
    dev_data = pd.read_json(os.path.join(dataset_path,'dev.json'),lines=True)
    test_data = pd.read_json(os.path.join(dataset_path,'test.json'),lines=True)

    
    print('Preparing data...')
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)
        

    train_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length_labels=args.max_length,
                                        type_id=args.type_id,position_id_restart=args.position_id_restart)
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,collate_fn=train_collator)

    eval_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True,max_length_labels=args.max_length,
                                        type_id=args.type_id,position_id_restart=args.position_id_restart)
    dev_dataloader = DataLoader(dev_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
    
    
    # load models 
    print(next(iter(train_dataloader)))
    print(next(iter(dev_dataloader)))


    if args.train:
        train(encoder,train_dataloader,dev_dataloader,args)

    if args.eval:
        if args.pretrained_model_artifact:
            print('Loading pretrained model artifact...')
            model_artifact = run.use_artifact(args.pretrained_model_artifact,type='model')
            model_artifact_dir = model_artifact.download()
            model_path = f"{model_artifact_dir}"
        else:
            model_path = args.out_dir
        #encoder = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        #encoder = TransformerModel.from_pretrained(model_path).to(device)
        encoder = NystromformerForTokenClassification.from_pretrained(model_path).to(device)

        test_dataset = Dataset.from_pandas(test_data)    

        all_ndcg = []
        all_map = []
        for i in range(5):
            
            test_dataloader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
            ndcg,mAP,_ = evaluate(encoder,test_dataloader,args)
            writer.add_scalar("charts/test_nDCG", ndcg,i)
            writer.add_scalar("charts/test_mAP", mAP,i)

            all_ndcg.append(ndcg)
            all_map.append(mAP)

        writer.add_scalar("charts/mean_test_nDCG", np.mean(all_ndcg), i+1)
        writer.add_scalar("charts/mean_test_mAP", np.mean(all_map), i+1)

