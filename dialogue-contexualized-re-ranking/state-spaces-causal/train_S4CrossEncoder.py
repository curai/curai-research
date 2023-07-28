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
from transformers import AutoTokenizer, AutoModel,NystromformerForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from S4Models import S4CrossEncoder
from sklearn.metrics import label_ranking_average_precision_score,ndcg_score
from transformers.utils import logging
from sklearn.utils import shuffle
logging.set_verbosity(40)

from tqdm import tqdm
import wandb



@dataclass
class DataCollatorWithPadding:
    '''
    Tokenizing and padding input texts.
    Adapted from HuggingFace script
    '''
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = 4096
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    eval: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        
        #token_type_ids = torch.zeros_like(len(contexts))
 
        if self.eval==True:
            contexts = [self.tokenizer(feature['contexts'])['input_ids'] for feature in features][0]
            candidates = [self.tokenizer(candidate)['input_ids']
                          for feature in features for candidate in feature['full_candidates']]
            scores = [feature['scores'] for feature in features]

            batch = [{'input_ids':contexts + q}
                     for q in candidates]

            batch = [{'input_ids':torch.tensor(i['input_ids'][-self.max_length_labels:]).long().unsqueeze(0)}
                      for i in batch]
            return batch, scores
        else:        
            
            batch = []
            labels = []
            for feature in features:
                context = self.tokenizer(feature['contexts'])['input_ids']
                question = self.tokenizer(feature['questions'])['input_ids']
                negative = self.tokenizer(feature['negatives'][0])['input_ids']

                positive = context + question
                positive = positive[-self.max_length_labels:]
                #print(positive)
                positive = {'input_ids': torch.tensor(positive).long()}
                batch.append(positive)
                labels.append(1)

                negative = context + negative
                negative = negative[-self.max_length_labels:]
                negative = {'input_ids': torch.tensor(negative).long()}
                batch.append(negative)
                labels.append(0)
            
            
            batch = tokenizer.pad(
                    batch,
                    padding=self.padding,
                    return_attention_mask=self.return_attention_mask,
                    return_tensors="pt"
                    )
            batch['labels'] = torch.tensor(labels).long()
            batch['lengths'] = torch.sum(batch['attention_mask'],dim=-1).squeeze()
            del batch['attention_mask']

            return batch


def train(encoder,train_loader, dev_loader, args):
   
    print(device)
    encoder.to(device) 
    
    optimizer = AdamW(encoder.parameters(), 
                    lr=args.lr,weight_decay=args.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=1000,num_training_steps=len(train_loader)*args.epoch)

    best_perf = 0
    iterations = 0
    grad_acc = args.grad_acc
    
    wandb.watch(encoder)
    running_loss = 0
    for e in range(args.epoch):
        for i, batch in tqdm(enumerate(train_loader)):
            encoder.train()

            =
        
            loss = encoder(input_ids=batch['input_ids'].to(device),
                            lengths=batch['lengths'].to(device),
                            labels=batch['labels'].to(device)).loss
            
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
                ndcg,mAP = evaluate(encoder,dev_loader,args)
                writer.add_scalar("charts/nDCG", ndcg, iterations)
                writer.add_scalar("charts/mAP", mAP, iterations)
                
                iterations += 1

                # only save the best model
                if best_perf <= mAP:
                    best_perf = mAP
                    if not os.path.exists(args.out_dir):
                        os.mkdir(args.out_dir)
                    torch.save(encoder.state_dict(),os.path.join(args.out_dir,'pytorch_model.bin'))
                    

                    if args.track:
                        artifact = wandb.Artifact(args.exp_name, type="model")
                        artifact.add_dir(args.out_dir)
                        run.log_artifact(artifact)


     
def evaluate(encoder,eval_dataloader,args):
    encoder.eval()

    ndcg = []
    mAP = []
    with torch.no_grad():
        for i, (batch, relevance) in tqdm(enumerate(eval_dataloader)):
            try:
                preds = [torch.softmax(encoder(input_ids=instance['input_ids'].to(device)).logits,dim=-1).squeeze()[1] for instance in batch] 
                # q0 = question_encoder(**question_batch.to(device)).pooler_output
                
                preds = np.array([i.cpu().detach().numpy() for i in preds])
                print(relevance)
                print(preds)
                relevance = np.array(relevance) 
                if len(relevance.shape)==1:
                    relevance = [relevance]
                ndcg.append(ndcg_score(relevance,[preds]))
                mAP.append(label_ranking_average_precision_score(relevance,[preds]))
            except Exception as e:
                with open('errors','a') as out:
                    out.write(f"{e}"+'\n')
                continue


    return np.mean(ndcg),np.mean(mAP)




def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_artifact', default=None', 
                        type=str,help='the url to the dataset artifact')
    parser.add_argument('--eval_pretrained_model_artifact',default=None,type=str)
    parser.add_argument('--pretrained_model_artifact',default=None,type=str)
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train_batch_size', default=2, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int, help='For now, this must be 1')
    parser.add_argument('--grad_acc', default=16, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=200,type=int)
    parser.add_argument('--weight_decay',default=1e-6,type=float)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--n_turns',default=0,type=int)
    parser.add_argument('--max_length',default=2048,type=int)
    parser.add_argument('--out_dir',default="S4CrossEncoders",type=str)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--include_findings',action='store_true')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.track:
        run = wandb.init()
        writer = SummaryWriter(f"runs/{args.exp_name}_{time.time()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with initialize(version_base=None, config_path="configs"):
    
        config=compose(config_name="config_wiki_lm.yaml")
        print(OmegaConf.to_yaml(config))
        OmegaConf.set_struct(config, False)

        
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
    config.embedding.n_tokens = len(tokenizer)


    if args.track:
        data_artifact = run.use_artifact(args.dataset_artifact, type='dataset')
        data_artifact_dir = data_artifact.download()
        dataset_path = f"{data_artifact_dir}"

    # load datasets
    
    train_data = pd.read_json(os.path.join(dataset_path,'train_hxt.json'),lines=True)
    dev_data = pd.read_json(os.path.join(dataset_path,'dev.json'),lines=True)
    test_data = pd.read_json(os.path.join(dataset_path,'test.json'),lines=True)

]
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)
    test_dataset = Dataset.from_pandas(test_data)        
 
    train_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True,max_length_labels=args.max_length)
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,collate_fn=train_collator)

    eval_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, eval=True,max_length_labels=args.max_length)
    dev_dataloader = DataLoader(dev_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
    test_dataloader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
    
   
    print(next(iter(train_dataloader)))
    print(next(iter(dev_dataloader)))

    # load models 
    encoder = S4CrossEncoder(config)

    if args.pretrained_model_artifact is not None:
        print('Loading pretrained model artifact...')
        model_artifact = run.use_artifact(args.pretrained_model_artifact,type='S4_pretrained_model')
        model_artifact_dir = model_artifact.download()
        model_path = f"{model_artifact_dir}"
        state_dict = torch.load(os.path.join(model_path,'pytorch_model.bin'))
        encoder.load_state_dict(state_dict,strict=False)
        print('Pretrained weights loaded.')
    
    
    if args.train:
        train(encoder,train_dataloader,dev_dataloader,args)

    if args.eval:
        if args.eval_pretrained_model_artifact:
            print('Loading eval pretrained mdoel artifact...')
            model_artifact = run.use_artifact(args.eval_pretrained_model_artifact,type='S4_pretrained_model')
            model_artifact_dir = model_artifact.download()
            model_path = f"{model_artifact_dir}"
        else:
            model_path = args.out_dir
        encoder = S4CrossEncoder(config)
        state_dict = torch.load(os.path.join(model_path,'pytorch_model.bin'))
        encoder.load_state_dict(state_dict,strict=False)
        encoder.to(device)
        ndcg,mAP = evaluate(encoder,test_dataloader,args)
        writer.add_scalar("charts/test_nDCG", ndcg, 1)
        writer.add_scalar("charts/test_mAP", mAP, 1)


