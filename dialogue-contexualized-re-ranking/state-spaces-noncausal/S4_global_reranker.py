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
import expm


import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel,NystromformerForMaskedLM,NystromformerForTokenClassification
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


from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from S4Models import S4GlobalRanker

from tqdm import tqdm
import wandb

import sys
sys.path.append('../../allRank/')
from allrank.models.losses import approxNDCG, neuralNDCG, binary_listNet, lambdaLoss




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

        contexts = [self.tokenizer(feature['contexts'])['input_ids'][1:] for feature in features]
        #token_type_ids = torch.zeros_like(len(contexts))
 
        
        questions = [self._question_masking(feature['full_candidates'], feature['scores'],self.eval)
                        for feature in features]

        batch = [{'input_ids':context+inp,
                  'type_ids':[0 for i in context]+[1 for i in inp],
                    'labels':[-100 for i in context]+tgt} 
                    for context,(inp,tgt,scores) in zip(contexts,questions)]
        batch = [{'input_ids':torch.tensor([self.tokenizer.cls_token_id]+b['input_ids'][-self.max_length_labels+1:]).long().unsqueeze(0),
                  'type_ids':torch.tensor([0]+b['type_ids'][-self.max_length_labels+1:]).long().unsqueeze(0),
                  'labels':torch.tensor([-100]+b['labels'][-self.max_length_labels+1:]).long().unsqueeze(0)} 
                    for b in batch]

        pointers = [b['labels'] for b in batch]
        

        scores = [scores for (inp,tgt,scores) in questions]
        return batch[0], pointers[0], scores
        

    
    def _question_masking(self,candidates,scores,eval=False):

        if eval == False:
            assert len(candidates)==len(scores)
            candidates, scores = shuffle(candidates,scores)

        inp = []
        tgt = []
        for i in candidates:
            e = self.tokenizer(i+': ',add_special_tokens=False)['input_ids']
            inp +=  e + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
            tgt += [-100 for i in e] + [1] + [-100]
        assert len(inp) == len(tgt)
        return (inp, tgt, scores)





def train(encoder,train_loader, dev_loader, args):
   

    encoder.to(device) 
    #accelerator = Accelerator(gradient_accumulation_steps=args.grad_acc)

    optimizer = AdamW(encoder.parameters(), 
                    lr=args.lr,weight_decay=args.weight_decay)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=1000,num_training_steps=len(train_loader)*args.epoch)
    #scaler = GradScaler()

    best_perf = 0
    iterations = 0
    grad_acc = args.grad_acc
    

    #encoder, optimizer, train_dataloader = accelerator.prepare(
    #         encoder, optimizer, train_loader
    #)
    wandb.watch(encoder)
    running_loss = 0
    for e in range(args.epoch):
        for i, (batch, pointers, relevance) in tqdm(enumerate(train_loader)):
            encoder.train()

            
            
            logits = encoder(input_ids=batch['input_ids'].to(device)
                            ).logits.squeeze()
            print(logits.shape)
            scores = logits.unsqueeze(0)
            mask = pointers!=-100
            scores = torch.masked_select(scores,mask.to(device))
            relevance = torch.tensor(relevance).float()
            print(scores)
            print(relevance)
            
            bce = torch.nn.BCEWithLogitsLoss()
            loss = bce(scores.unsqueeze(0),relevance.to(device))
            loss = loss/grad_acc
            running_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),args.grad_norm)
        
              

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
    with open('errors','w') as out:
        out.write('Errors: \n')
    ndcg = []
    mAP = []
    with torch.no_grad():
        for i, (batch, pointers, relevance) in tqdm(enumerate(eval_dataloader)):
            try:
                logits = encoder(input_ids=batch['input_ids'].to(device)
                                ).logits.squeeze()
                scores = logits
                preds = []
                for pred,pointer in zip(scores,pointers.squeeze()): 
                    if pointer!=-100:
                        preds.append(pred.cpu().detach().numpy())
                #print(relevance)
                preds = np.array([preds])
                print(preds)
                relevance = np.array(relevance) 
                print(relevance)
                if len(relevance.shape)==1:
                    relevance = [relevance]
                ndcg.append(ndcg_score(relevance,preds))
                mAP.append(label_ranking_average_precision_score(relevance,preds))
            except Exception as e:
                with open('errors','a') as out:
                    out.write(f"{e}"+'\n')
                continue


    return np.mean(ndcg),np.mean(mAP)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():

    parser = argparse.ArgumentParser(parents=[expm.base_parser()])
    parser.add_argument('--dataset_artifact', default=None, 
                        type=str,help='the url to the dataset artifact')
    parser.add_argument('--pretrained_model_artifact',default=None,type=str)
    parser.add_argument('--encoder',
                        default='bert-base-uncased',type=str)
    parser.add_argument('--pretrained_weights',default=None,type=str) 
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int, help='For now, this must be 1')
    parser.add_argument('--grad_acc', default=32, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=200,type=int)
    parser.add_argument('--weight_decay',default=0.1,type=float)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--n_turns',default=0,type=int)
    parser.add_argument('--max_length',default=4096,type=int)
    parser.add_argument('--out_dir',default="S4Ranker",type=str)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--include_findings',action='store_true')
    parser.add_argument('--grad_norm',default=10,type=float)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.track:
        run = wandb.init()
        writer = SummaryWriter(f"runs/{args.exp_name}_{time.time()}")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    
    with initialize(version_base=None, config_path="configs"):
    
        config=compose(config_name="wiki_noncausal_lm_config.yaml")
        print(OmegaConf.to_yaml(config))
        OmegaConf.set_struct(config, False)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    config.embedding.n_tokens = len(tokenizer)

   
    if args.track:
        data_artifact = run.use_artifact(args.dataset_artifact, type='dataset')
        data_artifact_dir = data_artifact.download()
        dataset_path = f"{data_artifact_dir}"


    # load datasets
    
    train_data = pd.read_json(os.path.join(dataset_path,'train_with_scores.json'),lines=True)
    dev_data = pd.read_json(os.path.join(dataset_path,'dev.json'),lines=True)
    test_data = pd.read_json(os.path.join(dataset_path,'test.json'),lines=True)


   
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)
    test_dataset = Dataset.from_pandas(test_data)        
 
    train_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length_labels=args.max_length)
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,collate_fn=train_collator)

    eval_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length_labels=args.max_length)
    dev_dataloader = DataLoader(dev_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
    test_dataloader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
    
    # load models 
    print(next(iter(train_dataloader)))
    print(next(iter(dev_dataloader)))

    encoder = S4GlobalRanker(config)
    print('There are %s M parameters.'%(count_parameters(encoder)/1000000))

    if args.pretrained_weights:
        print('Loading pretrained model weights...')
        model_artifact = run.use_artifact(args.pretrained_weights,type='S4_pretrained_model')
        model_artifact_dir = model_artifact.download()
        model_path = f"{model_artifact_dir}"

        state_dict = torch.load(os.path.join(model_path,'pytorch_model.bin'),map_location='cpu')
        encoder.load_state_dict(state_dict,strict=False)

    
    
    if args.train:
        train(encoder,train_dataloader,dev_dataloader,args)

    if args.eval:
        if args.pretrained_model_artifact:
            print('Loading trained model artifact...')
            model_artifact = run.use_artifact(args.pretrained_model_artifact,type='model')
            model_artifact_dir = model_artifact.download()
            model_path = f"{model_artifact_dir}"
        else:
            model_path = args.out_dir
        encoder = S4GlobalRanker(config)
        state_dict = torch.load(os.path.join(model_path,'pytorch_model.bin'))
        encoder.load_state_dict(state_dict,strict=True)
        encoder.to(device)

        test_dataloader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
    
        all_ndcg = []
        all_map = []
        for i in range(5):
            
            test_dataloader = DataLoader(test_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=eval_collator)
            ndcg,mAP = evaluate(encoder,test_dataloader,args)
            writer.add_scalar("charts/test_nDCG", ndcg,i)
            writer.add_scalar("charts/test_mAP", mAP,i)

            all_ndcg.append(ndcg)
            all_map.append(mAP)

        writer.add_scalar("charts/mean_test_nDCG", np.mean(all_ndcg), i+1)
        writer.add_scalar("charts/mean_test_mAP", np.mean(all_map), i+1)
