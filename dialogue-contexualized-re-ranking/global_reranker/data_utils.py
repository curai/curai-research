import re
import os
import json
import argparse
import pandas as pd
import numpy as np
from ast import literal_eval


import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Union
import torch.nn.functional as F
from transformers.utils import logging
from sklearn.utils import shuffle
logging.set_verbosity(40)

from tqdm import tqdm




def process_questions(df):
    '''
    Deprecated
    '''

    all_candidates = []
    for cand, neg, rel in tqdm(zip(df['candidates'].tolist(),df['negatives'].tolist(),df['rel_questions'])):
        if len(cand)==0 and len(neg)==0:
            all_candidates.append([])
        elif len(cand)>=len(neg):
            if type(cand[0])==tuple or type(cand[0])==list:
                all_candidates.append(list(set([i[0] for i in cand]).difference(set(rel))))
            else:
                all_candidates.append(list(set([i for i in cand]).difference(set(rel))))
        elif len(cand)<len(neg):
            all_candidates.append(list(set([i for i in neg]).difference(set(rel))))
        else:
            raise Exception

    return all_candidates

@dataclass
class DataCollatorWithPadding:
    '''
    Tokenizing and padding input texts.
    Adapted from HuggingFace script
    '''
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = False
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = 4096
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    eval: bool = False
    type_id: bool = True
    position_id_restart: bool = False


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        #tokenize the dialogue context
        contexts = [self.tokenizer(feature['contexts'])['input_ids'][1:] for feature in features]
        #token_type_ids = torch.zeros_like(len(contexts))
        context_position_ids = [list(range(1,len(context)+1))for context in contexts]
 
        # tokenize a list of questions
        questions = [self._question_masking(feature['full_candidates'], feature['scores'],self.eval)
                        for feature in features]

        # create inputs to the model by concatenating the context and the questons
        batch = [{'input_ids':context+inp,
                  'type_ids':[0 for i in context]+[1 for i in inp],
                  'position_ids': context_pos + pos, 
                    'labels':[-100 for i in context]+tgt} 
                    for context,context_pos,(inp,tgt,pos,scores) in zip(contexts,context_position_ids,questions)]
        batch = [{'input_ids':torch.tensor([self.tokenizer.cls_token_id]+b['input_ids'][-self.max_length_labels+1:]).long().unsqueeze(0),
                  'token_type_ids':torch.tensor([0]+b['type_ids'][-self.max_length_labels+1:]).long().unsqueeze(0),
                  'position_ids':torch.tensor([0]+b['position_ids'][-self.max_length_labels+1:]).long().unsqueeze(0),
                  'labels':torch.tensor([-100]+b['labels'][-self.max_length_labels+1:]).long().unsqueeze(0)} 
                    for b in batch]

        # padding
        if len(batch)==1:# and not self.padding:
            batch = batch[0]
        else:
            n = len(batch)
            batch_padded = {}
            input_ids = torch.zeros(n,self.max_length_labels)
            type_ids = torch.zeros(n,self.max_length_labels)
            labels = torch.zeros(n,self.max_length_labels)
            position_ids = torch.zeros(n,self.max_length_labels)
            mask = torch.ones(n,self.max_length_labels)

            for i in range(n):
                length = batch[i]['labels'].size(1) 

                
                input_ids[i,:length] = batch[i]['input_ids'] 
                
                type_ids[i,:length] = batch[i]['token_type_ids'] 

                position_ids[i,:length] = batch[i]['position_ids']

                labels[i,:length] = batch[i]['labels'] 
                labels[i,length:] = -100
                
                mask[i,:length] = 1.0
                mask[:,length:] = 0.0


                
            batch_padded['input_ids'] = input_ids.long()
            batch_padded['attention_mask'] = mask.long()
            batch_padded['labels'] = labels.long()
            batch_padded['token_type_ids'] = type_ids.long()
            batch_padded['position_ids'] = position_ids.long()

            batch = batch_padded

        scores = [scores for (inp,tgt,pos,scores) in questions]
        labels = batch['labels']
        del batch['labels']

        if not self.type_id:
            del batch['token_type_ids']
        if not self.position_id_restart:
            del batch['position_ids']

        return batch, labels, scores

    
    def _question_masking(self,candidates,scores,eval=False):
        '''
        # concatenate questions together and generate the masks
        Args: 
            candidates: list[str]
                    a list of questions
            scores: list[int]
                    a list of ground truth scores (binary)
        Return:
            inp: list[int]
                tokenized questions that will be used as model input
            tgt: list[int]
                a mask that indicates the location of mask tokens
            scores: list[int]
                a list of ground truth scores (binary)
        '''
        if eval == False:
            assert len(candidates)==len(scores)
            candidates, scores = shuffle(candidates,scores)

        inp = []
        tgt = []
        pos = []
        for i in candidates:
            e = self.tokenizer(i+': ',add_special_tokens=False)['input_ids']
            inp +=  e + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
            tgt += [-100 for i in e] + [1] + [-100]
            pos += list(range(len(e)+2))
        assert len(inp) == len(tgt)
        return (inp, tgt, pos, scores)


