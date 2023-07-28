import time
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
import expm
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel,  get_cosine_schedule_with_warmup
from transformers.utils import logging, ModelOutput
from accelerate import Accelerator
logging.set_verbosity(40)

import sys
sys.path.append("/home/ubuntu/experiments/state-spaces/")

import math
from omegaconf import DictConfig, OmegaConf
import hydra
import src.models.nn.utils as U
from src.utils import registry
import src.utils as utils

from tqdm import tqdm
import wandb


def load_data(config):
    if config.dataset._name_ == 'wiki':
        data = load_dataset('wikipedia',"20220301.en",cache_dir='../../../../cache')['train']
        data = data.train_test_split(test_size=config.dataset.test_size)
    return data['train'], data['test']


@dataclass
class CausalLMCollatorWithPadding:
    '''
    Tokenizing and padding input texts.
    Adapted from HuggingFace script
    '''
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 8192

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        texts = [feature['text'] for feature in features]

        batch = self.tokenizer(
                        texts,
                        padding=self.padding,
                        truncation=True,
                        max_length=self.max_length,
                        return_attention_mask=self.return_attention_mask,
                        return_tensors="pt",
                    )

        batch['labels'] = batch['input_ids'].masked_fill(batch.attention_mask.ne(1),-100)
        batch['labels'] = batch['labels'][:,1:]
        batch['input_ids'] = batch['input_ids'][:,:-1]
        batch['lengths'] = torch.sum(batch['attention_mask'],dim=-1).squeeze()
        del batch['attention_mask']
        assert batch['labels'].size(1)<=self.max_length
        
        return batch

@dataclass
class MaskedLMCollatorWithPadding:
    '''
    Tokenizing and padding input texts.
    Adapted from HuggingFace script
    '''
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 8192
    mlm_probability: Optional[float] = 0.15

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        texts = [feature['text'] for feature in features]

        batch = self.tokenizer(
                        texts,
                        padding=self.padding,
                        truncation=True,
                        max_length=self.max_length,
                        return_attention_mask=self.return_attention_mask,
                        return_tensors="pt",
                    )

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"])
            
        batch['lengths'] = torch.sum(batch['attention_mask'],dim=-1)
        del batch['attention_mask']
        del batch['token_type_ids']

        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class S4ModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    states: Optional[Tuple[torch.FloatTensor]] = None
    y: Optional[Tuple[torch.FloatTensor]] = None
    embeddings: Optional[torch.FloatTensor] = None




class S4Embedding(nn.Module):

    def __init__(self, n_tokens, d_model, rescale=True,  **kwargs):
        super().__init__()
        self.rescale = rescale
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(n_tokens, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-.5)

    def forward(self,x):
        x = self.embedding(x)
        
        if self.rescale:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            x = x * (self.d_model**0.5)
        return x

        
class S4Model(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        
        n_tokens = config.embedding.n_tokens
        d_model = config.model.d_model
        d_output = config.decoder.d_output
        tied = config.decoder.tied

        self.encoder = S4Embedding(**config.embedding)

        self.model = utils.instantiate(registry.model, config.model)

        self.decoder = nn.Linear(d_output,n_tokens)
        if tied:
            assert d_model == d_output
            self.decoder.weight = self.encoder.embedding.weight

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError


    def forward(self,input_ids: torch.LongTensor, lengths: torch.LongTensor=None, labels: torch.LongTensor=None) -> S4ModelOutput:
        '''
        For causal LM, the lengths parameter has no impact on the model
        '''

        embed = self.encoder(input_ids)
        
        y, state = self.model(embed, state=None, lengths=lengths)
        #print(y.shape)
        self._state = state
        logits = self.decoder(y)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return  S4ModelOutput(
                loss = loss,
                logits = logits,
                states = state,
                y = y,
                embeddings = embed
                )

    def generate(self, x_t: torch.tensor):
        '''
        function to generate a single token
        '''
        x_t = self.encoder(x_t).squeeze(0) # Potential edge case for encoders that expect (B, L, H)?
        #print(x_t.shape)
        x_t, state = self.model.step(x_t, state=self._state)
        #print(x_t.shape)
        self._state = state
        x_t = self.decoder(x_t)
        #print(x_t.shape)
        return x_t

def evaluate(model,dataloader,device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(**batch.to(device))
            val_loss.append(outputs.loss.cpu().detach().numpy())

    return np.mean(val_loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    '''
    This function is simply a placeholder to initialize expm to log experiments.
    All training arguments are stored in config_wiki_lm.yaml.
    '''
    parser = argparse.ArgumentParser(parents=[expm.base_parser()])
    parser.add_argument('--name', default='wiki_pretraining',type=str)
    return parser.parse_args()


@hydra.main(config_path="configs", config_name="config_wiki_lm.yaml")
def main(config: OmegaConf):

    
    args = parse_args()
    global run
    run = expm.init(args)
    global writer
    writer = SummaryWriter(f"runs/{args.exp_name}_{time.time()}")


    OmegaConf.set_struct(config, False)

    print(config.dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)
    config.embedding.n_tokens = len(tokenizer)

    train_data, eval_data = load_data(config)

    if config.trainer.task == 'causal':
        collator = CausalLMCollatorWithPadding(tokenizer=tokenizer,max_length=config.dataset.l_max)
    elif config.trainer.task == 'mlm':
        collator = MaskedLMCollatorWithPadding(tokenizer=tokenizer,max_length=config.dataset.l_max)

    train_dataloader = DataLoader(train_data,batch_size=1,collate_fn=collator)
    eval_dataloader = DataLoader(eval_data,batch_size=1,collate_fn=collator)
    
    print(next(iter(train_dataloader)))
    print(next(iter(eval_dataloader)))
    
    # training hyperparameters
    device = config.trainer.device
    grad_acc_step = config.trainer.accumulate_grad_batches
    fp16 = config.trainer.fp16
    eval_step = config.trainer.evaluation_step
    logging_step = config.trainer.log_every_n_steps

    

    accelerator = Accelerator(gradient_accumulation_steps=grad_acc_step,mixed_precision='fp16')

    model = S4Model(config)

    total_param = count_parameters(model)
    print('There are totally %s M parameters!'%(total_param/1000000))

    optimizer = AdamW(model.parameters(), 
                    lr=config.optimizer.lr,weight_decay=config.optimizer.weight_decay)
    
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                num_warmup_steps=config.scheduler.num_warmup_steps,
                                num_training_steps=config.scheduler.num_training_steps)


    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                                                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler    
                                                        )   

    total_steps = 0

    wandb.watch(model)
    for i,batch in tqdm(enumerate(train_dataloader)):
        with accelerator.accumulate(model):
            model.train()
            optimizer.zero_grad()
            outputs = model(**batch)

            if i%(logging_step*grad_acc_step)==0:
                total_steps += logging_step
                writer.add_scalar("charts/train_loss", outputs.loss.detach().cpu().numpy(),total_steps)
            
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()    

        
        if i%(eval_step*grad_acc_step) == 0:    
            eval_loss = evaluate(model,eval_dataloader,device)   
            writer.add_scalar("charts/eval_loss", eval_loss, total_steps)   
            accelerator.save_state(config.checkpoint.dirpath) 
                                            

            
            artifact = wandb.Artifact(config.checkpoint.dirpath, type="S4_pretrained_model")
            artifact.add_dir(config.checkpoint.dirpath)
            run.log_artifact(artifact)



if __name__ == "__main__":

    main()




    '''
    # code for generation

    input_ids = input_ids = torch.ones(1,1).long().to(device)
    model._reset_state(input_ids,device='cuda')
    for module in model.modules():
        if hasattr(module, 'setup_step'): 
            module.setup_step()
    model.eval()
    out = model.generate(input_ids)
    print(out)
    '''

# WANDB_PROJECT=history_taking_reranking WANDB_JOB_TYPE=Pretrain_S4 python train_S4_lm.py 