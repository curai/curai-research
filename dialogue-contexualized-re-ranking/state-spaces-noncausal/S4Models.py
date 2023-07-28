import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import registry
import src.utils as utils

from transformers.utils import logging, ModelOutput
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

@dataclass
class S4ModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    states: Optional[Tuple[torch.FloatTensor]] = None
    y: Optional[Tuple[torch.FloatTensor]] = None
    embeddings: Optional[torch.FloatTensor] = None


@dataclass
class S4CrossEncoderOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class S4RankerOutput(ModelOutput):
    logits: torch.FloatTensor = None




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

        
class S4LMModel(nn.Module):

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


    def forward(self,input_ids: torch.LongTensor, lengths: torch.LongTensor=None, labels: torch.LongTensor=None, rate=1) -> S4ModelOutput:
        '''
        For causal LM, the lengths parameter has no impact on the model
        '''

        embed = self.encoder(input_ids)
        
        y, state = self.model(embed, state=None, lengths=lengths, rate=rate)
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
        print(x_t.shape)
        x_t, state = self.model.step(x_t, state=self._state)
        print(x_t.shape)
        self._state = state
        x_t = self.decoder(x_t)
        print(x_t.shape)
        return x_t


   
class S4CrossEncoder(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        
        self.n_tokens = config.embedding.n_tokens
        self.d_model = config.model.d_model
        self.d_output = config.decoder.d_output
        

        self.encoder = S4Embedding(**config.embedding)

        self.model = utils.instantiate(registry.model, config.model)

        self.decoder = nn.Sequential(
                        nn.Linear(self.d_model,self.d_model),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.d_model,2)
                                        )


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


    def forward(self,input_ids: torch.LongTensor, lengths: torch.LongTensor=None, labels:torch.tensor=None, rate=4) -> torch.tensor:
        

        embed = self.encoder(input_ids)

        y, state = self.model(embed, state=None, lengths=lengths, rate=rate)
        #print(y.shape)
        #self._state = state
        if lengths is not None:
            y = torch.cat([torch.mean(rep[:l],dim=0).unsqueeze(0) for rep,l in zip(y,lengths)])
        else:
            y = torch.mean(y,dim=1).squeeze()
        logits = self.decoder(y)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.d_output), labels.view(-1))

        return  S4CrossEncoderOutput(
                    loss=loss,
                    logits=logits
                )
                


class S4GlobalRanker(nn.Module):

    def __init__(self,config) -> None:
        super().__init__()
        
        self.n_tokens = config.embedding.n_tokens
        self.d_model = config.model.d_model
        self.d_output = 1

        self.encoder = S4Embedding(**config.embedding)

        #self.type_embedding = nn.Embedding(config.dataset.n_types, self.d_model)

        self.model = utils.instantiate(registry.model, config.model)

        #self.dropout = nn.Dropout(config.model.dropout)
        self.projector = nn.Linear(self.d_model,1)
                                        

        

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


    def forward(self,input_ids: torch.LongTensor, 
                lengths: torch.LongTensor=None, 
                #token_type_ids:torch.tensor=None, 
                rate=1) -> torch.tensor:
        

        embed = self.encoder(input_ids)

        #type_emb = self.type_embedding(token_type_ids)

        #embed += type_emb

        y, state = self.model(embed, state=None, lengths=lengths, rate=rate)
        #print(y.shape)
        #self._state = state
        
        logits = self.projector(y)

        return  S4RankerOutput(
                    logits=logits
                )
                
