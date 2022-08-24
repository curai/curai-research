import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import collections
# from models.hard_concrete import HardConcrete
from models.focal_loss import BinaryFocalLossWithLogits

from transformers import AutoModel
import pdb

def pairwise_cosine_similarity(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def apply_along_dim(function, x, dim=0):
    return torch.stack([
        function(x_i) for x_i in torch.unbind(x, dim=dim)
    ], dim=dim)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, args):
        super(EncoderWrapper, self).__init__()
        self.normalize = args.normalize_final_hidden
        self.encoder = AutoModel.from_pretrained(args.encoder_name)
        self.pooling_method = args.hidden_states_pooling_method
        encoder_hidden_size = self.encoder.config.hidden_size
        self.projection_head = ProjectionHead(encoder_hidden_size, 128, 128)

    def forward(self, model_input):
        hidden = self.encoder(**model_input)[0][:, 0, :]
        projection = self.projection_head(hidden)
        normalized_projection = F.normalize(projection, dim=-1)
        return normalized_projection


class EntityMultiHeadAttention(nn.Module):
    def __init__(self, args, hidden_size):
        super(EntityMultiHeadAttention, self).__init__()

        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = args.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(args.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, entity_embeddings, hidden_states, attention_mask=None, output_attentions=False):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        query_layer = self.query(entity_embeddings).view(entity_embeddings.size(0), self.num_attention_heads, self.attention_head_size).permute(1, 0, 2)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer.unsqueeze(0), key_layer.transpose(-1, -2))


        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class EntityAttention(nn.Module):
    def __init__(self, args, hidden_size):
        super(EntityAttention, self).__init__()

        self.hidden_size = hidden_size
        self.attention_type = args.attention_type
        # self.attention_head_size = args.attention_head_size
        # self.all_head_size = self.num_attention_heads * self.attention_head_size

        if args.attention_type > 0:
            self.query = nn.Linear(hidden_size, hidden_size)

        if args.attention_type > 1:
            self.key = nn.Linear(hidden_size, hidden_size)
        # self.key = nn.Linear(hidden_size, self.all_head_size)
        # self.value = nn.Linear(hidden_size, )

        self.dropout = nn.Dropout(args.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, entity_embeddings, hidden_states, attention_mask=None, output_attentions=False):
        # value_layer = self.value(hidden_states)

        key = hidden_states

        if self.attention_type > 0:
            entity_embeddings = self.query(entity_embeddings)

        if self.attention_type > 1:
            key = self.key(key)


        attention_scores = torch.matmul(entity_embeddings.unsqueeze(0), key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer.unsqueeze(0), key_layer.transpose(-1, -2))


        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, hidden_states)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ContrastiveEntityExtractor(nn.Module):
    def __init__(self, args):
        super(ContrastiveEntityExtractor, self).__init__()
        self.normalize = args.normalize_final_hidden
        self.encoder = AutoModel.from_pretrained(args.encoder_name)
        self.use_contrastive_loss = args.use_contrastive_loss
        self.use_attention = args.use_attention
        self.use_classification_loss = args.use_classification_loss
        self.ignore_cls = args.ignore_cls
        self.pooling_method = args.hidden_states_pooling_method

        self.append_query = args.append_query
        self.use_projection_head = args.use_projection_head
        
        encoder_hidden_size = self.encoder.config.hidden_size

        if self.append_query:
            final_hidden_size = 2 * encoder_hidden_size
        else:
            final_hidden_size = encoder_hidden_size

        if self.use_attention:
            if args.use_multi_head:
                self.attention_layer = EntityMultiHeadAttention(args, encoder_hidden_size) 
                final_hidden_size = args.num_attention_heads * args.attention_head_size
            else:
                self.attention_layer = EntityAttention(args, encoder_hidden_size)

        if self.use_classification_loss:
            self.classifier = BinaryClassifier(final_hidden_size, args.classifier_hidden_size)

        if self.use_projection_head:
            self.projection_head = ProjectionHead(final_hidden_size, args.projection_hidden_size, args.projection_output_size)

        # self.classifier = MultipleBinaryClassifiers(final_hidden_size, args.classifier_hidden_size, self.n_classes, normalize=self.normalize)


        # self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # self.criterion = BinaryFocalLossWithLogits(reduction='mean')

    def forward(self, text_inputs, entities_inputs, return_hidden=False, return_loss_tensor=False, output_attentions=True):

        entity_representations = []
        attention = []
        logits = []


        hidden_states = self.encoder(**text_inputs)[0]
        attention_masks = text_inputs['attention_mask']


        for entity in entities_inputs:
            entity_cls = self.encoder(**entity)[0][:, 0, :]

            if self.ignore_cls:
                hidden_states = hidden_states[:, 1:, :]
                attention_masks = attention_masks[:, 1:]

            attention_output = self.attention_layer(entity_cls, hidden_states, attention_mask=attention_masks, output_attentions=output_attentions)
            attention_representation = attention_output[0]

            if self.append_query:
                attention_representation = torch.cat((attention_representation, entity_cls.unsqueeze(0)), dim=-1)

            if self.use_projection_head:
                projected_representation = self.projection_head(attention_representation)
                entity_representations.append(F.normalize(projected_representation, dim=-1).squeeze(0))
            else:
                entity_representations.append(F.normalize(attention_representation, dim=-1).squeeze(0))


            if output_attentions:
                attention.append(attention_output[1])

            if self.use_classification_loss:
                logits.append(self.classifier(attention_representation).squeeze(-1))




        output = collections.defaultdict(lambda x: None)

        if return_hidden:
            output['hidden_states'] = hidden_states

        output['entity_representations'] = entity_representations
        output['attention'] = attention
        output['logits'] = logits

        return output


class ContrastiveEntityClassifier(nn.Module):
    def __init__(self, args, labels, entity_embeddings):
        super(ContrastiveEntityClassifier, self).__init__()
        self.normalize = args.normalize_final_hidden
        self.encoder = AutoModel.from_pretrained(args.encoder_name)
        self.n_classes = len(labels)
        self.temperature = args.temperature
        self.use_contrastive_loss = args.use_contrastive_loss
        self.use_attention = args.use_attention
        # pdb.set_trace()
        self.pooling_method = args.hidden_states_pooling_method
        
        encoder_hidden_size = self.encoder.config.hidden_size

        self.entity_embeddings = nn.Parameter(entity_embeddings, requires_grad=True)

        final_hidden_size = encoder_hidden_size

        if self.use_attention:
            if args.use_multi_head:
                self.attention_layer = EntityMultiHeadAttention(args, encoder_hidden_size) 
                final_hidden_size = args.num_attention_heads * args.attention_head_size
            else:
                self.attention_layer = EntityAttention(args, encoder_hidden_size)

        if self.use_contrastive_loss:
            self.projection_head = ProjectionHead(encoder_hidden_size, 128, 128)

        self.classifier = MultipleBinaryClassifiers(final_hidden_size, args.classifier_hidden_size, self.n_classes, normalize=self.normalize)


        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # self.criterion = BinaryFocalLossWithLogits(reduction='mean')

    def forward(self, input_ids, token_type_ids, attention_mask, targets, return_hidden=False, return_loss_tensor=False, output_attentions=False):


        hidden_states = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        if self.use_attention:
            attention_output = self.attention_layer(self.entity_embeddings, hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = attention_output[0]
        else:
            if self.pooling_method == 'max': 
                hidden_states = torch.max(hidden_states * attention_mask.unsqueeze(-1), axis=1)[0]
            elif self.pooling_method == 'mean':
                hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)

        logits = self.classifier(hidden_states)

        output = collections.defaultdict(lambda x: None)

        if return_hidden:
            output['hidden_states'] = hidden_states

        if self.use_contrastive_loss:
            output['label_representations'] = F.normalize(self.projection_head(hidden_states), dim=-1)

        if output_attentions:
            output['attention'] = attention_output[1]

        if self.training:
            cls_loss = self.criterion(logits, targets)
            output['loss'] = (cls_loss,)
        else:
            output['logits'] = torch.sigmoid(logits)


        return output

class MultipleBinaryClassifiers(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, normalize=False):
        super(MultipleBinaryClassifiers, self).__init__()
        self.n_classes = n_classes
        self.normalize = normalize
        self.classifiers = nn.ModuleList(
            [BinaryClassifier(input_size, hidden_size, normalize=normalize) for _ in range(n_classes)]
        )

    def forward(self, x):
        logits = []

        for idx in range(self.n_classes):
            final_hidden_states = x[:, idx] if len(x.size()) == 3 else x
            y = self.classifiers[idx](final_hidden_states)
            logits.append(y)
        
        logits = torch.cat(logits, dim=-1) 
        

        return logits


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, normalize=False, shared=False):
        super(BinaryClassifier, self).__init__()
        self.shared = shared
        self.normalize = normalize

        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1, bias=True)
        self.init_parameters()
    
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0)

    def forward(self, x):
        if self.normalize:
            norm = x.norm(p=2, dim=-1, keepdim=True)
            x = x.div(norm.expand_as(x))

        x = self.linear1(x)
        x = self.relu(x)
        y = self.linear2(x)
        if self.shared:
            y = y.squeeze(-1)
        return y


class ProjectionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProjectionHead, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        y = self.linear2(x)
        return y
        