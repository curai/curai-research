import os
import csv
import json
import torch
import random
import argparse
import sqlalchemy
import collections
import numpy as np
import pandas as pd
import re

from os.path import join as pjoin
from utils.logger import init_logger
from utils.helpers import pairwise_cosine_similarity, merge_subword_tokens, add_name_ent
from utils.data import get_dataloaders_ner, process_json_file_ner
from utils.visualize import generate_heatmap, visualize_entity_synonyms
from models.contrastive_classifier import ContrastiveEntityExtractor, EncoderWrapper
from models.optimizers import build_optim
from models.losses import SupConLoss
from models.focal_loss import BinaryFocalLossWithLogits
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from itertools import combinations
# from torch.optim import SGD
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import spacy
from spaczz.matcher import FuzzyMatcher


# from curai.kb.utils import load_curai_kb, load_kb_item_dict
# from curai.kb.enums import SnomedAttrType, SourceType
# from curai.nlp.stop_words import stopwords
# from curai.nlp.text_preprocessor import TextPreprocessor
# from curai.entity_extractor.entity_recognizer import RuleBasedEntityRecognizer
import pdb


pruned_stopwords_list = {
    'no',
    'denies',
    'has',
    'she',
    'her',
    'unsure',
    'tried',
    'he',
    'had',
    'is',
    'was',
    'on',
    'been',
    'a',
    'the',
    'reports',
    'to',
    'being',
    'any',
    'for',
    'taking',
    'having',
}

# stopwords = stopwords.union(pruned_stopwords_list)
stopwords = pruned_stopwords_list

# def init_rule_based_ner():
#     # Initialize NER
#     curai_kb = load_curai_kb(load_dxplain=False, load_labs=False)
#     item_dict = curai_kb.get_item_dict([SnomedAttrType.FINDING, SnomedAttrType.CLASS, SnomedAttrType.BASE])
#     concept2synonyms = collections.defaultdict(list)
#     name2id = collections.defaultdict(lambda: None)
#     for cid, concept in item_dict.items():
#         name2id[concept.name] = cid
#         for synonym in concept.synonyms:
#             concept2synonyms[concept.name].append(synonym)
#             name2id[synonym] = cid
#     kb_item_dict = load_kb_item_dict(item_dict)
#     er = RuleBasedEntityRecognizer(kb_item_dict, TextPreprocessor(), stopwords)
#     return er, name2id, concept2synonyms

def pretrain_entity_embeddings(args):
    """
    Pretrain the concept embeddings using supervised contrastive learning
    """
    save_dir = pjoin(args.checkpoints_dir, 'encoders')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)


    model = EncoderWrapper(args)

    device = args.device
    model = model.to(device)

    if not os.path.isfile(args.processed_data_path):
        processed_data = process_json_file_ner(args.json_data_path, tokenizer, args.k_folds)
        torch.save(processed_data, args.processed_data_path)
    else:
        processed_data = torch.load(args.processed_data_path)

    entity_inputs = processed_data['entity_synonyms']

    KB_entities = [] 

    # Entities sorted by the number of synonyms
    sorted_entity_list = sorted([(k, len(v['input_ids'])) for k, v in entity_inputs.items()], key=lambda x: x[1], reverse=True)
    contrastive_criteria = SupConLoss(contrast_mode='one')

    model.train()
    args.lr = 0.002
    optimizer = build_optim(args, model)

    # TSNE plot for un-finetuned embeddings (CLS)
    # tsne_plot_path = pjoin('results', 'tsne', f'{args.encoder}_epoch_0.png')
    # visualize_entity_synonyms(sorted_entity_list, model.encoder, entity_inputs, tsne_plot_path)


    max_pos = 20
    n_neg = 100

    prev_best = 0.
    best_ckpt = None

    for epoch in range(20):
        label_order = [x[0] for x in sorted_entity_list]
        random.shuffle(label_order)
        loss = 0.
        with tqdm(total=len(label_order)) as pbar:
            for i, entity_label in enumerate(label_order):
                pos_inputs = entity_inputs[entity_label]

                pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}

                if len(pos_inputs['input_ids']) > max_pos:
                    synonym_idx = random.sample(range(len(pos_inputs['input_ids'])), max_pos)
                    pos_inputs = {k: v[synonym_idx] for k, v in pos_inputs.items()}

                # Sampling for negative training examples
                negative_entity_names = [sorted_entity_list[idx][0] for idx in random.sample(range(len(entity_inputs)), n_neg)]
                negative_entity_input = collections.defaultdict(list)
                for entity_name in negative_entity_names:
                    synonym_idx = random.sample(range(len(entity_inputs[entity_name]['input_ids'])), 1)[0]

                    for k, v in entity_inputs[entity_name].items():
                        negative_entity_input[k].append(entity_inputs[entity_name][k][synonym_idx])
                negative_entity_input['input_ids'] = pad_sequence(negative_entity_input['input_ids'], batch_first=True, padding_value=0).to(device)
                negative_entity_input['token_type_ids'] = pad_sequence(negative_entity_input['token_type_ids'], batch_first=True, padding_value=0).to(device)
                negative_entity_input['attention_mask'] = pad_sequence(negative_entity_input['attention_mask'], batch_first=True, padding_value=0).to(device)
                pos_output = model(pos_inputs)
                neg_output = model(negative_entity_input)

                embeddings = torch.cat((pos_output, neg_output))
                labels = torch.tensor([1 for _ in range(len(pos_output))] + [0 for _ in range(len(neg_output))]).cuda()
                loss = contrastive_criteria(embeddings.unsqueeze(1), labels=labels)
                pbar.set_description(f"Loss: {round(loss.item(), 3)}")
                loss.backward()

                if (i + 1) % 32 == 0:
                    optimizer.step()
                    model.zero_grad()

                pbar.update(1)

        # TSNE plot for each epoch
        # tsne_plot_path = pjoin('results', 'tsne', f'{args.encoder}_epoch_{epoch + 1}.png')
        # visualize_entity_synonyms(sorted_entity_list, model.encoder, entity_inputs, tsne_plot_path)



        all_entity_embeddings = []
        n_labels = len(entity_inputs)
        current_label = 0
        labels = []
        for entity_label, model_inputs in entity_inputs.items():
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            with torch.no_grad():
                entity_embeddings = model(model_inputs)
            all_entity_embeddings.append(entity_embeddings.cpu())
            labels.extend([current_label for i in range(len(entity_embeddings))])
            current_label += 1
        
        all_entity_embeddings = torch.cat(all_entity_embeddings, dim=0)
        n_embeddings = len(all_entity_embeddings)
        labels = torch.tensor(labels).view(-1, 1)
        similarities = pairwise_cosine_similarity(all_entity_embeddings, all_entity_embeddings)

        pos_mask = torch.eq(labels, labels.T).float()
        neg_mask = 1 - pos_mask
        pos_mask = pos_mask - torch.eye(n_embeddings)

        pos_sims = ((similarities * pos_mask).sum() / pos_mask.sum()).item()
        neg_sims = ((similarities * neg_mask).sum() / neg_mask.sum()).item()
        
        diff = pos_sims - neg_sims

        if diff > prev_best:
            prev_best = diff
            save_name = f"rfe_{args.encoder}_lr{args.lr}_epoch{epoch+1}_{round(diff, 3)}.pt"
            save_path = pjoin(save_dir, save_name)
            torch.save(model.encoder.state_dict(), save_path)
            print(diff)
            best_ckpt = save_path

    return best_ckpt


def evaluate_contrastive(model, tokenizer, test_loader, entity_inputs, eval_similarity=False, attention_heatmap_path=None, num_negatives=50):
    predictions = []
    results = {}

    model.eval()
    device = args.device

    entity_list = list(entity_inputs.keys())

    sum_pos_sim, sum_neg_sim, n_pos, n_neg = 0., 0., 0., 0.

    output_attentions = (attention_heatmap_path != None)

    if output_attentions:
        text_lists, attention_lists, headers = [], [], []


    for i, batch in enumerate(tqdm(test_loader)):
        
        batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}

        with torch.no_grad():

            # Sampling for negative examples
            negative_entity_names = [entity_list[idx] for idx in random.sample(range(len(entity_inputs)), num_negatives)]
            negative_entity_input = collections.defaultdict(list)
            for entity_name in negative_entity_names:
                synonym_idx = random.sample(range(len(entity_inputs[entity_name]['input_ids'])), 1)[0]

                for k, v in entity_inputs[entity_name].items():
                    negative_entity_input[k].append(entity_inputs[entity_name][k][synonym_idx])
            negative_entity_input['input_ids'] = pad_sequence(negative_entity_input['input_ids'], batch_first=True, padding_value=0).to(device)
            negative_entity_input['token_type_ids'] = pad_sequence(negative_entity_input['token_type_ids'], batch_first=True, padding_value=0).to(device)
            negative_entity_input['attention_mask'] = pad_sequence(negative_entity_input['attention_mask'], batch_first=True, padding_value=0).to(device)
            output = model(batch['input'], [negative_entity_input])
            negative_entity_embeddings = output['entity_representations'][0]

            # Compute embeddings for all entity synonyms
            for entity_name in batch['entities'][0]:
                output = model(batch['input'], [{k: v.to(device) for k, v in entity_inputs[entity_name].items()}]) # Assumes they all fit on device
                entity_embeddings = output['entity_representations'][0]
                all_embeddings = torch.cat((entity_embeddings, negative_entity_embeddings))
                labels = torch.tensor([1 for _ in range(len(entity_embeddings))] + [0 for _ in range(len(negative_entity_embeddings))]).to(device)

                n_embeddings = len(all_embeddings)
                labels = labels.view(-1, 1)
                similarities = pairwise_cosine_similarity(all_embeddings, all_embeddings)
                pos_mask = torch.eq(labels, labels.T).float()
                neg_mask = 1 - pos_mask
                pos_mask = pos_mask - torch.eye(n_embeddings).to(pos_mask.device)

                sum_pos_sim += (similarities * pos_mask).sum().item()
                sum_neg_sim += (similarities * neg_mask).sum().item()
                n_pos += pos_mask.sum().item()
                n_neg += neg_mask.sum().item()

    results['avg_pos_sim'] = sum_pos_sim / n_pos
    results['avg_neg_sim'] = sum_neg_sim / n_neg
    return results

def train_contrastive(args, model, tokenizer, entity_inputs, dataloader, save_path, test_loader=None, top_k_ckpts=3, load_from=None):
    """
    Train on the subset of RFE examples not in the evaluation set
    """

    logger = init_logger(f'contrastive_pretraining_ner_rfe.log')
    optimizer = build_optim(args, model)

    device = args.device
    model = model.to(device)

    # Load pre-pretrained encoder weights on entities
    if load_from != None:
        model.encoder.load_state_dict(torch.load(load_from))

    active_embeddings = []
    contrastive_criteria = SupConLoss(contrast_mode='one')
    # cls_criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')

    kept_ckpts = []

    entity_list = list(entity_inputs.keys())
    for epoch in range(args.epochs):

        model.train()

        epoch_loss = 0.
        epoch_pos_sim, epoch_neg_sim = 0., 0.
        epoch_n_pos, epoch_n_neg = 0., 0.

        for i, batch in enumerate(tqdm(dataloader)):

            # batch['entities'][0] = [name for name in batch['entities'][0] if name in entity_inputs.keys()]

            batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}

            entities = []
            for entity_name in batch['entities'][0]:
                if not entity_name in entity_inputs.keys():
                    pdb.set_trace()

                if len(entity_inputs[entity_name]['input_ids']) > 10:
                    synonym_idx = random.sample(range(len(entity_inputs[entity_name]['input_ids'])), 10)
                    entities.append({k: v[synonym_idx].to(device) for k, v in entity_inputs[entity_name].items()})
                else:
                    entities.append({k: v.to(device) for k, v in entity_inputs[entity_name].items()})

            # Sampling for negative training examples
            negative_entity_names = [entity_list[idx] for idx in random.sample(range(len(entity_inputs)), args.num_negatives)]
            negative_entity_input = collections.defaultdict(list)
            for entity_name in negative_entity_names:
                synonym_idx = random.sample(range(len(entity_inputs[entity_name]['input_ids'])), 1)[0]

                for k, v in entity_inputs[entity_name].items():
                    negative_entity_input[k].append(entity_inputs[entity_name][k][synonym_idx])

            negative_entity_input['input_ids'] = pad_sequence(negative_entity_input['input_ids'], batch_first=True, padding_value=0).to(device)
            negative_entity_input['token_type_ids'] = pad_sequence(negative_entity_input['token_type_ids'], batch_first=True, padding_value=0).to(device)
            negative_entity_input['attention_mask'] = pad_sequence(negative_entity_input['attention_mask'], batch_first=True, padding_value=0).to(device)
            entities.append(negative_entity_input)
            output = model(batch['input'], entities)

            # Combine all entity representations
            # num_synonyms = [len(x) for x in output['entity_representations']]
            # pdb.set_trace()

            loss = 0.
            for entity_idx, entity_name in enumerate(batch['entities'][0]):
                n_pos = min(10, len(entity_inputs[entity_name]['input_ids']))
                n_neg = args.num_negatives
                labels = torch.tensor([1 for _ in range(n_pos)] + [0 for _ in range(n_neg)]).to(device)

                # if args.use_contrastive_loss:
                pos = output['entity_representations'][entity_idx]
                neg = output['entity_representations'][-1]
                embeddings = torch.cat((pos, neg))
                loss += contrastive_criteria(embeddings.unsqueeze(1), labels=labels)

                # Compute average similarities between pos and neg embeddings
                n_embeddings = len(embeddings)
                labels = labels.view(-1, 1)
                similarities = pairwise_cosine_similarity(embeddings, embeddings)
                pos_mask = torch.eq(labels, labels.T).float()
                neg_mask = 1 - pos_mask
                pos_mask = pos_mask - torch.eye(n_embeddings).to(pos_mask.device)

                epoch_pos_sim += (similarities * pos_mask).sum().item()
                epoch_neg_sim += (similarities * neg_mask).sum().item()
                epoch_n_pos += pos_mask.sum().item()
                epoch_n_neg += neg_mask.sum().item()


                epoch_loss += loss.item()

                # generate_heatmap(text_lists, attention_lists, pjoin('results', 'attention_visualize_ner', f'epoch_{epoch+1}', f'visualize_attention_ex_{i + 1}.tex'), headers)

            loss = loss / len(batch['entities'][0])
            loss.backward()

            if (i + 1) % 32 == 0:
                optimizer.step()
                model.zero_grad()

            lr = optimizer.optimizer.param_groups[0]['lr']
            avg_pos_sim_train = round(epoch_pos_sim / epoch_n_pos, 3)
            avg_neg_sim_train = round(epoch_neg_sim / epoch_n_neg, 3)

            # avg_pos_sim_test = round(test_results['avg_pos_sim'], 3)
            # avg_neg_sim_test = round(test_results['avg_neg_sim'], 3)

        train_summary = f"(Epoch {epoch + 1}) Loss: {epoch_loss}, \
                          Avg Pos Sim: {avg_pos_sim_train}, Avg Neg Sim: {avg_neg_sim_train}|, LR: {lr}"

        logger.info(train_summary)
        print(train_summary)

    torch.save(model.state_dict(), save_path)
    
    return model, kept_ckpts

def evaluate_ner(model, tokenizer, dataframe, test_ids, ignore_cls=True, threshold=5, eval_er=False, visualize=False, seen_concepts=set(), seen=True):

    # For analysis
    replaced_input = {
        9: ('anxiety', 'solicitude'),
        16: ('nasal congestion', 'nose feel clogged'),
        17: ('unilateral', 'bloody nose'),
        28: ('gingivitis', 'periodontal disease'),
        30: ('penile discharge', 'penis pain'),
    }
    replaced_input = False

    # For computing IoU
    total_intersect, total_union, total_IoU = 0, 0, 0

    # For computing micro f-score
    true_positives, false_positives, false_negatives = 0, 0, 0

    # For computing macro f-score
    precisions, recalls, f_scores, n_entities = 0, 0, 0, 0


    # One-to-one comparison with rule-based model
    # if eval_er:
    #     er, name2id, concept2synonyms = init_rule_based_ner()
    #     _true_positives, _false_positives, _false_negatives = 0, 0, 0
    #     _precisions, _recalls, _f_scores = 0, 0, 0

    # For visualization
    if visualize:
        results_dir = pjoin('results', 'visualize_results_rfe_zeroshot')
        os.makedirs(results_dir, exist_ok=True)
        text_list, saliency_list, headers = [], [], []
        max_rows = 100 # Maximum number of rows per file for visualization
        file_count = 0

    device = args.device

    sampled = 0
    for row_idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        if row.example_id not in test_ids:
            continue

        if replaced_input and (row_idx not in replaced_input.keys()):
            continue
        
        entity_name = json.loads(row.annotation)['classes']
        annotations = json.loads(row.annotation)['annotations'][0]
        input_text = annotations[0].split(tokenizer.sep_token)[0].strip()
        annotated_tokens = list(TreebankWordTokenizer().tokenize(input_text))
        annotation_string = ' '.join(annotated_tokens)

        try:
            assert len(entity_name) == 1
        except:
            continue

        if seen and entity_name[0] not in seen_concepts:
            continue
        elif not seen and entity_name[0] in seen_concepts:
            continue


        # For aligning annotations to tokens
        text_len = 0
        start2idx, end2idx = {}, {}
        for tok_idx, tok in enumerate(annotated_tokens):
            tok_start = tok_idx + text_len
            start2idx[tok_start] = tok_idx
            end2idx[tok_start + len(tok)] = tok_idx
            text_len += len(tok)
        spans = [(x[0], x[1]) for x in annotations[1]['entities']]

        try:
            token_level_spans = [(start2idx[x[0]], end2idx[x[1]]) for x in spans]
        except:
            continue
            print("Couldn't find the token index of a span!")
            pdb.set_trace()

        # Ground-truth token mask for the entity
        ground_truth_mask = np.zeros(len(annotated_tokens))
        for span in token_level_spans:
            ground_truth_mask[span[0]: span[1] + 1] = 1
        ground_truth_mask = ground_truth_mask.astype(bool)

        # No annotation
        if np.count_nonzero(ground_truth_mask) == 0:
            continue


        # Creating inputs for PLM-based NER
        if replaced_input:
            entity_name = [replaced_input[row_idx][1]]

        entity_inputs = {k: v.to(device) for k, v in tokenizer(entity_name, return_tensors="pt", padding=True).items()} 
        text_input = {k: v.to(device) for k, v in \
                      tokenizer(annotated_tokens, return_tensors="pt", padding=True, is_split_into_words=True).items()}

        output = model(text_input, [entity_inputs])

        tokens = tokenizer.convert_ids_to_tokens(text_input['input_ids'][0].tolist())
        attentions = (output['attention'][0] * 100).squeeze(0)
        attentions[attentions < 1] = 0
        attentions = attentions.tolist()

        if ignore_cls:
            tokens = tokens[1:]

        tokens, attentions = merge_subword_tokens(tokens, attentions)

        if tokens[:-1] != annotated_tokens:
            continue
            print("Detokenized tokens does not equal to the input!")
            pdb.set_trace()

        attn = attentions[0] # One concept per example
        visualized_attn = attn.copy()

        # Create prediction masks over tokens
        prediction_mask = (np.array(attn[:-1]) > threshold)

        # Remove stop words
        stopword_idx = np.array([token_idx for (token_idx, token) in enumerate(tokens[:-1]) if token in stopwords], dtype=int)
        prediction_mask[stopword_idx] = False

        tp = np.count_nonzero(prediction_mask & ground_truth_mask)
        fp = np.count_nonzero(prediction_mask & ~ground_truth_mask)
        fn = np.count_nonzero(~prediction_mask & ground_truth_mask)
        f1 = tp / (tp + 0.5 * (fp + fn))


        true_positives += tp
        false_positives += fp
        false_negatives += fn

        try:
            precisions += tp / (tp + fp)
        except:
            pass
        try:
            recalls += tp / (tp + fn)
        except:
            pass

        f_scores += f1
        n_entities += 1

        # Evaluating rule-based NER model
        if eval_er:
            extraction = er.find_entities(annotation_string, unique_entities=True)
            rule_based_prediction = []
            rule_based_mask = np.zeros(len(annotated_tokens))

            for matched in extraction.entities:
                name = matched.kb_item.name

                # Only considering ground-truth entity
                if name == entity_name[0] or entity_name[0] in concept2synonyms[name]:
                    matched_name = matched.matched_name
                    char_start = annotation_string.find(matched_name)
                    token_start = len(annotation_string[:char_start].strip().split())
                    token_end = token_start + len(matched_name.split())
                    rule_based_mask[token_start: token_end] = 1

            rule_based_mask = rule_based_mask.astype(bool)
            _tp = np.count_nonzero(rule_based_mask & ground_truth_mask)
            _fp = np.count_nonzero(rule_based_mask & ~ground_truth_mask)
            _fn = np.count_nonzero(~rule_based_mask & ground_truth_mask)
            _f1 = _tp / (_tp + 0.5 * (_fp + _fn))

            _true_positives += _tp
            _false_positives += _fp
            _false_negatives += _fn

            try:
                _precisions += _tp / (tp + _fp)
            except:
                pass
            try:
                _recalls += _tp / (_tp + _fn)
            except:
                pass
            _f_scores += _f1

        # Write results to tex files
        if visualize:

            if replaced_input:
                headers.append(f"Index: {row_idx}, GT Entity Name: {json.loads(row.annotation)['classes'][0]}, Input Entity Name: {entity_name[0]}")
                text_list.append(annotated_tokens)
                saliency_list.append(visualized_attn[:-1])

            else:
                headers.append(f"Index: {row_idx}, GT Entity Name: {entity_name[0]}")
                text_list.append(annotated_tokens)
                saliency_list.append(ground_truth_mask.astype(float) * 50)

                headers.append(f"Index: {row_idx}, Constrastive-NER, F-Score: {round(f1, 3)}")
                text_list.append(annotated_tokens)
                saliency_list.append(visualized_attn[:-1])

                if eval_er:
                    headers.append(f"Index: {row_idx}, Rule-Based-NER, F-Score: {round(_f1, 3)}")
                    text_list.append(annotated_tokens)
                    saliency_list.append(rule_based_mask.astype(float) * 50)

                if len(text_list) > max_rows:
                    tex_file = pjoin(results_dir, f'results_file_{file_count + 1}.tex')
                    generate_heatmap(text_list, saliency_list, tex_file, headers)
                    text_list, saliency_list, headers = [], [], []
                    file_count += 1 


    # Rest of inputs
    if replaced_input:
        tex_file = pjoin(results_dir, f'results_file_replaced_input.tex')
        generate_heatmap(text_list, saliency_list, tex_file, headers)
    else:
        tex_file = pjoin(results_dir, f'results_file_{file_count + 1}.tex')
        generate_heatmap(text_list, saliency_list, tex_file, headers)

    
    


    results = {}
    print(f"Number of concept-example pairs: {n_entities}")
    results['micro_precision'] = round(true_positives / (true_positives + false_positives), 5)
    results['micro_recall'] = round(true_positives / (true_positives + false_negatives), 5)
    results['micro_f1'] = round(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)), 5)
    print(f"(Micro) P: {results['micro_precision']}, R: {results['micro_recall']}, F: {results['micro_f1']}")

    results['macro_precision'], results['macro_recall'] = round(precisions / n_entities, 5), round(recalls / n_entities, 5)
    results['macro_f1'] = round((f_scores / n_entities), 5)
    print(f"(Macro) P: {results['macro_precision']}, R: {results['macro_recall']}, F: {results['macro_f1']}")
    if eval_er:
        print("\n")
        print("Results for Rule-Based NER Model:")
        _micro_precision = round(_true_positives / (_true_positives + _false_positives), 5)
        _micro_recall = round(_true_positives / (_true_positives + _false_negatives), 5)
        _micro_f1 = round(_true_positives / (_true_positives + 0.5 * (_false_positives + _false_negatives)), 5)
        print(f"(Micro) P: {_micro_precision}, R: {_micro_recall}, F: {_micro_f1}")

        _macro_precision, _macro_recall = round(_precisions / n_entities, 5), round(_recalls / n_entities, 5)
        _macro_f1 = round((_f_scores / n_entities), 5)
        print(f"(Macro) P: {_macro_precision}, R: {_macro_recall}, F: {_macro_f1}")

    return results


def train_classifier(args, model, tokenizer, entity_inputs, entity_synonyms, train_loader, ckpt_save_path, test_set=None):
    logger = init_logger(f'classification_training_hnlp.log')
    optimizer = build_optim(args, model)

    device = args.device

    model = model.to(device)

    if args.freeze_weights:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.attention_layer.parameters():
            param.requires_grad = False            

    if args.classification_loss == 'bce':
        cls_criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif args.classification_loss == 'focal':
        cls_criteria = BinaryFocalLossWithLogits(reduction='mean')
    
    max_positives = 20
    entity_list = list(entity_inputs.keys())

    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()

        for data_idx, batch in enumerate(tqdm(train_loader)):

            batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}

            # text_input = {k: v.to(device) for k, v in example['text_inputs'].items()}
            batch_synonym_inputs = []

            for entity in batch['entities'][0]:
                synonym_inputs = entity_inputs[entity]
                if len(synonym_inputs['input_ids']) > max_positives:
                    indices = random.sample(range(len(synonym_inputs['input_ids'])), max_positives)
                    batch_synonym_inputs.append({k: v[indices].to(device) for k, v in synonym_inputs.items()})
                else:
                    batch_synonym_inputs.append({k: v.to(device) for k, v in synonym_inputs.items()})


            # Sampling for negative training examples
            negative_entity_names = [entity_list[idx] for idx in random.sample(range(len(entity_inputs)), args.num_negatives)]
            negative_entity_input = collections.defaultdict(list)
            for entity_name in negative_entity_names:
                synonym_idx = random.sample(range(len(entity_inputs[entity_name]['input_ids'])), 1)[0]
                for k, v in entity_inputs[entity_name].items():
                    negative_entity_input[k].append(entity_inputs[entity_name][k][synonym_idx])
            negative_entity_input['input_ids'] = pad_sequence(negative_entity_input['input_ids'], batch_first=True, padding_value=0).to(device)
            negative_entity_input['token_type_ids'] = pad_sequence(negative_entity_input['token_type_ids'], batch_first=True, padding_value=0).to(device)
            negative_entity_input['attention_mask'] = pad_sequence(negative_entity_input['attention_mask'], batch_first=True, padding_value=0).to(device)

            batch_synonym_inputs.append(negative_entity_input)

            output = model(batch['input'], batch_synonym_inputs)

            loss = 0.
            for input_idx, synonym_inputs in enumerate(batch_synonym_inputs[:-1]):
                n_pos = min(max_positives, len(synonym_inputs['input_ids']))
                n_neg = args.num_negatives
                labels = torch.tensor([1 for _ in range(n_pos)] + [0 for _ in range(n_neg)]).to(device)

                pos = output['logits'][input_idx]
                neg = output['logits'][-1]
                logits = torch.cat((pos, neg), dim=-1).squeeze(0)
                loss += cls_criteria(logits.unsqueeze(0), labels.unsqueeze(0).float())

            epoch_loss += loss.item()
            loss = loss / len(batch_synonym_inputs)
            loss.backward()

            if (data_idx + 1) % 32 == 0:
                optimizer.step()
                model.zero_grad()

        
        lr = optimizer.optimizer.param_groups[0]['lr']
        train_summary = f"(Epoch {epoch + 1}) Loss: {epoch_loss} LR: {lr}"
        logger.info(train_summary)
        print(train_summary)

    if test_set:

        def print_results(recalls):
            n_single = n_pairs - n_multi
            print(f"Single Span: {n_single}/{n_pairs}")
            test_summary = f"Top-1 Recall: {round(recalls[0][0]/n_single, 4)}, \
                             Top-5 Recall: {round(recalls[0][1]/n_single, 4)}, \
                             Top-10 Recall: {round(recalls[0][2]/n_single, 4)}"
            logger.info(test_summary)
            print(test_summary)
            print(f"Multi Span: {n_multi}/{n_pairs}")
            test_summary = f"Top-1 Recall: {round(recalls[1][0]/n_multi, 4)}, \
                             Top-5 Recall: {round(recalls[1][1]/n_multi, 4)}, \
                             Top-10 Recall: {round(recalls[1][2]/n_multi, 4)}"
            logger.info(test_summary)
            print(test_summary)
            print(f"All: {n_pairs}")
            test_summary = f"Top-1 Recall: {round((recalls[0][0] + recalls[1][0])/n_pairs, 4)}, \
                             Top-5 Recall: {round((recalls[0][1] + recalls[1][1])/n_pairs, 4)}, \
                             Top-10 Recall: {round((recalls[0][2] + recalls[1][2])/n_pairs, 4)}"
            logger.info(test_summary)
            print(test_summary)




        
        model.eval()
        concept2vectors = {}

        nlp = spacy.blank("en")
        matcher = FuzzyMatcher(nlp.vocab)


        # Precompute entity embeddings
        for concept, synonym_inputs in entity_inputs.items():
            synonym_inputs = {k: v.to(device) for k, v in synonym_inputs.items()}
            with torch.no_grad():
                synonym_vectors = model.encoder(**synonym_inputs)[0][:, 0, :].detach()
            concept2vectors[concept] = synonym_vectors

        n_multi = 0
        n_pairs = 0
        baseline_recalls = [[0, 0, 0], [0, 0, 0]]
        recalls = [[0, 0, 0], [0, 0, 0]]
        for example in tqdm(test_set):

            # Baseline
            baseline_probs = []
            for concept, synonyms in entity_synonyms.items():
                synonyms = list(set(synonyms))
                max_sim = 0

                for syn in synonyms:
                    matcher.add(syn, [nlp(syn)], on_match=add_name_ent)

                matches = matcher(nlp(example['text']))

                if matches:
                    for match in matches:
                        if match[-1] > max_sim:
                            max_sim = match[-1]

                for syn in synonyms:
                    matcher.remove(syn)

                if max_sim > 0:
                    baseline_probs.append((concept, max_sim))

            sorted_baseline_probs = sorted(baseline_probs, key=lambda x: x[1], reverse=True)
            sorted_baseline_ids = [prob[0] for prob in sorted_baseline_probs]

            # OSLAT-Linker
            probs = []
            with torch.no_grad():
                text_input = {k: v.to(device) for k, v in example['input'].items()}
                attention_masks = text_input['attention_mask']
                input_hidden = model.encoder(**text_input)[0]

                if model.ignore_cls:
                    input_hidden = input_hidden[:, 1:, :]
                    attention_masks = attention_masks[:, 1:]

                for concept, synonym_vectors in concept2vectors.items():
                    concept_representations = model.attention_layer(
                        synonym_vectors,
                        input_hidden,
                        attention_mask=attention_masks,
                    )[0]

                    if args.append_query:
                        concept_representations = torch.cat((concept_representations, synonym_vectors.unsqueeze(0)), dim=-1)

                    logits = model.classifier(concept_representations).squeeze(-1)
                    probs.append((concept, logits.max().item()))
            sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
            sorted_ids = [prob[0] for prob in sorted_probs]


            # Accumuate scores
            for entity_idx, name in enumerate(example['entities']):
                n_pairs += 1

                # TO DO: Multi-Span is obtained from annotations
                multi_span = False

                if multi_span:
                    n_multi += 1
                    recall_idx = 1
                else:
                    recall_idx = 0

                if name in sorted_baseline_ids[:1]:
                    baseline_recalls[recall_idx][0] += 1
                if name in sorted_baseline_ids[:5]:
                    baseline_recalls[recall_idx][1] += 1
                if name in sorted_baseline_ids[:10]:
                    baseline_recalls[recall_idx][2] += 1

                if name in sorted_ids[:1]:
                    recalls[recall_idx][0] += 1
                if name in sorted_ids[:5]:
                    recalls[recall_idx][1] += 1
                if name in sorted_ids[:10]:
                    recalls[recall_idx][2] += 1



        print("\nFuzzy-Matching:")
        print_results(baseline_recalls)

        print("\nOSLAT-Linker:")
        print_results(recalls)

            
                    



        

    # save_name = f"{args.encoder}_lr{args.lr}_epoch{args.epoch}.pth"
    # ckpt_save_path = pjoin(ckpt_dir, save_name)
    torch.save(model.state_dict(), ckpt_save_path)
    return model

def run_rfe(args):

    # Pretrain entity embeddings
    args.json_data_path = 'resources/CuRSA/CuRSA-FIXED-v0-processed-all.json'
    args.processed_data_path = 'resources/CuRSA/CuRSA-FIXED-v0-processed-all.pth'

    if not args.wo_pretraining:
        best_ckpt_path = pretrain_entity_embeddings(args)
        # best_ckpt_path = pjoin('checkpoints', 'encoders', 'hnlp_biobert_lr0.0002_epoch18_0.17.pt')
    else:
        best_ckpt_path = None
    args.lr = 0.0002

    # USERNAME = "postgres"
    # PASSWORD = "1yjEkbdCMkdv5cqDJB7h2yyh2rlBid"
    # SECRET_KEY = b"gDe5jNAKTu4NHtZJ"
    # DB_URI = "ml-adhoc-cluster.c6fksyrlat9v.us-west-2.rds.amazonaws.com"
    # DIALECT = "postgresql"
    # SQLALCHEMY_DATABASE_URI=f"{DIALECT}://{USERNAME}:{PASSWORD}@{DB_URI}"
    # SQLALCHEMY_TRACK_MODIFICATIONS=False
    # SCHEMA = "rfe_ner_annotator"

    # engine = sqlalchemy.create_engine(SQLALCHEMY_DATABASE_URI)
    # query = query = f'''
    #         SELECT * FROM {SCHEMA}.annotation
    #     '''
    # with engine.connect() as con:
    #     df = pd.read_sql(query,con)


    # Initialize Neural Model
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = ContrastiveEntityExtractor(args)


    device = args.device
    model = model.to(device)


    with open(pjoin('resources', 'CuRSA', 'cursa-train-test-seen-unseen.json')) as f:
        json_data = json.load(f)
        data_split = json_data['DATASET']
        seen_concepts = set(json_data['CONCEPTS']['SEEN'])


    if not os.path.isfile(args.processed_data_path):
        processed_data = process_json_file_ner(args.json_data_path, tokenizer, args.k_folds)
        torch.save(processed_data, args.processed_data_path)
    else:
        processed_data = torch.load(args.processed_data_path)

    train_encounter_ids = set([x[1] for x in data_split['TRAIN']])
    train_ids = set([i for i, ex in enumerate(processed_data['data']) if ex['id'] in train_encounter_ids])
    test_ids = set(range(len(processed_data['data']))) - train_ids
    train_loader, test_loader = get_dataloaders_ner(args, train_ids, test_ids)

    # er, name2id, concept2synonyms = init_rule_based_ner()
    entity_inputs = processed_data['entity_synonyms']

    contrastive_ckpt_dir = pjoin(args.checkpoints_dir, 'contrastive_ner_rfe')

    if args.wo_pretraining:
        contrastive_ckpt_dir += '_nopretrain'
    if args.append_query:
        contrastive_ckpt_dir += '_concatquery'

    os.makedirs(contrastive_ckpt_dir, exist_ok=True)

    ckpt_save_path = pjoin(contrastive_ckpt_dir, f"{args.encoder}_lr{args.lr}_epoch{args.epochs}.pth")

    if not args.wo_contrastive:
        if not os.path.isfile(ckpt_save_path):
            model = train_contrastive(args, model, tokenizer, entity_inputs, train_loader, ckpt_save_path, test_loader=test_loader, load_from=best_ckpt_path)
        else:
            model.load_state_dict(torch.load(ckpt_save_path, map_location=args.device), strict=False)
            print(f"Loaded Checkpoints at \"{ckpt_save_path}\"")


    # test_encounter_ids = set([x[1] for x in data_split['TEST']])
    # test_ids = set([i for i, ex in enumerate(processed_data['data']) if ex['id'] in test_encounter_ids])
    classifier_ckpt_dir = pjoin(args.checkpoints_dir, 'classifier')
    if args.append_query:
        classifier_ckpt_dir += '_concatquery'

    if args.wo_pretraining:
        classifier_ckpt_dir += '_nopretrain'
    if args.wo_contrastive:
        classifier_ckpt_dir += '_nocontrastive'

    if args.classification_loss == 'focal':
        classifier_ckpt_dir += '_focal'

    if args.freeze_weights:
        classifier_ckpt_dir += '_freeze'

    os.makedirs(classifier_ckpt_dir, exist_ok=True)
    ckpt_save_path = pjoin(classifier_ckpt_dir, f"{args.encoder}_lr{args.lr}_epoch{args.epochs}.pth")
    model = train_classifier(
        args,
        model,
        tokenizer,
        entity_inputs,
        processed_data['synonyms_names'],
        train_loader,
        ckpt_save_path,
        test_set=test_loader.dataset,
    )



    
