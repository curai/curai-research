import os
import csv
import json
import torch
import random
import collections
import numpy as np
import pandas as pd
import re

import spacy
from spaczz.matcher import FuzzyMatcher


from os.path import join as pjoin
from utils.logger import init_logger
from utils.helpers import pairwise_cosine_similarity, merge_subword_tokens, add_name_ent
from utils.data import HNLPContrastiveNERDataset
from utils.visualize import generate_heatmap, visualize_entity_synonyms
from models.contrastive_classifier import ContrastiveEntityExtractor, EncoderWrapper
from models.optimizers import build_optim
from models.losses import SupConLoss
from models.focal_loss import BinaryFocalLossWithLogits
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from tqdm import tqdm
import torch.nn.functional as F

import pdb


stopwords = {
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


def pretrain_entity_embeddings(args, data_path):
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


    with open(data_path, 'r') as f:
        data = json.load(f)
        id2synonyms = data['CONCEPT_TO_SYNS']
        id2synonyms = {k: v for k, v in id2synonyms.items() if k in data['CONCEPTS']['SEEN']}

    # Entities sorted by the number of synonyms
    contrastive_criteria = SupConLoss(contrast_mode='one')

    model.train()
    
    # Manually define the learning rate for the optimizer (different from NER training)
    old_lr = args.lr
    args.lr = 0.002
    optimizer = build_optim(args, model)
    args.lr = old_lr

    max_pos = 20
    n_neg = 50

    prev_best = 0.
    best_ckpt = None

    for epoch in range(20):
        label_order = list(id2synonyms.keys())
        random.shuffle(label_order)
        loss = 0.
        with tqdm(total=len(label_order)) as pbar:
            for i, entity_id in enumerate(label_order):

                # Positive training examples
                synonyms = id2synonyms[entity_id]
                if len(synonyms) > max_pos:
                    synonyms = random.sample(synonyms, max_pos)

                # Sampling for negative training examples
                negative_entity_ids = random.choices(label_order, k=n_neg)
                negative_entity_names = []
                for entity_id in negative_entity_ids:
                    negative_entity_names.append(random.sample(id2synonyms[entity_id], 1)[0])

                pos_inputs = tokenizer(synonyms, return_tensors="pt", padding=True)
                neg_inputs = tokenizer(negative_entity_names, return_tensors="pt", padding=True)

                pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
                neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}

                pos_output = model(pos_inputs)
                neg_output = model(neg_inputs)

                embeddings = torch.cat((pos_output, neg_output))
                labels = torch.tensor([1 for _ in range(len(pos_output))] + [0 for _ in range(len(neg_output))]).to(device)
                loss = contrastive_criteria(embeddings.unsqueeze(1), labels=labels)

                pbar.set_description(f"Loss: {round(loss.item(), 3)}")
                loss.backward()

                if (i + 1) % 32 == 0:
                    optimizer.step()
                    model.zero_grad()

                pbar.update(1)

        all_entity_embeddings = []
        n_labels = len(id2synonyms.keys())
        current_label = 0
        labels = []
        for entity_id, synonyms in id2synonyms.items():

            model_inputs = tokenizer(synonyms, return_tensors="pt", padding=True)
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
            save_name = f"hnlp_{args.encoder}_lr{args.lr}_epoch{epoch+1}_{round(diff, 3)}.pt"
            save_path = pjoin(save_dir, save_name)
            torch.save(model.encoder.state_dict(), save_path)
            print(diff)
            best_ckpt = save_path

    return best_ckpt


def train_contrastive(args, model, tokenizer, id2synonyms, train_set, save_path, top_k_ckpts=3, load_from=None):
    """
    Train the NER model using the supervised contrastive objective
    """

    logger = init_logger(f'contrastive_pretraining_ner_hnlp.log')
    optimizer = build_optim(args, model)

    device = args.device
    model = model.to(device)

    # Load pre-pretrained encoder weights on entities
    if load_from != None:
        model.encoder.load_state_dict(torch.load(load_from, map_location=device))

    active_embeddings = []
    contrastive_criteria = SupConLoss(contrast_mode='one')
    cls_criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')


    kept_ckpts = []
    max_positives = 20

    for epoch in range(args.epochs):

        model.train()

        epoch_loss = 0.
        epoch_pos_sim, epoch_neg_sim = 0., 0.
        epoch_n_pos, epoch_n_neg = 0., 0.

        # Shuffle the training set index for each epoch
        shuffled_indices = sorted(range(len(train_set)), key=lambda k: random.random())

        for data_idx in tqdm(shuffled_indices):

            example = train_set[data_idx]

            text_input = {k: v.to(device) for k, v in example['text_inputs'].items()}
            batch_synonym_inputs = []

            for synonym_inputs in example['synonym_inputs']:
                if len(synonym_inputs['input_ids']) > max_positives:
                    indices = random.sample(range(len(synonym_inputs['input_ids'])), max_positives)
                    batch_synonym_inputs.append({k: v[indices].to(device) for k, v in synonym_inputs.items()})
                else:
                    batch_synonym_inputs.append({k: v.to(device) for k, v in synonym_inputs.items()})

            # Sampling for negative training examples
            sampled_ids = random.choices(list(id2synonyms.keys()), k=args.num_negatives)
            negative_names = [random.sample(id2synonyms[concept_id], 1)[0] for concept_id in sampled_ids]
            negative_inputs = tokenizer(negative_names, return_tensors="pt", padding=True)
            negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}
            batch_synonym_inputs.append(negative_inputs)

            output = model(text_input, batch_synonym_inputs)

            loss = 0.
            for input_idx, synonym_inputs in enumerate(batch_synonym_inputs[:-1]):
                n_pos = min(max_positives, len(synonym_inputs['input_ids']))
                n_neg = args.num_negatives
                labels = torch.tensor([1 for _ in range(n_pos)] + [0 for _ in range(n_neg)]).to(device)

                if args.use_contrastive_loss:
                    pos = output['entity_representations'][input_idx]
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

            loss = loss / len(example['synonym_inputs'])
            loss.backward()

            if (data_idx + 1) % 32 == 0:
                optimizer.step()
                model.zero_grad()

        
        lr = optimizer.optimizer.param_groups[0]['lr']
        avg_pos_sim_train = round(epoch_pos_sim / epoch_n_pos, 3)
        avg_neg_sim_train = round(epoch_neg_sim / epoch_n_neg, 3)

        train_summary = f"(Epoch {epoch + 1}) Loss: {epoch_loss}, \
                          Avg Pos Sim: {avg_pos_sim_train}, Avg Neg Sim: {avg_neg_sim_train}|, LR: {lr}"

        logger.info(train_summary)
        print(train_summary)

    torch.save(model.state_dict(), save_path)
    return model


def evaluate_hnlp_ner(model, tokenizer, data_path, ignore_cls=True, baseline=None, threshold=5, concept_ids=set(), multi_span=False):
    device = next(model.parameters()).device
    nlp = spacy.load("en_core_web_sm")
    spacy_tokenizer = nlp.tokenizer

    with open(data_path, 'r') as f:
        data = json.load(f)
        id2synonyms = data['CONCEPT_TO_SYNS']
        test_set = data['DATASET']['TEST']

    n_synonyms = 10

    # For computing micro f-score
    true_positives, false_positives, false_negatives = 0, 0, 0

    # For computing macro f-score
    precisions, recalls, f_scores, n_entities = 0, 0, 0, 0

    # One-to-one comparison with rule-based model
    if baseline:
        _true_positives, _false_positives, _false_negatives = 0, 0, 0
        _precisions, _recalls, _f_scores = 0, 0, 0

        if baseline == 'fuzzy':
            nlp = spacy.blank("en")
            matcher = FuzzyMatcher(nlp.vocab)
        else:
            raise NotImplementedError

    for idx, example in enumerate(tqdm(test_set)):

        text = example['text']
        text = re.sub(r"\[\*(.*?)\*\]", '', text)
        tokens = [token.text for token in spacy_tokenizer(text)]

        entities = example['entities']

        for entity_id, entity_annotations in entities:
            if multi_span and len(entity_annotations) == 1:
                continue
            elif not multi_span and len(entity_annotations) > 1:
                continue

            if concept_ids and entity_id not in concept_ids:
                continue

            matched_str = [x[1] for x in entity_annotations]
            matched_str_len = [len(x[1].split()) for x in entity_annotations]

            try:
                synonyms = id2synonyms[entity_id]
            except:
                print(f"{entity_id} not in dictionary")
                continue

            if len(synonyms) > n_synonyms:
                synonyms = random.sample(synonyms, n_synonyms)

            synonym_inputs = tokenizer(synonyms, return_tensors="pt", padding=True)       
            text_input = tokenizer(tokens, return_tensors="pt", padding=True, is_split_into_words=True, truncation=True, max_length=512)
            synonym_inputs = {k: v.to(device) for k, v in synonym_inputs.items()}
            text_input = {k: v.to(device) for k, v in text_input.items()}
            with torch.no_grad():
                output = model(text_input, [synonym_inputs])

            attentions = (output['attention'][0] * 100).squeeze(0).mean(0)
            attentions = attentions.tolist()


            tokens = tokenizer.convert_ids_to_tokens(text_input['input_ids'][0].tolist())

            if ignore_cls:
                tokens = tokens[1:]

            tokens, attentions = merge_subword_tokens(tokens, [attentions])
            attn = attentions[0]


            # Ignore [SEP] token
            tokens = tokens[:-1]
            attn = attn[:-1]

            # Ground-Truth Mask
            ground_truth_mask = np.zeros(len(tokens))
            for tok_idx in range(len(tokens)):
                for matched_len in matched_str_len:
                    if (tok_idx + matched_len) <= len(tokens) and \
                       (' '.join(tokens[tok_idx: tok_idx + matched_len]) in matched_str):
                        ground_truth_mask[tok_idx: tok_idx + matched_len] = 1

            if sum(ground_truth_mask) == 0:
                continue

            ground_truth_mask = ground_truth_mask.astype(bool)

            # Create prediction masks over tokens
            prediction_mask = (np.array(attn) > threshold)

            # Remove stop words
            stopword_idx = np.array([token_idx for (token_idx, token) in enumerate(tokens) if token in stopwords], dtype=int)
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

            if baseline:
                rule_based_mask = np.zeros(len(tokens))
                if baseline == 'fuzzy':

                    for synonym in synonyms:
                        matcher.add(synonym, [nlp(synonym)], on_match=add_name_ent)

                    matches = matcher(nlp(' '.join(tokens)))

                    for synonym, start, end, score in matches:
                        rule_based_mask[start:end] = 1

                    for synonym in synonyms:
                        matcher.remove(synonym)

                else:
                    raise NotImplementedError



                rule_based_mask = rule_based_mask.astype(bool)

                _tp = np.count_nonzero(rule_based_mask & ground_truth_mask)
                _fp = np.count_nonzero(rule_based_mask & ~ground_truth_mask)
                _fn = np.count_nonzero(~rule_based_mask & ground_truth_mask)
                try:
                    _f1 = _tp / (_tp + 0.5 * (_fp + _fn))
                except:
                    pass

                _true_positives += _tp
                _false_positives += _fp
                _false_negatives += _fn
                try:
                    _precisions += _tp / (_tp + _fp)
                except:
                    pass
                try:
                    _recalls += _tp / (_tp + _fn)
                except:
                    pass
                _f_scores += _f1

    results = {}
    print(f"Number of concept-example pairs: {n_entities}")
    results['micro_precision'] = round(true_positives / (true_positives + false_positives), 5)
    results['micro_recall'] = round(true_positives / (true_positives + false_negatives), 5)
    results['micro_f1'] = round(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)), 5)
    print(f"(Micro) P: {results['micro_precision']}, R: {results['micro_recall']}, F: {results['micro_f1']}")

    results['macro_precision'], results['macro_recall'] = round(precisions / n_entities, 5), round(recalls / n_entities, 5)
    results['macro_f1'] = round((f_scores / n_entities), 5)
    print(f"(Macro) P: {results['macro_precision']}, R: {results['macro_recall']}, F: {results['macro_f1']}")
    
    if baseline:
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

def train_classifier(args, model, tokenizer, id2synonyms, train_set, ckpt_save_path, test_set=None):
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

    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()

        # Shuffle the training set index for each epoch
        shuffled_indices = sorted(range(len(train_set)), key=lambda k: random.random())

        for data_idx in tqdm(shuffled_indices):

            example = train_set[data_idx]

            text_input = {k: v.to(device) for k, v in example['text_inputs'].items()}
            batch_synonym_inputs = []

            for synonym_inputs in example['synonym_inputs']:
                if len(synonym_inputs['input_ids']) > max_positives:
                    indices = random.sample(range(len(synonym_inputs['input_ids'])), max_positives)
                    batch_synonym_inputs.append({k: v[indices].to(device) for k, v in synonym_inputs.items()})
                else:
                    batch_synonym_inputs.append({k: v.to(device) for k, v in synonym_inputs.items()})

            # Sampling for negative training examples
            sampled_ids = random.choices(list(id2synonyms.keys()), k=args.num_negatives)
            negative_names = [random.sample(id2synonyms[concept_id], 1)[0] for concept_id in sampled_ids]
            negative_inputs = tokenizer(negative_names, return_tensors="pt", padding=True)
            negative_inputs = {k: v.to(device) for k, v in negative_inputs.items()}
            batch_synonym_inputs.append(negative_inputs)

            output = model(text_input, batch_synonym_inputs)

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
            loss = loss / len(example['synonym_inputs'])
            loss.backward()

            if (data_idx + 1) % 32 == 0:
                optimizer.step()
                model.zero_grad()

        
        lr = optimizer.optimizer.param_groups[0]['lr']
        train_summary = f"(Epoch {epoch + 1}) Loss: {epoch_loss} LR: {lr}"
        logger.info(train_summary)
        print(train_summary)

        if test_set:
            recalls = [0, 0, 0]
            model.eval()
            id2vectors = {}

            # Precompute entity embeddings
            for concept_id, synonyms in id2synonyms.items():
                synonym_inputs = tokenizer(synonyms, return_tensors="pt", padding=True)
                synonym_inputs = {k: v.to(device) for k, v in synonym_inputs.items()}
                with torch.no_grad():
                    synonym_vectors = model.encoder(**synonym_inputs)[0][:, 0, :].detach()
                id2vectors[concept_id] = synonym_vectors


            for example in tqdm(test_set):
                probs = []
                with torch.no_grad():
                    text_input = {k: v.to(device) for k, v in example['text_inputs'].items()}
                    attention_masks = text_input['attention_mask']
                    input_hidden = model.encoder(**text_input)[0]

                    if model.ignore_cls:
                        input_hidden = input_hidden[:, 1:, :]
                        attention_masks = attention_masks[:, 1:]

                    for concept_id, synonym_vectors in id2vectors.items():
                        concept_representations = model.attention_layer(
                            synonym_vectors,
                            input_hidden,
                            attention_mask=attention_masks,
                        )[0]

                        if args.append_query:
                            concept_representations = torch.cat((concept_representations, synonym_vectors.unsqueeze(0)), dim=-1)

                        logits = model.classifier(concept_representations).squeeze(-1)
                        probs.append((concept_id, logits.max().item()))

                gt_concept = example['entity_ids'][0]
                sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
                sorted_ids = [prob[0] for prob in sorted_probs]
                if gt_concept in sorted_ids[:1]:
                    recalls[0] += 1
                if gt_concept in sorted_ids[:5]:
                    recalls[1] += 1
                if gt_concept in sorted_ids[:10]:
                    recalls[2] += 1


            test_summary = f"Top-1 Recall: {round(recalls[0]/len(test_set), 4)}, \
                             Top-5 Recall: {round(recalls[1]/len(test_set), 4)}, \
                             Top-10 Recall: {round(recalls[2]/len(test_set), 4)}"
            logger.info(test_summary)
            print(test_summary)

        

    # save_name = f"{args.encoder}_lr{args.lr}_epoch{args.epoch}.pth"
    # ckpt_save_path = pjoin(ckpt_dir, save_name)
    torch.save(model.state_dict(), ckpt_save_path)
    return model

def run_hnlp(args):

    # Pretrain entity embeddings
    hnlp_data_path = 'resources/hNLP/hNLP-train-test-seen-unseen.json'

    if not args.wo_pretraining:
        # best_ckpt_path = pretrain_entity_embeddings(args, hnlp_data_path)
        best_ckpt_path = pjoin('checkpoints', 'encoders', 'hnlp_biobert_lr0.0002_epoch18_0.17.pt')
    else:
        best_ckpt_path = None

    # Initialize Neural Model
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = ContrastiveEntityExtractor(args)

    model = model.to(args.device)

    with open(hnlp_data_path) as f:
        json_data = json.load(f)
        data_split = json_data['DATASET']
        id2synonyms = json_data['CONCEPT_TO_SYNS']
        seen_concepts = set(json_data['CONCEPTS']['SEEN'])
        unseen_concepts = set(json_data['CONCEPTS']['UNSEEN'])

    train_set = HNLPContrastiveNERDataset(data_split['TRAIN'], tokenizer, id2synonyms)
    test_set = HNLPContrastiveNERDataset(data_split['TEST'], tokenizer, id2synonyms)

    if args.wo_pretraining:
        contrastive_ckpt_dir = pjoin(args.checkpoints_dir, 'contrastive_ner_hnlp_concat_no_pretrain')
    else:
        contrastive_ckpt_dir = pjoin(args.checkpoints_dir, 'contrastive_ner_concat_hnlp')

    os.makedirs(contrastive_ckpt_dir, exist_ok=True)

    ckpt_save_path = pjoin(contrastive_ckpt_dir, f"{args.encoder}_lr{args.lr}_epoch{args.epochs}.pth")
    if not args.wo_contrastive:
        if not os.path.isfile(ckpt_save_path):
            model = train_contrastive(
                args,
                model,
                tokenizer,
                id2synonyms,
                train_set,
                ckpt_save_path,
                top_k_ckpts=3,
                load_from=best_ckpt_path,
            )
        else:
            model.load_state_dict(torch.load(ckpt_save_path, map_location=args.device), strict=False)

    classifier_ckpt_dir = pjoin(args.checkpoints_dir, 'classifier')
    if args.append_query:
        classifier_ckpt_dir += '_concat_query'

    if args.wo_pretraining:
        classifier_ckpt_dir += '_no_pretrain'
    if args.wo_contrastive:
        classifier_ckpt_dir += '_no_contrastive'

    os.makedirs(classifier_ckpt_dir, exist_ok=True)
    ckpt_save_path = pjoin(classifier_ckpt_dir, f"{args.encoder}_lr{args.lr}_epoch{args.epochs}.pth")
    model = train_classifier(
        args,
        model,
        tokenizer,
        id2synonyms,
        train_set,
        ckpt_save_path,
        test_set=test_set,
    )


    """
    Evaluation on hNLP dataset
    """

    # print("hNLP Results ---------------------------------------")
    # print("(Single-Span) hNLP Results for SEEN concepts:")
    # evaluate_hnlp_ner(model, tokenizer, hnlp_data_path, baseline='fuzzy', concept_ids=seen_concepts, multi_span=False)
    # print("(Single-Span) hNLP Results for UNSEEN concepts:")
    # evaluate_hnlp_ner(model, tokenizer, hnlp_data_path, baseline='fuzzy', concept_ids=unseen_concepts, multi_span=False)
    # print("(Multi-Span) hNLP Results for SEEN concepts:")
    # evaluate_hnlp_ner(model, tokenizer, hnlp_data_path, baseline='fuzzy', concept_ids=seen_concepts, multi_span=True)
    # print("(Multi-Span) hNLP Results for UNSEEN concepts:")
    # evaluate_hnlp_ner(model, tokenizer, hnlp_data_path, baseline='fuzzy', concept_ids=unseen_concepts, multi_span=True)




    
