import os
import json
import torch
import spacy
import collections

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
import pdb


class HNLPContrastiveNERDataset(Dataset):

    def __init__(self, data, tokenizer, id2synonyms):
        super(HNLPContrastiveNERDataset, self).__init__()
        nlp = spacy.load("en_core_web_sm")
        spacy_tokenizer = nlp.tokenizer

        

        self.data = []
        for example in data:
            text = example['text']
            text = re.sub(r"\[\*(.*?)\*\]", '', text)
            tokens = [token.text for token in spacy_tokenizer(text)]
            text_inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, max_length=512)
            synonym_inputs = []

            for entity_id, entity_annotations in example['entities']:
                try:
                    synonyms = id2synonyms[entity_id]
                except:
                    continue

                synonym_inputs.append(tokenizer(synonyms, return_tensors="pt", padding=True))

            # Keep examples with at least one concept
            if len(synonym_inputs) > 0:
                self.data.append({
                    'id': example['id'],
                    'text_inputs': text_inputs,
                    'synonym_inputs': synonym_inputs,
                    'entity_ids': [entity[0] for entity in example['entities']],
                    'tokens': tokens,
                    'multispan': [len(entity[1]) > 1 for entity in example['entities']],
                })

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def ner_collate_fn(batch):
    results = collections.defaultdict(list)
    results['input'] = collections.defaultdict(list)
    results['ids'] = []
    for example in batch:
        for k, v in example['input'].items():
            results['input'][k].append(v.squeeze(0))

        results['entities'].append(example['entities'])
        results['ids'].append(example['id'])

    results['input']['input_ids'] = pad_sequence(results['input']['input_ids'], batch_first=True, padding_value=0)
    results['input']['token_type_ids'] = pad_sequence(results['input']['token_type_ids'], batch_first=True, padding_value=0)
    results['input']['attention_mask'] = pad_sequence(results['input']['attention_mask'], batch_first=True, padding_value=0)
    results['tokens'] = [example['tokens'] for example in batch]

    return results

class ContrastiveNERDataset(Dataset):

    def __init__(self, args, indices=[]):
        super(ContrastiveNERDataset, self).__init__()

        processed_data = torch.load(args.processed_data_path)
        self.data = processed_data['data']

        if len(indices):
            self.indices = set(indices)
            self.data = [x for i, x in enumerate(self.data) if i in set(self.indices)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def ner_collate_fn(batch):
    results = collections.defaultdict(list)
    results['input'] = collections.defaultdict(list)
    results['ids'] = []
    for example in batch:
        for k, v in example['input'].items():
            results['input'][k].append(v.squeeze(0))

        results['entities'].append(example['entities'])
        results['ids'].append(example['id'])
    results['input']['input_ids'] = pad_sequence(results['input']['input_ids'], batch_first=True, padding_value=0)
    results['input']['token_type_ids'] = pad_sequence(results['input']['token_type_ids'], batch_first=True, padding_value=0)
    results['input']['attention_mask'] = pad_sequence(results['input']['attention_mask'], batch_first=True, padding_value=0)
    results['text'] = [example['text'] for example in batch]
    return results

def get_dataloaders_ner(args, train_ids, test_ids):

    train_set = ContrastiveNERDataset(args, train_ids)
    test_set = ContrastiveNERDataset(args, test_ids)
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=ner_collate_fn)

    test_loader = DataLoader(test_set,
                             batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=ner_collate_fn)

    return train_loader, test_loader



def process_json_file_ner(file_path, tokenizer, k_folds):
    with open(file_path) as f:
        json_data = json.load(f)

    data = json_data['data']
    for example in json_data['data']:
        example['input'] = tokenizer(example['text'], return_tensors="pt")


    entity_synonyms = {}
    for entity_name, synonyms in json_data['entity_synonyms'].items():
        entity_synonyms[entity_name] = tokenizer(synonyms, return_tensors="pt", padding=True)

    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    # folds = [x for x in kfold.split(data)]
    results = {'data': data, 'entity_synonyms': entity_synonyms, 'synonyms_names': json_data['entity_synonyms']}
    return results