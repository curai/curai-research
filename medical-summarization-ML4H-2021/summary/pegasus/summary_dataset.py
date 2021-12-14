import torch.utils.data as data

from ..utils.data.constants import Text, ResponseTypes


class SummaryDataset(data.Dataset):
    def __init__(self, snippet_dataset):
        self.sources, self.targets, self.encounter_ids = [], [], []
        for snippet in snippet_dataset:
            self.encounter_ids.append(snippet.uid)
            self.sources.append(Text.SPACE.join(snippet.get_formatted()))
            if snippet.summary is None:
                self.targets.append(Text.EMPTY_STRING)
            else:
                self.targets.append(Text.remove_neg_token(snippet.summary))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return (self.encounter_ids[index], self.sources[index],
            self.targets[index])


class SummaryCollate(): 
    def __init__(self, tokenizer, source_max_length, target_max_length): 
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __call__(self, batch):
        encounter_ids = [x[0] for x in batch]
        sources = [x[1] for x in batch]
        targets = [x[2] for x in batch]

#        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
#                sources,
#                targets,
#                max_length=self.source_max_length,
#                max_target_length=self.target_max_length,
#                padding="longest",
#                return_tensors="pt"
#        ).data
#        batch_encoding["encounter_ids"] = encounter_ids
#        return batch_encoding
        
        sources = self.tokenizer(sources, padding='longest',
            max_length=self.source_max_length, return_tensors='pt',
            truncation=True)
        targets = self.tokenizer(targets, padding='longest',
            max_length=self.target_max_length, return_tensors='pt',
            truncation=True)
        return {
            'encounter_ids': encounter_ids,
            'source_input_ids': sources['input_ids'],
            'source_attention_mask': sources['attention_mask'],
            'target_input_ids': targets['input_ids'],
            'target_attention_mask': targets['attention_mask'],
        }
