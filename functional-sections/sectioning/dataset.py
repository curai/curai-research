import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer

class TextTurnsDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        LABEL_COLUMNS:list,
        max_token_len: int = 128
      ):
        self.tokenizer = tokenizer
        self.data = data
        self.LABEL_COLUMNS = LABEL_COLUMNS
        self.max_token_len = max_token_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text_turn = data_row.text
        labels = data_row[self.LABEL_COLUMNS]
        encoding = self.tokenizer.encode_plus(
          text_turn,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return dict(
          text_turn=text_turn,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=torch.FloatTensor(labels)
        )