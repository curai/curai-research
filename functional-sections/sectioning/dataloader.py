import torch
from sectioning.dataset import TextTurnsDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

#wrap the custom dataset into a LightningDataModule
class TextTurnsDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, LABEL_COLUMNS, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.LABEL_COLUMNS = LABEL_COLUMNS
        self.max_token_len = max_token_len
    def setup(self, stage=None):
        self.train_dataset = TextTurnsDataset(
          self.train_df,
          self.tokenizer,
          self.LABEL_COLUMNS,
          self.max_token_len
        )
        self.test_dataset = TextTurnsDataset(
          self.test_df,
          self.tokenizer,
          self.LABEL_COLUMNS,
          self.max_token_len
        )
    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=2
        )
    def val_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=2
        )
    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=2
        )