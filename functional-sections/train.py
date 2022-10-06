import sys
import pandas as pd
import json
import os
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from distutils.util import strtobool
# import wandb
# from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sectioning.dataloader import TextTurnsDataModule
from sectioning.model import TextTurnsTagger

from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-base")
special_token_dict = {'additional_special_tokens':['<START>','<END>']}
tokenizer.add_special_tokens(special_token_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="datasets", type=str)
    parser.add_argument("--model_dir", default="models", type=str)
    parser.add_argument("--label_columns", nargs='+', default=[])
    parser.add_argument("--max_token_count", default=512, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--n_epochs", default=5, type=int)
    parser.add_argument("--using_gpu", default=True, type=lambda x:bool(strtobool(x)))
    parser.add_argument("--text_type", default="turn", type=str)
    return parser.parse_args()

args = parse_args()    

def run(TRAIN_DIR_TO_FILE, VAL_DIR_TO_FILE, MODEL_DIR_TO_FILE, MAX_TOKEN_COUNT, LABEL_COLUMNS, BATCH_SIZE, N_EPOCHS, gpu, multi_label): #multi_label is true on the turn-level model
       
    train_df = pd.read_pickle(TRAIN_DIR_TO_FILE)
    val_df = pd.read_pickle(VAL_DIR_TO_FILE)
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5
    
    trained_model = TextTurnsTagger(
      n_classes=len(LABEL_COLUMNS),
      tokenizer = tokenizer,
      n_warmup_steps=warmup_steps,
      n_training_steps=total_training_steps,
      multi_label = multi_label
    )
    data_module = TextTurnsDataModule(
      train_df = train_df,
      test_df = val_df,
      tokenizer = tokenizer,
      LABEL_COLUMNS = LABEL_COLUMNS,
      batch_size=BATCH_SIZE,
      max_token_len=MAX_TOKEN_COUNT
    )
    
#    # uncomment if using checkpoints
#     checkpoint_callback = ModelCheckpoint(
#       dirpath="checkpoints",
#       filename="best-checkpoint",
#       save_top_k=1,
#       verbose=True,
#       monitor="val_loss",
#       mode="min"
#     )
    
    if gpu:
        trainer = pl.Trainer(
          #logger=#choose your logger,
          #checkpoint_callback=checkpoint_callback,
          max_epochs=N_EPOCHS,
          gpus=1,
          deterministic=True,
          progress_bar_refresh_rate=30
        )
    else:
        trainer = pl.Trainer(
          #logger=#choose your logger,
          #checkpoint_callback=checkpoint_callback,
          max_epochs=N_EPOCHS,
          progress_bar_refresh_rate=30
        )
    trainer.fit(trained_model, data_module)
    torch.save(trained_model.state_dict(), MODEL_DIR)
    
    return


if __name__ == "__main__":
    
    args = parse_args()
    if args.text_type == 'turn':
        train_filename, val_filename, model_filename = "train_turn.pkl", "val_turn.pkl", "model_sent.ckpt"
        multi_label = True
    else:
        train_filename, val_filename, model_filename = "train_sent.pkl", "val_sent.pkl", "model_sent.ckpt"
        multi_label = False
        
        
    TRAIN_DIR_TO_FILE = os.path.join(args.dataset_dir, train_filename)
    VAL_DIR_TO_FILE = os.path.join(args.dataset_dir, val_filename)
    MODEL_DIR_TO_FILE = os.path.join(args.model_dir, model_filename)
    run(TRAIN_DIR_TO_FILE, VAL_DIR_TO_FILE, MODEL_DIR_TO_FILE, args.max_token_count, args.label_columns, args.batch_size, args.n_epochs, args.using_gpu, multi_label)
    
    
    
    
    
    