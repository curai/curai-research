import sys
import os
import argparse
from distutils.util import strtobool
import pandas as pd
import torch
import numpy as np
import json
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from sklearn.metrics import classification_report


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
    parser.add_argument("--threshold_for_multi_label", default=0.5, type=float)
    parser.add_argument("--using_gpu", default=True, type=lambda x:bool(strtobool(x)))
    parser.add_argument("--text_type", default="turn", type=str)
    return parser.parse_args()

args = parse_args()  

def run(DATA_DIR, MODEL_DIR, MAX_TOKEN_COUNT, LABEL_COLUMNS, BATCH_SIZE, THRESHOLD, gpu, multi_label): #multi_label is true on the turn-level model
    
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    
    test_df = pd.read_pickle(DATA_DIR)
    
    trained_model = TextTurnsTagger(
      n_classes=len(LABEL_COLUMNS),
      tokenizer = tokenizer,
      multi_label = multi_label
    )
    trained_model.load_state_dict(
      torch.load(MODEL_DIR)
    )
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    data_loader = TextTurnsDataModule(test_df, test_df, tokenizer, LABEL_COLUMNS = LABEL_COLUMNS, batch_size = BATCH_SIZE)
    data_loader.setup()
    
    predictions = []
    labels = []
    for batch in tqdm(data_loader.test_dataloader(),total=len(test_df)//BATCH_SIZE):
        with torch.no_grad():
            _, prediction = trained_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device)
                  )
        predictions.append(prediction)
        labels.append(batch["labels"])
    predictions = torch.cat(predictions).detach().cpu()
    labels = torch.cat(labels).detach().cpu().int()
    
    if multi_label:
        acc = accuracy(predictions, labels, threshold=THRESHOLD)
        y_true = labels.numpy()
        y_pred = predictions.numpy()
        upper, lower = 1, 0
        y_pred = np.where(y_pred > THRESHOLD, upper, lower)
        
    else:
        total_correct = 0
#         predictions = predictions.numpy()
#         labels = torch.stack(labels).numpy()
        for i in range(len(predictions)):
            if predictions[i].argmax() == labels[i].argmax():
                total_correct += 1
        acc = total_correct/len(predictions)
        
        pred_idxs = torch.Tensor(predictions).argmax(dim=1)
        pred_idxs = pred_idxs.reshape(pred_idxs.shape[0], 1)
        pred_from_idx = torch.zeros(predictions.shape)
        pred_from_idx.scatter_(1, pred_idxs, 1.)
        y_pred = pred_from_idx.numpy()
        y_true = labels.numpy()
        
        
    print('Accuracy is: ' + str(acc))
    print(classification_report(
            y_true,
            y_pred,
            target_names=LABEL_COLUMNS,
            zero_division=0
        ))
    
    return

if __name__ == "__main__":
    
    args = parse_args()
    if args.text_type == 'turn':
        test_filename, model_filename = "test_turn.pkl", "model_sent.ckpt"
        multi_label = True
    else:
        test_filename,model_filename = "test_sent.pkl", "model_sent.ckpt"
        multi_label = False
        
        
    TEST_DIR_TO_FILE = os.path.join(args.dataset_dir, test_filename)
    MODEL_DIR_TO_FILE = os.path.join(args.model_dir, model_filename)
    run(TEST_DIR_TO_FILE, MODEL_DIR_TO_FILE, args.max_token_count, args.label_columns, args.batch_size, args.threshold_for_multi_label, args.using_gpu, multi_label)
    
    
    
    
    
    