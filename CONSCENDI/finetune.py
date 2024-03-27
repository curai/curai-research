import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import subprocess
import os
import re

# argparse a string that is either "prep" or "finetune"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="prep or finetune")
# add optional argument for train_csv
parser.add_argument("--train_csv", help="path to train csv")
# add optional argument for model
parser.add_argument("--model", help="first letter of model to finetune, e.g. 'd' for davinci, 'c' for curie, etc.")

args = parser.parse_args()
if args.mode == "prep":
    if args.train_csv is None:
        raise ValueError("Please specify path to train csv")
    def prep_dataset(train_csv):
        # convos = pd.read_csv(train_csv)
        command = f"openai tools fine_tunes.prepare_data -f '{train_csv}' -q"
        subprocess.run(command, shell=True)
    prep_dataset(args.train_csv)
elif args.mode == "finetune":
    def finetune(model):
        command = f"""openai api fine_tunes.create -t "multi-rule-guardrail/data_mr_new_2/train_dataset_prepared_train.jsonl" -v "multi-rule-guardrail/data_mr_new_2/train_dataset_prepared_valid.jsonl" --compute_classification_metrics --classification_n_classes 11 -m {model}"""
        subprocess.run(command, shell=True)
    finetune(model=args.model)


