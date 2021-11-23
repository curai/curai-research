import random
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, PegasusConfig
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import argparse
import tqdm
import time
import gc
import json

from summary.utils.data.datasets import SnippetDataset
from summary.utils.data.filters import SnippetFilter
from summary.pegasus import Trainer

from torch.utils.tensorboard import SummaryWriter
from summary.utils import read_json, subset_dict, write_json

import summary.utils.metric_utils as metric_utils

from pynvml import *

ROOT = "<root directory>"
UNLABELED_DATA_PATH = 'example_data/unlabeled_data.json'
TRAIN_DATA_PATH = 'example_data/train_data.json'
TEST_DATA_PATH = 'example_data/test_data.json'

TRAIN_DATASET_MAX_SIZE = 6400

def prepare_dataset(samples=1000, seed=0, return_unused=False, unused_samples=-1):
    '''
    Prepares training and testing datasets for PEGASUS fine-tuning, with optional
    parameters to return leftover samples - which can be used for sample selection
    or pseudo-labeling.

    Parameters:

        samples (int): Number of samples to include in training dataset.
        seed (int): Seed to shuffle samples with.
        return_unused (bool): If true, will return samples not selected in the training set.
        unused_samples (int): Used with return_unused set to true, returns specific number
                                of samples in unused dataset. -1 returns all unused samples.

    '''

    print("Preparing snippet datasets...")

    dataset_paths = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    datasets = []

    for dataset_path in dataset_paths:
        dataset = SnippetDataset.from_json_file(dataset_path)
     
        if dataset_path == TRAIN_DATA_PATH:
            print(f"Setting seed to {seed}.")
            random.seed(seed)
            snippet_ids = random.sample(dataset.snippet_ids, len(dataset))
            datasets.append(dataset.subset(snippet_ids[:samples]))

            if return_unused:
                if unused_samples > len(dataset) - samples:
                    print(f"Adjusting number of samples in unused/pseduo-labeled dataset to: {len(dataset) - samples}")
                    unused_samples = len(dataset) - samples

                ids_to_return = snippet_ids[-unused_samples:] if unused_samples != -1 else snippet_ids[samples:]
                print("Ids to return: ", len(ids_to_return))
                datasets.append(dataset.subset(reversed(ids_to_return)))
        else:
            datasets.append(dataset)
        
    return datasets

def prepare_unl_dataset(args):
    dataset_paths = [UNLABELED_DATA_PATH]
    datasets = []

    for dataset_path in dataset_paths:
        dataset = SnippetDataset.from_json_file(dataset_path)
        dataset.clean(filter_by_ratio=args.filter_unlabeled_by_ratio)
        
        datasets.append(dataset)
        
    return datasets[0]

def parse_args(parser):
    parser.add_argument("--pretrained-model", type=str, required=True, help="Path or name of pre-trained model to pull PEGASUS from.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to train on.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=8, help="Size of batch.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=16, help="Number of batches to accumulate before computing update.")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name, used by WandB for experiment tracking.")
    parser.add_argument("--track", action="store_true", help="If true, WandB will track experiment.")
    parser.add_argument("--seed", type=str, default=0, help="Random sampling seed.")
    parser.add_argument("--from-pseudo-label", type=str, choices=['y','n'], help='Pseudo-labeled dataset path.')
    parser.add_argument("--pseudo-label-path", type=str, default="", help="Specify if overriding the path stored in last_pl_path.txt")
    parser.add_argument("--pseudo-label-strategy", type=str, choices=["recall", "sum_log_logits"], default="recall", help="Pseudo-label filtering strategy.")
    parser.add_argument("--sum-log-logits-threshold", type=float, default=-0.1, help="Threshold for sum log logits pseudo-labeling selection.")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration if training in a loop.")
    parser.add_argument("--label-unlabeled-set", action="store_true", help='Use unlabeled dataset for pseudo-labeling.')
    parser.add_argument("--filter-unlabeled-by-ratio", type=int, default=-1, help="Max ratio of unlabeled snippet to summary, -1 indicates no filtering.")
    parser.add_argument("--sll-human-threshold-window", nargs=2, type=float, default=[0.0, 0.0], help="Range of sll threshold ")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout of model.")
    parser.add_argument("--sparse-training", action="store_true", help="Use FISH sparse training mask for PEGASUS.")
    args = parser.parse_args()

    return args

def get_previous_model():
    if os.path.exists(os.path.join(ROOT, "previous_model_checkpoint.txt")):
        with open(os.path.join(ROOT, "previous_model_checkpoint.txt"), 'r') as f:
                past_exp_name = f.read()
                return os.path.join(os.getcwd(), "checkpoints", past_exp_name)
    else:
        raise FileNotFoundError("Previous model path not found.")

def run_experiment(args, experiment_name):
    if args.track:
        import wandb
        run = wandb.init(
            project='<project',
            entity="<your_wandb_entity>",
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            save_code=True)

    writer = SummaryWriter(f"runs/{experiment_name}")
    
    # Initialize Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset, test_dataset = prepare_dataset(samples=args.num_samples, seed=args.seed)
    filtered_dataset_json = None

    if args.from_pseudo_label == "y":

        to_log = {}

        # Read in last pseudo-labeled dataset if overriding path is not path provided.
        if args.pseudo_label_path == "":
            path_to_read = os.path.join(ROOT, "last_pl_path.txt")
            with open(path_to_read, 'r') as f:
                pseudo_label_path = f.read()
        else:
            assert os.path.isfile(args.pseudo_label_path), \
                f"Provided pseudo label path {args.pseudo_label_path} is not valid. "
            pseudo_label_path = args.pseudo_label_path
            to_log["pseudo-label-path"] = pseudo_label_path

        dataset = SnippetDataset.from_json_file(pseudo_label_path, is_pseudo_label=True)
        dataset.clean(filter_by_ratio=args.filter_unlabeled_by_ratio)

        if args.pseudo_label_strategy == "recall":
            # We filter out samples that have a concept and affirmation recall lower than 1.
            concept_recall_threshold = 1.0
            affirmation_recall_threshold = 1.0
            snippet_filter = SnippetFilter(
                concept_recall_threshold=concept_recall_threshold,
                affirmation_recall_threshold=affirmation_recall_threshold,
                keep_fixed=True)
            
            dataset.apply_snippet_filter(snippet_filter)
            assert dataset[dataset.snippet_ids[-1]].concept_recall >= concept_recall_threshold, "Sample not below threshold."
            assert dataset[dataset.snippet_ids[-1]].affirmation_recall >= affirmation_recall_threshold, "Sample not below threshold."

        elif args.pseudo_label_strategy == "sum_log_logits":

            sum_log_logits_threshold=args.sum_log_logits_threshold
            snippet_filter = SnippetFilter(
                sum_log_logits_threshold=sum_log_logits_threshold,
                keep_fixed=True,
            )

            dataset.apply_snippet_filter(snippet_filter)
            print(dataset[dataset.snippet_ids[-1]].predicted_summary_sum_log_logits)
            # assert dataset[dataset.snippet_ids[-1]].predicted_summary_sum_log_logits >= sum_log_logits_threshold

        print(f"Length of train dataset before adding ids: {len(train_dataset)}.")
        print(f"Adding {len(dataset)} ids to training set.")

        for snippet in dataset:
            train_dataset.add(snippet)
        
        to_log["num_pseudo_labeled_pts"] = len(dataset)
        
        if not args.label_unlabeled_set and args.sll_human_threshold_window[1] is not 0.0:
            hl_dataset = SnippetDataset.from_json_file(pseudo_label_path, is_pseudo_label=True)
            hl_dataset.clean(filter_by_ratio=args.filter_unlabeled_by_ratio)

            low, high = args.sll_human_threshold_window
            ids_to_add = hl_dataset.get_ids_in_confidence_window(low, high)
            full_dataset, _ = prepare_dataset(samples=TRAIN_DATASET_MAX_SIZE, seed=args.seed)

            to_log["num_human_labeled_pts"] = len(ids_to_add)

            for id in ids_to_add:
                
                train_dataset.add(full_dataset[id])
                dataset.add(full_dataset[id])

            print(f"Adding {len(ids_to_add)} human labels to training set.")

        else:
            to_log["num_human_labeled_pts"] = 0
            
        if args.track:
            wandb.log(to_log)

        print(f"After filtering, total train dataset is of size: {len(train_dataset)}")

        filtered_dataset_json = dataset.get_source_json(keep_only_existing_ids=True)

        for snippet_id in filtered_dataset_json:
            filtered_dataset_json[snippet_id]["fixed"] = "True"

    if args.track:
        wandb.log({
            "num_training_pts": len(train_dataset),
        })

    model_name_or_path = get_previous_model() if args.pretrained_model == "from_previous" else args.pretrained_model
    config = PegasusConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path, dropout=args.dropout)
    model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=config).to(device)
    
    while len(train_dataset) * args.epochs / args.batch_size < 200:
        args.epochs += 1
        print(f"Increasing total number of epochs to {args.epochs} to ensure at least 200 steps. Curr: {len(train_dataset) * args.epochs / args.batch_size}, {len(train_dataset)}")
    
    if args.sparse_training:
        # Brief attempt at incorporating sparse updating to regularize network.
        pass
        # trainer = SparseUpdateTrainer(snippet_dataset=train_dataset, test_dataset=test_dataset, model=model, batch_size=args.batch_size,
        #         accumulate_grad_batches=args.accumulate_grad_batches, n_epochs=args.epochs, device=device, experiment_name=experiment_name, 
        #         writer=writer, is_tracking=args.track)
        # data_collator = SummaryCollate(trainer.tokenizer, trainer.source_max_length,
        #     trainer.target_max_length)
        # data_loader = data.DataLoader(trainer.dataset, batch_size=1,
        #     num_workers=trainer.num_workers, collate_fn=data_collator, shuffle=False,
        #     pin_memory=True)
        # mask = create_mask_gradient(model=trainer.model, train_dataset=data_loader, data_collator=data_collator, num_workers=trainer.num_workers, 
        #         num_samples=args.num_samples, keep_ratio=0.005, sample_type='label', grad_type='square')
        # trainer.mask = mask
    
    else:
        trainer = Trainer(snippet_dataset=train_dataset, test_dataset=test_dataset, model=model, batch_size=args.batch_size,
                accumulate_grad_batches=args.accumulate_grad_batches, n_epochs=args.epochs, device=device, experiment_name=experiment_name, 
                writer=writer, is_tracking=args.track)

    
    trainer.train(checkpoint_save_path=f"checkpoints/{experiment_name}")

    trainer.evaluate()

    # Pseudo-Labeling - Label all samples from the full dataset that were not in the train dataset.
    if args.label_unlabeled_set:
        full_dataset = prepare_unl_dataset(args)
    else:
        full_dataset, _ = prepare_dataset(samples=TRAIN_DATASET_MAX_SIZE, seed=args.seed)

    train_dataset_ids = set(train_dataset.snippet_ids)
    ids_to_pseudo_label = []

    for snippet in full_dataset:
        if snippet.uid not in train_dataset_ids:
            ids_to_pseudo_label.append(snippet.uid)
    
    pseudo_label_dataset = full_dataset.subset(ids_to_pseudo_label)
    pseudo_label_save_path = f"results/pegasus/decoded_{experiment_name}.json"
    trainer.decode(pseudo_label_save_path, dataset=pseudo_label_dataset, existing_labels=filtered_dataset_json)

    # Save current experiment name as past model.
    if os.path.exists(os.path.join(ROOT, "previous_model_checkpoint.txt")):
        os.remove(os.path.join(ROOT, "previous_model_checkpoint.txt"))
    
    with open(os.path.join(ROOT, "previous_model_checkpoint.txt"), 'w') as f:
            f.write(experiment_name)

    # Remove existing file and replace with new pseudo-label dataset path.
    if args.from_pseudo_label == "y":
        os.remove(os.path.join(ROOT, "last_pl_path.txt"))
        
    with open(os.path.join(ROOT, "last_pl_path.txt"), 'w') as f:
            f.write(pseudo_label_save_path)

    print("Saved pseudo-labels to:", os.path.join(os.getcwd(), "last_pl_path.txt"))

    return pseudo_label_save_path

if __name__ == "__main__":

    # Setup training arguments and WandB
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    experiment_name = f"{args.exp_name}_{int(time.time())}" if args.from_pseudo_label != "y" else f"pseudolabel_{args.exp_name}_{int(time.time())}"
    pseudo_label_save_path = run_experiment(args, experiment_name)

    print(pseudo_label_save_path)
    



