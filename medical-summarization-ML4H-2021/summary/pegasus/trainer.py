"""Gradient Accumulation is based on
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/train.py\#L219-L250
"""
import torch
import torch.nn.functional as F
import torch.utils.data as data
import warnings
import pandas as pd

import IPython

from torch.utils.data import dataset
from transformers.utils.dummy_pt_objects import NoRepeatNGramLogitsProcessor
warnings.filterwarnings("ignore", "This overload of add_ is deprecated")

from .summary_dataset import SummaryDataset, SummaryCollate
from summary.utils.data.types.snippet import Snippet
from summary.pegasus import Decoder

from ..metrics.metrics_report import MetricsReport
from summary.utils import metric_utils, read_json, subset_dict, write_json, ConceptAffirmationTagger, KBConceptRecognizer
from tqdm import tqdm
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import (
    Adafactor,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)

from transformers.optimization import get_adafactor_schedule

from torch.utils.tensorboard import SummaryWriter

import wandb
import plotly.graph_objects as go


class Trainer:
    MODEL_NAME = "google/pegasus-cnn_dailymail"

    def __init__(
        self, snippet_dataset, test_dataset=None, model=None, batch_size=4, source_max_length=512,
        target_max_length=128, num_workers=4, device="cuda",
        accumulate_grad_batches=64, n_epochs=3, experiment_name="", writer=None, is_tracking=False
    ):
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = PegasusTokenizer.from_pretrained(self.MODEL_NAME)
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_snippet_dataset = snippet_dataset
        self.test_snippet_dataset = test_dataset

        self.init_train_data_loader()
        self.init_test_data_loader()

        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.MODEL_NAME).to(self.device) if model is None else model
        self.optimizer = Adafactor(self.model.parameters())
        self.scheduler = get_adafactor_schedule(self.optimizer)
        self.experiment_name = experiment_name
        self.writer = writer if writer is not None else SummaryWriter(f"runs/{self.experiment_name}")
        self.is_tracking = is_tracking
        self.entity_recognizer = KBConceptRecognizer()
        self.affirmation_tagger = ConceptAffirmationTagger()

    def init_train_data_loader(self):
        self.dataset = SummaryDataset(self.train_snippet_dataset)
        data_collator = SummaryCollate(self.tokenizer, self.source_max_length,
            self.target_max_length)
        self.data_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, collate_fn=data_collator, shuffle=False,
            pin_memory=True)
    
    def init_test_data_loader(self):
        if self.test_snippet_dataset is None:
            return
            
        self.test_dataset = SummaryDataset(self.test_snippet_dataset)
        data_collator = SummaryCollate(self.tokenizer, self.source_max_length,
            self.target_max_length)
        self.test_data_loader = data.DataLoader(self.test_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, collate_fn=data_collator, shuffle=False,
            pin_memory=True)

    def train(self, checkpoint_save_path, save_checkpoint=True):
        for epoch_i in range(1, self.n_epochs + 1):
            self._train_epoch(epoch_i)
        if save_checkpoint:
            self.model.save_pretrained(checkpoint_save_path)

    def _train_epoch(self, epoch_i):
        effective_batch_loss = 0.
        avg_epoch_loss = 0.

        # Make sure effective batch size is compatible with dataset size.
        while len(self.data_loader) < self.accumulate_grad_batches * self.batch_size:
            self.accumulate_grad_batches /= 2
            print(f"Adjusting effective batch size by changing accumulate_grad_batches to {self.accumulate_grad_batches}")

        assert self.accumulate_grad_batches * self.batch_size < len(self.data_loader)

        pbar = tqdm(enumerate(self.data_loader, start=1),
            desc=f"Epoch {epoch_i}", leave=True, total=len(self.data_loader))
        for step_count, batch in pbar:
            batch_loss = self._train_step(batch)
            batch_loss /= self.accumulate_grad_batches  # norm for grad accumulation
            self.writer.add_scalar("charts/loss_step", batch_loss, step_count)
            effective_batch_loss += batch_loss.item()

            batch_loss.backward()

            if step_count % self.accumulate_grad_batches == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                # self.writer.add_scalar("charts/lr", self.scheduler.get_last_lr(), step_count)

                pbar.set_description(f"Epoch {epoch_i} "
                    f"| step_loss = {effective_batch_loss:.4f}")
                avg_epoch_loss += effective_batch_loss
                effective_batch_loss = 0.

        avg_epoch_loss /= len(self.data_loader.dataset) / (
            self.batch_size * self.accumulate_grad_batches)
        
        self.writer.add_scalar("charts/loss_epoch", avg_epoch_loss, epoch_i)
        pbar.write(f"Epoch {epoch_i} "
            f"| avg_epoch_loss = {avg_epoch_loss:.4f}")

    def _train_step(self, batch):
        outputs = self._step(batch)
        lm_logits = outputs.logits
        labels = batch["target_input_ids"].to(self.device)

        # print(lm_logits.size(), lm_logits.view(-1, lm_logits.shape[-1]).size(), labels.size(), labels.view(-1).size())
        # print(torch.min(lm_logits), torch.max(lm_logits))

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]),
            labels.view(-1), ignore_index=0)
        

            
        return loss

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_input_ids = self.shift_tokens_right(
            batch["target_input_ids"], pad_token_id).to(self.device)
        decoder_input_ids[:, 0] = self.tokenizer.pad_token_id

        return self.model(
            input_ids=batch["source_input_ids"].to(self.device),
            attention_mask=batch["source_attention_mask"].to(self.device),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=batch["target_attention_mask"].to(
                self.device),
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

    def _step_generate(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_input_ids = self.shift_tokens_right(
            batch["target_input_ids"], pad_token_id).to(self.device)
        decoder_input_ids[:, 0] = self.tokenizer.pad_token_id

        return self.model.generate(
            input_ids=batch["source_input_ids"].to(self.device),
            attention_mask=batch["source_attention_mask"].to(self.device),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=batch["target_attention_mask"].to(
                self.device),
            num_beams=2,
            use_cache=True,
            repetition_penalty=2.0
        ).cpu()
    
    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int):
        """
        Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        """
        prev_output_tokens = input_ids.clone()
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
        index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
        prev_output_tokens[:, 0] = decoder_start_tokens
    
        return prev_output_tokens

    def evaluate(self):

        def _evaluate_dataset(self, decoder, log_filename, metric_filename, dataset, tag=""):
            '''
            Evaluate a dataset and compute metrics, with logging to wandb if self.is_tracking is True.
            '''
            ids, summaries, scores = decoder.decode(f"results/pegasus/{log_filename}.json", dataset)
            all_results = metric_utils.get_common_results_among_groups([log_filename])
            auto_metrics, gold_metrics = metric_utils.compute_metrics_for_all_groups(all_results,[log_filename], tag=tag)
            write_json(auto_metrics, metric_filename)

            data = {
                "snippet_id": ids,
                "snippets": [snippet.get_formatted(text=True) for snippet in dataset],
                "gt_summary": [snippet.summary for snippet in dataset],
                "predicted_summary": summaries,
                "confidence (sll)": scores
            }
            result_table = wandb.Table(dataframe=pd.DataFrame(data=data))

            sum_log_logits_hist = go.Figure(data=[go.Histogram(x=scores)])
            sum_log_logits_hist.update_layout(
                title_text='Histogram of Predicted Summary\'s Sum of Log Logits', # title of plot
                xaxis_title_text='Sum of Log Logits', # xaxis label
                yaxis_title_text='Count', # yaxis label
                bargap=0.2, # gap between bars of adjacent location coordinates
                bargroupgap=0.1 # gap between bars of the same location coordinates
            )

            if self.is_tracking:
                wandb.log(auto_metrics[log_filename])
                wandb.log(gold_metrics[log_filename])
                wandb.log({
                    f"Dataset ({tag}) Evaluation": result_table, 
                    f"Dataset ({tag}) Evaluation - Predicted Summary Confidence Histogram": wandb.Plotly(sum_log_logits_hist)
                    })
    
        def _get_concept_affirmation_recall_filtered_dataset(self):
            '''
            Get set of samples from self.test_snippet_dataset that have a concept and affirmation recall of 1.0 with 
            it's corresponding snippet. On the set of 501 test samples from the below path it results a set of 109 samples. 
            "varun/naacl_paper/datasets/test_datasets/human-501.json"
            '''
            snippets = [self.test_snippet_dataset[snippet_id].get_formatted(text=True) for snippet_id in self.test_snippet_dataset.snippet_ids]
            summaries = [self.test_snippet_dataset[snippet_id].summary for snippet_id in self.test_snippet_dataset.snippet_ids]

            test_snippet_concepts = [self.entity_recognizer.get_concepts(snippet) for snippet in snippets]
            test_summary_concepts = [self.entity_recognizer.get_concepts(summary) for summary in summaries]
            test_concept_recall = [self._compute_concept_recall(snippet_concepts, summary_concepts) \
                                    for snippet_concepts, summary_concepts in zip(test_snippet_concepts, test_summary_concepts)]

            test_snippet_affirmations = [self.affirmation_tagger.get_affirmations(snippet, concepts) \
                                    for snippet, concepts in zip(snippets, test_snippet_concepts)]
            test_summary_affirmations = [self.affirmation_tagger.get_affirmations(summary, concepts) \
                                    for summary, concepts in zip(summaries, test_summary_concepts)]
            test_affirmation_recall = [self._compute_affirmations_recall(snippet_affirmations, summary_affirmations) \
                                    for snippet_affirmations, summary_affirmations in zip(test_snippet_affirmations, test_summary_affirmations)]

            filtered_ids = [snippet_id for index, snippet_id in enumerate(self.test_snippet_dataset.snippet_ids) \
                                    if test_concept_recall[index] >= 1.0 and test_affirmation_recall[index] >= 1.0]
            print(f"Length of filtered Ids: {len(filtered_ids)}")

            filtered_test_snippet_dataset = self.test_snippet_dataset.subset(filtered_ids)

            return filtered_test_snippet_dataset


        decoder = Decoder(self.test_snippet_dataset, model=self.model, tokenizer=self.tokenizer)

         # Log test set metrics
        log_filename = f"decoded_{self.experiment_name}"
        metric_filename = f"metrics/{self.experiment_name}_metrics.json"
        _evaluate_dataset(self, decoder, log_filename, metric_filename, self.test_snippet_dataset)

        # Log train set metrics
        log_filename = f"decoded_{self.experiment_name}_train"
        metric_filename = f"metrics/{self.experiment_name}_train_metrics.json"
        _evaluate_dataset(self, decoder, log_filename, metric_filename, self.train_snippet_dataset, tag="train")

        # Log test set filtered for samples with only concept and affirmation recall of 1.0
        # filtered_test_snippet_dataset = _get_concept_affirmation_recall_filtered_dataset(self)
        # log_filename = f"decoded_{self.experiment_name}_filtered_test"
        # metric_filename = f"metrics/{self.experiment_name}_filtered_test_metrics.json"
        # _evaluate_dataset(self, decoder, log_filename, metric_filename, filtered_test_snippet_dataset, tag="filtered_test")


    def decode(self, result_save_path, dataset=None, existing_labels=None):
        if dataset is not None:
            self.test_snippet_dataset = dataset
            self.init_test_data_loader()

        all_snippet_ids, all_predictions, all_summary_sum_log_logits = [], [], []
        for batch in tqdm(self.test_data_loader, desc="Decoding", leave=True):
            snippet_ids, batch_predictions, sequence_scores = self._decode_batch(batch)
            all_snippet_ids += snippet_ids
            all_predictions += batch_predictions
            all_summary_sum_log_logits += sequence_scores


        results = self.test_snippet_dataset.to_json()
        for snippet_id, predicted_summary, summary_sum_log_logits in zip(all_snippet_ids,
                all_predictions, all_summary_sum_log_logits):

            snippet = self.test_snippet_dataset[snippet_id].get_formatted(text=True)
            summary = self.test_snippet_dataset[snippet_id].summary
            results[snippet_id]["snippet_concepts"] = self.entity_recognizer.get_concepts(snippet)

            results[snippet_id]["gt_summary"] = summary
            # results[snippet_id]["gt_summary_concepts"] = self.entity_recognizer.get_concepts(summary)

            results[snippet_id]["predicted_summary"] = predicted_summary
            results[snippet_id]["predicted_summary_concepts"] = self.entity_recognizer.get_concepts(predicted_summary)

            snippet_affirmations = self.affirmation_tagger.get_affirmations(
                snippet, results[snippet_id]["snippet_concepts"])
            predicted_summary_affirmations = self.affirmation_tagger.get_affirmations(
                results[snippet_id]["predicted_summary"], results[snippet_id]["predicted_summary_concepts"])
            results[snippet_id]["snippet_affirmations"] = snippet_affirmations
            results[snippet_id]["predicted_affirmations"] = predicted_summary_affirmations
            results[snippet_id]["concept_recall_w_snippet"] = self._compute_concept_recall(results[snippet_id]['snippet_concepts'], results[snippet_id]['predicted_summary_concepts'])
            results[snippet_id]["affirmation_recall_w_snippet"] = self._compute_affirmations_recall(results[snippet_id]["snippet_affirmations"], results[snippet_id]["predicted_affirmations"])
            results[snippet_id]["predicted_summary_sum_log_logits"] = float(summary_sum_log_logits)
        
        if existing_labels is not None:
            results.update(existing_labels)
        
        write_json(results, result_save_path)

        if self.is_tracking:
            snippet_ids = results.keys()
            data = {
                "snippet_id": snippet_ids,
                "snippets": [results[id]['snippet'] for id in snippet_ids],
                "summaries": [results[id]['gt_summary'] for id in snippet_ids],
                "predicted_summary": [results[id]['predicted_summary'] for id in snippet_ids],
                "predicted_summary_sum_log_logits": [results[id]['predicted_summary_sum_log_logits'] for id in snippet_ids],
                "concept_recall_w_snippet": [results[id]['concept_recall_w_snippet'] for id in snippet_ids],
                "affirmation_recall_w_snippet": [results[id]['affirmation_recall_w_snippet'] for id in snippet_ids],
                "snippet_concepts": [results[id]['snippet_concepts'] for id in snippet_ids],
                # "summary_concepts": [results[id]['gt_summary_concepts'] for id in snippet_ids],
                "predicted_summary_concepts": [results[id]['predicted_summary_concepts'] for id in snippet_ids],
                "snippet_affirmations": [str(results[id]['snippet_affirmations']) for id in snippet_ids],
                "predicted_affirmations": [str(results[id]['predicted_affirmations']) for id in snippet_ids]
            }

            result_table = wandb.Table(dataframe=pd.DataFrame(data=data))
            wandb.log({"Generated Pseudo-Labels": result_table})

            concept_hist = go.Figure(data=[go.Histogram(x=data['concept_recall_w_snippet'])])
            concept_hist.update_layout(
                title_text='Histogram of Concept Recall between Predicted Summary and Snippet', # title of plot
                xaxis_title_text='Recall Score', # xaxis label
                yaxis_title_text='Count', # yaxis label
                bargap=0.2, # gap between bars of adjacent location coordinates
                bargroupgap=0.1 # gap between bars of the same location coordinates
            )

            affirmation_hist = go.Figure(data=[go.Histogram(x=data['affirmation_recall_w_snippet'])])
            affirmation_hist.update_layout(
                title_text='Histogram of Affirmation Recall between Predicted Summary and Snippet', # title of plot
                xaxis_title_text='Affirmation Score', # xaxis label
                yaxis_title_text='Count', # yaxis label
                bargap=0.2, # gap between bars of adjacent location coordinates
                bargroupgap=0.1 # gap between bars of the same location coordinates
            )

            sum_log_logits_hist = go.Figure(data=[go.Histogram(x=data['predicted_summary_sum_log_logits'])])
            sum_log_logits_hist.update_layout(
                title_text='Histogram of Predicted Summary\'s Sum of Log Logits', # title of plot
                xaxis_title_text='Sum of Log Logits', # xaxis label
                yaxis_title_text='Count', # yaxis label
                bargap=0.2, # gap between bars of adjacent location coordinates
                bargroupgap=0.1 # gap between bars of the same location coordinates
            )
            
            wandb.log({"Concept Recall Histogram": wandb.Plotly(concept_hist), 
                        "Affirmation Recall Histogram": wandb.Plotly(affirmation_hist),
                        "Predicted Summary's Sum of Log Logits Histogram": wandb.Plotly(sum_log_logits_hist)})

        return all_snippet_ids, all_predictions

    def _compute_concept_recall(self, gold_concepts, pred_concepts):
        gold_concepts = set(gold_concepts)
        pred_concepts = set(pred_concepts)
        true_pos = gold_concepts.intersection(pred_concepts)
        return len(true_pos) / len(gold_concepts) if len(gold_concepts) > 0 else 0
    
    def _compute_affirmations_recall(self, gold_affirmations,
            pred_affirmations):
        true_pos = 0
        for concept, negation in gold_affirmations.items():
            if pred_affirmations.get(concept, "") == negation:
                true_pos += 1
        return true_pos / len(gold_affirmations) if len(gold_affirmations) > 0 else 0

    def _decode_batch(self, batch):
        encounter_ids = batch['encounter_ids']
        source_input_ids = batch['source_input_ids'].to(self.device)
        source_attention_mask = batch['source_attention_mask'].to(self.device)

        generated_ids_dict = self.model.generate(
            source_input_ids,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            attention_mask=source_attention_mask,
            use_cache=True,
            num_beams=4,
            max_length=150,
            repetition_penalty=2.5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            min_length=0,
            return_dict_in_generate=True,
            output_scores=True,
        )

        batch_predictions = self._ids_to_clean_text(generated_ids_dict['sequences'])
        sequence_scores = list(generated_ids_dict['sequences_scores'].cpu().numpy())
        return encounter_ids, batch_predictions, sequence_scores

    def _ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return list(map(str.strip, gen_text))

class SparseUpdateTrainer(Trainer):
    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def _train_step(self, batch):
        outputs = self._step(batch)
        lm_logits = outputs.logits
        labels = batch["target_input_ids"].to(self.device)

        # print(lm_logits.size(), lm_logits.view(-1, lm_logits.shape[-1]).size(), labels.size(), labels.view(-1).size())
        # print(torch.min(lm_logits), torch.max(lm_logits))

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]),
            labels.view(-1), ignore_index=0)
        
        for name, params in self.model.named_parameters():
            device = params.device
            self.mask[name] = self.mask[name].to(device)

            params.grad.data.copy_(params.grad.data * self.mask[name].data)
        
            
        return loss

    # def training_step(self, *args, **kwargs):
    #     loss = super().training_step(*args, **kwargs)

    #     # mask out the gradients
    #     for name, params in self.model.named_parameters():
    #         device = params.device
    #         self.mask[name] = self.mask[name].to(device)

    #         params.grad.data.copy_(params.grad.data * self.mask[name].data)

    #     return loss

def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    pbar = tqdm(enumerate(data_loader, start=1),
            desc=f"Computing mask...", leave=True, total=len(data_loader))
            
    for idx, inputs in pbar:
        print(idx)
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data
        
        model.zero_grad()

    return gradients_dict

def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        logits = return_dicts["logits"]

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict

def create_mask_gradient(model, train_dataset, data_collator, num_workers, num_samples, keep_ratio, sample_type, grad_type):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=num_workers,
        collate_fn=data_collator, shuffle=True)

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


