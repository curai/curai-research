from summary.utils.data.datasets.snippet_dataset import SnippetDataset
import torch.utils.data as data

from .summary_dataset import SummaryDataset, SummaryCollate
from summary.utils import write_json
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
from tqdm import tqdm


class Decoder:
    MODEL_NAME = "google/pegasus-cnn_dailymail"

    def __init__(
        self, snippet_dataset, model=None, tokenizer=None, num_beams=8,
        repetition_penalty=2.5, no_repeat_ngram_size=2,
        early_stopping=True, source_max_length=512,
        target_max_length=128, device="cuda", batch_size=8,
        num_workers=4
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping
        self.target_max_length = target_max_length
        self.source_max_length = source_max_length
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.tokenizer = PegasusTokenizer.from_pretrained(
            self.MODEL_NAME) if tokenizer is None else tokenizer
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.MODEL_NAME).to(self.device) if model is None else model

        self.init_dataset(snippet_dataset)

    def init_dataset(self, dataset):
        assert isinstance(dataset, SnippetDataset), "Please provide dataset as a SnippetDataset object."
        self.snippet_dataset = dataset
        self.dataset = SummaryDataset(self.snippet_dataset)
        data_collator = SummaryCollate(self.tokenizer, self.source_max_length,
            self.target_max_length)
        self.data_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, collate_fn=data_collator, shuffle=False,
            pin_memory=True)

    def decode(self, results_path, dataset=None):
        if dataset is not None:
            self.init_dataset(dataset)

        all_snippet_ids, all_predictions, all_summary_sum_log_logits = [], [], []
        for batch in tqdm(self.data_loader, desc="Decoding", leave=True):
            snippet_ids, batch_predictions, sequence_scores = self._decode_batch(batch)
            all_snippet_ids += snippet_ids
            all_predictions += batch_predictions
            all_summary_sum_log_logits += sequence_scores

        results = self.snippet_dataset.to_json()
        for snippet_id, predicted_summary, sequence_score in zip(all_snippet_ids,
                all_predictions, all_summary_sum_log_logits):
            results[snippet_id]["predicted_summary"] = predicted_summary
            results[snippet_id]["predicted_summary_sum_log_logits"] = float(sequence_score)
        write_json(results, results_path)

        return all_snippet_ids, all_predictions, all_summary_sum_log_logits

    # def _decode_batch(self, batch):
    #     encounter_ids = batch['encounter_ids']
    #     source_input_ids = batch['source_input_ids'].to(self.device)
    #     source_attention_mask = batch['source_attention_mask'].to(self.device)

    #     generated_ids = self.model.generate(
    #         source_input_ids,
    #         decoder_start_token_id=self.tokenizer.pad_token_id,
    #         attention_mask=source_attention_mask,
    #         use_cache=True,
    #         num_beams=4,
    #         max_length=128,
    #         repetition_penalty=2.5,
    #         no_repeat_ngram_size=2,
    #         early_stopping=True,
    #         min_length=0,
    #     )
    #     batch_predictions = self._ids_to_clean_text(generated_ids)
    #     return encounter_ids, batch_predictions
    
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
