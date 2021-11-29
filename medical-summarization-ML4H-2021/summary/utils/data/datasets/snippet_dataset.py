import os
import numpy as np
from transformers.utils.dummy_pt_objects import NoRepeatNGramLogitsProcessor

from .... import is_history_gathering as hist_clf
from ..constants import ResponseTypes, Speakers, Text
from ...dict import AttrDict
from ...file_io import read_json
from ..filters import SnippetFilter
from .helpers import read_json_snippet
from ..types.chat import Chat
from ..types.snippet import Snippet

import random


class SnippetDataset:
    """
    Dataset of Snippet

    Parameters
    ----------
    snippets : list of Snippet
    classify_history_gathering : bool
        Whether to classify if snippets are gathering history or not
    """

    def __init__(self, snippets, source_files=None):
        self._snippets = self._filter_empty_summaries_with_no_tags(snippets)
        self.snippet_ids = [snippet.uid for snippet in self._snippets]
        self.source_files = source_files
        self._snippets_data = {}
        for snippet in self._snippets:
            self._snippets_data[snippet.uid] = snippet

    

    # ------------------ PUBLIC METHODS ------------------
    def add(self, snippet):
        self._snippets.append(snippet)
        self.snippet_ids.append(snippet.uid)
        self._snippets_data[snippet.uid] = snippet

    def remove(self, snippet):
        self._snippets.remove(snippet)
        self.snippet_ids.remove(snippet.uid)
        del self._snippets_data[snippet.uid]

    def subset(self, snippet_ids):
        snippets = []
        for snippet_id in snippet_ids:
            snippets.append(self._snippets_data[snippet_id])
        return SnippetDataset(snippets)

    def apply_snippet_filter(self, snippet_filter, remove=True):
        """
        Applies a SnippetFilter

        Parameters
        ----------
        snippet_filter : SnippetFilter
        remove : bool
            Whether to remove or keep things specified in
            `snippet_filter`
        """
        if snippet_filter.keep_fixed:
            snippets_to_keep = self.get_fixed_snippets()
            ids_to_keep = set([snippet.uid for snippet in snippets_to_keep])
            fixed_snippet_dataset = self.subset(ids_to_keep)

            for id in ids_to_keep:
                self.remove(self._snippets_data[id])
            print(f"Number of fixed Ids: {len(ids_to_keep)}")

        else:
            ids_to_keep = []

        bad_snippet_ids = set(
            self._filter_length(snippet_filter.max_num_turns) +
            self._filter_source(snippet_filter.source) +
            self._filter_tags(snippet_filter.tags) +
            self._filter_response_types(snippet_filter.response_types) +
            self._filter_only_speaker(snippet_filter.only_speaker) + 
            self._filter_concept_recall_threshold(snippet_filter.concept_recall_threshold) + 
            self._filter_affirmation_recall_threshold(snippet_filter.affirmation_recall_threshold) + 
            self._filter_sum_log_logits_threshold(snippet_filter.sum_log_logits_threshold, ids_to_keep)
        )

        if snippet_filter.snippet_ids is not None:
            bad_snippet_ids.union(set(snippet_filter.snippet_ids))

        if not remove:
            bad_snippet_ids = list(set(self.snippet_ids) - bad_snippet_ids)

        added = 0
        for bad_id in bad_snippet_ids:
            self.remove(self._snippets_data[bad_id])
            added += 1
        print(f"Removed {added} snippets.")

        # Re-add fixed snippets
        if snippet_filter.keep_fixed:
            for snippet in fixed_snippet_dataset:
                self.add(snippet)

    def remove_non_history_gathering(self):
        for snippet_id in self.snippet_ids.copy():
            snippet = self._snippets_data[snippet_id]
            if not snippet.is_history_gathering:
                self.remove(snippet)

    def to_json(self):
        res = {}
        for snippet_id, snippet in self._snippets_data.items():
            res.update(snippet.to_json())
        return res
    
    def get_source_json(self, keep_only_existing_ids=False):
        if self.source_files is None:
            return None
        
        json_data = {}
        for source_file in self.source_files:
            json_data.update(read_json(source_file))

        if keep_only_existing_ids:
            snippet_ids_to_remove = []

            for snippet_id in json_data:
                if snippet_id not in self.snippet_ids:
                    snippet_ids_to_remove.append(snippet_id)
            
            for snippet_id in snippet_ids_to_remove:
                del json_data[snippet_id]
        
        return json_data
    
    def get_fixed_snippets(self):
        fixed_snippets = [snippet for snippet in self._snippets.copy() if snippet.fixed == "True"]
        return fixed_snippets

    def clean(self, filter_by_ratio=-1):
        '''
        Clean snippet dataset by running de-duplication on snippets, removing
        snippets shorter than length 3.
        '''

        uncleaned_dataset_length = len(self)

        formatted_snippets = [self[id].get_formatted(text=True) for id in self.snippet_ids]
        _, unique_idxs = np.unique(np.asarray(formatted_snippets), return_index=True)
        unique_idxs = set(unique_idxs)

        def to_remove(index, filter_by_ratio=-1) -> bool:
            if index not in unique_idxs:
                return True
            elif len(formatted_snippets[index].split(" ")) <= 3:
                return True
            elif filter_by_ratio > 0 and len(formatted_snippets[index]) >= len(self[self.snippet_ids[index]].summary) * filter_by_ratio:
                return True
            
            return False

        for index, id in enumerate(self.snippet_ids):
            if to_remove(index, filter_by_ratio=filter_by_ratio):
                self.remove(self[id])
        
        cleaned_dataset_length = len(self)

        print(f"Finished cleaning dataset. Old Length: {uncleaned_dataset_length}, New Length: {cleaned_dataset_length}.")
    
    def get_ids_in_confidence_window(self, low, high, is_descending=True):
        '''
        Given a percentile range, return the snippet ids in that percentile of confidence scores.
        Default is to have confidence scores sorted in descending order.

        Ex: low=0.05, high=0.20 - Returns snippets in between the top 5% and top 20% of confidence scores.
        '''
        ids = []
        snippets = [snippet for snippet in self if snippet.predicted_summary_sum_log_logits is not None]

        if low != high:
            snippets.sort(key=lambda x: x.predicted_summary_sum_log_logits, reverse=is_descending)

            low_threshold_index = int(len(snippets) * low)
            high_threshold_index = int(len(snippets) * high)

            print(len(snippets), low_threshold_index, high_threshold_index)

            ids = [snippets[index].uid for index in range(low_threshold_index, high_threshold_index)]
        else:
            # Select random subset of labels across confidence distribution.
            random.shuffle(snippets)
            num_ids_to_take = int(len(snippets) * low)
            ids = [snippet.uid for snippet in snippets[:num_ids_to_take]]

        return ids


    # ------------------ MAGIC METHODS ------------------
    def __len__(self):
        return len(self._snippets)

    def __getitem__(self, snippet_id):
        return self._snippets_data[snippet_id]

    def __iter__(self):
        for snippet in self._snippets:
            yield snippet

    # ------------------ CLASS METHODS ------------------
    @classmethod
    def from_json_file(cls, json_files, is_pseudo_label=False):
        if not isinstance(json_files, list):
            json_files = [json_files]
        
        snippets = []

        for json_file in json_files:
            print(f"Reading snippet dataset from {json_file}")
            data = read_json(json_file)
            for snippet_id, data_pt in data.items():
                snippets.append(read_json_snippet(snippet_id, data_pt, is_pseudo_label=is_pseudo_label))
                
        return cls(snippets=snippets, source_files=json_files)
    
    @classmethod
    def from_dict(cls, dicts):
        if not isinstance(dicts, list):
            dicts = [dicts]
        
        snippets = []

        for dict in dicts:
            for snippet_id, data_pt in dict.items():
                snippets.append(read_json_snippet(snippet_id, data_pt))
                
        return cls(snippets=snippets)
    

  
    def _filter_length(self, max_num_turns):
        bad_ids = []
        if max_num_turns is not None:
            for snippet in self:
                if len(snippet) > max_num_turns:
                    bad_ids.append(snippet.uid)
        return bad_ids

    def _filter_source(self, source):
        bad_ids = []
        if source is not None:
            for snippet in self:
                if snippet.source is not None and snippet.source == source:
                    bad_ids.append(snippet.uid)
        return bad_ids

    def _filter_tags(self, tags):
        bad_ids = []
        if tags is not None:
            tags = set(tags)
            for snippet in self:
                if (snippet.tags is not None and
                        any(x in tags for x in snippet.tags)):
                    bad_ids.append(snippet.uid)
        return bad_ids

    def _filter_response_types(self, response_types):
        bad_ids = []
        if response_types is not None:
            response_types = set(response_types)
            for snippet in self:
                if (len(set(snippet.response_types).intersection(
                        response_types)) > 0):
                    bad_ids.append(snippet.uid)
        return bad_ids

    def _filter_only_speaker(self, speaker):
        bad_ids = []
        for snippet in self:
            snippet_speakers = set([turn.speaker for turn in
                snippet.turns])
            if (len(snippet_speakers) == 1 and
                    speaker == snippet_speakers.pop()):
                bad_ids.append(snippet.uid)
        return bad_ids

    def _filter_empty_summaries_with_no_tags(self, snippets):
        good_snippets = []
        for snippet in snippets:
            # unlabeled
            if snippet.summary is None:
                good_snippets.append(snippet)
            # labeled
            else:
                if snippet.summary == Text.EMPTY_STRING:
                    if snippet.tags is not None and len(snippet.tags) != 0:
                        good_snippets.append(snippet)
                else:
                    good_snippets.append(snippet)
        return good_snippets
    
    def _filter_concept_recall_threshold(self, threshold):
        bad_ids = []
        if threshold is not None:
            for snippet in self:
                if snippet.concept_recall is not None and snippet.concept_recall < threshold:
                    bad_ids.append(snippet.uid)
        return bad_ids
    
    def _filter_affirmation_recall_threshold(self, threshold):
        bad_ids = []
        if threshold is not None:
            for snippet in self:
                if snippet.affirmation_recall is not None and snippet.affirmation_recall < threshold:
                    bad_ids.append(snippet.uid)
        return bad_ids
    
    def _filter_sum_log_logits_threshold(self, threshold, fixed_ids):
        bad_ids = []
        if threshold is not None:

            snippets = [snippet for snippet in self if snippet.predicted_summary_sum_log_logits is not None and snippet.uid not in fixed_ids]
            snippets.sort(key=lambda x: x.predicted_summary_sum_log_logits, reverse=True)

            threshold_index = int(len(snippets) * threshold)
            threshold_value = snippets[threshold_index].predicted_summary_sum_log_logits

            for snippet in snippets:
                if snippet.predicted_summary_sum_log_logits < threshold_value:
                    bad_ids.append(snippet.uid)
 
        return bad_ids
