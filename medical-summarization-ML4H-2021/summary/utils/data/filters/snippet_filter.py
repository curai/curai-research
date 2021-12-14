class SnippetFilter:
    """ 
    Snippet Filter for filtering a SnippetDataset

    Parameters
    ----------
    max_num_turns : int
        Maximum number of turns a snippet can have
    source : str
        Source of summary labels
    only_speaker : str
        Speaker in snippets where only one person speaks
    tags : list of str
    response_types : list of str
    snippet_ids : list of str
    only_speaker: TODO
    concept_recall_threshold: float
        Threshold above which a sample is selected if its summary's 
        concept recall to the snippet is provided.
    """

    def __init__(
        self, max_num_turns=None, source=None, tags=None,
        response_types=None, snippet_ids=None, only_speaker=None,
        concept_recall_threshold=None, affirmation_recall_threshold=None,
        sum_log_logits_threshold=None, keep_fixed=False, sll_human_threshold_window=None
    ):
        self.max_num_turns=max_num_turns
        self.source = source
        self.tags = tags
        self.response_types = response_types
        self.snippet_ids = snippet_ids
        self.only_speaker = only_speaker
        self.concept_recall_threshold = concept_recall_threshold
        self.affirmation_recall_threshold = affirmation_recall_threshold
        self.sum_log_logits_threshold=sum_log_logits_threshold
        self.keep_fixed=keep_fixed
        self.sll_human_threshold_window=sll_human_threshold_window

