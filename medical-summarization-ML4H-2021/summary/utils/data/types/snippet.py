import numpy as np

from ..constants import ResponseTypes, Speakers, Text


class Snippet: 
    """ 
    Class entailing details of a snippet

    Parameters
    ----------
    uid : int
        Snippet ID 
    turns : list of Turns
        Utterances/Turns which make up the chat
    is_history_gathering : bool
        Whether the snippet is gathering history or not
    source: str
        Source of the summary label
    tags : list of str
    summary: str
    """
    def __init__(
            self, uid, turns, is_history_gathering=None,
            tags=None, summary=None, source=None, concept_recall=None,
            affirmation_recall=None, predicted_summary_sum_log_logits=None, 
            fixed="False"
        ): 
        self.uid = uid
        self.turns = turns
        self.is_history_gathering = is_history_gathering
        self.kb_concepts = [turn.kb_concept for turn in self.turns]
        self.response_types = [turn.response_type for turn in self.turns]

        self.tags = tags
        self.source = source
        if summary is not None:
            self.summary = Text.process(summary)
        else:
            self.summary = summary
        
        self.concept_recall = concept_recall
        self.affirmation_recall = affirmation_recall
        self.predicted_summary_sum_log_logits = predicted_summary_sum_log_logits
        self.fixed = fixed
    
    # ------------------ PUBLIC METHODS ------------------
    def get_formatted(self, speaker_prefixes=False, text=False):
        """ 
        Gets snippet in a formatted manner

        Parameters
        ----------
        speaker_prefixes : bool
            Whether to include speaker prefixes or not
        text : bool
            Whether to return a list of turns or text of turns

        Returns
        ----------
        turns : list or str
        """
        turns = []
        for turn in self.turns:
            if speaker_prefixes:
                turns.append(str(turn))
            else: 
                turns.append(turn.text)
        if text:
            turns = Text.NEW_LINE.join(turns)
        return turns

    def word_distribution_by_speaker(self):
        """ 
        Distribution of words by each speaker in a snippet

        Returns
        ----------
        dist : np.array
            First position is percentage of words by professional
            and second position is percentage of words by customer
        """
        dist = [0., 0.]
        for turn in self.turns:
            n_words = len(turn.text.split())
            if turn.speaker == Speakers.PROFESSIONAL:
                dist[0] += n_words
            else: 
                dist[1] += n_words
        dist = np.array(dist)
        dist /= dist.sum()
        return dist

    def to_json(self):
        res = {self.uid: {}}
        res[self.uid] = {
            "snippet": self.get_formatted(speaker_prefixes=True),
            "response_types": self.response_types,
            "kb_concepts": self.kb_concepts,
            "tags": self.tags,
            "summary": self.summary,
            "source": self.source,
            "is_history_gathering": self.is_history_gathering
        }
        return res

    # ------------------ MAGIC METHODS ------------------
    def __repr__(self): 
        if (self.is_history_gathering is not None and
                not self.is_history_gathering):
            serialized = (f"Snippet (Not History Gathering): "
                f"{self.uid}{Text.NEW_LINE}")
        else:
            serialized = (f"-------- Snippet {self.uid} "
                f"--------{Text.NEW_LINE}")
        str_turns = [str(turn) for turn in self.turns]
        serialized += Text.NEW_LINE.join(str_turns)
        return serialized

    def __len__(self): 
        return len(self.turns)
