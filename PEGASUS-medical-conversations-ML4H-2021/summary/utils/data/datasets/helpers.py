from ..constants import  Speakers, Text
from ..types.snippet import Snippet
from ..types.turn import Turn

def read_json_snippet(snippet_id, data_pt, is_pseudo_label=False):
    """Read a snippet in json format into a Snippet object"""
    turns = []
    turn_texts = data_pt["snippet"]
    response_types = data_pt["response_types"]
    kb_concepts = data_pt.get("kb_concepts", [None] * len(turn_texts))
    for turn_text, response_type, kb_concept in zip(turn_texts,
            response_types, kb_concepts):
        if Text.PROFESSIONAL_PREFIX in turn_text:
            turn_text = turn_text.replace(Text.PROFESSIONAL_PREFIX,
                Text.EMPTY_STRING)
            speaker = Speakers.PROFESSIONAL
        else:
            turn_text = turn_text.replace(Text.CUSTOMER_PREFIX,
                Text.EMPTY_STRING)
            speaker = Speakers.CUSTOMER

        turns.append(Turn(text=turn_text, speaker=speaker,
            response_type=response_type, kb_concept=kb_concept))
    summary = data_pt['summary'] if not is_pseudo_label else data_pt["predicted_summary"]
    concept_recall = data_pt['concept_recall_w_snippet'] if 'concept_recall_w_snippet' in data_pt else None
    affirmation_recall = data_pt['affirmation_recall_w_snippet'] if 'affirmation_recall_w_snippet' in data_pt else None
    predicted_summary_sum_log_logits = data_pt['predicted_summary_sum_log_logits'] if 'predicted_summary_sum_log_logits' in data_pt else None
    fixed = data_pt['fixed'] if 'fixed' in data_pt else "False"

    return Snippet(uid=snippet_id, turns=turns, 
        is_history_gathering=data_pt.get('is_history_gathering'), 
        tags=data_pt['tags'], summary=summary,
        source=data_pt['source'], concept_recall=concept_recall,
        affirmation_recall=affirmation_recall,
        predicted_summary_sum_log_logits=predicted_summary_sum_log_logits,
        fixed=fixed)
