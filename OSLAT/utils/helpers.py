import torch
import numpy as np
from spacy.tokens import Span
from spacy.util import filter_spans

import pdb

def pairwise_cosine_similarity(a, b, eps=1e-8):
    """
    added eps for numerical stability

    a: torch.tensor (n x d)
    b: torch.tensor (m x d)
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def merge_subword_tokens(tokens, attentions, method='sum'):
    """
    Merge sub-word tokens along with their attention values,
    the attention value of the merged word is computed using
    the specified method (e.g. sum, mean, max).

    tokens: list of strings
    attentions: list of list of numbers (float/int)
    """
    assert method in ['mean', 'max', 'sum']

    merged_tokens = []
    merged_attention = [[] for _ in range(len(attentions))]

    curr_word = [0]
    for idx, tok in enumerate(tokens[1:]):

        idx += 1
        is_subword = tok.startswith('##')
        if is_subword:
            curr_word.append(idx)

        if (not is_subword) or (idx + 1) == len(tokens):
            word = [tokens[i].replace('##', '')  for i in curr_word]
            merged_tokens.append(''.join(word))

            for i, attn in enumerate(attentions):
                attn_values = [attn[j] for j in curr_word]
                if method == 'mean':
                    merged_attention[i].append(sum(attn_values) / len(attn_values))
                elif method == 'max':
                    merged_attention[i].append(max(attn_values))
                elif method == 'sum':
                    merged_attention[i].append(sum(attn_values))

            if (idx + 1) < len(tokens):
                curr_word = [idx]
            elif not is_subword: # When the last token is not a subword (e.g. [SEP])
                merged_tokens.append(tokens[idx])
                [merged_attention[i].append(attentions[i][-1]) for i in range(len(attentions))]

    return merged_tokens, merged_attention

def add_name_ent(matcher, doc, i, matches):
    """Callback on match function. Adds "NAME" entities to doc."""
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    _match_id, start, end, _ratio = matches[i]
    entity = Span(doc, start, end, label="NAME")
    try:
        doc.ents += (entity,)
    except ValueError:
        doc.ents = tuple(filter_spans(list(doc.ents) + [entity]))
