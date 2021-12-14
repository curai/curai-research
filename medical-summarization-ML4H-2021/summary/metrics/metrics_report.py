import numpy as np

from rouge_score import rouge_scorer
from summary.utils import ConceptAffirmationTagger, KBConceptRecognizer


class MetricsReport:
    """ 
    Report Metrics on predicted summaries given gold summaries

    Parameters
    ----------
    gold_summaries : list of str
        List of gold summmaries
    predicted_summaries : list of str
        List of predicted summmaries
    """

    def __init__(self, gold_summaries, predicted_summaries):
        self.gold_summaries = gold_summaries
        self.predicted_summaries = predicted_summaries
    
        self.kb_concept_recognizer = KBConceptRecognizer()
        self.concept_affirmation_tagger = ConceptAffirmationTagger()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True)
        self._compute_metrics()

    def to_json(self, tag=""):
        rouge_key = "rouge" if tag is "" else f"rouge.{tag}"
        concept_key = "concept" if tag is "" else f"concept.{tag}"
        affirmation_key = "affirmation" if tag is "" else f"affirmation.{tag}"

        return {
            rouge_key: self.rouge_metrics,
            concept_key: self.concept_metrics,
            affirmation_key: self.affirmation_metrics
        }

    def _compute_metrics(self):
        all_affirmation_f1s, all_concept_f1s = [], []
        all_rouge_f1s = []
        for gold_summary, pred_summary in zip(self.gold_summaries,
                self.predicted_summaries):
            rouge_metrics = self.rouge_scorer.score(gold_summary,
                    pred_summary)
            concept_confusion, affirmation_confusion = self._get_confusions(
                gold_summary, pred_summary)

            n_tp, n_fp, n_tn, n_fn = affirmation_confusion
            c_tp, c_fp, c_fn = concept_confusion
            affirmation_f1 = self._compute_f1(n_tp, n_fp, n_fn)
            concept_f1 = self._compute_f1(c_tp, c_fp, c_fn)
    
            rouge_l_f1 = rouge_metrics["rougeL"].fmeasure
            all_affirmation_f1s.append(affirmation_f1)
            all_concept_f1s.append(concept_f1)
            all_rouge_f1s.append(rouge_l_f1)

        all_concept_f1s = np.array(all_concept_f1s)
        all_affirmation_f1s = np.array(all_affirmation_f1s)
        all_rouge_f1s = np.array(all_rouge_f1s)

        self.rouge_metrics = {
            "f1": all_rouge_f1s.mean(),
            "std": all_rouge_f1s.std(),
        }
        self.concept_metrics = {
            "f1": all_concept_f1s.mean(),
            "std": all_concept_f1s.std(),
        }
        self.affirmation_metrics = {
            "f1": all_affirmation_f1s.mean(),
            "std": all_affirmation_f1s.std(),
        }

    def _get_confusions(self, gold_summary, pred_summary):
        gold_concepts = self.kb_concept_recognizer.get_concepts(gold_summary)
        pred_concepts = self.kb_concept_recognizer.get_concepts(pred_summary)
        gold_affirmations = self.concept_affirmation_tagger.get_affirmations(
            gold_summary, gold_concepts)
        pred_affirmations = self.concept_affirmation_tagger.get_affirmations(
            pred_summary, pred_concepts)

        concept_confusion = self._concept_confusion(gold_concepts,
            pred_concepts)
        affirmation_confusion = self._affirmations_confusion(
            gold_affirmations, pred_affirmations)
        return concept_confusion, affirmation_confusion
    
    def _compute_f1(self, tp, fp, fn):
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else: 
            precision = 0.

        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else: 
            f1 = 0.
        return f1
    
    def _concept_confusion(self, gold_concepts, pred_concepts):
        gold_concepts = set(gold_concepts)
        pred_concepts = set(pred_concepts)
        tp = len(gold_concepts.intersection(pred_concepts))
        fp = len(pred_concepts - gold_concepts)
        fn = len(gold_concepts - pred_concepts)
        return tp, fp, fn
    
    def _affirmations_confusion(self, gold_affirmations, pred_affirmations):
        tp, fp, tn, fn = 0, 0, 0, 0
        for concept, negation in pred_affirmations.items():
            if concept not in gold_affirmations:
                continue
            concept_agree = negation == gold_affirmations[concept]
            if concept_agree:
                if not negation:
                    tp += 1
                else:
                    tn += 1
            else:
                if not negation:
                    fp += 1
                else:
                    fn += 1
        return tp, fp, tn, fn
