# NOTE: NegationTagger is not included
#  Use custom negation tagger that has method get_negations
# get_negations takes as input a text and a set of concepts for which we need to find affirmations
# An example negation tagger is negex.py
# https://code.google.com/archive/p/negex/
import NegationTagger


class ConceptAffirmationTagger:
    """ Recognize affirmations of concepts in text"""
    def __init__(self):
        self.negation_tagger = NegationTagger()

    def get_affirmations(self, text, concepts):
        """ 
        Entity linking on a given text

        Parameters
        ----------
        text : str
        concepts: list
            List of knowledge base concepts

        Returns
        -------
        affirmations : dict
            Dict mapping concept to its negation in text
        """
        affirmations = {}
        for affirmation in self.negation_tagger.get_negations(text,
                                                              concepts)[0]:
            concept = affirmation["name"]
            negation = affirmation["negation"]
            affirmations[concept] = negation
        return affirmations
