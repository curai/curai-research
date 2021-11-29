import EntityRecognizer
# NOTE: EntityRecognizer is not included
#  Use custom entity recognizer that has method extract_entities
# extract_entities takes as input a text and returns a set of entities (phrases in text that match to an entity)


class KBConceptRecognizer:
    """ Recognize knowledge base concepts in text"""
    def __init__(self):
        self.entity_recognizer = EntityRecognizer()

    def get_concepts(self, text):
        """ 
        Entity linking on a given text

        Parameters
        ----------
        text : str

        Returns
        -------
        concepts : list
            List of recognized concepts 
        """
        concepts = [
            x.matched_name
            for x in self.entity_recognizer.extract_entities(text)
        ]
        return concepts
