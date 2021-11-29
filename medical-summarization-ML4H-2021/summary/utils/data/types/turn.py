from ..constants import ResponseTypes, Speakers, Text


class Turn:
    """ 
    Class entailing details of a Turn

    Parameters
    ----------
    text : str
    speaker : str 
    response_type : str
    kb_concept : str
    """
    def __init__(self, text, speaker, response_type, kb_concept=None):
        self.text = Text.process(text)
        self.speaker = speaker
        self.response_type = response_type
        self.kb_concept = kb_concept

    def __repr__(self):
        if self.speaker == Speakers.PROFESSIONAL:
            prefix = Text.PROFESSIONAL_PREFIX
        else: 
            prefix = Text.CUSTOMER_PREFIX
        return f"{prefix}{self.text}"
