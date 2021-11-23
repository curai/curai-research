"""Text handling class for snippets/chats"""
import re

class Text: 
    APOSTROPHE = "'"
    CUSTOMER_PREFIX = "CUSTOMER: "
    DASH = "-"
    DOUBLE_QUOTE = "\""
    EMPTY_STRING = ""
    NEG_TOKEN = "[NO]"
    NEW_LINE = "\n"
    PERIOD = "."
    PROFESSIONAL_PREFIX = "PROFESSIONAL: "
    SLASH = "\\"
    SPACE = " "
    UNDERSCORE = "_"
    UNICODE_APOSTROPHE = "\u2019"
    UNICODE_DASH = "\u2013"
    UNICODE_LEFT_QUOTE = "\u201c"
    UNICODE_OBJECT_REPLACEMENT = "\ufffc"
    UNICODE_REPLACEMENT = "\ufffd"
    UNICODE_RIGHT_QUOTE = "\u201d"
    UNICODE_SINGLE_QUOTE = "\u2018"

    @classmethod
    def process(cls, text): 
        text = text.replace(cls.NEW_LINE, cls.EMPTY_STRING)
        text = text.replace(cls.SLASH, cls.EMPTY_STRING)
        text = text.replace(cls.UNICODE_APOSTROPHE, cls.APOSTROPHE)
        text = text.replace(cls.UNICODE_DASH, cls.DASH)
        text = text.replace(cls.UNICODE_LEFT_QUOTE, cls.DOUBLE_QUOTE)
        text = text.replace(cls.UNICODE_OBJECT_REPLACEMENT, cls.EMPTY_STRING)
        text = text.replace(cls.UNICODE_REPLACEMENT, cls.EMPTY_STRING)
        text = text.replace(cls.UNICODE_RIGHT_QUOTE, cls.DOUBLE_QUOTE)
        text = text.replace(cls.UNICODE_SINGLE_QUOTE, cls.APOSTROPHE)

        text = re.sub("\s+", cls.SPACE, text)
        text = text.strip()
        return text

    @classmethod
    def remove_neg_token(cls, text):
        return text.replace(cls.NEG_TOKEN + cls.SPACE, cls.EMPTY_STRING)
