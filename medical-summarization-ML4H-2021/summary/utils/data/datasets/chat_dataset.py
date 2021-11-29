from ..constants import ResponseTypes, Speakers, Text
from ...file_io import read_json
from .helpers import read_json_snippet
from .snippet_dataset import SnippetDataset
from ..types.chat import Chat
from ..types.snippet import Snippet
from ..types.turn import Turn


class ChatDataset(SnippetDataset): 
    """ 
    Dataset of Chats

    Parameters
    ----------
    chats : list of Chat
    classify_history_gathering : bool
        Whether to classify if snippets are gathering history or not
    """

    def __init__(self, chats, classify_history_gathering=False): 
        super().__init__([snippet for chat in chats
            for snippet in chat.snippets], classify_history_gathering)

        self.chat_ids = set([chat.uid for chat in chats])

        self._chats = chats
        self._chats_data = {}
        for chat in chats:
            self._chats_data[chat.uid] = chat

    # ------------------ PUBLIC METHODS ------------------
    def add(self, chat):
        for snippet in chat.snippets:
            super().add(snippet)
        self.chat_ids.add(chat.uid)
        self._chats_data[chat.uid] = chat

    def remove(self, chat): 
        for snippet in chat.snippets:
            super().remove(snippet)
        self._chats.remove(chat)
        self.chat_ids.remove(chat.uid)
        del self._chats_data[chat.uid]

    def subset(self, chat_ids):
        chats = []
        for chat_id in chat_ids:
            chats.append(self._chats_data[chat_id])
        return ChatDataset(chats)
    
    def to_json(self, chat_format=False, rfes_as_snippet=False):
        if chat_format:
            res = {}
            for chat_id, chat in self._chats_data.items():
                res.update(chat.to_json())
        else: 
            res = super().to_json()
            if rfes_as_snippet:
                res = self._prepend_rfes_to_chats_snippets(res)
        return res

    # ------------------ MAGIC METHODS ------------------
    def __len__(self): 
        return len(self._chats)

    def __getitem__(self, lookup_id):
        if Text.UNDERSCORE in lookup_id:
            chat_id, snippet_id = lookup_id.split(Text.UNDERSCORE)
            if snippet_id == "rfe": 
                res = self._chats_data[chat_id].reason_for_encounter
            else: 
                res = self._snippets_data[lookup_id]
        else:
            res = self._chats_data[lookup_id]
        return res

    def __iter__(self):
        for chat in self._chats: 
            yield chat

    # ------------------ CLASS METHODS ------------------
    @classmethod
    def from_json_file(cls, json_file, chat_format=False):
        data = read_json(json_file)
        if chat_format:
            chats = self._read_chat_formatted_json(data)
        else:
            chats = self._read_snippet_formatted_json(data) 
        return cls(chats=chats)

    # ------------------ PRIVATE METHODS ------------------
    def _read_chat_formatted_json(self, chats_json):
        chats = []
        for chat_id, chat_data_pt in data.items(): 
            snippets = []
            for snippet_id, snippet_data_pt in chat_data_pt.items(): 
                if snippet_id == "reason_for_encounter":
                    rfe = snippet_data_pt
                else: 
                    snippets.append(read_json_snippet(snippet_id,
                        snippet_data_pt))
            chats.append(Chat.from_snippets(uid=chat_id,
                reason_for_encounter=rfe, snippets=snippets))
        return chats

    def _read_snippet_formatted_json(self, snippets_json):
        chats, snippets = [], []
        prev_chat_id = None
        for snippet_id, data_pt in data.items(): 
            chat_id, snippet_num = snippet_id.split(Text.UNDERSCORE)
            if snippet_num == "rfe": 
                if prev_chat_id is not None:
                    chats.append(Chat.from_snippets(uid=chat_id,
                        reason_for_encounter=rfe, snippets=snippets))
                    snippets = []
                rfe = data_pt 
            else: 
                snippets.append(read_json_snippet(snippet_id, data_pt))
            prev_chat_id = chat_id
        chats.append(Chat.from_snippets(uid=chat_id,
            reason_for_encounter=rfe, snippets=snippets))
        return chats
