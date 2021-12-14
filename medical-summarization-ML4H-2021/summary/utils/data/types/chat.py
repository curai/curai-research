from .snippet import Snippet
from ..constants import ResponseTypes, Speakers, Text


class Chat:
    """ 
    Class entailing details of a chat

    Parameters
    ----------
    uid : int
        Encounter ID 
    reason_for_encounter : str
        Reason for the encounter
    turns : list of Turns
        Utterances/Turns which make up the chat
    snippets : list of Snippets
        Snippets which make up the chat
    """
    def __init__(self, uid, reason_for_encounter, turns, snippets):
        self.uid = uid
        self.reason_for_encounter = reason_for_encounter
        self.turns = turns

        if snippets is None:
            self.snippets = self._snippetize()
            self._trailing_ar_cust_turns_as_free_text()
        else:
            self.snippets = snippets

    # ------------------ PUBLIC METHODS ------------------
    def to_json(self): 
        res = {self.uid: {"reason_for_encounter": self.reason_for_encounter}}
        for snippet in self.snippets: 
            res[self.uid].update(snippet.to_json())
        return res


    # ------------------ MAGIC METHODS ------------------
    def __repr__(self):
        serialized = f"Chat: {self.uid}{Text.NEW_LINE}"
        serialized += f"RFE: {self.reason_for_encounter}{Text.NEW_LINE}"
        if self.snippets is None:
            content = [str(turn) for turn in self.turns]
        else:
            content = [str(turn) for turn in self.snippets]
        serialized += Text.NEW_LINE.join(content)
        return serialized 

    def __len__(self): 
        return len(self.snippets)

    # ------------------ CLASS METHODS ------------------
    @classmethod
    def from_turns(cls, uid, reason_for_encounter, turns):
        return cls(uid=uid, reason_for_encounter=reason_for_encounter,
            turns=turns, snippets=None)
    
    @classmethod
    def from_snippets(cls, uid, reason_for_encounter, snippets):
        return cls(uid=uid, reason_for_encounter=reason_for_encounter,
            snippets=snippets, turns=None)

    # ------------------ PRIVATE METHODS ------------------
    def _trailing_ar_cust_turns_as_free_text(self):
        """
        Put trailing customer chat turns for an AR response
        into original AR response free text
        """
        i, j = 0, 2
        turn_changes, del_indices = [], set() 
        while i + 1 < len(self.turns):
            curr_turn, next_turn = self.turns[i], self.turns[i+1]

            prof_asks_ar_question = (curr_turn.speaker == Speakers.PROFESSIONAL
                and curr_turn.response_type != ResponseTypes.CHAT)
            cust_answers_ar_question = (next_turn.speaker == Speakers.CUSTOMER
                and next_turn.response_type != ResponseTypes.CHAT)

            if next_turn.response_type == ResponseTypes.AR_FT:
                free_text = next_turn.text
            else: 
                free_text = next_turn.text + Text.PERIOD
        
            if prof_asks_ar_question and cust_answers_ar_question:
                init_j = j 
                while (j < len(self.turns) and
                        self.turns[j].speaker == Speakers.CUSTOMER and
                        self.turns[j].response_type == ResponseTypes.CHAT):
                    free_text += ' ' + self.turns[j].text + Text.PERIOD
                    j += 1
                
                # Condense customer trailing chat text into free text portion
                # of AR question response
                if j > init_j:
                    turn_changes.append((list(range(i, j)), [i, i+1]))
                    self.turns[i+1].text = free_text
                    self.turns[i+1].response_type = ResponseTypes.CHAT
                    del_indices.update(range(init_j, j))
            else: 
                j += 1
            i = j 
            j = i + 2
    
        self.turns = [turn for i, turn in enumerate(self.turns)
            if i not in del_indices]

    def _snippetize(self):
        """Splits a chat into snippets"""
        i, count = 0, 0
        content, snippets = [], []
        prev_turn_is_cust  = False
        while i < len(self.turns): 
            turn = self.turns[i]
            if turn.speaker == Speakers.PROFESSIONAL: 
                # Professional AR Question
                if turn.response_type != ResponseTypes.CHAT:
                    snippets.append(Snippet(f"{self.uid}_{count}",
                        [self.turns[i], self.turns[i + 1]]))
                    count += 1
                    i += 1

                # Professional CHAT Response
                else:
                    if '?' in turn.text and len(content) != 0:
                        snippets.append(
                            Snippet(f"{self.uid}_{count}", content))
                        content = []
                        count += 1
                    else:
                        # Professional educational or acknowledgment
                        if prev_turn_is_cust and len(content) != 0:
                            snippets.append(
                                Snippet(f"{self.uid}_{count}", content))
                            content = []
                            count += 1
                    content.append(turn)
                prev_turn_is_cust = False
            else:
                prev_turn_is_cust = True
                content.append(turn)
            i += 1
        return snippets
