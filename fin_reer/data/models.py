from typing import List, Tuple

import stanza
from flair.models import SequenceTagger


class PreTokenizer:
    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize')


class Whitespace(PreTokenizer):
    def pre_tokenize_str(self, s: str) -> List[Tuple[str, Tuple[int, int]]]:
        tokens: List[Tuple[str, Tuple[int, int]]] = []

        doc = self.nlp(s)
        for sent in doc.sentences:
            for token in sent.tokens:
                tokens.append((token.text, (token.start_char, token.end_char)))

        return tokens


class Models:
    pre_tokenizer = Whitespace()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
    tagger = SequenceTagger.load("ner")
