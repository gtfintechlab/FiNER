from flair.data import Sentence
from snorkel.preprocess import preprocessor

from fin_reer import Models


pre_tokenizer = Models.pre_tokenizer
nlp = Models.nlp
tagger = Models.tagger


@preprocessor(memoize=True)
def pre_tokenize_text(x):
    if not x.entity_tokens:
        x.entity_tokens = pre_tokenizer.pre_tokenize_str(x.text)
    return x


@preprocessor(memoize=True)
def predict_ner_tags_library(x):
    if not x.entity_tokens:
        doc = nlp(str(x.text))
        tokens = []
        for sent in doc.sentences:
            for token in sent.tokens:
                tokens.append((token.text, (token.start_char, token.end_char)))
        x.entity_tokens = tokens

    if not x.flair_spans:
        sentence = Sentence(x.text)
        tagger.predict(sentence)
        x.flair_spans = sentence.get_spans('ner')

    return x
