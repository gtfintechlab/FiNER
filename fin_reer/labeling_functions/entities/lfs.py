import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from snorkel.labeling import labeling_function

from fin_reer.config.constants import UtilityConstants
from fin_reer.data.models import Models
from fin_reer.enums.labels import Labels
from fin_reer.labeling_functions.utils.utils import filter_spans, generate_labels, extract_flair_entity
from fin_reer.preprocessing.preprocessors import pre_tokenize_text, predict_ner_tags_library

from fin_reer.data.gazetteers import Gazetteers

names = Gazetteers.NAMES
popular_names = UtilityConstants.POPULAR_NAMES
cities = Gazetteers.CITIES
states = Gazetteers.STATES
countries = Gazetteers.COUNTRIES
world_cities = Gazetteers.WORLD_CITIES
stop_words = Gazetteers.STOP_WORDS
pre_tokenizer = Models.pre_tokenizer


@labeling_function(pre=[predict_ner_tags_library]) ##
def label_per_library_flair(x):
    """
        Label ORG based on pre-trained model
    """

    flair_entity_label: str = "PER"
    lf_entity: str = "PER"

    return extract_flair_entity(x, flair_entity_label, lf_entity)


@labeling_function(pre=[predict_ner_tags_library]) ##
def label_loc_library_flair(x):
    """
        Label ORG based on pre-trained model
    """

    flair_entity_label: str = "LOC"
    lf_entity: str = "LOC"

    return extract_flair_entity(x, flair_entity_label, lf_entity)


@labeling_function(pre=[predict_ner_tags_library]) ##
def label_org_library_flair(x):
    """
        Label ORG based on pre-trained model
    """

    flair_entity_label: str = "ORG"
    lf_entity: str = "ORG"

    return extract_flair_entity(x, flair_entity_label, lf_entity)


@labeling_function(pre=[pre_tokenize_text])
def label_loc_senator(x):
    entity_tokens = x.entity_tokens
    spans = []

    append_next = False
    for token, token_span in entity_tokens:        
        if token.startswith("D-") or token.startswith("R-"):
            append_next = True
            continue
        if append_next:
            spans.append(token_span)
            append_next=False

    spans = filter_spans(x.text, spans)

    return x.uuid, generate_labels(x, spans, "LOC", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_loc_based(x):
    entity_tokens = x.entity_tokens
    spans = []
    
    previous = None
    for token, token_span in entity_tokens:
        if token.endswith("-based") and token[0].isupper():
            spans.append(previous)
        previous = token_span

    spans = filter_spans(x.text, spans)
    return x.uuid, generate_labels(x, spans, "LOC", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_org_heuristic_1(x):
    """
    labels organizations by analysis of suffixes
    """
    suffixes = {"LLC", "Ltd", "Inc", "Co", "Bank", "Corporation", "LLC", "Company", "Incorporated", "Limited",
                "Association", "Board"}
    tokenss = x.entity_tokens

    spans = []

    for i in range(0, len(tokenss)):
        if tokenss[i][0] in suffixes:
            j = i
            end = tokenss[i][1][1]
            s = ""
            start = None
            while j > 0:
                if tokenss[j][0][0].isupper():
                    s = tokenss[j][0] + " " + s
                    start = tokenss[j][1][0]
                    j = j - 1
                else:
                    break

            if start and len(s.split()) > 1:
                spans.append((start, end))

    spans = filter_spans(x.text, spans)
    return x.uuid, generate_labels(x, spans, "ORG", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_loc_heuristic_1(x):
    """
    labels location by finding some patterns
    """

    tokenss = x.entity_tokens
    spans = []
    for i in range(0, len(tokenss)):
        if tokenss[i][0] == "of" and tokenss[i - 1][0].lower() == "state":
            j = i + 1
            start = tokenss[j][1][0]
            s = ""
            end = None
            while j < len(tokenss):
                if tokenss[j][0][0].isupper():
                    s = s + tokenss[j][0]
                    end = tokenss[j][1][1]
                    j = j + 1
                else:
                    break

            if end and len(s) > 0:
                spans.append((start, end))
        elif tokenss[i][0] == "of" and tokenss[i - 1][0].lower() == "district":
            j = i + 1
            start = tokenss[j][1][0]
            s = ""
            end = None
            while j < len(tokenss):
                if tokenss[j][0][0].isupper():
                    s = s + tokenss[j][0]
                    end = tokenss[j][1][1]
                    j = j + 1
                else:
                    break

            if end and len(s) > 0:
                spans.append((start, end))
        elif tokenss[i][0] == "near":
            j = i + 1
            start = tokenss[j][1][0]
            s = ""
            end = None
            while j < len(tokenss):
                if tokenss[j][0][0].isupper():
                    s = s + tokenss[j][0]
                    end = tokenss[j][1][1]
                    j = j + 1
                else:
                    break
            if end and len(s) > 0:
                spans.append((start, end))
        elif tokenss[i][0] == "in" and tokenss[i - 1][0].lower() == "based":
            j = i + 1
            start = tokenss[j][1][0]
            s = ""
            end = None
            while j < len(tokenss):
                if tokenss[j][0][0].isupper():
                    s = s + tokenss[j][0]
                    end = tokenss[j][1][1]
                    j = j + 1
                else:
                    break

            if end and len(s) > 0:
                spans.append((start, end))
    spans = filter_spans(x.text, spans)
    return x.uuid, generate_labels(x, spans, "LOC", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_org_heuristic_role(x):
    """
    Description:
    If the phrase like "(in C) of" is detected, next two tokens are tagged as an organization.
    C = ["CEO", "COO", "CTO", "CXO", "CFO", "CIO", "CCO", "CHRM", "CSO", "CGO", "CAO", "CMO", "CDO"]
    """

    roles = {"CEO", "COO", "CTO", "CXO", "CFO", "CIO", "CCO", "CHRM", "CSO", "CGO", "CAO", "CMO", "CDO"}
    tokens = x.entity_tokens
    spans = []
    i = 0
    if len(tokens) == 1:
        return x.uuid, generate_labels(x, spans, "ORG", "ENTITY")
    while i < len(tokens):
        if tokens[i][0] in roles and (i + 3) < len(tokens) and tokens[i + 1][0] == "of":
            spans.append((tokens[i + 2][1][0], tokens[i + 3][1][1]))
            i += 3
        i += 1

    spans = filter_spans(x.text, spans)
    return x.uuid, generate_labels(x, spans, "ORG", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_org_heuristic_abbr(x):
    """
    Description: Detect an abbreviation usually used at the end of an organization's legal name (Assn, Co, Corp,
    Dept, Inc, Ltd, LLC, St, US, Univ). If any of these is detected, previous two words are tagged as an organization.
    """
    spans = []
    abbr_list = {"Assn", "Co", "Corp", "Dept", "Inc", "Ltd", "LLC", "St", "US", "Univ"}
    tokens = x.entity_tokens
    i = 0
    while i < len(tokens):
        if tokens[i][0] in abbr_list and len(tokens) > 2:
            if (i - 3) >= 0 and tokens[i - 1][0] == ',':
                spans.append(tokens[i - 3][1])
                spans.append(tokens[i - 2][1])
            elif (i - 2) >= 0:
                spans.append(tokens[i - 2][1])
                spans.append(tokens[i - 1][1])
        i += 1

    spans = filter_spans(x.text, spans)
    return x.uuid, generate_labels(x, spans, "ORG", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_per_heuristic_1(x):
    """
    Find person based on executive titles
    """
    basket = {'CEO', 'CFO', 'CTO', 'CIO', 'CFO', 'President', 'Chairman'}
    spans = []
    titles = []
    titles_start = []

    entity_tokens = x.entity_tokens

    for i, (token, token_span) in enumerate(entity_tokens):

        if i + 2 < len(entity_tokens) and token in basket and entity_tokens[i + 1][0][0].isupper() and \
                entity_tokens[i + 2][0][0].isupper():
            title_start, title_end = token_span
            start, title_end = entity_tokens[i + 2][1]

            titles.append(x.text[title_start:title_end])
            titles_start.append(title_start)
        if i - 2 >= 0 and token in basket and entity_tokens[i - 1][0][0].isupper() and \
                entity_tokens[i - 2][0][0].isupper():
            title_start, title_end = token_span
            title_start, end = entity_tokens[i - 2][1]

            titles.append(x.text[title_start:title_end])
            titles_start.append(title_start)

    for title_idx in np.arange(len(titles)):
        spans.append((titles_start[title_idx], titles_start[title_idx] + len(titles[title_idx])))

    spans = filter_spans(x.text, spans)
    return x.uuid, generate_labels(x, spans, "PER", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_per_heuristic_suffix(x):
    """
    Description:
    Detect a name suffix (CPA, DDS, Esq, JD, Jr, LLD, MD, PhD, Ret, RN, Sr, DO).
    If any of these is detected, previous two words are tagged as a person.
    """
    titles = {"CPA", "DDS", "Esq", "JD", "Jr", "LLD", "MD", "PhD", "Ret", "RN", "Sr", "DO"}
    tokens = x.entity_tokens
    spans = []
    i = 0
    while i < len(tokens):
        if tokens[i][0] in titles and len(tokens) > 2:
            if (i - 3) >= 0 and tokens[i - 1][0] == ',':
                spans.append((tokens[i - 3][1][0], tokens[i - 2][1][1]))
            elif (i - 2) >= 0:
                spans.append((tokens[i - 2][1][0], tokens[i - 1][1][1]))
        i += 1

    spans = filter_spans(x.text, spans)
    return x.text, generate_labels(x, spans, "PER", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_org_heuristic_abbr(x):
    """
    Description: Detect an abbreviation usually used at the end of an organization's legal name (Assn, Co, Corp,
    Dept, Inc, Ltd, LLC, St, US, Univ). If any of these is detected, previous two words are tagged as an organization.
    """
    spans = []
    abbr_list = {"Assn", "Co", "Corp", "Dept", "Inc", "Ltd", "LLC", "St", "US", "Univ"}
    tokens = x.entity_tokens
    i = 0
    while i < len(tokens):
        if tokens[i][0] in abbr_list and len(tokens) > 2:
            if (i - 3) >= 0 and tokens[i - 1][0] == ',':
                spans.append(tokens[i - 3][1])
                spans.append(tokens[i - 2][1])
            elif (i - 2) >= 0:
                spans.append(tokens[i - 2][1])
                spans.append(tokens[i - 1][1])
        i += 1

    spans = filter_spans(x.text, spans)
    return x.text, generate_labels(x, spans, "ORG", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_org_heuristic_partner(x):
    """
    Description:
    Detect the phrase "partnered with" and tag the next two words as an organization.
    """
    spans = []
    tokens = x.entity_tokens
    i = 0
    if len(tokens) == 1:
        return x.text, generate_labels(x, spans, "ORG", "ENTITY")
    while i < len(tokens):
        if (i + 3) < len(tokens) and (
                tokens[i][0] == "partner" or tokens[i][0] == "partnered" or tokens[i][0] == "partnering") and \
                tokens[i + 1][0] == "with":
            spans.append((tokens[i + 2][1][0], tokens[i + 3][1][1]))
            i += 3
        i += 1

    spans = filter_spans(x.text, spans)
    return x.text, generate_labels(x, spans, "ORG", "ENTITY")


@labeling_function(pre=[pre_tokenize_text])
def label_org_heuristic_trademark(x):
    """
    Description:
    Detect the phrase "trademarks of" and tag the next two words as an organization.
    """
    spans = []
    tokens = x.entity_tokens
    i = 0
    if len(tokens) == 1:
        return x.text, generate_labels(x, spans, "ORG", "ENTITY")
    while i < len(tokens):
        if ((i + 3) < len(tokens)) and (tokens[i][0] == "trademark" or tokens[i][0] == "trademarks") and tokens[i + 1][
            0] == "of":
            spans.append(tokens[i + 2][1])
            spans.append(tokens[i + 3][1])
            i += 3
        i += 1

    spans = filter_spans(x.text, spans)
    return x.text, generate_labels(x, spans, "ORG", "ENTITY")