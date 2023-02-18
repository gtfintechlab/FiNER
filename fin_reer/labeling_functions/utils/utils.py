import json
import re
from typing import List, Tuple

import pandas as pd
from flair.data import Span
from pandas.core.series import Series
from collections import defaultdict
from tqdm import tqdm

from fin_reer import Models
from fin_reer.config.constants import UtilityConstants
from fin_reer.enums.labels import Labels

pre_tokenizer = Models.pre_tokenizer


def generate_labels(x: Series,
                    labelled_spans: List[Tuple[int, int]],
                    label: str,
                    label_type: str) -> List[Tuple[Tuple[int, int], int]]:
    """

    :param x: Data Point from which span is taken
    :param labelled_spans: Spans in text to label
    :param label: Must belong to list of entities or list of relationships in enums module of this project
    :param label_type: Only one of ENTITY or RELATIONSHIP
    :return: Returns a list of all labels for a given span. Right now only one label will be given for each span
    """
    if label_type == "ENTITY":
        b_label: int = Labels.ENTITIES[f"{label}_B"].value
        i_label: int = Labels.ENTITIES[f"{label}_I"].value
    else:
        b_label: int = Labels.RELATIONSHIPS[f"{label}_B"].value
        i_label: int = Labels.RELATIONSHIPS[f"{label}_I"].value

    o_label: int = -1

    is_entity_and_is_start = {}
    labels = []
    entity_tokens = x.entity_tokens
    span_idx = [-1 for _ in range(len(x.text))]

    for i, (_, (s, e)) in enumerate(entity_tokens):
        for j in range(s, e):
            span_idx[j] = i

    labelled_spans_positional = []
    for (s, e) in labelled_spans:
        sub_spans = pre_tokenizer.pre_tokenize_str(x.text[s:e]) # we might to improve logic of "pre_tokenize_str" for . case in person
        for i, (_, sub_span) in enumerate(sub_spans):
            s1, e1 = sub_span
            is_start = True if i == 0 else False
            labelled_spans_positional.append(((s + s1, s + e1), is_start))

    for (s, e), is_start in labelled_spans_positional:
        for j in range(s, e):
            if span_idx[j] != -1 and span_idx[j] not in is_entity_and_is_start:
                is_entity_and_is_start[span_idx[j]] = is_start

    for (_, (s, e)) in entity_tokens:
        is_set = False
        for j in range(s, e):
            assert span_idx[j] != -1
            if span_idx[j] in is_entity_and_is_start:
                if is_entity_and_is_start[span_idx[j]]:
                    labels.append(((s, e), b_label))
                else:
                    labels.append(((s, e), i_label))
                is_set = True
                break

        if not is_set:
            labels.append(((s, e), o_label))

    return labels


def mask_ner_spans(ner_spans, text: str) -> str:
    for span in reversed(ner_spans):
        if span.annotation_layers['ner'][0].value in {"PER", "LOC", "ORG"}:
            start = span[0]
            end = span[1]

            text = text.replace(text[start:end], "#" * (end - start))
    return text


def extract_flair_entity(x: Series,
                         flair_entity_label: str,
                         lf_entity: str) -> Tuple[str, List[Tuple[Tuple[int, int], int]]]:
    ner_spans: List[Span] = x.flair_spans

    entity_parts = []
    for span in ner_spans:
        if span.annotation_layers['ner'][0].value == flair_entity_label:
            entity_parts.append((span.start_pos, span.end_pos))

    return x.uuid, generate_labels(x, entity_parts, lf_entity, label_type="ENTITY")


def search_key_executive(x: Series,
                         pattern: str,
                         relationship: str):
    """

    :param x: Data point to be labelled
    :param pattern: Regular expression that will be used to label data point
    :param relationship: Must belong to list of relations in enums module of this project
    :return: List of relationship labels for key executives

    Description:
        This function extracts out common logic used to identify key executives based on regular expressions.
        Many labelling functions extracting relationships like CEO, CTO, COO can use this common utility function
        in order to avoid writing duplicate code for each labelling function
    """

    text: str = x.text
    match = re.search(pattern, text, re.IGNORECASE)
    entities = json.loads(x.snorkel_entities)

    labels = []
    if match:
        # Spans that were labelled as person by entities labelling functions
        spans = entities["PER"]

        start: int = match.start()
        end: int = match.end()

        # Persons on the left of the match and on the right of the match
        left_persons = list(filter(lambda t: t[1] < start, spans))
        right_persons = list(filter(lambda t: t[0] >= end, spans))

        if left_persons:
            span = left_persons[-1]
            labels.append(span)
            # labels.extend(generate_labels(x, (span.start_pos, span.end_pos), relationship, "RELATIONSHIP"))
        elif right_persons:
            span = right_persons[0]
            labels.append(span)
            # labels.extend(generate_labels(x, (span.start_pos, span.end_pos), relationship, "RELATIONSHIP"))

    labels = filter_spans(x.text, labels)
    return x.uuid, generate_labels(x, labels, relationship, "RELATIONSHIP")


def search_org_relation(x: Series,
                        pattern: str,
                        relationship: str):
    """

    :param x: Data point to be labelled
    :param pattern: Regular expression that will be used to label data point
    :param relationship: Must belong to list of relations in enums module of this project
    :return: List of relationship labels for company relations like supplier, competitor, customer, etc.

    Description: This function extracts out common logic used to identify relations between companies based on
    regular expressions.
    """

    text: str = x.text
    match = re.search(pattern, text, re.IGNORECASE)
    entities = json.loads(x.snorkel_entities)

    labels = []
    org_spans = []
    if match:
        spans = entities["ORG"]
        for span in spans:
            org_spans.append(span)
        # labels.extend(generate_labels(x, (span.start_pos, span.end_pos), relationship, "RELATIONSHIP"))

    org_spans = filter_spans(x.text, org_spans)
    return x.uuid, generate_labels(x, org_spans, relationship, "RELATIONSHIP")


def search_loc_relation(x: Series,
                        pattern: str,
                        relationship: str):
    """

    :param x: Data point to be labelled
    :param pattern: Regular expression that will be used to label data point
    :param relationship: Must belong to list of relations in enums module of this project
    :return: List of relationships corresponding to locations. Like HQ location, factory loc etc.

    Description: This function extracts out common logic used to identify relations between companies and locations
    """

    text: str = x.text
    match = re.search(pattern, text, re.IGNORECASE)
    entities = json.loads(x.snorkel_entities)

    loc_spans = []
    if match:

        spans = entities["LOC"]
        for span in spans:
            loc_spans.append((span[0], span[1]))
            # labels.extend(generate_labels(x, (span.start_pos, span.end_pos), relationship, "RELATIONSHIP"))

    loc_spans = filter_spans(x.text, loc_spans)
    return x.uuid, generate_labels(x, loc_spans, relationship, "RELATIONSHIP")


def search_per_relation(x: Series,
                        pattern: str,
                        relationship: str):
    """
    :param x: Data point to be labelled
    :param pattern: Regular expression that will be used to label data point
    :param relationship: Must belong to list of relations in enums module of this project
    :return: List of relationshiop labels for person relations like terminations, resignations, promotions, etc.

    Description: This function extracts out common logic used to identify relations between companies based on
    regular expressions.
    """

    text: str = x.text
    match = re.search(pattern, text, re.IGNORECASE)
    entities = json.loads(x.snorkel_entities)

    per_spans = []
    if match:

        spans = entities["PER"]
        for span in spans:
            per_spans.append((span[0], span[1]))
            # labels.extend(generate_labels(x, (span.start_pos, span.end_pos), relationship, "RELATIONSHIP"))

    return x.uuid, generate_labels(x, per_spans, relationship, "RELATIONSHIP")


def get_spans_word_wise(txt: str,
                        spans: List[Tuple[int, int]],
                        id_b: int,
                        id_i: int):
    """
    This function breaks the name string of entities at every whitespace and gives output in _B and _I format
    """

    lst = []
    for ent in spans:
        s = ent[0]
        tmp = txt[ent[0]:ent[1]].split()

        if len(tmp) > 0:
            lst.append(((s, s + len(tmp[0])), id_b))
            s = s + len(tmp[0]) + 1
            for ele in tmp[1:]:
                lst.append(((s, s + len(ele)), id_i))
                s = s + len(ele) + 1

    return lst


def find_spans(text: str,
               patterns: list):
    """
    :param text: text string of our data point
    :param patterns: list of names of entities for wich we need to find spans
    Helper function to find spans of a particular sub string present in a larger text string

    :return: span of the entities
    """

    dic = defaultdict(list)
    patterns = list(set(patterns))

    for pattern in patterns:

        match = re.finditer(pattern, text)

        for m in match:
            dic[pattern].append(m.span())

    return dic


def common_logic_function(x_ner: list,
                          txt: str,
                          chk_set: set,
                          label: str):
    """
    :param x_ner: spacy tokenized text (returned by ner_tockenize())
    :param txt: our text attribute of the datapoint
    :param chk_set: set which contains keywords to check relationships present in the datapoint
    :param label: label to give to our datapoint

    :return: tupple containing (txt, labels)
    """

    length = len(x_ner)
    labels = []

    names = []
    for term, tag in x_ner:
        if tag == "PERSON":
            names.append(term)

    spans_dic = find_spans(txt, names)

    for idx, (term, tag) in enumerate(x_ner):
        if tag == "PERSON":

            tmp = [q[0] for q in x_ner[idx + 1:min(idx + 6, length)]]
            my_set = set(tmp)

            if len(chk_set.intersection(my_set)) > 0:
                span = spans_dic[term].pop(0)
                s, e = span[0], span[1]
                b_label = Labels.RELATIONSHIPS[f"{label}_B"].value
                i_label = Labels.RELATIONSHIPS[f"{label}_I"].value
                labels.extend(get_spans_word_wise(txt, [(s, e)], b_label, i_label))

    return txt, labels


def merge_gold_labels(df_raw, df_gold):
    new_rows = []
    raw_list = df_raw.apply(lambda x: x.text[x.span[0]:x.span[1]], axis=1).tolist()
    # raw_list = df_raw.raw_tok.tolist()
    gold_list = df_gold.gold_token.tolist()
    gold_label_list = df_gold.gold_label.tolist()

    i, j, k = 0, 0, 0

    n = len(raw_list)
    m = len(gold_list)

    with tqdm(total=n) as pbar:
        while i < n and j < m:
            try:
                if raw_list[i] == gold_list[j]:
                    new_rows.append((gold_list[j], gold_label_list[k]))
                    i += 1
                    j += 1
                    k += 1
                    pbar.update(1)
                elif len(raw_list[i]) < len(gold_list[j]):
                    new_rows.append((raw_list[i], gold_label_list[k]))
                    rem = gold_list[j][len(raw_list[i]):]
                    gold_list[j] = rem
                    i += 1
                    pbar.update(1)
                else:
                    gold_list[j+1] = gold_list[j] + gold_list[j+1]
                    gold_label_list[k+1] = gold_label_list[k]
                    j += 1
                    k += 1
            except:
                print(raw_list[i], gold_list[j], df_gold.iloc[j],df_raw.iloc[j])
                i += 1
                j += 1

    assert i == n
    assert j == m, f"j != m, {j} != {m}"
    assert k == m

    new_gold_df = pd.DataFrame(new_rows, columns=["gold_token", "gold_label"])
    new_df = pd.concat((df_raw, new_gold_df), axis=1)
    return new_df


def filter_spans(text, spans):
    """

    Args:
        text: Raw text
        spans: Spans to be filtered

    Description:
        Filters spans containing blacklisted words and initial lower case letters

    Returns:

    """

    filtered_spans = []
    for span in spans:
        s, e = span
        token = text[s:e]
        if token[0].islower() or token in UtilityConstants.BLACKLIST:
            continue
        filtered_spans.append(span)

    return filtered_spans
