import json
import os
from typing import List, Any, Tuple, Set, Union, Dict

import numpy as np
import pandas as pd
import string
from sklearn.metrics import classification_report
from tqdm import tqdm

from fin_reer.config.constants import PathConstants, UtilityConstants
from fin_reer.data.models import Models
from fin_reer.enums.labels import Labels


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

pre_tokenizer = Models.pre_tokenizer


def get_data_files(path: str, extensions: List[str]) -> List[str]:
    """
    :param path: Path of the directory containing all 10-k files
    :param extensions: List of extensions of files whose paths are desired.

    :return: List of paths of all .txt files in a directory

    This function recursively reads a directory, and returns all .txt files present in that directory.
    """
    extensions = set(extensions)
    files: List[str] = []
    for current_path, _, current_files in os.walk(path):
        for file_ in current_files:
            abs_path = os.path.join(current_path, file_)

            _, ext = os.path.splitext(abs_path)
            if ext in extensions:
                files.append(abs_path)

    return files


def label_uncovered_text(text: str, start: int, end: int, sentence_idx) -> List[Any]:
    unlabelled_text = text[start:end]
    unlabelled_tokens = pre_tokenizer.pre_tokenize_str(unlabelled_text)

    rows = []
    for (token, span) in unlabelled_tokens:
        actual_span = (start + span[0], start + span[1])
        assert text[actual_span[0]:actual_span[1]] == token
        rows.append((text, sentence_idx, actual_span, token, 'O'))
    return rows


# Two pointer method to combine and rearrange sentences in their original order.
def combine_and_reorder_rows(row_list_1: List[Tuple[Any]], row_list_2: List[Tuple[Any]]):
    """

    :param row_list_1: List 1 containing rows of final result (labelled case)
    :param row_list_2: List 2 containing rows of final result (unlabelled case)
    :return: List of final result data combined from two lists above.

    Description: It is assumed that the both lists are sorted according to sentence indices.
                 Thus, both row_list_1 and row_list_1 are sorted in increasing order of sentence indices.
    """

    all_data_reordered = []
    i, j = 0, 0

    n1 = len(row_list_1)
    n2 = len(row_list_2)

    while i < n1 and j < n2:
        if row_list_1[i][1] < row_list_2[j][1]:
            all_data_reordered.append(row_list_1[i])
            i += 1
        elif row_list_1[i][1] > row_list_2[j][1]:
            all_data_reordered.append(row_list_2[j])
            j += 1
        else:
            raise Exception("Index can never be same in two lists. Something is wrong :/")

    while i < n1:
        all_data_reordered.append(row_list_1[i])
        i += 1

    while j < n2:
        all_data_reordered.append(row_list_2[j])
        j += 1

    return all_data_reordered


def generate_training_data(labelled_sentences: pd.DataFrame,
                           unlabelled_sentences: pd.DataFrame,
                           predictions: np.array,
                           label_type: str,
                           data_path: str = PathConstants.EXAMPLE_ENTITIES_TRAINING_DATA_SAVE_PATH) -> None:
    """

    :param labelled_sentences: Dataframe containing sentences in which at least one token is labelled
    :param unlabelled_sentences: Dataframe containing sentences which are not labelled at all
    :param predictions: Output of model.predict on label_df
    :param label_type: "ENTITY" or "RELATIONSHIP"
    :param data_path: Path where training data will be saved.
    :return: Returns None. Saves training data to
    """

    labelled_sentences['predicted'] = pd.Series(predictions, index=labelled_sentences.index)
    grps = labelled_sentences.groupby(['idx'])

    # This loop labels all unlabelled portions of sentences with "OTHER" label
    other_tokens_in_labelled = []
    for sentence_idx, sentnece_df in grps:
        text: str = sentnece_df.head(1).text.tolist()[0]
        s: int = 0
        n: int = len(text)

        # sort sentnece_df to keep span in order to avoid duplicate labels
        sentnece_df = sentnece_df.sort_values(by='span', key=lambda col: col.map(lambda x: x[0]))

        new_rows = []
        for _, row in sentnece_df.iterrows():
            (start, end) = row.span

            # Label uncovered text before labelled span
            new_rows.extend(label_uncovered_text(text, s, start, sentence_idx))

            # Retain actual labelled span and append to new list of rows
            token = text[start:end]
            if label_type == UtilityConstants.LABEL_TYPE_ENTITY:
                new_rows.append((row.text, row.idx, row.span, token, Labels.ENTITIES(row.predicted).name))
            elif label_type == UtilityConstants.LABEL_TYPE_RELATIONSHIP:
                new_rows.append((row.text, row.idx, row.span, token, Labels.RELATIONSHIPS(row.predicted).name))
            else:
                raise ValueError(f"Incorrect label type. Expected either 'ENTITY' or 'RELATIONSHIP'")

            s = end

        # Label uncovered text after rightmost labelled span
        new_rows.extend(label_uncovered_text(text, s, n, sentence_idx))
        other_tokens_in_labelled.extend(new_rows)

    # In sentences in which no token is labelled, all tokens are labelled as "OTHER"
    other_tokens_in_unlabelled = []
    for _, row in unlabelled_sentences.iterrows():
        text = row.text
        tokens = pre_tokenizer.pre_tokenize_str(text)
        idx = row.idx

        new_rows = []
        for token, span in tokens:
            new_rows.append((text, idx, span, token, 'O'))
        other_tokens_in_unlabelled.extend(new_rows)

    all_data_reordered = combine_and_reorder_rows(other_tokens_in_labelled, other_tokens_in_unlabelled)

    # Save the dataframe to excel file
    column_list = ["sentence", "idx", "span", "token", "label"]
    training_data_df = pd.DataFrame(all_data_reordered, columns=column_list)
    training_data_df.to_excel(data_path + ".xlsx", index=False)


def compute_report(df: pd.DataFrame, pred_columns: List[str]) -> pd.DataFrame:
    # function name, total predicted, total correct, false positive, false negative, true positive, category
    print("Computing report on gold data")

    df_report = pd.DataFrame([])
    for col in pred_columns:
        vals = set(df[col])
        for val in vals:
            if UtilityConstants.INT_TO_STR_MAP[val] == 'O':
                continue
            col_total = f"total_{UtilityConstants.INT_TO_STR_MAP[val].lower()}"
            col_total_predicted = f"total_{UtilityConstants.INT_TO_STR_MAP[val].lower()}_predicted"
            col_true_positives = f"{UtilityConstants.INT_TO_STR_MAP[val].lower()}_true_positives"
            col_false_positives = f"{UtilityConstants.INT_TO_STR_MAP[val].lower()}_false_positives"
            col_false_negatives = f"{UtilityConstants.INT_TO_STR_MAP[val].lower()}_false_negatives"

            val_total = len(df[df.apply(lambda x:val in x.gold_label, axis=1)])

            df_val = df[df[col] == val]
            val_totalpredicted = len(df_val)
            val_true_positive = len(df_val[df_val.apply(lambda x:val in x.gold_label, axis=1)])
            val_false_positive = val_totalpredicted - val_true_positive

            df_not_val = df[df[col] != val]
            val_false_negatives = len(df_not_val[df_not_val.apply(lambda x:val in x.gold_label, axis=1)])
            val_accuracy = val_true_positive / val_totalpredicted

            sub_df = pd.DataFrame({
                'lf_name': col,
                'accuracy': [val_accuracy],
                col_total: [val_total],
                col_total_predicted: [val_totalpredicted],
                col_true_positives: [val_true_positive],
                col_false_positives: [val_false_positive],
                col_false_negatives: [val_false_negatives]
            })

            df_report = pd.concat((df_report, sub_df), axis=0)

    print("Reported computed")
    return df_report


def map_label_to_std_label(label: Union[str, int], label_type: str):
    if label == -1:
        return 0
    if isinstance(label, int):
        if label_type == UtilityConstants.LABEL_TYPE_ENTITY:
            label = Labels.ENTITIES(label).name
        else:
            label = Labels.RELATIONSHIPS(label).name

    return UtilityConstants.STR_TO_INT_LABEL_MAP[label]


def map_labels_df_to_standard_labels(df: pd.DataFrame,
                                     label_columns: Set[str],
                                     label_type: str):
    """
        Description: Maps labels of each column to one of the four labels
                     O, PER, LOC or ORG
    """

    # non_lf_cols: Set[str] = {"text", "idx", "span", "gold_token"}

    for col in df.columns:
        if col in label_columns:
            df[col] = df.apply(lambda x:map_label_to_std_label(x[col], label_type), axis=1)

    return df


def expand_boundaries(text: str, spans: List[List]):
    punctuations = set(string.punctuation)
    punctuations.add('’')
    """

    Args:
        text: Input labelled sentence
        spans: List of labelled spans.


    Description: This function expands labelled spans to include non-whitespace boundary characters. It is possible
                 that we miss boundary characters while manual labelling. This function expands all spans to include
                 any non-whitespace boundary characters.

                 For example, consider sentence "Elon Musk is CEO of Tesla" and manual annotation [1, 9, PER]. This
                 function will change this manual annotation to [0, 9, PER]

    Returns:

    """
    n = len(text)

    def expand_left(start):
        # Decrease left position until you encounter start of sentence or a whitespace
        while start >= 0 and not text[start].isspace() and text[start] not in punctuations:
            start -= 1

        # Return the corrected start position
        return start + 1

    def expand_right(end):
        # Increase right position until you encounter end of sentence or a whitespace
        while end <= n and not text[end - 1].isspace() and text[end - 1] not in punctuations:
            end += 1

        # Return the corrected end position
        return end - 1

    # List to remove some location related adjectives from gold data
    exception_list = {"Chinese", "Taiwanese", "Malaysian", "European", "Asian", "South African"}

    expanded_spans = []
    for span in spans:
        s, e, label = span
        s_expanded = expand_left(s - 1)
        e_expanded = expand_right(e + 1)

        new_span = [s_expanded, e_expanded, label]

        if text[s_expanded:e_expanded] in exception_list:
            print(f"Skipping {text[s:e]} label because it is part of {text[s_expanded:e_expanded]} token present in "
                  f"exception list")
            continue

        if s != s_expanded or e != e_expanded:
            print(f"Changed {text[s:e]} to {text[s_expanded:e_expanded]}")

        expanded_spans.append(new_span)

    return expanded_spans


def strip_trailing_spaces(text: str, spans: List[List]):
    """

    Args:
        text: Input labelled sentence
        spans: List of labelled spans

    Returns: Spans after removing trailing whitespaces

    """

    def trim_left(start):
        # Increase start until you encounter a non-whitespace character
        while text[start].isspace():
            start += 1

        # Return the correct left position after trimming left whitespaces
        return start

    def trim_right(end):
        # Decrease end position until you encounter a non-whitespace character
        while text[end - 1].isspace():
            end -= 1

        # Return the correct right position after trimming right whitespaces
        return end

    trimmed_spans = []
    for span in spans:
        s, e, label = span
        s_trimmed = trim_left(s)
        e_trimmed = trim_right(e)

        new_span = [s_trimmed, e_trimmed, label]

        if s != s_trimmed or e != e_trimmed:
            print(f"Changed {text[s:e]} to {text[s_trimmed:e_trimmed]}")

        trimmed_spans.append(new_span)

    return trimmed_spans


def aggregate_snorkel_pipeline_entities(df: pd.DataFrame) -> Dict[int, str]:
    df = df.copy()
    df.preds = df.apply(lambda x:map_label_to_std_label(x.preds, UtilityConstants.LABEL_TYPE_ENTITY), axis=1)

    grps_by_idx = df.groupby('idx')
    grps_by_idx_map = {k: v for k, v in grps_by_idx}

    rows = {}
    print("Groupping snorkel entities to feed in relationship pipeline")
    for k, v in tqdm(grps_by_idx_map.items()):
        entities_map = {"PER": [], "LOC": [], "ORG": []}
        grps_by_label = v.groupby('preds')
        grps_by_label_map = {k1: v1 for k1, v1 in grps_by_label}

        for k1, v1 in grps_by_label_map.items():
            entity = UtilityConstants.INT_TO_STR_MAP[k1]
            if entity in entities_map:
                span_list = v1.span.tolist()
                span_list = sorted(span_list, key=lambda x: x[0])
                entities_map[entity] = span_list

        rows[k] = json.dumps(entities_map)

    assert len(set(df.idx.tolist())) == len(rows), f"Length mismatch between {len(set(df.idx.tolist()))} and {len(rows)}"

    return rows


def replace_item_in_list(ls, a, b):
    return [b if x == a else x for x in ls]