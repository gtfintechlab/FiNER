import ast
import json
import os
import traceback

import numpy as np
import pandas as pd
from typing import List, Set
from nltk.tokenize import sent_tokenize
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import LabelModel

from fin_reer.apply.ner_applier import PandasLFApplierForNER
from fin_reer.labeling_functions.utils.utils import merge_gold_labels
from fin_reer.config.constants import PathConstants, ModelConstants, UtilityConstants
from fin_reer.enums.labels import Labels
from fin_reer.labeling_functions import entity_lfs
from fin_reer.utils import map_labels_df_to_standard_labels


def run_snorkel_pipeline(df: pd.DataFrame,
                         labelling_functions: List[LabelingFunction],
                         cardinality: int,
                         file_name: str = "chunks.parquet"):
    applier: PandasLFApplierForNER = PandasLFApplierForNER(labelling_functions)
    label_df: pd.DataFrame = applier.apply(df, fault_tolerant=True)

    print("Label matrix generated")

    label_df.to_parquet(file_name)

    lf_columns = [lf.name for lf in labelling_functions]
    label_matrix: np.array = label_df[lf_columns].to_numpy()

    label_model = LabelModel(cardinality=cardinality, verbose=True)
    print(f"Training snorkel model for {ModelConstants.NUM_EPOCHS} epochs")
    label_model.fit(L_train=label_matrix,
                    s=ModelConstants.NUM_EPOCHS,
                    log_freq=ModelConstants.LOG_FREQ,
                    seed=ModelConstants.SEED)

    train_predictions = label_model.predict(label_matrix)
    preds = pd.DataFrame(train_predictions, columns=['preds'])
    merged_preds: pd.DataFrame = pd.concat((label_df, preds), axis=1)

    merged_preds.to_csv("label_matrix_with_predictions_msft.csv.gz", header=True, index=False, compression="gzip")
    return merged_preds


def map_int_label_to_str_labels(df: pd.DataFrame,
                                label_columns: Set[str],
                                label_type: str):
    return map_labels_df_to_standard_labels(df, label_columns, label_type)


def merge_gold(df: pd.DataFrame,
               gold_df: pd.DataFrame,
               label_columns: Set[str],
               label_type: str):
    if label_type == UtilityConstants.LABEL_TYPE_RELATIONSHIP:
        df.to_csv("df_while_merging.csv.gz", header=True, index=False, compression="gzip")
        gold_df.to_csv("gold_df_while_merging.csv.gz", header=True, index=False, compression="gzip")
    merged_gold: pd.DataFrame = merge_gold_labels(df, gold_df)
    mapped_to_std = map_labels_df_to_standard_labels(merged_gold, label_columns, label_type)

    return mapped_to_std


# def compute_accuracy_(df):
#     compute_accuracy(df)


def aggregate_entity_labelling_output(df: pd.DataFrame, label_type):
    df = map_int_label_to_str_labels(df, {"preds"}, label_type)
    groups = df.groupby(by=["idx", "preds"]).agg({"span": lambda x: list(x), "text": lambda x: list(set(x))})

    rows = {}
    idx_to_txt = {}
    for grp in groups.index:
        idx, label = grp
        val = groups.loc[grp]
        spans = val.span
        if isinstance(spans[0], str):
            spans = [ast.literal_eval(span) for span in spans]

        text = val.text
        text = text[0]
        idx_to_txt[idx] = text

        if idx not in rows:
            rows[idx] = {}

        rows[idx][label] = spans

    rel_input_df = []
    for k, v in rows.items():
        rows[k] = json.dumps(v)
        rel_input_df.append({"idx": k, "text": idx_to_txt[k], "snorkel_entities": rows[k]})

    return pd.DataFrame(rel_input_df)


if __name__ == "__main__":
    entity_cardinality: int = Labels.get_entity_cardinality()
    relationship_cardinality: int = Labels.get_relationship_cardinality()

    # run over text files listed in master file
    start_loc = 0
    end_loc = 5
    master_df = pd.read_excel(PathConstants.MASTER_INPUT_TEXT_FILE).iloc[start_loc:end_loc, :]
    res_list_entity = []
    res_list_relationship = []
    err_list = []
    column_list = list(master_df.columns)
    column_list.append('output_path')
    for index, master_row in master_df.iterrows():
        try:
            text_file_path = os.path.join(os.getcwd(),
                                          PathConstants.PREFIX_INPUT_TEXT_FILE_PATH + master_row['file_name'])
            text_file_name_split = master_row['file_name'].split('.')
            unique_sec_file_id = text_file_name_split[0]

            print(text_file_path)
            f = open(text_file_path, 'r', encoding="utf-8", errors='ignore')
            text_file = f.read()
            f.close()

            lines: List[str] = sent_tokenize(text_file)[:10]
            raw_data_df: pd.DataFrame = pd.DataFrame(lines, columns=["text"])

            curr_row = list(master_row)

            output_file_path = PathConstants.PREFIX_DATA_SAVE_PATH + "entity_" + str(unique_sec_file_id)

            entity_data = run_snorkel_pipeline(raw_data_df, entity_lfs, entity_cardinality)
            entity_data.to_csv("entity_data.csv", header=True, index=False)

            # res_list_entity.append(curr_row + [output_file_path])
            #
            # output_file_path = PathConstants.PREFIX_DATA_SAVE_PATH + "relationship_" + str(unique_sec_file_id)
            # relationship_data = run_snorkel_pipeline(entity_data, relationship_lfs, relationship_cardinality)
            # relationship_data.to_csv("relationship_data.csv", header=True, index=False)
            #
            # res_list_relationship.append(curr_row + [output_file_path])
            break

        except Exception as e:
            curr_row = list(master_row)
            err_list.append(curr_row)
            print(curr_row, e)
            traceback.print_exc()

    result_master_df = pd.DataFrame(res_list_entity, columns=column_list)
    result_master_df.to_excel(
        PathConstants.PREFIX_MASTER_OUTPUT_TEXT_FILE_ENTITIES + str(start_loc) + '_' + str(end_loc) + '.xlsx',
        index=False)

    result_master_df = pd.DataFrame(res_list_relationship, columns=column_list)
    result_master_df.to_excel(
        PathConstants.PREFIX_MASTER_OUTPUT_TEXT_FILE_RELATIONSHIPS + str(start_loc) + '_' + str(end_loc) + '.xlsx',
        index=False)

    err_df = pd.DataFrame(err_list, columns=column_list[:-1])
    err_df.to_excel(PathConstants.PREFIX_MASTER_OUTPUT_TEXT_FILE_ERROR + str(start_loc) + '_' + str(end_loc) + '.xlsx',
                    index=False)
