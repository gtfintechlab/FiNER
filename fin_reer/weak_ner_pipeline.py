import ast
import json
import logging
import os
import pickle
import shutil
import sys
from enum import EnumMeta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from snorkel.labeling.model import LabelModel
from tqdm import tqdm

from fin_reer.apply.ner_applier import PandasLFApplierForNER
from fin_reer.enums.labels import Labels
from fin_reer.labeling_functions import entity_lfs
from fin_reer.labeling_functions.utils.utils import merge_gold_labels

from utils import replace_item_in_list

logging.basicConfig()


class Pipeline:
    def __init__(self, config_path, labelling_functions, label_enums: EnumMeta, cardinality: int):
        self.labelling_functions = labelling_functions
        self.lf_columns = [lf.name for lf in self.labelling_functions]
        self.label_enums = label_enums

        config = json.load(open(config_path))
        self.input_df = None
        self.test_input_df = None
        self.generated_label_df = None
        self.input_df_path = config['input_df_path']
        self.unpickle_columns = config['unpickle_columns']
        self.cardinality = cardinality

        self.epochs = config['epochs']
        self.log_freq = config['log_frequency']
        self.seed = config['seed']

        self.experiment_name = config['experiment_name']
        self.experiment_version = config['experiment_version']

        self.label_matrix_save_path = config['label_matrix_save_path']

        self.evaluate_generated_labels: bool = config['evaluate_generated_labels']
        self.test_gold_data_path: str = config['test_gold_data_path']
        self.train_gold_data_path: str = config['train_gold_data_path']
        self.test_input_df_path: str = config['test_input_df_path']

        self.split_generated_data = config['split_generated_data']

        if 'run_only_flair' in config and config['run_only_flair']:
            self.lf_columns = [lf for lf in self.lf_columns if 'flair' in lf]

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def run_ner_pipeline(self, input_df, label_model: Optional[LabelModel] = None):
        """
        
        Args:
            input_df: Input raw texts for which weak labels needs to be generated
            label_model: If the label_model is provided, it will not be trained on label
                        matrix generated for input_df. It will be used to aggregate labels 
                        in this label matrix to generate unique labels. 

        Returns:
            label_df: Generated label matrix
            label_model: Returns label model which can later be used for aggregating new label matrices

        """

        self.logger.info("Applying labelling functions")
        applier: PandasLFApplierForNER = PandasLFApplierForNER(self.labelling_functions)
        label_df: pd.DataFrame = applier.apply(input_df, fault_tolerant=True)
        label_df['text'] = label_df.uuid.map(input_df.set_index('uuid').text.to_dict())

        self.logger.info("Preparing label matrix and Training snorkel aggregator")

        label_matrix: np.array = label_df[self.lf_columns].to_numpy()

        # If label model is provided, we assume that it should be used for aggregation instead
        # of initialising a new model and training it on label_matrix generated for the input_df
        if not label_model:
            # This means that we need to initialise and train the snorkel label model
            label_model = LabelModel(cardinality=self.cardinality, verbose=True)
            label_model.fit(L_train=label_matrix,
                            n_epochs=self.epochs,
                            log_freq=self.log_freq,
                            seed=self.seed)
            train_predictions = label_model.predict(label_matrix).tolist()
            label_df['predictive_label'] = train_predictions
        else:
            # This means that we have to use pretrained model on test set
            test_predictions = label_model.predict(label_matrix).tolist()
            label_df['predictive_label'] = test_predictions

        for col in self.lf_columns + ['predictive_label']:
            label_df[col].replace(-1, 0, inplace=True)

        return label_df, label_model

    def run_generative_part(self):
        label_df, label_model = self.run_ner_pipeline(self.input_df)
        unique_labels = list(set(label_df[self.lf_columns].values.flatten().tolist()))
        unique_labels.sort()
        str2int = {self.label_enums(lab).name: lab for lab in unique_labels}
        int2str = {v: k for k, v in str2int.items()}

        self.logger.info(f"int2str: {int2str}, str2int: {str2int}")
        label_df['label_name'] = label_df['predictive_label'].map(int2str)

        dir_name = f"{self.experiment_name}_{self.experiment_version}".replace(".", "_")
        dir_path = os.path.join(self.label_matrix_save_path, dir_name)

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)

        if self.train_gold_data_path:
            input_gold_df = pd.read_csv(self.train_gold_data_path)
            input_gold_df.rename(columns={'sentence_id': 'doc_idx', 'token': 'gold_token', 'label': 'gold_label'},
                                 inplace=True)
            label_df = merge_gold_labels(label_df, input_gold_df)

        # label_df.to_csv("label_df_temp.csv.gz", index=False, compression="gzip")

        if self.evaluate_generated_labels and self.test_gold_data_path:
            self.logger.info(f"Evaluating generated model on test split")
            test_label_df, _ = self.run_ner_pipeline(self.test_input_df, label_model)
            test_gold_df = pd.read_csv(self.test_gold_data_path)
            test_gold_df.rename(columns={'sentence_id': 'doc_idx', 'token': 'gold_token', 'label': 'gold_label'},
                                inplace=True)

            test_label_df = merge_gold_labels(test_label_df, test_gold_df)

            y_true = test_label_df.gold_label.tolist()
            y_pred = test_label_df.predictive_label.tolist()

            report = classification_report(y_true,
                                           y_pred,
                                           labels=list(int2str.keys()),
                                           target_names=list(int2str.values()),
                                           digits=4,
                                           zero_division=0,
                                           output_dict=True)

            report_df = pd.DataFrame(report)
            report_df.to_csv(os.path.join(dir_path, "classification_report.csv"))

            y_true_1 = replace_item_in_list(y_true, 2, 1)
            y_true_1 = replace_item_in_list(y_true_1, 4, 3)
            y_true_1 = replace_item_in_list(y_true_1, 6, 5)
            y_pred_1 = replace_item_in_list(y_pred, 2, 1)
            y_pred_1 = replace_item_in_list(y_pred_1, 4, 3)
            y_pred_1 = replace_item_in_list(y_pred_1, 6, 5)

            report_1 = classification_report(y_true_1,
                                             y_pred_1,
                                             labels=list(int2str.keys()),
                                             target_names=list(int2str.values()),
                                             digits=4,
                                             zero_division=0,
                                             output_dict=True)
            report_df_1 = pd.DataFrame(report_1)
            report_df_1.to_csv(os.path.join(dir_path, "classification_report_merged.csv"))

            self.logger.info(f"Classification report: {report_df}")

            test_label_df_save_path = os.path.join(dir_path, f"test_label_matrix_{dir_name}.csv.gz")
            test_label_df.to_csv(test_label_df_save_path, index=False, compression="gzip")

        label_df_save_path = os.path.join(dir_path, f"label_matrix_{dir_name}.csv.gz")
        label_df.to_csv(label_df_save_path, index=False, compression="gzip")
        self.generated_label_df = label_df
        json.dump(int2str, open(os.path.join(dir_path, f"int2str.json"), "w"))
        json.dump(str2int, open(os.path.join(dir_path, f"str2int.json"), "w"))

        self._save_adversarial_model_compatible_data()

    def load_data(self):
        self.input_df = pd.read_csv(self.input_df_path)
        self.test_input_df = pd.read_csv(self.test_input_df_path)
        if self.unpickle_columns:
            self.logger.info("Unpickling columns")
        for col in self.unpickle_columns:
            tqdm.pandas(desc=f"Unpickling {col}")
            self.input_df[col] = self.input_df[col].progress_apply(lambda x: pickle.loads(ast.literal_eval(x)))
            self.test_input_df[col] = self.test_input_df[col].progress_apply(
                lambda x: pickle.loads(ast.literal_eval(x)))

    def _save_adversarial_model_compatible_data(self):
        uuid_to_doc_id = self.input_df.set_index('uuid').doc_idx.to_dict()
        uuid_to_sent_id = self.input_df.set_index('uuid').sent_idx.to_dict()

        # Save two datasets (gold and weak) if gold data is provided
        datasets = []
        if self.evaluate_generated_labels:
            weak_gold_df = self.generated_label_df[['uuid', 'gold_token', 'predictive_label', 'gold_label']].copy()
            weak_gold_df['document_id'] = weak_gold_df.uuid.map(uuid_to_doc_id)
            weak_gold_df['sentence_id'] = weak_gold_df.uuid.map(uuid_to_sent_id)

            weak_data = weak_gold_df[['uuid', 'document_id', 'sentence_id', 'gold_token', 'predictive_label']].copy()
            weak_data.rename(columns={'gold_token': 'token', 'predictive_label': 'label'}, inplace=True)

            gold_data = weak_gold_df[['uuid', 'document_id', 'sentence_id', 'gold_token', 'gold_label']].copy()
            gold_data.rename(columns={'gold_token': 'token', 'gold_label': 'label'}, inplace=True)

            datasets.append({
                'name': 'weak_data',
                'dataset': weak_data
            })
            datasets.append({
                'name': 'gold_data',
                'dataset': gold_data
            })
        else:
            weak_gold_df = self.generated_label_df[['uuid', 'predictive_label']]
            weak_gold_df['document_id'] = weak_gold_df.uuid.map(uuid_to_doc_id)
            weak_gold_df['sentence_id'] = weak_gold_df.uuid.map(uuid_to_sent_id)
            weak_gold_df['token'] = self.generated_label_df.apply(lambda x: x.text[x.span[0]:x.span[1]], axis=1)
            weak_gold_df.rename(columns={'predictive_label': 'label'}, inplace=True)

            datasets.append({
                'name': 'weak_data',
                'dataset': weak_gold_df
            })

        dir_name = f"{self.experiment_name}_{self.experiment_version}".replace(".", "_")
        dir_path = os.path.join(self.label_matrix_save_path, dir_name)
        if self.split_generated_data:
            documents = weak_gold_df.document_id.unique().tolist()

            train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=self.seed)
            train_docs, val_docs = train_test_split(train_docs, test_size=0.2, random_state=self.seed)

            for ds in datasets:
                name = ds['name']
                dataset: pd.DataFrame = ds['dataset']

                train = dataset[dataset.document_id.isin(train_docs)]
                val = dataset[dataset.document_id.isin(val_docs)]
                test = dataset[dataset.document_id.isin(test_docs)]

                train_path = os.path.join(dir_path, f"{dir_name}_{name}_train_split.csv.gz")
                val_path = os.path.join(dir_path, f"{dir_name}_{name}_val_split.csv.gz")
                test_path = os.path.join(dir_path, f"{dir_name}_{name}_test_split.csv.gz")

                train.to_csv(train_path, index=False, compression="gzip")
                val.to_csv(val_path, index=False, compression="gzip")
                test.to_csv(test_path, index=False, compression="gzip")
        else:
            for ds in datasets:
                name = ds['name']
                dataset: pd.DataFrame = ds['dataset']

                save_path = os.path.join(dir_path, f"{dir_name}_{name}_generated_data.csv.gz")
                dataset.to_csv(save_path, index=False, compression="gzip")


def main():
    config_path = sys.argv[1]
    cardinality = Labels.get_entity_cardinality()
    pipeline = Pipeline(config_path, entity_lfs, Labels.ENTITIES, cardinality)
    pipeline.load_data()
    pipeline.run_generative_part()


if __name__ == "__main__":
    main()
