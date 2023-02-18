import json
import logging
import os
import shutil
import sys

import pandas as pd
from flair.data import Sentence
from sklearn.metrics import classification_report

from fin_reer import Models

logging.basicConfig()


class VanillaFlairPipeline:
    def __init__(self, config_path: str):
        config = json.load(open(config_path))
        self.test_data_path = config['test_data_path']
        self.tagger = Models.tagger
        self.experiment_name = config['experiment_name']
        self.experiment_version = config['experiment_version']
        self.results_save_path = config['results_save_path']

        self.str2int = {'<unk>': 0,
                        'O': 0,
                        'S-ORG': 5,
                        'S-MISC': 0,
                        'B-PER': 1,
                        'E-PER': 2,
                        'S-LOC': 3,
                        'B-ORG': 5,
                        'E-ORG': 6,
                        'I-PER': 2,
                        'S-PER': 1,
                        'B-MISC': 0,
                        'I-MISC': 0,
                        'E-MISC': 0,
                        'I-ORG': 6,
                        'B-LOC': 3,
                        'E-LOC': 4,
                        'I-LOC': 4,
                        '<START>': 0,
                        '<STOP>': 0}

        self.label_list = {"O": 0, "PER_B": 1, "PER_I": 2, "LOC_B": 3, "LOC_I": 4, "ORG_B": 5, "ORG_I": 6}

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def get_flair_predictions(self, flair_sentence):
        return [self.str2int[token.annotation_layers['ner'][0].value] for token in flair_sentence.tokens]

    def run(self):
        df = pd.read_csv(self.test_data_path)

        sentences = df.groupby('uuid', as_index=False).agg({'token': list, 'label': list})
        flair_sentences = [Sentence(sentence) for sentence in sentences.token.tolist()]
        self.tagger.predict(flair_sentences, verbose=True)

        preds = [self.get_flair_predictions(sent) for sent in flair_sentences]
        sentences['preds'] = preds
        len_before = len(sentences)
        sentences = sentences[sentences.apply(lambda x: len(x.label) == len(x.preds), axis=1)]
        len_after = len(sentences)
        self.logger.info(
            f"Dropped {len_before - len_after} sentences because of length mismatch in predictions and gold labels")

        preds_flattened = []
        for label_list in sentences.preds.tolist():
            preds_flattened.extend(label_list)
        actual_flattened = []
        for label_list in sentences.label.tolist():
            actual_flattened.extend(label_list)

        report = classification_report(actual_flattened,
                                       preds_flattened,
                                       digits=4,
                                       labels=list(self.label_list.values()),
                                       target_names=list(self.label_list.keys()),
                                       output_dict=True)

        dir_name = f"{self.experiment_name}_{self.experiment_version}".replace(".", "_")
        dir_path = os.path.join(self.results_save_path, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        report = pd.DataFrame(report)
        report.to_csv(os.path.join(dir_path, "classification_report.csv"))
        self.logger.info(report)


def main():
    config_path = sys.argv[1]
    pipeline = VanillaFlairPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
