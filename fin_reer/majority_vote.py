import json
import logging
import os
import shutil
import sys
from collections import Counter

import pandas as pd
from sklearn.metrics import classification_report

from fin_reer.labeling_functions import entity_lfs

logging.basicConfig()


class MajorityVoteModel:
    def __init__(self, config_path: str):
        config = json.load(open(config_path))

        self.label_matrix_path = config['label_matrix_path']
        self.experiment_name = config['experiment_name']
        self.experiment_version = config['experiment_version']
        self.results_save_path = config['results_save_path']

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def run(self):
        df = pd.read_csv(self.label_matrix_path)
        lf_cols = [lf.name for lf in entity_lfs]
        vals = df[lf_cols].values

        preds = []
        for val in vals:
            nz = [x for x in val.tolist() if x != 0]
            x = Counter(nz)
            mc = x.most_common()
            if len(mc) == 0:
                preds.append(0)
                continue
            if len(mc) == 1:
                preds.append(mc[0][0])
            else:
                if mc[0][1] > mc[1][1]:
                    preds.append(mc[0][0])
                else:
                    preds.append(0)

        labels = {"O": 0, "PER_B": 1, "PER_I": 2, "LOC_B": 3, "LOC_I": 4, "ORG_B": 5, "ORG_I": 6}
        report = pd.DataFrame(
            classification_report(df.gold_label.tolist(), preds, digits=5, output_dict=True,
                                  labels=list(labels.values()),
                                  target_names=list(labels.keys())))

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
    pipeline = MajorityVoteModel(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
