from functools import partial
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from dask import dataframe as dd
from dask.distributed import Client
from snorkel.labeling.lf import LabelingFunction
from snorkel.labeling.apply.core import BaseLFApplier, _FunctionCaller
from snorkel.types import DataPoint

Scheduler = Union[str, Client]
LabelsType = Dict[Tuple[str, int, Tuple[int, int]], List[int]]


def apply_lfs_to_data_point(x: DataPoint, lfs: List[LabelingFunction], f_caller: _FunctionCaller, idx: List[int]):
    labels = []
    labelled = False
    text = None
    for j, lf in enumerate(lfs):
        # print(f"applying: ", lf.name)
        text, ner_tags = f_caller(lf, x)

        if ner_tags:
            labels.append((text, idx[0], ner_tags, j))
            labelled = True

    if not labelled:
        labels.append((x.uuid, idx[0], [], -1))

    idx[0] += 1
    return labels


def flatten_labels(labels, num_lfs) -> LabelsType:
    d: LabelsType = {}
    for label_list in labels:
        for (text, idx, ner_tags, j) in label_list:
            if ner_tags:
                assert j != -1
                for tag in ner_tags:
                    span, label = tag
                    if (text, idx, span) in d:
                        d[text, idx, span][j] = label
                    else:
                        d[text, idx, span] = [-1] * num_lfs
                        d[text, idx, span][j] = label
            else:
                assert j == -1
                span = (-1, -1)
                d[text, idx, span] = [-1] * num_lfs

    return d


def convert_to_pandas(labels: LabelsType, lfs: List[LabelingFunction]) -> pd.DataFrame:
    rows: List[List[Any]] = []
    for k, v in labels.items():
        text, idx, span = k
        row: List[Any] = [text, idx, span]

        row_labels: List[int] = v
        row.extend(row_labels)

        rows.append(row)

    columns: List[str] = ["uuid", "idx", "span"] + [f.name for f in lfs]
    label_df = pd.DataFrame(rows, columns=columns)
    return label_df


class PandasLFApplierForNER(BaseLFApplier):

    def apply(self, df: pd.DataFrame, progress_bar: bool = True, fault_tolerant: bool = False) -> pd.DataFrame:
        f_caller = _FunctionCaller(fault_tolerant)
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller, idx=[0])
        call_fn = df.apply

        if progress_bar:
            tqdm.pandas()
            call_fn = df.progress_apply

        
        labels = call_fn(apply_fn, axis=1)
        labels_flattened = flatten_labels(labels, len(self._lfs))

        labels_df: pd.DataFrame = convert_to_pandas(labels_flattened, self._lfs)

        return labels_df


class DaskLFApplier(BaseLFApplier):
    def apply(
        self,
        df: dd.DataFrame,
        scheduler: Scheduler = "processes",
        fault_tolerant: bool = False,
    ):
        f_caller = _FunctionCaller(fault_tolerant)
        tqdm.pandas()
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller, idx=[0])
        map_fn = df.map_partitions(lambda p_df: p_df.progress_apply(apply_fn, axis=1))
        labels = map_fn.compute(scheduler=scheduler)
        labels_flattened = flatten_labels(labels, len(self._lfs))

        labels_df: pd.DataFrame = convert_to_pandas(labels_flattened, self._lfs)

        return labels_df


class PandasParallelLFApplier(DaskLFApplier):
    def apply(  # type: ignore
        self,
        df: pd.DataFrame,
        n_parallel: int = 2,
        scheduler: Scheduler = "processes",
        fault_tolerant: bool = False,
    ):

        if n_parallel < 2:
            raise ValueError(
                "n_parallel should be >= 2. "
                "For single process Pandas, use PandasLFApplier."
            )
        df = dd.from_pandas(df, npartitions=n_parallel)
        return super().apply(df, scheduler=scheduler, fault_tolerant=fault_tolerant)