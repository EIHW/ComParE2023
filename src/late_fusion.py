#!/usr/bin/env python
import json
import yaml
import glob
import os
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix
from functools import reduce
from collections import Counter
from ml.ci import CI

FUSION_RESULTS_PATH = "results/fusion"
FUSION_METRICS_PATH = "metrics/fusion"


if __name__ == "__main__":
    params = {}
    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params["fusion"]

    result_dirs = sorted(
        [os.path.join("./results", f) for f in params["to_fuse"]]
    )

    os.makedirs(FUSION_RESULTS_PATH, exist_ok=True)
    os.makedirs(FUSION_METRICS_PATH, exist_ok=True)
    metrics = {"devel": {}, "test": {}}

    all_devel_predictions = reduce(
        lambda left, right: pd.merge(left, right, on=["filename", "true"]),
        [
            pd.read_csv(os.path.join(result_dir, "predictions.devel.csv"))
            for result_dir in result_dirs
        ],
    )
    all_devel_predictions["prediction"] = all_devel_predictions[
        [p for p in all_devel_predictions.columns if "prediction" in p]
    ].agg(lambda x: Counter(x).most_common(1)[0][0], axis=1)
    all_devel_predictions[["filename", "prediction", "true"]].to_csv(
        os.path.join(FUSION_RESULTS_PATH, "predictions.devel.csv"), index=False
    )


    all_test_predictions = reduce(
        lambda left, right: pd.merge(left, right, on=["filename", "true"]),
        [
            pd.read_csv(os.path.join(result_dir, "predictions.test.csv"))
            for result_dir in result_dirs
        ],
    )
    all_test_predictions["prediction"] = all_test_predictions[
        [p for p in all_test_predictions.columns if "prediction" in p]
    ].agg(lambda x: Counter(x).most_common(1)[0][0], axis=1)
    all_test_predictions[["filename", "prediction", "true"]].to_csv(
        os.path.join(FUSION_RESULTS_PATH, "predictions.test.csv"), index=False
    )
