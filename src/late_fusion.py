#!/usr/bin/env python
import ast

import json
import yaml
import os
import numpy as np
import pandas as pd
from functools import reduce

FUSION_RESULTS_PATH = "results/fusion"
FUSION_METRICS_PATH = "metrics/fusion"


if __name__ == "__main__":
    params = {}
    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params

    result_dirs = sorted(
        [os.path.join("./results/svm", f) for f in params["fusion"]["to_fuse"]]
    )

    os.makedirs(FUSION_RESULTS_PATH, exist_ok=True)
    os.makedirs(FUSION_METRICS_PATH, exist_ok=True)

    for partition in ["devel", "test"]:
        all_predictions = reduce(
            lambda left, right: pd.merge(left, right, on=["filename", "true"]),
            [
                pd.read_csv(os.path.join(result_dir, f"predictions.{partition}.csv"))
                for result_dir in result_dirs
            ],
        )

        prediction_cols = all_predictions.filter(like="prediction").columns.tolist()

        all_predictions["predictions_all"] = (
            all_predictions[prediction_cols]
            .apply(
                lambda row: [
                    round(x, 6)
                    for x in np.mean(np.vstack(row.apply(ast.literal_eval)), axis=0)
                ],
                axis=1,
            )
            .astype(str)
        )
        all_predictions = all_predictions[["filename", "predictions_all", "true"]]
        all_predictions = all_predictions.rename(
            columns={"predictions_all": "prediction"}
        )
        all_predictions[["filename", "prediction", "true"]].to_csv(
            os.path.join(FUSION_RESULTS_PATH, f"predictions.{partition}.csv"),
            index=False,
            lineterminator="",
        )
        all_predictions = all_predictions.applymap(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        all_predictions.to_csv(
            os.path.join(FUSION_RESULTS_PATH, f"predictions.{partition}.csv"),
            index=False,
        )
