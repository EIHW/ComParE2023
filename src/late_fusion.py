#!/usr/bin/env python
import ast

import json
import yaml
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix
from functools import reduce
from collections import Counter
from ml.ci import CI
from pathlib import Path
from os.path import join

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
    test_file = Path(join("data/lab/", "test.unmasked.csv"))
    if test_file.is_file():
        test_df = True
    else: 
        print('No Test labels')
        test_df = None
        
             
    COLUMN_NAMES = [
        "Anger",
        "Boredom",
        "Calmness",
        "Concentration",
        "Determination",
        "Excitement",
        "Interest",
        "Sadness",
        "Tiredness",
    ]

    for partition in ['devel', 'test']: 

        all_predictions = reduce(
        lambda left, right: pd.merge(left, right, on=["filename"]),
        [
            pd.read_csv(os.path.join(result_dir, f"predictions.{partition}.csv"))
            for result_dir in result_dirs
        ],
        )

        pred_means = pd.DataFrame()
        for emotion in COLUMN_NAMES:
            pred_cols = [col for col in all_predictions.columns if 'pred' in col and col.startswith(emotion)]
            pred_mean = all_predictions[pred_cols].mean(axis=1)
            pred_means[f'{emotion}_pred'] = pred_mean

        true_cols = [col for col in all_predictions.columns if 'true_x' in col]
        all_true = all_predictions[true_cols]

        all_true.columns = [col.rstrip('_x') for col in all_true.columns]
        filename_df = all_predictions['filename']
        all_predictions_fusion = pd.concat([filename_df,all_true,pred_means],axis=1)


        all_predictions_fusion.to_csv(os.path.join(FUSION_RESULTS_PATH, f"predictions.{partition}.csv"), index=False)



