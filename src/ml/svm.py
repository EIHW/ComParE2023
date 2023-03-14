#!/usr/bin/env python
import json
import os
import re
import sys
from unittest import result
import matplotlib.pyplot as plt
from os.path import splitext

import numpy as np
import pandas as pd
import pycm
import yaml
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from ci import CI


RANDOM_SEED = 42

GRID = [
    {
        "scaler": [StandardScaler(), MinMaxScaler()],
        "estimator": [LinearSVC(random_state=RANDOM_SEED)],
        "estimator__loss": ["squared_hinge"],
        "estimator__C": np.logspace(-2, -6, num=5),
        "estimator__class_weight": ["balanced"],
        "estimator__max_iter": [10000],
    }
]

PIPELINE = Pipeline([("scaler", None), ("estimator", LinearSVC())])


def make_dict_json_serializable(meta_dict: dict) -> dict:
    cleaned_meta_dict = meta_dict.copy()
    for key in cleaned_meta_dict:
        if type(cleaned_meta_dict[key]) not in [str, float, int]:
            cleaned_meta_dict[key] = str(cleaned_meta_dict[key])
    return cleaned_meta_dict


def load_features(feature_file, label_base, target="complaint"):
    labels = {
        partition: pd.read_csv(f"{label_base}/{partition}.csv")
        for partition in ["train", "devel", "test"]
    }
    feature_delimiter = ";" if "opensmile" in feature_file else ","
    df = pd.read_csv(feature_file, delimiter=feature_delimiter, quotechar="'")
    joined_dfs = {
        partition: df.merge(
            _labels, left_on=df.columns[0], right_on=_labels.columns[0]
        ).sort_values(by=df.columns[0])
        for partition, _labels in labels.items()
    }
    train_devel_df = pd.concat([joined_dfs["train"], joined_dfs["devel"]])
    test_df = joined_dfs["test"]
    # subset of only specified groups and water rounds
    feature_names = list(df.columns[1:])
    train_devel_names = train_devel_df.values[:, 0].tolist()
    train_devel_features = train_devel_df[feature_names].values
    train_devel_labels = train_devel_df[target].values
    _split_indices = [-1] * len(joined_dfs["train"]) + [0] * len(joined_dfs["devel"])
    train_devel_names = np.array(train_devel_names)
    train_devel_X = train_devel_features
    train_devel_y = np.array(train_devel_labels)
    split = np.array(_split_indices)

    test_names = test_df.values[:, 0].tolist()
    test_features = test_df[feature_names].values
    test_labels = test_df[target].values
    test_names = np.array(test_names)
    test_X = test_features
    test_y = np.array(test_labels)
    return (
        (train_devel_names, train_devel_X, train_devel_y, split),
        (test_names, test_X, test_y),
        feature_names,
    )


def run_svm(feature_folder, label_base, result_folder, metrics_folder, target="complaint"):
    feature_files = os.path.join(feature_folder, "features.csv")
    (
        (train_devel_names, train_devel_X, train_devel_y, _split),
        (test_names, test_X, test_y),
        feature_names,
    ) = load_features(feature_files, label_base, target=target)

    train_X = train_devel_X[np.argwhere(_split == -1)].squeeze()
    train_y = train_devel_y[np.argwhere(_split == -1)].squeeze()

    devel_X = train_devel_X[np.argwhere(_split == 0)].squeeze()
    devel_y = train_devel_y[np.argwhere(_split == 0)].squeeze()
    devel_names = train_devel_names[np.argwhere(_split == 0)].squeeze()

    scoring = "recall_macro"
    grid = GRID

    split = PredefinedSplit(_split)

    grid_search = GridSearchCV(
        estimator=PIPELINE,
        param_grid=grid,
        scoring=scoring,
        n_jobs=-1,
        cv=split,
        refit=True,
        verbose=1,
        return_train_score=False,
    )

    # fit on data. train -> devel first, then train+devel implicit
    grid_search.fit(train_devel_X, train_devel_y)
    test_preds = grid_search.predict(test_X)
    best_estimator = grid_search.best_estimator_

    # fit clone of best estimator on train again for devel predictions
    estimator = clone(best_estimator, safe=False)
    estimator.fit(train_X, train_y)
    devel_preds = estimator.predict(devel_X)


    svm_params = make_dict_json_serializable(grid_search.best_params_)

    df_predictions_devel = pd.DataFrame(
        {
            "filename": devel_names,
            "prediction": devel_preds.tolist(),
            "true": devel_y.tolist(),
        }
    )
    df_predictions_devel.to_csv(
        os.path.join(result_folder, "predictions.devel.csv"), index=False
    )

    df_predictions_test = pd.DataFrame(
        {
            "filename": test_names,
            "prediction": test_preds.tolist(),
            "true": test_y.tolist(),
        }
    )
    df_predictions_test.to_csv(
        os.path.join(result_folder, "predictions.test.csv"), index=False
    )

    with open(os.path.join(result_folder, "best_params.json"), "w") as f:
        json.dump(svm_params, f)

    pd.DataFrame(grid_search.cv_results_).to_csv(
        os.path.join(result_folder, "grid_search.csv"), index=False
    )


if __name__ == "__main__":
    feature_type = sys.argv[1]
    feature_base = f"./data/features/{feature_type}/"
    result_folder = f"./results/svm/{feature_type}/"
    metrics_folder = f"./metrics/svm/{feature_type}/"
    label_base = f"./data/lab"
    with open("params.yaml") as f:
        target = yaml.safe_load(f)["target"]

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    run_svm(feature_base, label_base, result_folder, metrics_folder, target=target)
