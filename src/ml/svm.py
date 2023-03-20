#!/usr/bin/env python
import os
import json
import sys
import yaml
from pathlib import Path
from os.path import join


import audmetric
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

RANDOM_SEED = 42


def make_dict_json_serializable(meta_dict: dict) -> dict:
    cleaned_meta_dict = meta_dict.copy()
    for key in cleaned_meta_dict:
        if type(cleaned_meta_dict[key]) not in [str, float, int]:
            cleaned_meta_dict[key] = str(cleaned_meta_dict[key])
    return cleaned_meta_dict


def load_features(feature_file, label_base):
    
    test_file_exist = Path(join(label_base, "test.unmasked.csv"))
    if test_file_exist.is_file():
        labels = {
        partition: pd.read_csv(f"{label_base}/{partition}.csv")
        for partition in ["train", "devel", "test.unmasked"]
        }
    else: 
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
    if test_file_exist.is_file():
        test_df = joined_dfs["test.unmasked"]
    else: 
        test_df = joined_dfs["test"]

        # subset of only specified groups and water rounds
    feature_names = list(df.columns[1:])
    train_devel_names = train_devel_df.values[:, 0].tolist()
    train_devel_features = train_devel_df[feature_names].values
    train_devel_labels = train_devel_df[COLUMN_NAMES].values

    _split_indices = [-1] * len(joined_dfs["train"]) + [0] * len(joined_dfs["devel"])
    train_devel_names = np.array(train_devel_names)
    train_devel_X = train_devel_features
    train_devel_y = np.array(train_devel_labels)
    split = np.array(_split_indices)

    test_names = test_df.values[:, 0].tolist()
    test_features = test_df[feature_names].values
    test_labels = test_df[COLUMN_NAMES].values
    test_names = np.array(test_names)
    test_X = test_features
    test_y = np.array(test_labels)
    return (
        (train_devel_names, train_devel_X, train_devel_y, split),
        (test_names, test_X, test_y),
        feature_names,
    )


def run_svm(feature_folder, label_base, result_folder, metrics_folder, params):

    if params["type"] == "regression":
        clf_class = LinearSVR

        scoring = "neg_mean_absolute_error"
        grid = [
            {
                "scaler": [StandardScaler(), MinMaxScaler()],
                "svm__estimator": [clf_class(random_state=RANDOM_SEED)],
                "svm__estimator__loss": [
                    "epsilon_insensitive",
                    "squared_epsilon_insensitive",
                ],
                "svm__estimator__C": np.logspace(-2, -6, num=5),
                "svm__estimator__max_iter": [50000],
            }
        ]
    else:
        raise NotImplementedError(f"{params['type']} not supported.")

    pipeline = Pipeline([("scaler", None), ("svm", MultiOutputRegressor(clf_class()))])

    feature_files = os.path.join(feature_folder, "features.csv")
    (
        (train_devel_names, train_devel_X, train_devel_y, _split),
        (test_names, test_X, test_y),
        feature_names,
    ) = load_features(feature_files, label_base)

    train_X = train_devel_X[np.argwhere(_split == -1)].squeeze()
    train_y = train_devel_y[np.argwhere(_split == -1)].squeeze()

    devel_X = train_devel_X[np.argwhere(_split == 0)].squeeze()
    devel_y = train_devel_y[np.argwhere(_split == 0)].squeeze()
    devel_names = train_devel_names[np.argwhere(_split == 0)].squeeze()

    split = PredefinedSplit(_split)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring=scoring,
        n_jobs=-1,
        cv=split,
        refit=True,
        verbose=0,
        return_train_score=False,
    )

    # fit on data. train -> devel first, then train+devel implicit
    grid_search.fit(train_devel_X, train_devel_y)
    test_preds = grid_search.predict(test_X)
    best_estimator = grid_search.best_estimator_
    print(best_estimator)
    # fit clone of best estimator on train again for devel predictions
    estimator = clone(best_estimator, safe=False)
    estimator.fit(train_X, train_y)
    devel_preds = estimator.predict(devel_X)

    # Evaluate on development set
    mean_pearsons = [
        audmetric.pearson_cc(devel_y[:, i], devel_preds[:, i])
        for i in range(len(COLUMN_NAMES))
    ]
    mean_cc = np.average(mean_pearsons)

    svm_params = make_dict_json_serializable(grid_search.best_params_)

    df_predictions_devel = pd.DataFrame(
        {
            "filename": devel_names,
            "prediction": devel_preds.tolist(),
            "true": devel_y.tolist(),
        }
    )


    df_predictions_test = pd.DataFrame(
        {
            "filename": test_names,
            "prediction": test_preds.tolist(),
            "true": test_y.tolist(),
        }
    )

    with open(os.path.join(result_folder, "best_params.json"), "w") as f:
        json.dump(svm_params, f)

    pd.DataFrame(grid_search.cv_results_).to_csv(
        os.path.join(result_folder, "grid_search.csv"), index=False
    )

    devel_names = df_predictions_devel["filename"]

    for label_type in ["true", "prediction"]:
        if label_type == "true":
            col_type = [str(i) + "_true" for i in COLUMN_NAMES]
        if label_type == "prediction":
            col_type = [str(i) + "_pred" for i in COLUMN_NAMES]
        for index, row in df_predictions_devel.iterrows():
            res = df_predictions_devel[label_type][index]
            for res, col in zip(res, col_type):
                df_predictions_devel.at[index, col] = float(res)
        if label_type == "true":
            devel_y = df_predictions_devel[col_type]
        if label_type == "prediction":
            devel_preds = df_predictions_devel[col_type]

    test_names = df_predictions_test["filename"]

    test_file_exist = Path(join(label_base, "test.unmasked.csv"))
    if test_file_exist.is_file():
        test_df = pd.read_csv(join(label_base, "test.unmasked.csv"))
    else: 
        print('No Test file, loading masked')
        test_df = pd.read_csv(join(label_base, "test.csv"))
    for label_type in ["true", "prediction"]:
        if label_type == "true":
            col_type = [str(i) + "_true" for i in COLUMN_NAMES]
        if label_type == "prediction":
            col_type = [str(i) + "_pred" for i in COLUMN_NAMES]

        for index, row in df_predictions_test.iterrows():
            res = df_predictions_test[label_type][index]
            for res, col in zip(res, col_type):
                if test_file_exist.is_file():
                    df_predictions_test.at[index, col] = float(res)
                else:
                    df_predictions_test.at[index, col] = res
        if label_type == "true":
            test_y = df_predictions_test[col_type]
        if label_type == "prediction":
            test_preds = df_predictions_test[col_type]

    df_predictions_devel = df_predictions_devel.drop(['prediction', 'true'],axis=1)
    df_predictions_devel.to_csv(
        os.path.join(result_folder, "predictions.devel.csv"), index=False
    )
    df_predictions_test = df_predictions_test.drop(['prediction', 'true'],axis=1)
    df_predictions_test.to_csv(
        os.path.join(result_folder, "predictions.test.csv"), index=False
    )


if __name__ == "__main__":
    feature_type = sys.argv[1]
    feature_base = f"./data/features/{feature_type}/"
    result_folder = f"./results/svm/{feature_type}/"
    metrics_folder = f"./metrics/svm/{feature_type}/"
    label_base = f"./data/lab"

    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params["ml"]

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    run_svm(feature_base, label_base, result_folder, metrics_folder, params)
