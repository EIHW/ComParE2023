#!/usr/bin/env python
import sys
import yaml
import audmetric
import numpy as np
import pandas as pd
import scipy.stats as stats
from os import makedirs
from os.path import join
from pathlib import Path

from ci import CI

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


def compute_metrics(devel_df, test_df, params):

    devel_names = devel_df["filename"]

    for label_type in ["true", "prediction"]:
        if label_type == "true":
            col_type = [str(i) + "_true" for i in COLUMN_NAMES]
        if label_type == "prediction":
            col_type = [str(i) + "_pred" for i in COLUMN_NAMES]

        if label_type == "true":
            devel_y = devel_df[col_type]
        if label_type == "prediction":
            devel_preds = devel_df[col_type]
    if test_df is not None:
        test_names = test_df["filename"]

        for label_type in ["true", "prediction"]:
            if label_type == "true":
                col_type = [str(i) + "_true" for i in COLUMN_NAMES]
            if label_type == "prediction":
                col_type = [str(i) + "_pred" for i in COLUMN_NAMES]

            if label_type == "true":
                test_y = test_df[col_type]
            if label_type == "prediction":
                test_preds = test_df[col_type]
    else:
        print("No Test Evaluation ...")



    metrics = {"devel": {}, "test": {}}

    if params["type"] == "regression":

        print("Best Devel:\n")
        all_rho_devel = []
        all_ci_low_devel = []
        all_ci_high_devel = []

        for label in COLUMN_NAMES:
            rho_devel = stats.spearmanr(
                devel_y[f"{label}_true"], devel_preds[f"{label}_pred"]
            )[0]

            print(f"Calculating CIs: {label}")
            ci_low, ci_high = CI(
                devel_preds[f"{label}_pred"], devel_y[f"{label}_true"], stats.spearmanr
            )
            print(
                f"{label} Rho: {round(rho_devel,4)} ({round(ci_low,4)} - {round(ci_high,4)})"
            )

            all_rho_devel.append(rho_devel)
            all_ci_low_devel.append(float(ci_low))
            all_ci_high_devel.append(float(ci_high))

            metrics["devel"][f"{label}_ci_low_rho"] = float(ci_low)
            metrics["devel"][f"{label}_ci_high_rho"] = float(ci_high)
            metrics["devel"][f"{label}_rho"] = float(rho_devel)

        mean_rho = np.average(all_rho_devel)
        mean_low_ci = np.average(all_ci_low_devel)
        mean_high_ci = np.average(all_ci_high_devel)

        print(
            f"Mean Rho: {np.round(np.average(all_rho_devel),4)} ({float(mean_low_ci)} - {float(mean_high_ci)})"
        )

        metrics["devel"][f"[Mean_rho]"] = float(mean_rho)
        metrics["devel"][
            f"[Mean_Rho_CI]"
        ] = f"({float(mean_low_ci)} - {float(mean_high_ci)})"

        if test_df is not None:

            print("Test:\n")
            all_rho_test = []
            all_ci_low_test = []
            all_ci_high_test = []

            for label in COLUMN_NAMES:
                rho_test = stats.spearmanr(
                    test_y[f"{label}_true"], test_preds[f"{label}_pred"]
                )[0]

                print(f"Calculating CIs: {label}")
                ci_low, ci_high = CI(
                    test_preds[f"{label}_pred"], test_y[f"{label}_true"], stats.spearmanr
                )

                print(
                    f"{label} Rho: {round(rho_test,4)} ({round(ci_low,4)} - {round(ci_high,4)})"
                )

                all_rho_test.append(rho_test)
                all_ci_low_test.append(float(ci_low))
                all_ci_high_test.append(float(ci_high))

                metrics["test"][f"{label}_ci_low_rho"] = float(ci_low)
                metrics["test"][f"{label}_ci_high_rho"] = float(ci_high)
                metrics["test"][f"{label}_rho"] = float(rho_devel)

            mean_rho = np.average(all_rho_test)
            mean_low_ci = np.average(all_ci_low_test)
            mean_high_ci = np.average(all_ci_high_test)

            print(
                f"Mean Rho: {np.round(np.average(all_rho_test),4)} ({float(mean_low_ci)} - {float(mean_high_ci)})"
            )

            metrics["test"][f"[Mean_Rho]"] = float(mean_rho)
            metrics["test"][
                f"[Mean_Rho_CI]"
            ] = f"({float(mean_low_ci)} - {float(mean_high_ci)})"
        else: 
            print("No Test Evaluation ...")

    else:
        raise NotImplementedError(f"{params['type']} not supported.")
    return metrics


if __name__ == "__main__":
    experiment = sys.argv[1]
    result_folder = f"./results/{experiment}"
    metrics_folder = f"./metrics/{experiment}"
    lab_folder = f"./data/lab/"
    devel_df = pd.read_csv(join(result_folder, "predictions.devel.csv"))


    test_file = Path(join(lab_folder, "test.unmasked.csv"))
    if test_file.is_file():
        test_df = pd.read_csv(join(result_folder, "predictions.test.csv"))
    else: 
        print('No Test file')
        test_df = None

    with open("params.yaml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params["ml"]
    metrics = compute_metrics(devel_df, test_df, params)
    makedirs(metrics_folder, exist_ok=True)

    with open(join(metrics_folder, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)
