import yaml
import pycm
import sys
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from ci import CI
from os import makedirs
from os.path import join


def compute_metrics(devel_df, test_df):
    devel_names, devel_preds, devel_y = (
        devel_df["filename"].values,
        devel_df["prediction"].values,
        devel_df["true"].values,
    )
    test_names, test_preds, test_y = (
        test_df["filename"].values,
        test_df["prediction"].values,
        test_df["true"].values,
    )
    metrics = {"devel": {}, "test": {}}

    # metrics
    print("Devel:\n")
    uar_devel = recall_score(devel_y, devel_preds, average="macro")
    cm_devel = confusion_matrix(devel_y, devel_preds)
    print("Computing CI...")
    ci_low, ci_high = CI(devel_preds, devel_y)
    print(f"UAR: {uar_devel:.2%} ({ci_low:.2%} - {ci_high:.2%})")
    cr_devel = classification_report(devel_y, devel_preds)
    mega_report_devel = pycm.ConfusionMatrix(devel_y, devel_preds)
    print(cr_devel)

    metrics["devel"]["uar"] = float(uar_devel)
    metrics["devel"]["ci_low"] = float(ci_low)
    metrics["devel"]["ci_high"] = float(ci_high)
    metrics["devel"]["cm"] = cm_devel.tolist()

    if len(set(test_y)) > 1:
        print("Test:\n")
        uar_test = recall_score(test_y, test_preds, average="macro")
        cm_test = confusion_matrix(test_y, test_preds)
        print("Computing CI...")
        ci_low, ci_high = CI(test_preds, test_y)
        print(f"UAR: {uar_test:.2%} ({ci_low:.2%} - {ci_high:.2%})")
        cr_test = classification_report(test_y, test_preds)
        mega_report_test = pycm.ConfusionMatrix(test_y, test_preds)
        print(cr_test)

        metrics["test"]["uar"] = float(uar_test)
        metrics["test"]["cm"] = cm_test.tolist()
        metrics["test"]["ci_low"] = float(ci_low)
        metrics["test"]["ci_high"] = float(ci_high)

    return metrics


if __name__ == "__main__":
    experiment = sys.argv[1]
    result_folder = f"./results/{experiment}"
    metrics_folder = f"./metrics/{experiment}"

    devel_df = pd.read_csv(join(result_folder, "predictions.devel.csv"))
    test_df = pd.read_csv(join(result_folder, "predictions.test.csv"))

    metrics = compute_metrics(devel_df, test_df)

    makedirs(metrics_folder, exist_ok=True)

    with open(join(metrics_folder, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)
