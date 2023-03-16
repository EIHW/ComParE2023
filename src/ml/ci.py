import numpy as np
import pandas as pd

def CI(preds, labels, score=None):

    score_vals = []
    for s in range(1000):
        np.random.seed(s)
        sample = np.random.choice(
            range(len(preds)), len(preds), replace=True
        )  # boost with replacement
        sample_preds = preds[sample]
        sample_labels = labels[sample]

        uar = score(sample_labels, sample_preds)[0]
        score_vals.append(uar)

    q_0 = pd.DataFrame(np.array(score_vals)).quantile(0.025)[0]  # 2.5% percentile
    q_1 = pd.DataFrame(np.array(score_vals)).quantile(0.975)[0]  # 97.5% percentile

    return q_0, q_1
