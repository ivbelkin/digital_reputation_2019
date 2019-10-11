import pickle
import pandas as pd

from sklearn.svm import SVC
from dataset import DRCDataset
from copy import copy

from config import *

RESULT_FILES = ["output/result_0-19.pkl", "output/result_20-39.pkl", "output/result_40-59.pkl"]


def find_best_params(results):
    M = [0, 0, 0, 0, 0]
    p = [None, None, None, None, None]
    d = [None, None, None, None, None]
    for data_params, r_list in results:
        for r in r_list:
            t = int(r["target"]) - 1
            if r["valid_roc_auc_mean"] > M[t]:
                M[t] = r["valid_roc_auc_mean"]
                p[t] = r
                d[t] = data_params
    return M, p, d


def main():
    results = []
    for name in RESULT_FILES:
        with open(name, "rb") as f:
            results.extend(pickle.load(f))

    M, p, d = find_best_params(results)

    submission = {}
    models = []
    for target in range(1, 6):
        print("Target", target)
        data_params = d[target - 1]
        ds = DRCDataset(data_root=DATA_ROOT, **data_params)

        model_params = p[target - 1]
        mp = copy(model_params)

        del model_params["train_roc_auc_mean"], model_params["valid_roc_auc_mean"]
        del model_params["train_roc_auc_std"], model_params["valid_roc_auc_std"]
        del model_params["target"]
        model = SVC(verbose=0, probability=True, max_iter=100000, **model_params)

        for fold in range(N_FOLDS):
            print("Fold", fold)
            X_train = ds.train_x1_df[ds.fold != fold].drop("id", axis=1)
            y_train = ds.train_y_df[ds.fold != fold][str(target)]

            X_test = ds.test_x1_df.drop("id", axis=1)
            submission["id"] = ds.test_x1_df["id"]

            model.fit(X_train, y_train)
            models.append({"data_params": data_params, "model_params": mp, "model": model})

            if str(target) not in submission:
                submission[str(target)] = 0
            submission[str(target)] += model.predict_proba(X_test)[:, 1]

    for target in range(1, 6):
        submission[str(target)] /= N_FOLDS

    submission = pd.DataFrame(submission)
    submission.to_csv("submissions/smbt_svm.csv", index=False)

    with open("output/svm_models.pkl", "wb") as f:
        pickle.dump(models, f)


if __name__ == "__main__":
    main()
