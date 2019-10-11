from dataset import DRCDataset
from multiprocessing import cpu_count, Pool
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from copy import copy
from sklearn.metrics import roc_auc_score

import time
import numpy as np
import pickle

from config import *

ds = None


def create_model_params_list():
    pg = ParameterGrid(MODEL_PARAM_GRID)
    params = []
    for p in pg:
        if p["kernel"] != "poly":
            del p["degree"]
        if p["kernel"] not in ["rbf", "poly", "sigmoid"]:
            del p["gamma"]
        if p["kernel"] not in ["poly", "sigmoid"]:
            del p["coef0"]
        params.append(tuple(sorted(p.items())))
    params_list = [dict(p) for p in set(params)]
    return params_list


def create_data_params_list():
    pg = ParameterGrid(DATA_PARAM_GRID)
    params = []
    for p in pg:
        if p["compress_binary"] > 0:
            p["rm_X1_cat_rare"] = False
        if p["rm_X1_cat_rare"]:
            p["compress_binary"] = 0
        if p["X1_num_zero_one"]:
            p["X1_num_std"] = False
        if p["X1_num_std"]:
            p["X1_num_zero_one"] = False
        params.append(tuple(sorted(p.items())))
    params_list = [dict(p) for p in set(params)]
    return params_list


def train_score(params):
    params = copy(params)
    target = params["target"]
    del params["target"]

    train_scores = []
    valid_scores = []
    for fold in range(N_FOLDS):
        X_train = ds.train_x1_df[ds.fold != fold].drop("id", axis=1)
        X_valid = ds.train_x1_df[ds.fold == fold].drop("id", axis=1)

        y_train = ds.train_y_df[ds.fold != fold][target]
        y_valid = ds.train_y_df[ds.fold == fold][target]

        model = SVC(verbose=0, probability=True, max_iter=100000, **params)
        model.fit(X_train, y_train)

        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_valid_pred_proba = model.predict_proba(X_valid)[:, 1]

        train_score = roc_auc_score(y_train, y_train_pred_proba)
        valid_score = roc_auc_score(y_valid, y_valid_pred_proba)

        train_scores.append(train_score)
        valid_scores.append(valid_score)

    return {
        "train_roc_auc_mean": np.mean(train_scores),
        "valid_roc_auc_mean": np.mean(valid_scores),
        "train_roc_auc_std": np.std(train_scores),
        "valid_roc_auc_std": np.std(valid_scores)
    }


def main():
    global ds
    n_cpu = cpu_count()
    print("Running on", n_cpu, "cores")

    model_params_list = create_model_params_list()
    data_params_list = create_data_params_list()

    result = []
    total_start = time.clock_gettime(time.CLOCK_MONOTONIC)
    with open("output/progress.txt", "w") as f:
        f.write("Running on " + str(n_cpu) + " cores\n")
    for i, data_params in enumerate(data_params_list):
        np.random.seed(SEED + i)
        np.random.shuffle(model_params_list)

        ds = DRCDataset(data_root=DATA_ROOT, **data_params)

        start = time.clock_gettime(time.CLOCK_MONOTONIC)
        pool = Pool(n_cpu)
        res = pool.map(train_score, model_params_list[:MODEL_PARAM_N])
        end = time.clock_gettime(time.CLOCK_MONOTONIC)

        with open("output/progress.txt", "a") as f:
            f.write("Completed " + str(i + 1) + " of " + str(len(data_params_list)) + "\n")
            f.write("Time elapsed: " + str("{:.1f}\n\n".format(end - start)))

        for r, mp in zip(res, model_params_list):
            r.update(mp)
        result.append((data_params, res))
    total_end = time.clock_gettime(time.CLOCK_MONOTONIC)

    with open("output/progress.txt", "a") as f:
        f.write("Total time: " + str("{:.1f}\n".format(total_end - total_start)))

    with open("output/result.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
