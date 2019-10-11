DATA_ROOT = "/home/ivb/nvme/digital_reputation_2019/data/"

SEED = 42

N_FOLDS = 5

MODEL_PARAM_GRID = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4],  # only for 'poly' kernel
        "gamma": ["auto", "scale"],  # only for 'rbf', 'poly', 'sigmoid'
        "coef0": [0.0, 0.01, 0.1, 1.0, 10],  # 'poly', 'sigmoid'
        "shrinking": [True, False],
        "class_weight": [None, "balanced"],

        "target": ["1", "2", "3", "4", "5"],
    }

# сколько комбинаций проверить
MODEL_PARAM_N = 500  # из 21500
# на самом деле для разных предобработок будет своя 1000 наборов параметров

DATA_PARAM_GRID = {
    "rm_19": [True],
    "compress_binary": [0, 1, 2, 3],
    "X1_cat_to_bin": [True, False],
    "X1_num_log": [True],
    "X1_num_zero_one": [True, False],
    "X1_num_std": [True, False],
    "outliers_X1_num": [True, False],
    "rm_X1_cat_rare": [True, False],
    "seed": [SEED],
    "n_folds": [N_FOLDS]
}
