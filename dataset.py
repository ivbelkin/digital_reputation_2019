import pandas as pd
import os
import numpy as np


class DRCDataset:

    def __init__(
            self,
            data_root,
            rm_19,
            compress_binary,
            X1_cat_to_bin,
            X1_num_log,
            X1_num_zero_one,
            X1_num_std,
            outliers_X1_num,
            rm_X1_cat_rare,
            seed,
            n_folds
    ):
        self.data_root = data_root
        self.rm_19 = rm_19  # просто удалить бинарный признак 19

        # объединить некоторые группы взаимноискючающих признаков в один
        self.compress_binary = compress_binary

        # преобразовать категориальные переменные в бинарные
        self.X1_cat_to_bin = X1_cat_to_bin

        # прологарифмировать числовые признаки
        self.X1_num_log = X1_num_log

        # шкалировать в [0, 1]
        self.X1_num_zero_one = X1_num_zero_one

        # среднее = 0, СКО = 1
        self.X1_num_std = X1_num_std

        # фильтруем по признаку 9
        self.outliers_X1_num = outliers_X1_num

        # удалить редкие категориальные признаки: 14, 16 - 21
        self.rm_X1_cat_rare = rm_X1_cat_rare

        self.seed = seed

        self.n_folds = n_folds

        self._load_train()
        self._load_test()

    def _load_train(self):
        self._load_train_X1()
        # self._load_train_X2()
        # self._load_train_X3()
        self._load_train_y()

        self._split_on_folds()

        self._filter_outliers()

    def _load_test(self):
        self._load_test_X1()
        # self._load_test_X2()
        # self._load_test_X3()

    def _load_train_X1(self):
        path = os.path.join(self.data_root, "train", "X1.csv")
        train_x1_df = pd.read_csv(path)

        train_x1_df = self._proc_X1_binary(train_x1_df)
        train_x1_df = self._proc_X1_categorical(train_x1_df)
        train_x1_df = self._proc_X1_numerical(train_x1_df)

        self.train_x1_df = train_x1_df

    def _load_test_X1(self):
        path = os.path.join(self.data_root, "test", "X1.csv")
        test_x1_df = pd.read_csv(path)

        test_x1_df = self._proc_X1_binary(test_x1_df)
        test_x1_df = self._proc_X1_categorical(test_x1_df)
        test_x1_df = self._proc_X1_numerical(test_x1_df)

        self.test_x1_df = test_x1_df

    def _proc_X1_binary(self, x1_df):
        binary_features = ["1"] + list(map(str, range(10, 26)))

        if self.rm_19:
            x1_df = x1_df.drop("19", axis=1)

        if self.compress_binary:
            GROUPS = [
                ["14", "15", "16", "17"],
                ["18", "19", "20"], ["21", "22", "23"],
                ["24", "25"]
            ]
            if self.compress_binary >= 1:
                x1_df = self._compress_binary_columns(x1_df, GROUPS[0])
            if self.compress_binary >= 2:
                for i in [1, 2]:
                    x1_df = self._compress_binary_columns(x1_df, GROUPS[i])
            if self.compress_binary >= 3:
                x1_df = self._compress_binary_columns(x1_df, GROUPS[3])

        return x1_df

    def _proc_X1_categorical(self, x1_df):
        N = len(x1_df)
        categorical_features = ["2", "3"]

        if self.rm_X1_cat_rare:
            for feature in ["14", "16", "17", "18", "19", "20", "21"]:
                if feature in x1_df.columns:
                    x1_df = x1_df.drop(feature, axis=1)

        if self.X1_cat_to_bin:
            for feature in categorical_features:
                if feature not in x1_df.columns:
                    continue
                columns = [feature + "==" + str(i) for i in range(-2, 3)]
                for c in columns:
                    x1_df[c] = 0
                one_hot = np.zeros((N, 5), dtype=int)
                one_hot[np.arange(N), x1_df[feature].astype(int) + 2] = 1
                x1_df[columns] = one_hot

        return x1_df

    def _proc_X1_numerical(self, x1_df):
        numeric_features = list(map(str, range(4, 10)))

        if self.X1_num_log:
            x1_df[numeric_features] = np.log1p(x1_df[numeric_features])

        if self.X1_num_zero_one:
            m = x1_df[numeric_features].min()
            M = x1_df[numeric_features].max()
            x1_df[numeric_features] = (x1_df[numeric_features] - m) / (M - m)

        if self.X1_num_std:
            mean = x1_df[numeric_features].mean()
            std = x1_df[numeric_features].std()
            x1_df[numeric_features] = (x1_df[numeric_features] - mean) / std

        return x1_df

    def _compress_binary_columns(self, df, columns):
        if any(c not in df.columns for c in columns):
            return df
        c = df[columns[0]].map(str)
        for column in columns[1:]:
            c += df[column].map(str)
        c = c.map(lambda x: int(x, base=2))
        df = df.drop(columns, axis=1)
        df["_".join(columns)] = c
        return df

    def _load_train_X2(self):
        path = os.path.join(self.data_root, "train", "X2.csv")
        train_x2_df = pd.read_csv(path)
        raise NotImplementedError

    def _load_train_X3(self):
        path = os.path.join(self.data_root, "train", "X3.csv")
        train_x3_df = pd.read_csv(path)
        raise NotImplementedError

    def _load_train_y(self):
        path = os.path.join(self.data_root, "train", "Y.csv")
        train_y_df = pd.read_csv(path)
        self.train_y_df = train_y_df

    def _split_on_folds(self):
        idx = np.arange(len(self.train_x1_df))
        np.random.seed(self.seed)
        np.random.shuffle(idx)
        self.fold = idx % self.n_folds

    def _filter_outliers(self):
        outliers = None
        if self.outliers_X1_num:
            if self.X1_num_log:
                values = self.train_x1_df["9"]
            else:
                values = np.log1p(self.train_x1_df["9"])
            outliers = values > 4.5
        if outliers is not None:
            self.train_x1_df = self.train_x1_df[~outliers]
            self.train_y_df = self.train_y_df[~outliers]
            self.fold = self.fold[~outliers]
