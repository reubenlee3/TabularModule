from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from ..feature import FeatureSelection
from ..utils import get_logger

logger = get_logger(__name__)


class DataLoader(object):
    def __init__(self, train: (str, pd.DataFrame) = None, test: (str, pd.DataFrame) = None,
                 is_reduce_memory: bool = False, infer_datatype: bool = False,
                 categorical_columns: Union[list, str] = None, test_ratio: float = None,
                 target_label: str = None, run_feature_selection: bool = False, rfe_estimator: str = None,
                 task: str = None, n_features: Union[int, float] = None,
                 multi_colinear_threshold: float = None, keep_features: Union[list, str] = None,
                 text_features: Union[list, str] = None, seed: int = None):
        """"""
        if isinstance(train, str):
            # Read dataframe from csv path
            self.mode = 'training'
            self.train_path = Path(train)
            self.train_df = pd.read_csv(self.train_path)
            if test is not None:
                assert isinstance(test, str)
                self.test_path = Path(test)
                self.test_df = pd.read_csv(self.test_path)
            else:
                self.test_path = None
                self.test_df = None
        elif isinstance(train, pd.DataFrame):
            # Dataframe was directly fed in
            self.mode = 'training'
            self.train_path = None
            self.train_df = train
            if test is not None:
                assert isinstance(test, pd.DataFrame)
                self.test_path = None
                self.test_df = test
            else:
                self.test_path = None
                self.test_df = None
        else:
            self.train_df = None
            assert isinstance(test, str)
            self.test_path = Path(test)
            self.test_df = pd.read_csv(self.test_path)
            self.mode = 'prediction'
        self.target = target_label
        self.seed = seed
        if isinstance(n_features, float):
            n_features = int(n_features * (self.train_df.shape[1] - 1))
        if is_reduce_memory:
            logger.info('Reducing DataFrame memory')
            self.reduce_mem_usages()
        if not self.mode == 'prediction':
            # Infer datatype
            if not infer_datatype:
                logger.info('Inferring column types from the input')
                self.categorical_cols, self.numerical_cols = self.get_col_types(data=self.train_df, auto=False,
                                                                                categorical_columns=categorical_columns,
                                                                                target=self.target)
            else:
                logger.info('Inferring column types automatically')
                self.categorical_cols, self.numerical_cols = self.get_col_types(data=self.train_df, auto=True,
                                                                                categorical_columns=None,
                                                                                target=self.target)

            # Auto select top features for training
            if not run_feature_selection:
                logger.info('Auto feature selection not selected')
                self.important_columns = None
            else:
                feature_selection = FeatureSelection(train_df=self.train_df, rfe_estimator=rfe_estimator, task=task,
                                                     target=target_label, n_features=n_features,
                                                     keep_features=keep_features, num_features=self.numerical_cols,
                                                     text_features=text_features, cat_features=self.categorical_cols,
                                                     multi_colinear_threshold=multi_colinear_threshold, seed=self.seed)
                self.important_columns = feature_selection.run_feature_selection()
            if self.important_columns is not None:
                self.train_df = self.train_df[self.important_columns]
            self.train_test_split(test_ratio=test_ratio)
            logger.info('train shape: {}, test shape: {}'.format(self.train_df.shape, self.test_df.shape))
        else:
            logger.info('prediction shape: {}'.format(self.test_df.shape))

    @staticmethod
    def get_col_types(data: pd.DataFrame, auto: bool = False,
                      categorical_columns: list = None, target: str = None):
        """"""
        if not auto:
            columns_list = list(data.columns)
            numerical_cols = list(set(columns_list) - set(categorical_columns))
            categorical_cols = categorical_columns
            numerical_cols = numerical_cols
        else:
            categorical_cols = []
            numerical_cols = []
            for col in data.columns:
                if data[col].dtype == 'O':
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            categorical_cols = categorical_cols
            numerical_cols = numerical_cols
        # Remove target column from the list
        try:
            categorical_cols.remove(target)
        except ValueError:
            pass
        try:
            numerical_cols.remove(target)
        except ValueError:
            pass
        logger.info('Number of categorical columns: {}, numerical columns: {}'
                    .format(len(categorical_cols), len(numerical_cols)))
        return categorical_cols, numerical_cols

    def reduce_mem_usages(self):
        """"""
        if self.train_df is not None:
            self.train_df = reduce_mem_usage(self.train_df)
        if self.test_df is not None:
            self.test_df = reduce_mem_usage(self.test_df)

    def train_test_split(self, test_ratio: float = 0.2):
        """"""
        if self.test_df is not None:
            logger.info('Test dataframe is passed separately')
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=self.seed)
            train_index, test_index = next(splitter.split(self.train_df, self.train_df[self.target]))
            self.test_df = self.train_df.loc[self.train_df.index[test_index]]
            self.train_df = self.train_df.loc[self.train_df.index[train_index]]
            logger.info('Creating test(hold-out) by splitting the train dataframe')

    def return_values(self):
        """"""
        if not self.mode == 'prediction':
            # update categorical and numerical feature list
            if self.important_columns is not None:
                logger.info('Updating categorical and numerical cols after feature selection')
                self.categorical_cols = list(set(self.important_columns).intersection(self.categorical_cols))
                self.numerical_cols = list(set(self.important_columns).intersection(self.numerical_cols))
            return self.train_df, self.test_df, self.categorical_cols, self.numerical_cols, self.target
        else:
            return self.test_df


def reduce_mem_usage(df: pd.DataFrame, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
