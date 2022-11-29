from pathlib import Path
import pandas as pd
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger

logger = get_logger(__name__)


class DataLoader(object):
    def __init__(self, train: (str, pd.DataFrame) = None, test: (str, pd.DataFrame) = None,
                 is_reduce_memory: bool = False, infer_datatype: bool = False, categorical_columns: list = None,
                 test_ratio: float = None, target_label: str = None, seed: int = None):
        """"""
        if isinstance(train, str):
            # Read dataframe from csv path
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
            self.train_path = None
            self.train_df = train
            if test is not None:
                assert isinstance(test, pd.DataFrame)
                self.test_path = None
                self.test_df = test
            else:
                self.test_path = None
                self.test_df = None
        self.target = target_label
        self.seed = seed
        if is_reduce_memory:
            logger.info('Reducing DataFrame memory')
            self.reduce_mem_usages()
        if not infer_datatype:
            logger.info('Inferring column types from the input')
            self.get_col_types(auto=False, categorical_columns=categorical_columns)
        else:
            logger.info('Inferring column types automatically')
            self.get_col_types(auto=True, categorical_columns=None)
        self.train_test_split(test_ratio=test_ratio)
        logger.info('train shape: {}, test shape: {}'.format(self.train_df.shape, self.test_df.shape))

    def get_col_types(self, auto: bool = False, categorical_columns: list = None):
        """"""
        if not auto:
            columns_list = list(self.train_df.columns)
            numerical_cols = list(set(columns_list) - set(categorical_columns))
            self.categorical_cols = categorical_columns
            self.numerical_cols = numerical_cols
        else:
            categorical_cols = []
            numerical_cols = []
            for col in self.train_df.columns:
                if self.train_df[col].dtype == 'O':
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            self.categorical_cols = categorical_cols
            self.numerical_cols = numerical_cols
        # Remove target column from the list
        try:
            self.categorical_cols.remove(self.target)
        except ValueError:
            pass
        try:
            self.numerical_cols.remove(self.target)
        except ValueError:
            pass
        logger.info('Number of categorical columns: {}, numerical columns: {}'
                    .format(len(self.categorical_cols), len(self.numerical_cols)))

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
        return self.train_df, self.test_df, self.categorical_cols, self.numerical_cols, self.target


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


def data_drift_report(training_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                      save_dir: str = None):
    """"""
    logger.info('Calculating Data drift between training and testing dataframes')
    t0 = time.time()
    drift_report = Report(metrics=[DataDriftPreset(),
                                        ])
    drift_report.run(current_data=test_data, reference_data=training_data)
    file_path = os.path.join(save_dir, 'data_drift_report.html')
    drift_report.save_html(file_path)
    t1 = time.time()
    logger.info('time taken to generate drift is {:.2f} secs'.format(t1-t0))
    logger.info('Saved Data drift file in {}'.format(file_path))
