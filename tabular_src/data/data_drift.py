from typing import Union, NoReturn, Dict
import pandas as pd
import time
import os
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import RegressionTestPreset
from evidently.test_preset import MulticlassClassificationTestPreset
from evidently.test_preset import BinaryClassificationTopKTestPreset
from evidently.test_preset import BinaryClassificationTestPreset

from .data_driftaction import DriftActions
from ..utils import get_logger

logger = get_logger(__name__)


class DataIntegrityTest(object):
    def __init__(self, df: pd.DataFrame = None, categorical_columns: list = None, numerical_columns: list = None,
                 datetime_columns: list = None, target_label: str = None, task: str = None, seed: int = None):
        assert isinstance(df, pd.DataFrame), 'Data is not in Pandas DataFrame'
        self.df = df
        self.cat_colummns = categorical_columns
        self.num_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.target = target_label
        self.seed = seed
        self.task = task
        self.col_mapping = create_colmapping(target=target_label, prediction='None', datetime_columns=datetime_columns,
                                             num_columns=numerical_columns, cat_colummns=categorical_columns,
                                             task=self.task)

    def run_integrity_checks(self, save_html: bool = False, save_dir: str = None,
                             return_dict: bool = False) -> Union[NoReturn, Dict]:
        """"""
        # dataset-level tests
        t0 = time.time()
        # data_integrity_dataset_tests = TestSuite(tests=[
        #     TestAllColumnsShareOfMissingValues(),
        #     TestNumberOfConstantColumns(),
        #     TestNumberOfDuplicatedColumns(),
        #     TestNumberOfDuplicatedRows(),
        #     TestHighlyCorrelatedColumns(),
        #     # TODO: Check Target feature correlation in the report
        # ])
        data_integrity_dataset_tests = Report(metrics=[DataQualityPreset(),
                                                       ])
        data_integrity_dataset_tests.run(current_data=self.df, reference_data=None, column_mapping=None)
        if not save_html:
            logger.info('Not saving data integrity in html/json format')
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, 'data_integrity_report.html')
            data_integrity_dataset_tests.save_html(file_path)
            file_path = os.path.join(save_dir, 'data_integrity_report.json')
            data_integrity_dataset_tests.save_json(file_path)
            logger.info('Saved Data drift file in {}'.format(file_path))
        if return_dict:
            logger.info('Returning drift as dictionary to process dataframe')
            test_results = data_integrity_dataset_tests.as_dict()
            return test_results['metrics']
        else:
            logger.info('Not returning drift as dictionary')
        t1 = time.time()
        logger.info('time taken to generate data-integrity report is {:.2f} secs'.format(t1 - t0))

    def act_testresults(self, test_results: dict = None) -> pd.DataFrame:
        """"""
        integrity_result = test_results.copy()
        drift_actions = DriftActions(data=self.df, test_results=integrity_result)
        data = drift_actions.run_integrity_action()
        return data


class TrainingDataDrift(object):
    def __init__(self, train_df: pd.DataFrame = None, test_df: pd.DataFrame = None,
                 categorical_columns: list = None, numerical_columns: list = None,
                 datetime_columns: list = None, target_label: str = None, prediction_label: str = None,
                 task: str = None, seed: int = None):
        assert isinstance(train_df, pd.DataFrame), 'Training data is not in Pandas DataFrame'
        assert isinstance(test_df, pd.DataFrame), 'Test data is not in Pandas DataFrame'
        self.train_df = train_df
        self.test_df = test_df
        self.cat_columns = categorical_columns
        self.num_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.target = target_label
        self.seed = seed
        # TODO: Hard-coded the task to regression so that Target is considered as numeric
        self.task = 'regression'
        self.col_mapping = create_colmapping(target=self.target, prediction=prediction_label,
                                             datetime_columns=self.datetime_columns, task=self.task,
                                             num_columns=self.num_columns, cat_colummns=self.cat_columns)

    def run_drift_checks(self, save_html: bool = False, save_dir: str = None, return_dict: bool = False,
                         filename: str = 'training_datadrift') -> Union[NoReturn, Dict]:
        """"""
        # dataset-level tests
        t0 = time.time()
        # datadrift_tests = TestSuite(tests=[
        #     TestNumberOfDriftedColumns(),
        # ])
        datadrift_tests = Report(metrics=[DataDriftTable(),
                                          ])
        if self.target is not None:
            test_df = self.test_df.drop(columns=self.target, inplace=False).sort_index(axis=1, ascending=True,
                                                                                       inplace=False)
            train_df = self.train_df.drop(columns=self.target, inplace=False).sort_index(axis=1, ascending=True,
                                                                                         inplace=False)
        else:
            test_df = self.test_df.copy()
            train_df = self.train_df.copy()
        datadrift_tests.run(current_data=test_df, reference_data=train_df, column_mapping=self.col_mapping)
        if not save_html:
            logger.info('Not saving data integrity in html/json format')
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, '{}_report.html'.format(filename))
            datadrift_tests.save_html(file_path)
            file_path = os.path.join(save_dir, '{}_report.json'.format(filename))
            datadrift_tests.save_json(file_path)
            logger.info('Saved Data drift file in {}'.format(file_path))
        if return_dict:
            logger.info('Returning drift as dictionary to process dataframe')
            test_results = datadrift_tests.as_dict()
            return test_results['metrics'][0]
        t1 = time.time()
        logger.info('time taken to generate data-drift is {:.2f} secs'.format(t1 - t0))

    def run_target_drift_checks(self, save_html: bool = False, save_dir: str = None,
                                return_dict: bool = False) -> Union[NoReturn, Dict]:
        """"""
        # dataset-level tests
        t0 = time.time()
        # datadrift_tests = TestSuite(tests=[
        #     TestValueDrift(),
        # ])
        target_drift_tests = Report(metrics=[TargetDriftPreset(),
                                             ])
        target_drift_tests.run(current_data=self.test_df, reference_data=self.train_df, column_mapping=self.col_mapping)
        if not save_html:
            logger.info('Not saving data integrity in html/json format')
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, 'training_targetdrift_report.html')
            # target_drift_tests.save_html(file_path)
            file_path = os.path.join(save_dir, 'training_targetdrift_report.json')
            target_drift_tests.save_json(file_path)
            logger.info('Saved Data drift file in {}'.format(file_path))
        if return_dict:
            logger.info('Returning drift as dictionary to process dataframe')
            test_results = target_drift_tests.as_dict()
            return test_results['metrics']
        t1 = time.time()
        logger.info('time taken to generate target-drift is {:.2f} secs'.format(t1 - t0))

    @staticmethod
    def act_drift_results(test_results: dict = None, drift_thresh: float = 0.5,
                          important_drift_columns: list = None) -> bool:
        """"""
        drift_actions = DriftActions(test_results=test_results, important_drift_columns=important_drift_columns)
        continue_process = drift_actions.run_datadrift_action(drift_thresh=drift_thresh)
        return continue_process

    @staticmethod
    def act_target_drift_results(test_results: dict = None, drift_thresh: float = 0.5,
                                 important_drift_columns: list = None) -> pd.DataFrame:
        """"""
        drift_actions = DriftActions(test_results=test_results, important_drift_columns=important_drift_columns)
        continue_process = drift_actions.run_conceptdrift_action(drift_thresh=drift_thresh)
        return continue_process


class PerformanceDrift(object):
    def __init__(self, prediction_latest: pd.DataFrame = None, prediction_earlier: pd.DataFrame = None,
                 categorical_columns: list = None, numerical_columns: list = None,
                 datetime_columns: list = None, target_label: str = None, prediction_label: str = None,
                 task: str = None, seed: int = None):
        assert isinstance(prediction_latest, pd.DataFrame), 'Latest prediction is not in Pandas DataFrame'
        assert isinstance(prediction_earlier, pd.DataFrame), 'Earlier prediction data is not in Pandas DataFrame'
        self.pred_latest = prediction_latest
        self.pred_earlier = prediction_earlier
        self.cat_columns = categorical_columns
        self.num_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.target = target_label
        self.seed = seed
        self.task = task
        self.col_mapping = create_colmapping(target=self.target, prediction=prediction_label,
                                             datetime_columns=self.datetime_columns, task=self.task,
                                             num_columns=self.num_columns, cat_colummns=self.cat_columns)

    def run_drift_checks(self, top_k: int = 5, multi_label: bool = False,
                         save_html: bool = False, save_dir: str = None, return_dict: bool = False,
                         filename: str = 'performance_drift') -> Union[NoReturn, Dict]:
        """"""
        # dataset-level tests
        t0 = time.time()
        if not self.task == 'regression':
            import pdb; pdb.set_trace()
            if not multi_label:
                logger.info('Running Binary classification result drift reports between current and past prediction')
                datadrift_tests = TestSuite(tests=[
                    BinaryClassificationTestPreset(prediction_type='labels', stattest='psi'),
                ])
            else:
                logger.info('Running Multi classification result drift reports between current and past prediction')
                datadrift_tests = TestSuite(tests=[
                    MulticlassClassificationTestPreset(stattest='psi'),
                ])
        else:
            logger.info('Running Regression result drift reports between current and past prediction')
            datadrift_tests = TestSuite(tests=[
                RegressionTestPreset(),
            ])
        datadrift_tests.run(current_data=self.pred_latest, reference_data=self.pred_earlier, column_mapping=self.col_mapping)
        if not save_html:
            logger.info('Not result drift in html/json format')
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            import pdb; pdb.set_trace()
            file_path = os.path.join(save_dir, '{}_report.html'.format(filename))
            datadrift_tests.save_html(file_path)
            file_path = os.path.join(save_dir, '{}_report.json'.format(filename))
            datadrift_tests.save_json(file_path)
            logger.info('Saved performance drift file in {}'.format(file_path))
        if return_dict:
            logger.info('Returning performance drift as dictionary to process dataframe')
            test_results = datadrift_tests.as_dict()
            return test_results['metrics'][0]
        t1 = time.time()
        logger.info('time taken to generate data-drift is {:.2f} secs'.format(t1 - t0))


def create_colmapping(target: str = None, prediction: str = None, datetime_columns: list = None,
                      num_columns: list = None, cat_colummns: list = None, task: str = None) -> ColumnMapping:
    """"""
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.datetime_features = datetime_columns
    column_mapping.numerical_features = num_columns
    column_mapping.categorical_features = cat_colummns
    column_mapping.task = task
    return column_mapping
