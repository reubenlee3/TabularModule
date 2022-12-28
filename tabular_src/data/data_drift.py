import pandas as pd
import time
import os
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset

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
        self.col_mapping = create_colmapping(target=target_label, prediction=None, datetime_columns=datetime_columns,
                                             num_columns=numerical_columns, cat_colummns=categorical_columns)

    def run_integrity_checks(self, save_html: bool = False, save_dir: str = None):
        """"""
        # dataset-level tests
        t0 = time.time()
        data_integrity_dataset_tests = TestSuite(tests=[
            TestAllColumnsShareOfMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedColumns(),
            TestNumberOfDuplicatedRows(),
            TestHighlyCorrelatedColumns(),
            # TODO: Check Target feature correlation in the report
            TestTargetFeaturesCorrelations(),
        ])
        data_integrity_dataset_tests.run(current_data=self.df, reference_data=None,
                                         column_mapping=self.col_mapping)
        if not save_html:
            logger.info('Not saving data integrity in html/json format')
        else:
            file_path = os.path.join(save_dir, 'data_integrity_report.html')
            data_integrity_dataset_tests.save_html(file_path)
            file_path = os.path.join(save_dir, 'data_integrity_report.json')
            data_integrity_dataset_tests.save_json(file_path)
            logger.info('Saved Data drift file in {}'.format(file_path))
        # TODO: Modify the dataframe
        # test_results = data_integrity_dataset_tests.as_dict()
        # self.df = self.act_testresults(test_results=test_results)
        t1 = time.time()
        logger.info('time taken to generate data-integrity report is {:.2f} secs'.format(t1 - t0))

    def act_testresults(self, test_results: dict = None) -> pd.DataFrame:
        """"""
        # TODO WORK ON CLEANING TRAINING DATA BASED ON THE TEST RESULTS


class TrainingDataDrift(object):
    def __init__(self, train_df: pd.DataFrame = None, test_df: pd.DataFrame = None,
                 categorical_columns: list = None, numerical_columns: list = None,
                 datetime_columns: list = None, target_label: str = None,
                 task: str = None, seed: int = None):
        assert isinstance(train_df, pd.DataFrame), 'Training data is not in Pandas DataFrame'
        assert isinstance(test_df, pd.DataFrame), 'Test data is not in Pandas DataFrame'
        self.train_df = train_df
        self.test_df = test_df
        self.cat_colummns = categorical_columns
        self.num_columns = numerical_columns
        self.datetime_columns = datetime_columns
        self.target = target_label
        self.seed = seed
        self.task = task
        self.col_mapping = create_colmapping(target=target_label, prediction=target_label,
                                             datetime_columns=datetime_columns, task=task,
                                             num_columns=numerical_columns, cat_colummns=categorical_columns)

    def run_drift_checks(self, save_html: bool = False, save_dir: str = None, filename: str = 'training_datadrift'):
        """"""
        # dataset-level tests
        t0 = time.time()
        datadrift_tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(),
        ])
        # datadrift_tests = Report(metrics=[DataDriftPreset(),
        #                                    ])
        import pdb; pdb.set_trace()
        datadrift_tests.run(current_data=self.test_df, reference_data=self.train_df, column_mapping=self.col_mapping)
        if not save_html:
            logger.info('Not saving data integrity in html/json format')
        else:
            file_path = os.path.join(save_dir, '{}_report.html'.format(filename))
            datadrift_tests.save_html(file_path)
            file_path = os.path.join(save_dir, '{}_report.json'.format(filename))
            datadrift_tests.save_json(file_path)
            logger.info('Saved Data drift file in {}'.format(file_path))
        # TODO: Modify the dataframe
        # test_results = data_integrity_dataset_tests.as_dict()
        # self.df = self.act_testresults(test_results=test_results)
        t1 = time.time()
        logger.info('time taken to generate data-drift is {:.2f} secs'.format(t1 - t0))
        # return self.df

    def run_target_drift_checks(self, save_html: bool = False, save_dir: str = None):
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
            file_path = os.path.join(save_dir, 'training_targetdrift_report.html')
            target_drift_tests.save_html(file_path)
            file_path = os.path.join(save_dir, 'training_targetdrift_report.json')
            target_drift_tests.save_json(file_path)
            logger.info('Saved Data drift file in {}'.format(file_path))
        # TODO: Modify the dataframe
        # test_results = data_integrity_dataset_tests.as_dict()
        # self.df = self.act_testresults(test_results=test_results)
        t1 = time.time()
        logger.info('time taken to generate target-drift is {:.2f} secs'.format(t1 - t0))

    # def data_drift_report(training_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
    #                       save_dir: str = None):
    #     """"""
    #     logger.info('Calculating Data drift between training and testing dataframes')
    #     t0 = time.time()
    #     drift_report = Report(metrics=[DataDriftPreset(),
    #                                    ])
    #     drift_report.run(current_data=test_data, reference_data=training_data)
    #     file_path = os.path.join(save_dir, 'data_drift_report.html')
    #     drift_report.save_html(file_path)
    #     t1 = time.time()
    #     logger.info('time taken to generate drift is {:.2f} secs'.format(t1 - t0))
    #     logger.info('Saved Data drift file in {}'.format(file_path))

    def act_drift_results(self, test_results: dict = None) -> pd.DataFrame:
        """"""
        # TODO WORK ON CLEANING TRAINING DATA BASED ON THE TEST RESULTS


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
