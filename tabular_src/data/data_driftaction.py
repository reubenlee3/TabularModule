from typing import Dict, List, NoReturn, Union
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


class DriftActions(object):
    """"""

    def __init__(self, data: pd.DataFrame = None, test_results: Dict = None,
                 important_drift_columns: list = None):
        """"""
        self.data = data
        self.test_results = test_results
        self.drift_columns_check = important_drift_columns

    def run_integrity_action(self) -> pd.DataFrame:
        """"""
        data = self.data.copy()
        logger.info('Modifying dataframe based on the drift report')
        data_integrity_action = DataIntegrityAction(data=data, test_results=self.test_results)
        missing_columns = data_integrity_action.get_missingshare_columns(thresh=0.3)
        constant_columns = data_integrity_action.get_constant_columns()
        data_integrity_action.get_duplicate_colums()
        data_integrity_action.get_duplicate_rows()
        data_integrity_action.get_correlation_column()
        columns_dropped = list(set(missing_columns + constant_columns))
        if columns_dropped is not None and len(columns_dropped) > 0:
            logger.info('Issue detected in columns: {}'.format(columns_dropped))
            data = self.data.drop(columns=columns_dropped)
        else:
            logger.info('No issues with the columns in the training dataframe')
        return data

    def run_datadrift_action(self, drift_thresh: float = 0.5) -> bool:
        """"""
        logger.info('Analysing drifted columns')
        datadrift_action = DataDriftAction(test_results=self.test_results)
        drifted_columns, drift_perc = datadrift_action.drifted_features()
        logger.info('Drifted columns percentage is: {:.2f}%'.format(drift_perc * 100))
        logger.info('Drifted columns are: {}'.format(drifted_columns))
        if not drift_perc >= drift_thresh:
            logger.info(
                'Drifted column percentage({:.2f}%) is less than threshold({:.2f}%)'.format((drift_perc * 100),
                                                                                            (drift_thresh * 100)))
            continue_process = True
            logger.info('Continuing the model training')
        else:
            logger.info(
                'Drifted column percentage({:.2f}%) is more than than threshold({:.2f}%)'.format((drift_perc * 100),
                                                                                                 (drift_thresh * 100)))
            continue_process = False
            logger.info('Stopping the model training')
        return continue_process

    def run_conceptdrift_action(self, drift_thresh: float = 0.5, corr_method: str = 'pearson',
                                corr_thresh: float = 0.25) -> bool:
        """"""
        logger.info('Analysing concept with prediction/target and feature columns')
        conceptdrift_action = ConceptDriftAction(test_results=self.test_results)
        drifted_columns, drift_perc, drift_yesno = conceptdrift_action.concept_drifted_features(method=corr_method,
                                                                                                corr_thresh=corr_thresh)
        logger.info('Drifted columns percentage is: {:.2f}%'.format(drift_perc * 100))
        logger.info('Drifted columns are: {}'.format(drifted_columns))
        if not drift_perc >= drift_thresh:
            logger.info(
                'Drifted column percentage({:.2f}%) is less than threshold({:.2f}%)'.format((drift_perc * 100),
                                                                                            (drift_thresh * 100)))
            continue_process = True
            logger.info('Continuing the model training')
        else:
            logger.info(
                'Drifted column percentage({:.2f}%) is more than than threshold({:.2f}%)'.format((drift_perc * 100),
                                                                                                 (drift_thresh * 100)))
            continue_process = False
            logger.info('Stopping the model training')
        return continue_process


class DataIntegrityAction(object):
    """"""

    def __init__(self, data: pd.DataFrame = None, test_results: Dict = None):
        self.data = data
        self.test_results = test_results

    def get_missingshare_columns(self, function_name: str = 'ColumnSummaryMetric', thresh: float = 0.3) -> List:
        """"""
        missing_cols = []
        for result_dict in self.test_results:
            if result_dict['metric'] == function_name and \
                    result_dict['result']['current_characteristics']['missing_percentage'] >= thresh:
                missing_cols.append(str(result_dict['result']['column_name']))
        if len(missing_cols) > 0:
            logger.info('Columns with missing value more than threshold is {}'.format(missing_cols))
        return missing_cols

    def get_constant_columns(self) -> List:
        """"""
        constant_cols = self.data.columns[self.data.nunique() <= 1].to_list()
        if len(constant_cols) > 0:
            logger.info('Columns with constant value is {}'.format(constant_cols))
        return constant_cols

    def get_duplicate_colums(self, function_name: str = 'Number of Duplicate Columns') -> NoReturn:
        """"""
        if self.test_results[0]['result']['current']['number_of_duplicated_columns'] > 0:
            logger.info('There are duplicate columns in the data. Dropping it')
            # TODO: To remove duplicate columns in pandas
        pass

    def get_duplicate_rows(self, function_name: str = 'Number of Duplicate Rows') -> NoReturn:
        """"""
        if self.test_results[0]['result']['current']['number_of_duplicated_rows'] > 0:
            logger.info('There are duplicate rows in the data. Dropping it')
            self.data = self.data.drop_duplicates(keep='first', inplace=False)

    def get_correlation_column(self, function_name: str = 'Highly Correlated Columns') -> NoReturn:
        """"""
        # TODO: To remove highly correlated numeric feature
        pass


class DataDriftAction(object):
    """"""

    def __init__(self, test_results: Dict = None):
        self.test_results = test_results

    def drifted_features(self) -> Union[List, float]:
        """"""
        drifted_cols = []
        drift_perc = self.test_results['result']['share_of_drifted_columns']
        for key, value in self.test_results['result']['drift_by_columns'].items():
            if value['drift_detected']:
                drifted_cols.append(value['column_name'])
        return drifted_cols, drift_perc


class ConceptDriftAction(object):
    """"""

    def __init__(self, test_results: Dict = None):
        self.test_results = test_results

    def concept_drifted_features(self, method: str = 'pearson',
                                 corr_thresh: float = 0.25) -> Union[List, float, bool]:
        """"""
        target_shifted = self.test_results[0]['result']['drift_detected']
        if target_shifted:
            logger.info('Prediction or Target drifted between train and test/prediction')
        current_dict = self.test_results[2]['result']['current'][method]['values']
        current_dict['y1'] = self.test_results[2]['result']['reference'][method]['values']['y']
        corr_data = pd.DataFrame.from_dict(current_dict)
        corr_data['corr_difference'] = abs(corr_data['y1'] - corr_data['y'])
        drifted_cols = corr_data.loc[corr_data['corr_difference'] >= corr_thresh]['x'].tolist()
        drift_perc = len(drifted_cols)/corr_data.shape[0]
        return drifted_cols, drift_perc, target_shifted
