from typing import Dict, Union, Tuple
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pycaret.distributions import UniformDistribution, IntUniformDistribution, CategoricalDistribution
import copy
from .log import get_logger

logger = get_logger(__name__)


def pick_custom_grid(estimator: str = None, custom_grid_dict: dict = None) -> Union[Dict, None]:
    """"""
    # TODO 1: Implement params for more models
    if estimator == 'catboost':
        params_grid = dict(custom_grid_dict['model']['catboost'])
        logger.info('Selecting CatBoost Hyper-Params for tuning')
    elif estimator == 'lightgbm':
        params_grid = dict(custom_grid_dict['model']['lightgbm'])
        logger.info('Selecting LightGBM Hyper-Params for tuning')
    elif estimator == 'xgboost':
        params_grid = dict(custom_grid_dict['model']['xgboost'])
        logger.info('Selecting XGBoost Hyper-Params for tuning')
    else:
        logger.error('The params grid is not available for estimator {} is not available, '
                     'Returning None'.format(estimator))
        params_grid = None
    return clean_dict(custom_dict=params_grid)


def clean_dict(custom_dict: dict = None) -> Union[Dict, None]:
    """"""
    if custom_dict is not None:
        modified_dict = copy.deepcopy(custom_dict)
        for key, value in modified_dict.items():
            if isinstance(value, str):
                modified_dict[key] = eval(modified_dict[key])
        return modified_dict
    else:
        return None


def monotonic_feature_list(columns: list = None, monotonic_inc_list: list = None,
                           monotonic_dec_list: list = None) -> Union[Tuple, None]:
    """"""
    if monotonic_inc_list is not None or monotonic_dec_list is not None:
        monotonic_dict = {}
        if monotonic_inc_list is not None:
            monotonic_inc_list = list(set(columns).intersection(monotonic_inc_list))
            for key in monotonic_inc_list:
                monotonic_dict[key] = 1
        if monotonic_dec_list is not None:
            monotonic_dec_list = list(set(columns).intersection(monotonic_dec_list))
            for key in monotonic_dec_list:
                monotonic_dict[key] = -1
        return monotonic_dict
    else:
        logger.info('There are no monotonic features in the data')
        return None


def latest_file(folder_path: str = None, file_format: str = '.csv') -> str:
    """"""
    files = os.listdir(folder_path)
    paths = [os.path.join(folder_path, basename) for basename in files if basename.endswith(file_format)]
    return max(paths, key=os.path.getmtime)


def save_parquet(df: pd.DataFrame = None, path: str = None, file_name: str = 'data.parquet'):
    """"""
    logger.info('Saving data in parquet format')
    half_floats = df.select_dtypes(include="float16")
    df[half_floats.columns] = half_floats.astype("float32")
    table = pa.Table.from_pandas(df=df, preserve_index=True)
    file_path = os.path.join(path, file_name)
    pq.write_table(table, file_path)
    logger.info('Saved data in parquet format in {}'.format(file_path))
