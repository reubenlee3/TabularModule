from typing import Union
import pandas as pd
from category_encoders.count import CountEncoder
from ..utils import get_logger

logger = get_logger(__name__)


class FeatureSelection(object):
    def __init__(self, train_df: pd.DataFrame = None, rfe_estimator: str = None, task: str = None,
                 target: str = None, n_features: Union[int, float] = None, keep_features: Union[list, str] = None,
                 num_features: Union[list, str] = None, text_features: Union[list, str] = None,
                 cat_features: Union[list, str] = None, multi_colinear_threshold: float = None,
                 seed: int = None):
        self.train_df = train_df
        self.estimator = rfe_estimator
        self.target = target
        self.nfeatures = n_features
        self.task = task
        self.multi_colinear_ratio = multi_colinear_threshold
        self.keep_features = keep_features
        self.num_features = num_features
        self.text_features = text_features
        self.cat_features = cat_features
        self.seed = seed

    def _encode_columns(self):
        """"""
        count_enc = CountEncoder()
        encoded_column = count_enc.fit_transform(self.train_df[self.cat_features])
        num_list = self.num_features.copy()
        num_list.append(self.target)
        train_df = self.train_df[num_list].join(encoded_column, how='left')
        self.train_df = None
        self.train_df = train_df

    def run_feature_selection(self) -> list:
        """"""
        self._encode_columns()
        if not self.task == 'classification':
            logger.info('Selecting regression task to do auto-rfe')
            from pycaret.regression import setup, set_config, get_config
            classifier = setup(data=self.train_df, target=self.target, n_features_to_select=self.nfeatures,
                               feature_selection=True, remove_multicollinearity=True,
                               feature_selection_method='classic', keep_features=None,
                               numeric_features=list(self.train_df.columns).remove(self.target),
                               categorical_features=None, text_features=None, max_encoding_ohe=2,
                               encoding_method=None, multicollinearity_threshold=self.multi_colinear_ratio)
            set_config('seed', self.seed)
            feature_columns = list(get_config('X_train').columns)
            logger.info('Total columns after running feature selection is {}'.format(len(feature_columns)))
        else:
            logger.info('Selecting classification task to do auto-rfe')
            from pycaret.classification import setup, set_config, get_config
            classifier = setup(data=self.train_df, target=self.target, n_features_to_select=self.nfeatures,
                               feature_selection=True, remove_multicollinearity=True,
                               feature_selection_method='classic', keep_features=None,
                               numeric_features=list(self.train_df.columns).remove(self.target),
                               categorical_features=None, text_features=None, max_encoding_ohe=2,
                               encoding_method=None, multicollinearity_threshold=self.multi_colinear_ratio)
            set_config('seed', self.seed)
            feature_columns = list(get_config('X_train').columns)
            logger.info('Total columns after running feature selection is {}'.format(len(feature_columns)))
        feature_columns.append(self.target)
        if self.text_features is not None:
            top_columns = feature_columns + self.text_features
        else:
            top_columns = feature_columns
        return list(set(top_columns))
