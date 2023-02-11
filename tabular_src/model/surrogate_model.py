import pandas as pd
import joblib
import os

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..utils import get_logger

logger = get_logger(__name__)


class SurrogateModel(object):
    """"""

    def __init__(self, task: str = None, is_multilabel: bool = False, categorical_columns: list = None,
                 train: pd.DataFrame = None, target: str = None, numerical_columns: list = None,
                 test: pd.DataFrame = None, estimator: str = None, seed: int = None):
        """"""
        self.model_pipeline = None
        self.estimator = estimator
        self.cat_columns = categorical_columns
        self.num_columns = numerical_columns
        self.seed = seed
        self.task = task
        self.test = test
        self.train = train
        self.target = target
        self.is_multilabel = is_multilabel

    def __pipeline(self):
        """"""
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
        )
        # categorical_pipe = SimpleImputer(strategy='most_frequent')
        numerical_pipe = SimpleImputer(strategy="mean")

        preprocessing = ColumnTransformer(
            [
                ("cat", categorical_encoder, self.cat_columns),
                # ('cat_impute', categorical_pipe, self.cat_columns),
                ("num_impute", numerical_pipe, self.num_columns),
            ],
            verbose_feature_names_out=False,
        )
        return preprocessing

    def fit(self):
        """"""
        preprocessing = self.__pipeline()
        if not self.task == 'regression':
            # setup for classification
            if self.estimator == 'random_forest':
                logger.info('Selecting random forest as surrogate model')
                self.model_pipeline = Pipeline(
                    [
                        ("preprocess", preprocessing),
                        ("classifier", RandomForestClassifier(random_state=self.seed, n_estimators=125, n_jobs=-1,
                                                              max_depth=4, criterion='entropy')),
                    ]
                )
                self.model_pipeline.fit(X=self.train, y=self.target)
            elif self.estimator == 'decision_tree':
                logger.info('Selecting Decision tree as surrogate model')
                self.model_pipeline = Pipeline(
                    [
                        ("preprocess", preprocessing),
                        ("classifier", DecisionTreeClassifier(random_state=self.seed, max_depth=4, criterion='gini')),
                    ]
                )
                self.model_pipeline.fit(X=self.train, y=self.target)
            else:
                logger.info('estimator {} is not implemented'.format(self.estimator))
                raise ValueError('estimator {} is not implemented'.format(self.estimator))
        else:
            # TODO 1: Set up for regression
            logger.info('setup for regression')
            if self.estimator == 'random_forest':
                logger.info('Selecting random forest as surrogate model')
                self.model_pipeline = Pipeline(
                    [
                        ("preprocess", preprocessing),
                        ("classifier", RandomForestRegressor(random_state=self.seed, n_estimators=125, n_jobs=-1,
                                                             max_depth=4, criterion='friedman_mse')),
                    ]
                )
                self.model_pipeline.fit(X=self.train, y=self.target)
            elif self.estimator == 'decision_tree':
                logger.info('Selecting Decision tree as surrogate model')
                self.model_pipeline = Pipeline(
                    [
                        ("preprocess", preprocessing),
                        ("classifier", DecisionTreeRegressor(random_state=self.seed, max_depth=4, criterion='friedman_mse')),
                    ]
                )
                self.model_pipeline.fit(X=self.train, y=self.target)
            else:
                logger.info('estimator {} is not implemented'.format(self.estimator))
                raise ValueError('estimator {} is not implemented'.format(self.estimator))

    def save(self, save_path: str = None, only_model: bool = False):
        """"""
        model_name = os.path.join(save_path, 'surrogate_model')
        if not only_model:
            logger.info('Saving model with pipeline for scoring')
            model_name = f"{model_name}.pkl"
            joblib.dump(self.model_pipeline, model_name)
        else:
            logger.info('Saving only model')
            model_name = f"{model_name}.pkl"
            model = self.model_pipeline['classifier']
            joblib.dump(model, model_name)

    def load(self, load_path: str = None):
        """"""
        model_name = os.path.join(load_path, 'surrogate_model.pkl')
        logger.info('Loading model from: {}'.format(model_name))
        self.model_pipeline = joblib.load(model_name)

    def predict(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """"""
        if data is not None:
            logger.info('prediction using the given data')
            result_array = self.model_pipeline.predict(data)
        elif self.test is not None:
            logger.info('using test dataframe to predict')
            try:
                test = self.test.drop(columns=self.target, inplace=False)
            except:
                test = self.test
            result_array = self.model_pipeline.predict(test)
        else:
            logger.info('No dataframe to predict')
            result_array = None
        return result_array
