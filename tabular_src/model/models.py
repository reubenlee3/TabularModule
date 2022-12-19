from typing import Union
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .evaluation import EvaluateClassification
from ..utils import get_logger

logger = get_logger(__name__)


class TabularModels(object):
    """"""
    def __init__(self, train: pd.DataFrame = None, test: pd.DataFrame = None,
                 task: str = None, target: str = None, is_multilabel: bool = False):
        self.is_train = False
        self.is_val = False
        self.is_test = False
        self.task = task
        self.is_multilabel = is_multilabel
        self._check_data(train, test, target)

    def _check_data(self, train: pd.DataFrame = None, test: pd.DataFrame = None, target: str = None):
        """"""
        dataset = [train, test]
        for data in dataset:
            if data is not None:
                assert isinstance(data, (pd.DataFrame, np.ndarray))

        if train is not None:
            self.is_train = True
        else:
            if test is not None:
                self.is_test = True
            else:
                logger.error('train_target needs to be specified for train')
                raise ValueError()
        if self.task == 'classification':
            if not self.is_test:
                if self.is_multilabel:
                    assert train[target].nunique() > 2, \
                        'Unique value in the target column is less than 3, cannot use multi-label classification'
                else:
                    assert train[target].nunique() == 2, \
                        'Unique value in the target column is not 2, Cannot use binary classification'


class PyCaretModel(TabularModels):
    """"""
    def __init__(self, train: pd.DataFrame = None, target: str = None, task: str = None, test: pd.DataFrame = None,
                 estimator_list: list = None, params: dict = None, n_jobs: int = -1, use_gpu: bool = False,
                 is_multilabel=False, categorical_cols: Union[list, str] = None,
                 numerical_cols: Union[list, str] = None, keep_features: Union[list, str] = None,
                 text_cols: Union[list, str] = None, seed: int = None, verbose: bool = True):
        super().__init__(train=train, test=test, task=task, target=target, is_multilabel=is_multilabel)
        self.train = train
        self.test = test
        self.target = target
        self.estimator = estimator_list
        self.params = params
        self.n_jobs = n_jobs
        self.model = None
        self.tuned_model = None
        self.calibrated_model = None
        self.model_pipeline = None
        self.gpu = use_gpu
        # TODO: Hard-coded gpu to be false
        if self.gpu:
            logger.info('Checking for GPU installation')
            gpu = False
            if not gpu:
                logger.info('GPU not found')
            else:
                self.gpu = gpu
        self.num_cols = numerical_cols
        self.cat_cols = categorical_cols
        self.text_cols = text_cols
        self.keep_features = keep_features
        self.seed = seed
        self.verbose = verbose
        self.model = None

    def fit(self, apply_pca: bool = False, remove_outliers: bool = False, fold_strategy: str = 'stratifiedkfold',
            cv_fold_size: int = 4, calibrate: bool = False, probability_threshold: float = 0.5,
            optimize: bool = False, custom_grid: dict = None, n_iter: int = 20, search_library: str = None,
            search_algorithm: str = None, search_metric: str = None, early_stopping: bool = False,
            early_stopping_max_iters: int = None, ensemble_model: bool = False, ensemble_type: str = None):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import setup, set_config, create_model, tune_model, models, \
                blend_models, stack_models, calibrate_model, finalize_model, compare_models
            # Set up classifier
            classifier = setup(data=self.train, target=self.target, test_data=self.test, feature_selection=False,
                               remove_multicollinearity=True, multicollinearity_threshold=0.6,
                               pca=apply_pca, remove_outliers=remove_outliers, fold_strategy=fold_strategy,
                               fold=cv_fold_size, keep_features=self.keep_features, numeric_features=self.num_cols,
                               categorical_features=self.cat_cols, text_features=self.text_cols,
                               max_encoding_ohe=40, encoding_method=None, verbose=self.verbose,
                               n_jobs=self.n_jobs, use_gpu=self.gpu)
            set_config('seed', self.seed)
            # create model
            if len(self.estimator) == 1:
                logger.info('Only one {} estimator is passed for modelling'.format(self.estimator[0]))
                self.model = create_model(estimator=self.estimator[0], fold=cv_fold_size, round=2,
                                          cross_validation=True, probability_threshold=probability_threshold,
                                          verbose=self.verbose)
                if not optimize:
                    self.tuned_model = self.model
                else:
                    self.tuned_model, tuner = self._optimize_model(model=self.model, cv_fold_size=cv_fold_size,
                                                                   n_iter=n_iter, custom_grid=custom_grid,
                                                                   search_metric=search_metric,
                                                                   search_library=search_library,
                                                                   search_algorithm=search_algorithm,
                                                                   early_stopping=early_stopping,
                                                                   early_stopping_max_iters=early_stopping_max_iters,
                                                                   )
            else:
                logger.info('There are {} algorithms passed for modelling'.format(len(self.estimator)))
                # validate estimator list
                available_models = models().index.tolist()
                self.estimator = list(set(available_models).intersection(self.estimator))
                if not ensemble_model:
                    logger.info('Ensemble model is not selected , so selecting the best model')
                    self.model = compare_models(include=self.estimator, fold=cv_fold_size, round=2,
                                                cross_validation=True, sort=search_metric, n_select=1,
                                                errors='ignore', probability_threshold=probability_threshold,
                                                verbose=self.verbose
                                                )
                    if not optimize:
                        self.tuned_model = self.model
                    else:
                        self.tuned_model, tuner = self._optimize_model(model=self.model, cv_fold_size=cv_fold_size,
                                                                       n_iter=n_iter, custom_grid=custom_grid,
                                                                       search_metric=search_metric,
                                                                       search_library=search_library,
                                                                       search_algorithm=search_algorithm,
                                                                       early_stopping=early_stopping,
                                                                       early_stopping_max_iters=early_stopping_max_iters,
                                                                       )

                    logger.info('The best model is {}'.format(type(self.model)))
                else:
                    logger.info('Selected ensemble model and ensemble type is {}'.format(ensemble_type))
                    logger.info('Model ensemble is selected')
                    try:
                        model_list = []
                        for estimator in self.estimator:
                            model = create_model(estimator=estimator, fold=cv_fold_size, round=2,
                                                 cross_validation=True, probability_threshold=probability_threshold,
                                                 verbose=self.verbose
                                                 )
                            if not optimize:
                                tuned_model = model
                            else:
                                tuned_model, tuner = self._optimize_model(model=model, cv_fold_size=cv_fold_size,
                                                                          n_iter=n_iter, custom_grid=custom_grid,
                                                                          search_metric=search_metric,
                                                                          search_library=search_library,
                                                                          search_algorithm=search_algorithm,
                                                                          early_stopping=early_stopping,
                                                                          early_stopping_max_iters=early_stopping_max_iters,
                                                                          )
                            model_list.append(tuned_model)

                        if ensemble_type == 'stack':
                            logger.info('Model stacking is selected')
                            self.tuned_model = stack_models(estimator_list=model_list, meta_model=None,
                                                            fold=cv_fold_size, round=2, optimize=search_metric,
                                                            method='auto', restack=False, verbose=self.verbose,
                                                            probability_threshold=probability_threshold,
                                                            choose_better=False)
                        elif ensemble_type == 'blend':
                            logger.info('Blending ML model')
                            self.tuned_model = blend_models(estimator_list=model_list, fold=cv_fold_size,
                                                            round=2, optimize=search_metric, method='auto',
                                                            verbose=self.verbose,
                                                            probability_threshold=probability_threshold,
                                                            choose_better=False)
                    except AssertionError as error:
                        logger.info('The issue in stacking the model is {}'.format(error))

            # calibrate model
            if not calibrate:
                self.calibrated_model = self.tuned_model
            else:
                logger.info('Model calibration is selected')
                self.calibrated_model = calibrate_model(self.tuned_model, calibrate_fold=cv_fold_size,
                                                        fold=cv_fold_size, round=2, method='sigmoid',
                                                        verbose=self.verbose)
            # finalise model and train on holdout set
            self.model_pipeline = finalize_model(self.calibrated_model, model_only=False)
        else:
            from pycaret.regression import setup, set_config
            regressor = setup(data=self.train, target=self.target, test_data=self.test, feature_selection=False,
                              remove_multicollinearity=True, multicollinearity_threshold=0.6,
                              pca=apply_pca, remove_outliers=remove_outliers, fold_strategy=fold_strategy,
                              fold=cv_fold_size, keep_features=self.keep_features, numeric_features=self.num_cols,
                              categorical_features=self.cat_cols, text_features=self.text_cols,
                              max_encoding_ohe=2, encoding_method=None, verbose=self.verbose,
                              n_jobs=self.n_jobs,
                              use_gpu=self.gpu)
            set_config('seed', self.seed)
        logger.info('Training completed for PyCaret model')
        # TODO: Implement for regression

    def _optimize_model(self, model, cv_fold_size: int = 4, n_iter: int = 20, custom_grid: dict = None,
                        search_metric: str = None, early_stopping: bool = False, search_library: str = None,
                        search_algorithm: str = None, early_stopping_max_iters: int = None, ):
        """"""
        # TODO: Optimise grid based on the estimator
        # tune model
        if not self.task == 'regression':
            from pycaret.classification import tune_model
            logger.info('Model optimization is selected using method: {} and metric:'.format(search_algorithm,
                                                                                             search_metric))
            # from pycaret.distributions import UniformDistribution, IntUniformDistribution
            # custom_grid = {
            #     "max_depth": IntUniformDistribution(4, 6, 1),
            #     "learning_rate": UniformDistribution(0.05, 0.2, 1),
            # }
            model, tuner = tune_model(estimator=model, fold=cv_fold_size, n_iter=n_iter, round=2,
                                      custom_grid=custom_grid, optimize=search_metric,
                                      search_library=search_library, search_algorithm=search_algorithm,
                                      early_stopping=early_stopping, verbose=self.verbose,
                                      early_stopping_max_iters=early_stopping_max_iters,
                                      choose_better=True, return_tuner=True)
        else:
            # TODO: Modify for regression
            from pycaret.regression import tune_model
            logger.info('Model optimization is selected using method: {} and metric:'.format(search_algorithm,
                                                                                             search_metric))
            # from pycaret.distributions import UniformDistribution, IntUniformDistribution
            # custom_grid = {
            #     "max_depth": IntUniformDistribution(4, 6, 1),
            #     "learning_rate": UniformDistribution(0.05, 0.2, 1),
            # }
            model, tuner = tune_model(estimator=model, fold=cv_fold_size, n_iter=n_iter, round=2,
                                      custom_grid=custom_grid, optimize=search_metric,
                                      search_library=search_library, search_algorithm=search_algorithm,
                                      early_stopping=early_stopping, verbose=self.verbose,
                                      early_stopping_max_iters=early_stopping_max_iters,
                                      choose_better=True, return_tuner=True)
        return model, tuner

    def model_evaluation(self, path: str = None, plot: bool = False):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import predict_model
            predict_result = predict_model(estimator=self.calibrated_model, data=self.test.drop(columns=[self.target]),
                                           probability_threshold=0.5, raw_score=True, round=2,
                                           verbose=self.verbose)['prediction_score_1']
            evaluate = EvaluateClassification(estimator=self.calibrated_model, labels=self.test[self.target],
                                              pred_proba=predict_result, prob_threshold=0.5,
                                              multi_label=self.is_multilabel)
            file_path = os.path.join(path, 'evaluation_report.json')
            plot_path = os.path.join(path, 'evaluation_plots')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            evaluate.save(filepath=file_path, plot=plot, plot_path=plot_path)
        else:
            # TODO: Implement for regression
            from pycaret.regression import predict_model
            predict_result = predict_model(estimator=self.calibrated_model, data=self.test.drop(columns=[self.target]),
                                           round=2, verbose=self.verbose)['prediction_label']
            evaluate = EvaluateClassification(labels=self.test[self.target], pred_proba=predict_result,
                                              multi_label=self.is_multilabel)
            file_path = os.path.join(path, 'classification')
            evaluate.save(filepath=file_path, plot=True)

    def feature_explanation(self, path: str = None):
        """"""
        # TODO: Skip feature explanation for ensemble methods
        if not self.task == 'regression':
            from pycaret.classification import interpret_model
            # TODO: Sample Xtrain and Ytrain for larger dataset
            # X_train = get_config('X_Train')
            # Y_train = get_config('Y_Train')

            # default summary plot
            # file_path = os.path.join(path, 'shap_summary_plot.png')
            interpret_model(estimator=self.model, plot='summary', use_train_data=True, save=path)
            # PDP plots for top features list
            # TODO: implement top features and loop it for pdp
            # file_path = os.path.join(path, 'pdp_feature_1.png')
            interpret_model(estimator=self.model, plot='pfi', use_train_data=True, save=path)
            logger.info('feature importance artifacts is saved in the location : {}'.format(path))
        else:
            from pycaret.regression import interpret_model
            # TODO: Sample Xtrain and Ytrain for larger dataset
            # X_train = get_config('X_Train')
            # Y_train = get_config('Y_Train')
            # default summary plot
            file_path = os.path.join(path, 'shap_summary_plot.png')
            interpret_model(estimator=self.model, plot='summary', use_train_data=True, save=file_path)
            # PDP plots for top features list
            # TODO: implement top features and loop it 
            file_path = os.path.join(path, 'pdp_feature_1.png')
            interpret_model(estimator=self.model, plot='pdp', use_train_data=True, save=file_path)
            logger.info('feature importance artifacts is saved in the location : {}'.format(path))

    def check_fairness(self):
        """"""
        pass

    def save(self, path: str = None, model_stats: bool = False,
             training_data: bool = False, extract_tree: bool = False):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import save_model, pull
            file_path = os.path.join(path, 'classification')
            save_model(self.model_pipeline, model_name=file_path)
            if model_stats:
                logger.info('saving model performance')
                model_performance = pull(pop=False)
                file_path = os.path.join(path, 'model_performance.csv')
                model_performance.to_csv(file_path, index=False)
            if training_data:
                logger.info('Saving training data in parquet format')
                table = pa.Table.from_pandas(df=self.train, preserve_index=True)
                file_path = os.path.join(path, 'training_data.parquet')
                pq.write_table(table, file_path)
            if extract_tree:
                logger.info('Extracting tree from pycaret model')
                # TODO: Extract decision tree from the pycaret using API
            logger.info('model is saved in the location : {}'.format(path))
        else:
            from pycaret.classification import save_model, pull
            file_path = os.path.join(path, 'regression')
            save_model(self.model_pipeline, model_name=file_path)
            if model_stats:
                logger.info('saving model performance')
                model_performance = pull(pop=False)
                file_path = os.path.join(path, 'model_performance.csv')
                model_performance.to_csv(file_path, index=False)
            if training_data:
                logger.info('Saving training data in parquet format')
                table = pa.Table.from_pandas(df=self.train, preserve_index=True)
                file_path = os.path.join(path, 'training_data.parquet')
                pq.write_table(table, file_path)
            logger.info('model is saved in the location : {}'.format(path))

    def load(self, path: str = None):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import load_model
            file_path = os.path.join(path, 'classification')
            self.model_pipeline = load_model(model_name=file_path, verbose=False)
            logger.info('Trained pycaret model is loaded from: {}'.format(path))
        else:
            from pycaret.regression import load_model
            file_path = os.path.join(path, 'regression')
            self.model_pipeline = load_model(model_name=file_path, verbose=False)
            logger.info('Trained pycaret model is loaded from: {}'.format(path))

    def predict(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """"""
        assert isinstance(data, (pd.DataFrame, np.ndarray))
        if not self.task == 'regression':
            from pycaret.classification import predict_model
            predict_result = predict_model(estimator=self.model_pipeline, data=data, raw_score=True, round=2,
                                           verbose=self.verbose)
            logger.info('prediction done for pycaret classification model')
        else:
            from pycaret.regression import predict_model
            predict_result = predict_model(estimator=self.model_pipeline, data=data, round=2, verbose=self.verbose)
            logger.info('prediction done for pycaret regression model')
        return predict_result['prediction_score_1']
