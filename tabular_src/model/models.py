from typing import Union
import numpy as np
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd

from .evaluation import EvaluateClassification, EvaluateRegression
from ..feature import ModelExplainer
from ..fairness import FairnessClassification
from ..utils import pick_custom_grid, monotonic_feature_list, save_parquet, get_logger

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
                 monotone_inc_cols: Union[list, str] = None, monotone_dec_cols: Union[list, str] = None,
                 text_cols: Union[list, str] = None, seed: int = None, verbose: bool = True):
        super().__init__(train=train, test=test, task=task, target=target, is_multilabel=is_multilabel)
        if target is not None:
            self.train = train.drop(columns=target, inplace=False).sort_index(axis=1, ascending=True, inplace=False)
            self.target_col = train[target]
        else:
            self.train = train
            self.target_col = target
        self.test = test
        self.target_label = target
        self.estimator = estimator_list
        self.params = params
        self.n_jobs = n_jobs
        self.model = None
        self.tuned_model = None
        self.calibrated_model = None
        self.model_pipeline = None
        self.test_result = None
        self.gpu = use_gpu
        # TODO 1: Hard-coded gpu to be false
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
        self.model_str = None
        self.ensemble_model = False
        if self.train is not None:
            self.monotone_constraints = monotonic_feature_list(columns=self.train.columns.tolist(),
                                                               monotonic_inc_list=monotone_inc_cols,
                                                               monotonic_dec_list=monotone_dec_cols)
        else:
            self.monotone_constraints = None

    def fit(self, apply_pca: bool = False, remove_outliers: bool = False, fold_strategy: str = 'stratifiedkfold',
            cv_fold_size: int = 4, calibrate: bool = False, probability_threshold: float = 0.5,
            optimize: bool = False, custom_grid: dict = None, n_iter: int = 20, search_library: str = None,
            search_algorithm: str = None, search_metric: str = None, early_stopping: bool = False,
            early_stopping_max_iters: int = None, ensemble_model: bool = False, ensemble_type: str = None):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import setup, set_config, create_model, tune_model, models, \
                blend_models, stack_models, calibrate_model, finalize_model, compare_models, pull
            # Set up classifier
            classifier = setup(data=self.train, target=self.target_col, test_data=self.test, feature_selection=False,
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
                self.model = self.create_custom_model(estimator=self.estimator[0], cv_fold_size=cv_fold_size,
                                                      cross_validation=True,
                                                      probability_threshold=probability_threshold, verbose=self.verbose)
                self.model_str = self.estimator[0]
                if not optimize:
                    self.tuned_model = self.model
                else:
                    self.tuned_model, tuner = self._optimize_model(model=self.model, estimator_str=self.model_str,
                                                                   cv_fold_size=cv_fold_size,
                                                                   n_iter=n_iter, custom_grid_path=custom_grid,
                                                                   search_metric=search_metric,
                                                                   search_library=search_library,
                                                                   search_algorithm=search_algorithm,
                                                                   early_stopping=early_stopping,
                                                                   early_stopping_max_iters=early_stopping_max_iters)
            else:
                logger.info('There are {} algorithms passed for modelling'.format(len(self.estimator)))
                # validate estimator list
                available_models = models().index.tolist()
                self.estimator = list(set(available_models).intersection(self.estimator))
                if not ensemble_model:
                    logger.info('Ensemble model is not selected , so selecting the best model')
                    compare_models(include=self.estimator, fold=cv_fold_size, round=2,
                                   cross_validation=True, sort=search_metric, n_select=1,
                                   errors='ignore', probability_threshold=probability_threshold,
                                   verbose=self.verbose)
                    self.model_str = pull().index[0]
                    logger.info('The best model based on {mtr} is {model}'.format(mtr=search_metric,
                                                                                  model=self.model_str))
                    self.model = self.create_custom_model(estimator=self.model_str, cv_fold_size=cv_fold_size,
                                                          cross_validation=True, verbose=self.verbose,
                                                          probability_threshold=probability_threshold)
                    if not optimize:
                        self.tuned_model = self.model
                    else:
                        self.tuned_model, tuner = self._optimize_model(model=self.model, estimator_str=self.model_str,
                                                                       cv_fold_size=cv_fold_size,
                                                                       n_iter=n_iter, custom_grid_path=custom_grid,
                                                                       search_metric=search_metric,
                                                                       search_library=search_library,
                                                                       search_algorithm=search_algorithm,
                                                                       early_stopping=early_stopping,
                                                                       early_stopping_max_iters=early_stopping_max_iters,
                                                                       )
                else:
                    self.ensemble_model = ensemble_model
                    logger.info('Selected ensemble model and ensemble type is {}'.format(ensemble_type))
                    logger.info('Model ensemble is selected')
                    try:
                        model_list = []
                        for estimator in tqdm(self.estimator):
                            model = self.create_custom_model(estimator=estimator, cv_fold_size=cv_fold_size,
                                                             cross_validation=True, verbose=self.verbose,
                                                             probability_threshold=probability_threshold)
                            if not optimize:
                                tuned_model = model
                            else:
                                tuned_model, tuner = self._optimize_model(estimator_str=estimator, model=model,
                                                                          cv_fold_size=cv_fold_size,
                                                                          n_iter=n_iter, custom_grid_path=custom_grid,
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
            from pycaret.regression import setup, set_config, create_model, tune_model, models, \
                blend_models, stack_models, finalize_model, compare_models, pull
            regressor = setup(data=self.train, target=self.target_col, test_data=self.test, feature_selection=False,
                              remove_multicollinearity=True, multicollinearity_threshold=0.6,
                              pca=apply_pca, remove_outliers=remove_outliers, fold_strategy=fold_strategy,
                              fold=cv_fold_size, keep_features=self.keep_features, numeric_features=self.num_cols,
                              categorical_features=self.cat_cols, text_features=self.text_cols,
                              max_encoding_ohe=2, encoding_method=None, verbose=self.verbose,
                              n_jobs=self.n_jobs, use_gpu=self.gpu)
            set_config('seed', self.seed)
            # create regression model
            if len(self.estimator) == 1:
                logger.info('Only one {} estimator is passed for modelling'.format(self.estimator[0]))
                self.model = self.create_custom_model(estimator=self.estimator[0], cv_fold_size=cv_fold_size,
                                                      cross_validation=True, verbose=self.verbose,
                                                      probability_threshold=None)
                self.model_str = self.estimator[0]
                if not optimize:
                    self.tuned_model = self.model
                else:
                    self.tuned_model, tuner = self._optimize_model(model=self.model, estimator_str=self.model_str, cv_fold_size=cv_fold_size,
                                                                   n_iter=n_iter, custom_grid_path=custom_grid,
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
                    compare_models(include=self.estimator, fold=cv_fold_size, round=2, cross_validation=True,
                                   sort=search_metric, n_select=1, errors='ignore', verbose=self.verbose)
                    self.model_str = pull().index[0]
                    logger.info('The best model based on {mtr} is {model}'.format(mtr=search_metric,
                                                                                  model=self.model_str))
                    self.model = self.create_custom_model(estimator=self.model_str, cv_fold_size=cv_fold_size,
                                                          cross_validation=True, verbose=self.verbose,
                                                          probability_threshold=None)
                    if not optimize:
                        self.tuned_model = self.model
                    else:
                        self.tuned_model, tuner = self._optimize_model(model=self.model, estimator_str=self.model_str, cv_fold_size=cv_fold_size,
                                                                       n_iter=n_iter, custom_grid_path=custom_grid,
                                                                       search_metric=search_metric,
                                                                       search_library=search_library,
                                                                       search_algorithm=search_algorithm,
                                                                       early_stopping=early_stopping,
                                                                       early_stopping_max_iters=early_stopping_max_iters,
                                                                       )
                else:
                    logger.info('Selected ensemble model and ensemble type is {}'.format(ensemble_type))
                    logger.info('Model ensemble is selected')
                    try:
                        model_list = []
                        for estimator in self.estimator:
                            model = self.create_custom_model(estimator=estimator, cv_fold_size=cv_fold_size,
                                                             cross_validation=True, verbose=self.verbose,
                                                             probability_threshold=None)
                            if not optimize:
                                tuned_model = model
                            else:
                                tuned_model, tuner = self._optimize_model(estimator_str=estimator, model=model, cv_fold_size=cv_fold_size,
                                                                          n_iter=n_iter, custom_grid_path=custom_grid,
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
                                                            restack=False, verbose=self.verbose,
                                                            choose_better=False)
                        elif ensemble_type == 'blend':
                            logger.info('Blending ML model')
                            self.tuned_model = blend_models(estimator_list=model_list, fold=cv_fold_size,
                                                            round=2, optimize=search_metric,
                                                            verbose=self.verbose, choose_better=False)
                    except AssertionError as error:
                        logger.info('The issue in stacking the model is {}'.format(error))

            # finalise model and train on holdout set
            self.model_pipeline = finalize_model(self.tuned_model, model_only=False)
        logger.info('Training completed for PyCaret model')

    def _optimize_model(self, model, estimator_str: str = None, cv_fold_size: int = 4, n_iter: int = 20,
                        custom_grid_path: dict = None, search_metric: str = None, early_stopping: bool = False,
                        search_library: str = None, search_algorithm: str = None,
                        early_stopping_max_iters: int = None):
        """"""
        # tune model
        if not self.task == 'regression':
            from pycaret.classification import tune_model
            logger.info('Model optimization is selected using method: {} and metric {}:'.format(search_algorithm,
                                                                                             search_metric))
            custom_grid_dict = OmegaConf.load(custom_grid_path)
            custom_params = pick_custom_grid(estimator=estimator_str, custom_grid_dict=custom_grid_dict)
            model, tuner = tune_model(estimator=model, fold=cv_fold_size, n_iter=n_iter, round=2,
                                      custom_grid=custom_params, optimize=search_metric,
                                      search_library=search_library, search_algorithm=search_algorithm,
                                      early_stopping=early_stopping, verbose=self.verbose,
                                      early_stopping_max_iters=early_stopping_max_iters,
                                      choose_better=True, return_tuner=True
                                      )
        else:
            # TODO-Regression: Modify for regression
            from pycaret.regression import tune_model
            logger.info('Model optimization is selected using method: {} and metric {}:'.format(search_algorithm,
                                                                                             search_metric))
            custom_grid_dict = OmegaConf.load(custom_grid_path)
            custom_params = pick_custom_grid(estimator=estimator_str, custom_grid_dict=custom_grid_dict)
            model, tuner = tune_model(estimator=model, fold=cv_fold_size, n_iter=n_iter, round=2,
                                      custom_grid=custom_params, optimize=search_metric,
                                      search_library=search_library, search_algorithm=search_algorithm,
                                      early_stopping=early_stopping, verbose=self.verbose,
                                      early_stopping_max_iters=early_stopping_max_iters,
                                      choose_better=True, return_tuner=True)
        return model, tuner

    def model_evaluation(self, path: str = None, plot: bool = False, prior_model_result: str = None):
        """"""
        eval_results_path = os.path.join(path, 'evaluation')
        if not os.path.exists(eval_results_path):
            os.makedirs(eval_results_path)

        if not self.task == 'regression':
            from pycaret.classification import predict_model
            predict_result = predict_model(estimator=self.calibrated_model,
                                           data=self.test.drop(columns=[self.target_label]),
                                           probability_threshold=0.5, raw_score=True, round=2,
                                           verbose=self.verbose).filter(regex="prediction_.*", axis=1)
            self.test_result = pd.concat([self.test.reset_index(drop=True, inplace=False),
                                          predict_result.reset_index(drop=True, inplace=False)], axis=1)
            evaluate = EvaluateClassification(estimator=self.calibrated_model, labels=self.test[self.target_label],
                                              preds_label=predict_result[['prediction_label']], pred_proba=None,
                                              prob_threshold=None, multi_label=self.is_multilabel, seed=self.seed)
            file_path = os.path.join(eval_results_path, 'evaluation_report.json')

            plot_path = os.path.join(eval_results_path, 'plots')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            evaluate.save(filepath=file_path, plot=plot, plot_path=plot_path)
            # evaluate with earlier test performances
            if prior_model_result is not None:
                try:
                    evaluate.drift(predict_df=self.test_result, prior_model_result=prior_model_result,
                                   report_path=eval_results_path, target_label=self.target_label,
                                   prediction_label='prediction_label')
                except Exception as error:
                    logger.error('Issue in evaluating with previous model results: {}'.format(error))
            else:
                logger.info('There is no evaluation with prior model results')
        else:
            
            # TODO-Regression: Implement for regression
            from pycaret.regression import predict_model
            predict_result = predict_model(estimator=self.tuned_model,
                                           data=self.test.drop(columns=[self.target_label]),
                                           round=2,
                                           verbose=self.verbose).filter(regex="prediction_.*", axis=1)
            self.test_result = pd.concat([self.test.reset_index(drop=True, inplace=False),
                                          predict_result.reset_index(drop=True, inplace=False)], axis=1)
            evaluate = EvaluateRegression(estimator=self.tuned_model, labels=self.test[self.target_label],
                                              preds=predict_result['prediction_label'], seed=self.seed)
            file_path = os.path.join(eval_results_path, 'evaluation_report.json')

            plot_path = os.path.join(eval_results_path, 'plots')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            evaluate.save(filepath=file_path, plot=plot, plot_path=plot_path)
            # evaluate with earlier test performances
            if prior_model_result is not None:
                try:
                    evaluate.drift(predict_df=self.test_result, prior_model_result=prior_model_result,
                                   report_path=eval_results_path, target_label=self.target_label,
                                   prediction_label='prediction_label')
                except Exception as error:
                    logger.error('Issue in evaluating with previous model results: {}'.format(error))
            else:
                logger.info('There is no evaluation with prior model results')
            

    def feature_explanation(self, path: str = None):
        """"""
        if not self.ensemble_model:
            feature_path = os.path.join(path, 'feature_explanation')
            if not os.path.exists(os.path.join(feature_path)):
                os.makedirs(feature_path)
            if not self.task == 'regression':
                from pycaret.classification import interpret_model
                # TODO 2: Sample Xtrain and Ytrain for larger dataset
                # X_train = get_config('X_Train')
                # Y_train = get_config('Y_Train')

                # default summary plot
                # file_path = os.path.join(path, 'shap_summary_plot.png')
                interpret_model(estimator=self.model, plot='summary', use_train_data=True, save=feature_path)
                # PDP plots for top features list
                # TODO 3: implement top features and loop it for pdp
                # file_path = os.path.join(path, 'pdp_feature_1.png')
                interpret_model(estimator=self.model, plot='pfi', use_train_data=True, save=feature_path)
                logger.info('feature importance artifacts is saved in the location : {}'.format(feature_path))
            else:
                from pycaret.regression import interpret_model
                # TODO-Regression: Sample Xtrain and Ytrain for larger dataset
                # X_train = get_config('X_Train')
                # Y_train = get_config('Y_Train')
                # default summary plot
                interpret_model(estimator=self.model, plot='summary', use_train_data=True, save=feature_path)
                # PDP plots for top features list
                # TODO-Regression: implement top features and loop it
                interpret_model(estimator=self.model, plot='pdp', use_train_data=True, save=feature_path)
                logger.info('feature importance artifacts is saved in the location : {}'.format(feature_path))
        else:
            logger.info('No SHAP explanation for ensemble models')

    def check_fairness(self, sensitive_features: list = None, path: str = None):
        """"""
        if sensitive_features is not None:
            if not self.task == 'regression':
                from pycaret.classification import predict_model
                sensitive_cols = list(set(self.test.columns) & set(sensitive_features))
                if len(sensitive_cols) >= 1:
                    logger.info('The sensitive features are: {}'.format(sensitive_cols))
                    test_dataset = self.test[sensitive_cols]
                    y_test = self.test[self.target_label]
                    predict_result = predict_model(estimator=self.calibrated_model,
                                                   data=self.test.drop(columns=[self.target_label]),
                                                   probability_threshold=0.5, raw_score=True, round=2,
                                                   verbose=self.verbose)['prediction_score_1']
                    # loop through sensitive features to provide report
                    fairness = FairnessClassification(labels=y_test, pred_proba=predict_result, prob_threshold=0.5,
                                                      multi_label=False, sensitive_df=test_dataset, report_path=path)
                    fairness.calculate_metrics(plot=True)
                else:
                    logger.info('The sensitive features:{} not utilised in for '
                                'model training'.format(sensitive_features))
            else:
                # from pycaret.regression import get_config, predict_model
                # sensitive_cols = list(set(self.test.columns) & set(sensitive_features))
                # if len(sensitive_cols) >= 1:
                #     logger.info('The sensitive features are: {}'.format(sensitive_cols))
                #     test_dataset = self.test[sensitive_cols]
                #     y_test = self.test[self.target_label]
                #     predict_result = predict_model(estimator=self.calibrated_model,
                #                                    data=self.test.drop(columns=[self.target_label]), round=2,
                #                                    verbose=self.verbose)['prediction_score_1']
                # TODO-Regression: Implement for regression
                pass
        else:
            logger.info('Sensitive feature list is empty')

    def save(self, path: str = None, save_with_data: bool = False):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import save_model, pull
        else:
            from pycaret.regression import save_model, pull
        file_path = os.path.join(path, str(self.task))
        save_model(self.model_pipeline, model_name=file_path)
        if save_with_data:
            data_path = os.path.join(path, 'dataset')
            if not os.path.exists(data_path):
                os.makedirs(data_path)
                logger.info('Storing trained data-frame at {}'.format(data_path))
            save_parquet(df=self.train, path=data_path, file_name='training_data.parquet')
            if self.test_result is not None:
                save_parquet(df=self.test_result, path=data_path, file_name='test_data_with_result.parquet')
            else:
                save_parquet(df=self.test, path=data_path, file_name='test_data.parquet')
        logger.info('model is saved in the location : {}'.format(path))

    def load(self, path: str = None):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import load_model
        else:
            from pycaret.regression import load_model
        file_path = os.path.join(path, str(self.task))
        self.model_pipeline = load_model(model_name=file_path, verbose=False)
        logger.info('Trained pycaret model is loaded from: {}'.format(path))

    def predict(self, data: pd.DataFrame = None, proba: bool = False) -> pd.DataFrame:
        """"""
        assert isinstance(data, (pd.DataFrame, np.ndarray))
        if not self.task == 'regression':
            from pycaret.classification import predict_model
            predict_result = predict_model(estimator=self.model_pipeline, data=data, raw_score=True, round=2,
                                           verbose=self.verbose)
            logger.info('prediction done for pycaret classification model')
            if not proba:
                return predict_result['prediction_score_1']
            else:
                return predict_result['prediction_score_1']

        else:
            from pycaret.regression import predict_model
            # TODO-Regression: Implement for regression
            from pycaret.regression import predict_model
            predict_result = predict_model(estimator=self.model_pipeline, data=data, round=2,
                                           verbose=self.verbose)
            logger.info('prediction done for pycaret regression model')
            return predict_result['prediction_label']


    def create_custom_model(self, estimator: str = None, cv_fold_size: int = 4, probability_threshold: float = 0.5,
                            cross_validation: bool = True, verbose: bool = False):
        """"""
        if not self.task == 'regression':
            from pycaret.classification import create_model
            if estimator not in ['catboost', 'xgboost']:
                logger.info('Model is {}, So cannot apply monotonic function'.format(estimator))
                model = create_model(estimator=estimator, fold=cv_fold_size, round=2,
                                     cross_validation=cross_validation, probability_threshold=probability_threshold,
                                     verbose=verbose)
            else:
                logger.info('Model is {}, So apply monotonic function'.format(estimator))
                model = create_model(estimator=estimator, fold=cv_fold_size, round=2,
                                     cross_validation=cross_validation, probability_threshold=probability_threshold,
                                     monotone_constraints=self.monotone_constraints, verbose=verbose)
        else:
            from pycaret.regression import create_model
            if estimator not in ['catboost', 'xgboost']:
                logger.info('Model is {}, So cannot apply monotonic function'.format(estimator))
                model = create_model(estimator=estimator, fold=cv_fold_size, round=2,
                                     cross_validation=cross_validation,  verbose=verbose)
            else:
                logger.info('Model is {}, So apply monotonic function'.format(estimator))
                model = create_model(estimator=estimator, fold=cv_fold_size, round=2, cross_validation=cross_validation,
                                     monotone_constraints=self.monotone_constraints, verbose=verbose)
        return model
