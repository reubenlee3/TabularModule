import os
import pandas as pd
import time
import hydra
from datetime import datetime
from tabular_src import DataIntegrityTest, DataLoader, TrainingDataDrift
from tabular_src import PyCaretModel, SurrogateModel
from tabular_src import get_logger

logger = get_logger(__name__)

config_name = 'data_config_regression'

@hydra.main(config_path='config', config_name=config_name)
def execute_main(cfg) -> None:
    """"""
    # Set-up parameters
    t0 = time.time()
    output_folder = os.path.join(cfg.paths.output, cfg.process.exp_id)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger.info('Storing results at {}'.format(output_folder))

    if cfg.paths.train is not None:
        logger.info('Running Training mode')
        data_loader = DataLoader(train=cfg.paths.train, test=cfg.paths.test, is_reduce_memory=cfg.process.memory_reduce,
                                 infer_datatype=cfg.process.infer_datatype,
                                 categorical_columns=cfg.columns.categorical_columns,
                                 test_ratio=cfg.process.test_ratio, target_label=cfg.columns.target_label,
                                 run_feature_selection=cfg.feature_selection.auto,
                                 rfe_estimator=cfg.feature_selection.estimator, task=cfg.process.task,
                                 multi_colinear_threshold=cfg.feature_selection.multi_colinear_threshold,
                                 n_features=cfg.feature_selection.n_features, keep_features=cfg.columns.keep_columns,
                                 text_features=cfg.columns.text_columns, seed=cfg.process.seed)
        train_df, test_df, categorical_cols, numerical_cols, target_label = data_loader.return_values()
        if cfg.data_validation.data_integrity:
            # Data integrity tests
            data_integrity = DataIntegrityTest(df=train_df, categorical_columns=categorical_cols,
                                               numerical_columns=numerical_cols, datetime_columns=None,
                                               target_label=target_label, task=cfg.process.task,
                                               seed=cfg.process.seed
                                               )
            data_integrity_report = data_integrity.run_integrity_checks(save_html=cfg.data_validation.save_html,
                                                                        return_dict=True,
                                                                        save_dir=os.path.join(output_folder, 'reports'))
            train_df = data_integrity.act_testresults(test_results=data_integrity_report)
            # update data
            test_df = test_df[train_df.columns]
            categorical_cols = list(set(categorical_cols) & set(train_df.columns))
            numerical_cols = list(set(numerical_cols) & set(train_df.columns))

        if cfg.data_validation.data_drift:
            data_drift = TrainingDataDrift(train_df=train_df, test_df=test_df,
                                           categorical_columns=categorical_cols,
                                           numerical_columns=numerical_cols, datetime_columns=None,
                                           target_label=target_label, task=cfg.process.task,
                                           seed=cfg.process.seed)
            # Data-Drift
            data_drift_report = data_drift.run_drift_checks(save_html=cfg.data_validation.save_html,
                                                            save_dir=os.path.join(output_folder, 'reports'),
                                                            return_dict=True)
            datadrift_status = data_drift.act_drift_results(test_results=data_drift_report, drift_thresh=0.5)
            # Concept-Drift
            concept_drift_report = data_drift.run_target_drift_checks(save_html=cfg.data_validation.save_html,
                                                                      save_dir=os.path.join(output_folder, 'reports'),
                                                                      return_dict=True)
            conceptdrift_status = data_drift.act_target_drift_results(test_results=concept_drift_report,
                                                                      drift_thresh=0.5)
            # Check status to continue the model training
            if datadrift_status and conceptdrift_status:
                logger.info('Continuing model training. Model passed both data and concept drift')
            else:
                logger.info('Data drift status is {} and Concept drift status is {}'.format(datadrift_status,
                                                                                            conceptdrift_status))
                logger.info('Stopping Model training!!!!')
                # TODO 1: if process_status is false stop the training

        # build surrogate model
        if cfg.process.surrogate_model:
            logger.info('Building surrogate model for debug')
            surrogate_model = SurrogateModel(train=train_df.drop(columns=target_label, inplace=False),
                                             target=train_df[target_label],
                                             task=cfg.process.task, test=test_df,
                                             estimator=cfg.model.surrogate_algorithm, is_multilabel=False,
                                             categorical_columns=categorical_cols, numerical_columns=numerical_cols,
                                             seed=cfg.process.seed)
            surrogate_model.fit()
            surrogate_model_path = os.path.join(output_folder, 'surrogate_model')
            if not os.path.exists(surrogate_model_path):
                os.makedirs(surrogate_model_path)
            surrogate_model.save(save_path=surrogate_model_path, only_model=True)

        if not cfg.process.only_surrogate:
            # Train Model
            pycaret_model = PyCaretModel(train=train_df, target=target_label, task=cfg.process.task, test=test_df,
                                         estimator_list=cfg.model.algorithm, params=None, n_jobs=-1, use_gpu=False,
                                         is_multilabel=cfg.process.multi_label, categorical_cols=categorical_cols,
                                         numerical_cols=numerical_cols, keep_features=None, text_cols=None,
                                         monotone_inc_cols=cfg.columns.monotonic_increase_columns,
                                         monotone_dec_cols=cfg.columns.monotonic_decrease_columns,
                                         seed=cfg.process.seed, verbose=cfg.process.verbose)

            if not cfg.process.task == 'regression':
                custom_params_grid = cfg.hyperparams.classification_params
            else:
                custom_params_grid = cfg.hyperparams.regression_params

            pycaret_model.fit(apply_pca=cfg.model.pca, remove_outliers=False, fold_strategy=cfg.model.fold_strategy,
                              cv_fold_size=cfg.model.cv_fold, calibrate=cfg.model.calibrate,
                              probability_threshold=cfg.model.prob_thresh, optimize=cfg.model.tuning,
                              custom_grid=custom_params_grid, n_iter=cfg.model.iteration, search_library='optuna',
                              search_algorithm='tpe', search_metric=cfg.model.search_metric, early_stopping=True,
                              early_stopping_max_iters=4, ensemble_model=cfg.model.ensemble,
                              ensemble_type=cfg.model.ensemble_type)

            pycaret_model.model_evaluation(path=output_folder, plot=True,
                                           prior_model_result=cfg.data_validation.prior_model_result)

            if cfg.model.feature_importance:
                # TODO 3: Write Custom SHAP, ICE or Feature Permutation plots
                pycaret_model.feature_explanation(path=output_folder)

            if cfg.model.fairness:
                pycaret_model.check_fairness(sensitive_features=cfg.columns.sensitive_columns, path=output_folder)

            pycaret_model.save(path=output_folder, save_with_data=True)
        else:
            logger.info('only surrogate model training is selected')
        t1 = time.time()
        logger.info('Total training time is {:.2f} Secs'.format(t1 - t0))
    else:
        logger.info('Running Prediction mode')
        data_loader = DataLoader(train=None, test=cfg.paths.test, is_reduce_memory=cfg.process.memory_reduce,
                                 infer_datatype=cfg.process.infer_datatype, categorical_columns=None,
                                 test_ratio=None, target_label=None, run_feature_selection=False,
                                 rfe_estimator=None, task=cfg.process.task, multi_colinear_threshold=None,
                                 n_features=None, keep_features=None, text_features=None, seed=cfg.process.seed)

        test_df = data_loader.return_values()
        if cfg.data_validation.prediction_drift:
            logger.info('Running prediction-training drift')
            try:
                file_path = os.path.join(cfg.paths.result, 'training_data.parquet')
                trained_data = pd.read_parquet(path=file_path, engine='auto')
                # trained_data = trained_data.drop(columns=[cfg.columns.target_label])
                categorical_cols, numerical_cols = data_loader.get_col_types(data=trained_data, auto=True)
                data_drift = TrainingDataDrift(train_df=trained_data, test_df=test_df,
                                               categorical_columns=categorical_cols,
                                               numerical_columns=numerical_cols, datetime_columns=None,
                                               target_label=None, task=cfg.process.task,
                                               seed=cfg.process.seed)
                data_drift_report = data_drift.run_drift_checks(save_html=cfg.data_validation.save_html,
                                                                save_dir=os.path.join(output_folder, 'reports'),
                                                                filename='prediction_datadrift',
                                                                return_dict=True)
                datadrift_status = data_drift.act_drift_results(test_results=data_drift_report, drift_thresh=0.5)
                # TODO 2: Stop if there is too much drift between train and prediction

            except Exception as error:
                logger.error('Issue in loading trained data: {}'.format(error))

        # load model
        pycaret_model = PyCaretModel(task=cfg.process.task, test=test_df, n_jobs=-1, use_gpu=False,
                                     is_multilabel=cfg.process.multi_label, seed=cfg.process.seed,
                                     verbose=cfg.process.verbose)
        pycaret_model.load(path=output_folder)
        # score model
        pred_col = 'prob_score' if cfg.process.task == 'classification' else 'prediction_label'
        test_df[pred_col] = pycaret_model.predict(data=test_df)

        # save result in the folder
        pred_folder = os.path.join(output_folder, 'scoring')
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
            logger.info('Storing prediction at {}'.format(pred_folder))
        prediction_file = 'prediction{}.csv'.format(datetime.now().strftime('%d%b%Y_%H::%M'))
        file_path = os.path.join(pred_folder, prediction_file)
        test_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    execute_main()
