import os
import pandas as pd
import hydra
from tabular_src import DataIntegrityTest, DataLoader, TrainingDataDrift
from tabular_src import PyCaretModel, SurrogateModel
from tabular_src import get_logger

logger = get_logger(__name__)


@hydra.main(config_path='config', config_name='data_config')
def execute_main(cfg) -> None:
    """"""
    # Set-up parameters
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
            data_integrity = DataIntegrityTest(df=train_df, categorical_columns=categorical_cols,
                                               numerical_columns=numerical_cols, datetime_columns=None,
                                               target_label=target_label, task=cfg.process.task,
                                               seed=cfg.process.seed
                                               )
            data_integrity.run_integrity_checks(save_html=cfg.data_validation.save_html, save_dir=output_folder)
            # TODO: Action upon data integrity report

        if cfg.data_validation.data_drift:
            data_drift_report = TrainingDataDrift(train_df=train_df, test_df=test_df,
                                                  categorical_columns=categorical_cols,
                                                  numerical_columns=numerical_cols, datetime_columns=None,
                                                  target_label=target_label, task=cfg.process.task,
                                                  seed=cfg.process.seed)
            data_drift_report.run_drift_checks(save_html=cfg.data_validation.save_html, save_dir=output_folder)
            data_drift_report.run_target_drift_checks(save_html=cfg.data_validation.save_html, save_dir=output_folder)
            # TODO: Action upon data data drift report

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
            surrogate_model.save(save_path=output_folder, only_model=True)

        if not cfg.process.only_surrogate:
            # Train Model
            pycaret_model = PyCaretModel(train=train_df, target=target_label, task=cfg.process.task, test=test_df,
                                         estimator_list=cfg.model.algorithm, params=None, n_jobs=-1, use_gpu=False,
                                         is_multilabel=False, categorical_cols=categorical_cols,
                                         numerical_cols=numerical_cols, keep_features=None,
                                         text_cols=None, seed=cfg.process.seed, verbose=cfg.process.verbose)

            pycaret_model.fit(apply_pca=cfg.model.pca, remove_outliers=False, fold_strategy='stratifiedkfold',
                              cv_fold_size=cfg.model.cv_fold, calibrate=cfg.model.calibrate,
                              probability_threshold=cfg.model.prob_thresh, optimize=cfg.model.tuning,
                              # custom_grid=None,
                              n_iter=cfg.model.iteration, search_library='optuna', search_algorithm='tpe',
                              search_metric='F1', early_stopping=True, early_stopping_max_iters=4,
                              ensemble_model=cfg.model.ensemble, ensemble_type=cfg.model.ensemble_type)

            pycaret_model.model_evaluation(path=output_folder, plot=True)

            if cfg.model.feature_importance:
                pycaret_model.feature_explanation(path=output_folder)

            if cfg.model.fairness:
                pycaret_model.check_fairness(sensitive_features=cfg.columns.sensitive_columns, path=output_folder)

            pycaret_model.save(path=output_folder, training_data=True, model_stats=True)
        else:
            logger.info('only surrogate model training is selected')
    else:
        logger.info('Running Prediction mode')
        data_loader = DataLoader(train=None, test=cfg.paths.test, is_reduce_memory=cfg.process.memory_reduce,
                                 infer_datatype=cfg.process.infer_datatype, categorical_columns=None,
                                 test_ratio=None, target_label=None, run_feature_selection=False,
                                 rfe_estimator=None, task=cfg.process.task, multi_colinear_threshold=None,
                                 n_features=None, keep_features=None, text_features=None, seed=cfg.process.seed)

        test_df = data_loader.return_values()
        if cfg.data_validation.prediction_drift:
            logger.info('Running prediction drift')
            try:
                file_path = os.path.join(cfg.paths.result, 'training_data.parquet')
                trained_data = pd.read_parquet(path=file_path, engine='auto')
                trained_data = trained_data.drop(columns=[cfg.columns.target_label])
                categorical_cols, numerical_cols = data_loader.get_col_types(data=trained_data, auto=True)
                data_drift_report = TrainingDataDrift(train_df=trained_data, test_df=test_df,
                                                      categorical_columns=categorical_cols,
                                                      numerical_columns=numerical_cols, datetime_columns=None,
                                                      target_label=None, task=cfg.process.task,
                                                      seed=cfg.process.seed)
                import pdb; pdb.set_trace()
                data_drift_report.run_drift_checks(save_html=cfg.data_validation.save_html, save_dir=output_folder,
                                                   filename='prediction_datadrift')
            except Exception as error:
                logger.error('Issue in loading trained data: {}'.format(error))
        # TODO Analyse drift between trained and prediction data
        # Stop if there is too much drift

        # load model
        pycaret_model = PyCaretModel(task=cfg.process.task, test=test_df, n_jobs=-1, use_gpu=False,
                                     is_multilabel=False, seed=cfg.process.seed, verbose=cfg.process.verbose)

        pycaret_model.load(path=cfg.paths.result)
        # score model
        test_df['prob_score'] = pycaret_model.predict(data=test_df)

        # save result in the folder
        output_folder = os.path.join(cfg.paths.result, 'scoring')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info('Storing prediction at {}'.format(output_folder))
        file_path = os.path.join(output_folder, 'prediction.csv')
        test_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    execute_main()
