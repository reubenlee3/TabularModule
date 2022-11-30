import os.path

import hydra
from tabular_src import DataIntegrityTest, DataLoader, TrainingDataDrift
from tabular_src import get_logger

logger = get_logger(__name__)


@hydra.main(config_path='config', config_name='data_config')
def execute_main(cfg) -> None:
    """"""
    # Set-up parameters
    output_folder = os.path.join(cfg.paths.output, 'exp_1')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info('Storing results at {}'.format(output_folder))

    if cfg.paths.train is not None:
        logger.info('Running Training mode')
        data_loader = DataLoader(train=cfg.paths.train, test=cfg.paths.test, is_reduce_memory=cfg.process.memory_reduce,
                                 infer_datatype=cfg.process.infer_datatype,
                                 categorical_columns=cfg.columns.categorical_columns,
                                 test_ratio=cfg.process.test_ratio, seed=cfg.process.seed,
                                 target_label=cfg.columns.target_label)
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

    else:
        logger.info('Running Prediction mode')


if __name__ == "__main__":
    execute_main()
