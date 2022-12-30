import numpy as np
import pandas as pd
import json
import os
import functools
from copy import deepcopy
from fairlearn.metrics import (MetricFrame, count, selection_rate, demographic_parity_difference,
                               demographic_parity_ratio, false_positive_rate, false_negative_rate,
                               true_negative_rate, true_positive_rate,
                               equalized_odds_difference, equalized_odds_ratio)
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, log_loss,
                             roc_auc_score, fbeta_score)

from tabular_src.utils import get_logger

logger = get_logger(__name__)


class PerformanceMetric(object):
    """"""

    def __init__(self, labels, preds, sensitive_feature: pd.Series = None, multi_label: bool = False):
        assert isinstance(labels, (pd.Series, np.ndarray))
        assert isinstance(preds, (pd.Series, np.ndarray))
        if isinstance(labels, pd.Series):
            labels = np.asarray(labels)

        if not multi_label:
            logger.info('Evaluation is set for binary-classification')
            self.labels = labels
            self.preds = preds
            fbeta_06 = functools.partial(fbeta_score, beta=0.6, zero_division=1)
            metric_fns = {"selection_rate": selection_rate,
                          "accuracy": accuracy_score,
                          "precision": precision_score,
                          "recall": recall_score,
                          "f1_score": f1_score,
                          "tnr": true_negative_rate,
                          "fnr": false_negative_rate,
                          "roc_auc_score": roc_auc_score,
                          "log_loss": log_loss,
                          "fbeta_06": fbeta_06,
                          "count": count}
            grouped_on_feature = MetricFrame(metrics=metric_fns, y_true=self.labels, y_pred=self.preds,
                                             sensitive_features=sensitive_feature)
            self.overall_score = grouped_on_feature.overall
            self.groups = grouped_on_feature.by_group
        else:
            logger.info('Evaluation is set for multi-classification')
            # TODO 1: Implement performance for multiclass
            # self.confusion_matrix = multilabel_confusion_matrix(labels, self.preds)

    def show(self):
        """"""
        logger.info('Selection rate: {}'.format(self.overall_score["selection_rate"]))
        logger.info('Accuracy: {}'.format(self.overall_score["accuracy"]))
        logger.info('Precision: {}'.format(self.overall_score["precision"]))
        logger.info('Recall: {}'.format(self.overall_score["recall"]))
        logger.info('F1 Score: {}'.format(self.overall_score["f1_score"]))
        logger.info('ROC-AUC Score: {}'.format(self.overall_score["roc_auc_score"]))
        logger.info('Log loss: {}'.format(self.overall_score["log_loss"]))
        logger.info('Count: {}'.format(self.overall_score["count"]))
        print(self.groups)

    def to_dict(self):
        """"""
        res = {
            'selection_rate': self.overall_score["selection_rate"],
            'accuracy': self.overall_score["accuracy"],
            'precision': self.overall_score["precision"],
            'recall': self.overall_score["recall"],
            'f1_score': self.overall_score["f1_score"],
            'tnr': self.overall_score["tnr"],
            'fnr': self.overall_score["fnr"],
            'roc_auc_score': self.overall_score["roc_auc_score"],
            'log_loss': self.overall_score["log_loss"],
            'count': self.overall_score["count"],
        }
        group = self.groups.to_dict('dict')
        result_dict = {'overall metrics': res, 'group_metrics': group}
        return result_dict

    def to_json(self):
        """"""
        dic = self.to_dict()
        res = deepcopy(dic)
        return res

    def save(self, filepath: str = None):
        """"""
        res = self.to_json()
        with open(filepath, 'w') as fp:
            json.dump(res, fp)
            logger.info('Fairness report saved to: {}'.format(filepath))

    def to_plot(self, tile: str = 'Show all metrics in Accent colormap'):
        # Customize plots with colormap
        fig = self.groups.plot(kind="bar", subplots=True, layout=[3, 4], legend=False,
                               figsize=[12, 8], colormap="Accent", title=tile)
        return fig[0][0].figure


class FairnessMetric(object):
    """"""

    def __init__(self, labels, preds, sensitive_feature: pd.Series = None, fairness_method: str = 'between_groups',
                 parity_method: str = 'difference', multi_label: bool = False):
        assert isinstance(labels, (pd.Series, np.ndarray))
        assert isinstance(preds, (pd.Series, np.ndarray))
        if isinstance(labels, pd.Series):
            labels = np.asarray(labels)

        if not multi_label:
            logger.info('Evaluation is set for binary-classification')
            self.labels = labels
            self.preds = preds
            
            self.demographic_parity = demographic_parity_difference(y_true=self.labels, y_pred=self.preds,
                                                                    sensitive_features=sensitive_feature)
            self.disparate_impact = demographic_parity_ratio(y_true=self.labels, y_pred=self.preds,
                                                             sensitive_features=sensitive_feature)
            self.equal_odds = equalized_odds_difference(y_true=self.labels, y_pred=self.preds,
                                                        sensitive_features=sensitive_feature)
            
            metric_fns = {"fpr": false_positive_rate,
                          "tnr": true_negative_rate,
                          "fnr": false_negative_rate}
            grouped_on_feature = MetricFrame(metrics=metric_fns, y_true=self.labels, y_pred=self.preds,
                                             sensitive_features=sensitive_feature)
            
            if parity_method == 'difference':
                self.overall_score = grouped_on_feature.difference(method=fairness_method)
            elif parity_method == 'ratio':
                self.overall_score = grouped_on_feature.ratio(method=fairness_method)
            else:
                logger.error('Invalid parity method {}'.format(parity_method))
                raise ValueError('Invalid parity method {}. Method should be difference or ratio'.format(parity_method))
        else:
            logger.info('Evaluation is set for multi-classification')
            # TODO 2: Implement fairness metrics for multiclass
            # self.confusion_matrix = multilabel_confusion_matrix(labels, self.preds)

    def show(self):
        """"""
        logger.info('Disparate Impact: {}'.format(self.disparate_impact))
        logger.info('Demographic Parity: {}'.format(self.demographic_parity))
        logger.info('Equal odds: {}'.format(self.equal_odds))
        logger.info('False positive rate parity: {}'.format(self.overall_score["fpr"]))
        logger.info('True negative rate parity: {}'.format(self.overall_score["tnr"]))
        logger.info('False negative rate parity: {}'.format(self.overall_score["fnr"]))

    def to_dict(self):
        """"""
        res = {
            'disparate_impact': self.disparate_impact,
            'demographic_parity': self.demographic_parity,
            'equal_odds': self.equal_odds,
            'false_positive_rate_parity': self.overall_score["fpr"],
            'true_negative_rate_parity': self.overall_score["tnr"],
            'false_negative_rate_parity': self.overall_score["fnr"],
        }
        return res

    def to_json(self):
        """"""
        dic = self.to_dict()
        res = deepcopy(dic)
        return res

    def save(self, filepath: str = None):
        """"""
        res = self.to_json()
        with open(filepath, 'w') as fp:
            json.dump(res, fp)
            logger.info('Fairness report saved to: {}'.format(filepath))


class FairnessClassification(object):
    """"""

    def __init__(self, labels, pred_proba, prob_threshold, multi_label=False,
                 sensitive_df: pd.DataFrame = None, report_path: str = None):
        assert isinstance(labels, (pd.Series, np.ndarray))
        assert isinstance(pred_proba, (pd.Series, np.ndarray))
        assert isinstance(sensitive_df, (pd.Series, pd.DataFrame))
        self.labels = labels
        self.pred_proba = pred_proba
        self.sensitive_df = sensitive_df.copy()
        self.multi_label = multi_label
        if isinstance(labels, pd.Series):
            self.labels = np.asarray(self.labels)
        if isinstance(pred_proba, pd.Series):
            self.pred_proba = np.asarray(self.pred_proba)
            self.preds = np.where(self.pred_proba < prob_threshold, 0, 1)
        self.output_folder = os.path.join(report_path, 'fairness_reports')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            logger.info('Storing fairness reports at {}'.format(self.output_folder))

    def calculate_metrics(self, plot: bool = False):
        # loop through sensitive features in dataframe to calculate fairness
        for column in self.sensitive_df.columns:
            performance_metric = PerformanceMetric(labels=self.labels, preds=self.preds,
                                                   sensitive_feature=self.sensitive_df[column],
                                                   multi_label=self.multi_label)
            performance_dict = performance_metric.to_json()
            fairness_metric = FairnessMetric(labels=self.labels, preds=self.preds,
                                             parity_method='difference', fairness_method='between_groups',
                                             sensitive_feature=self.sensitive_df[column], multi_label=self.multi_label)
            fairness_dict = fairness_metric.to_json()
            
            result_dict = {'performance_metrics': performance_dict, 'fairness_metrics': fairness_dict}
            filepath = os.path.join(self.output_folder, 'fairness_report_{}.json'.format(column))
            with open(filepath, 'w') as fp:
                json.dump(result_dict, fp)
                logger.info('Fairness report saved to: {}'.format(filepath))
            if not plot:
                logger.info('Plotting the fairness metric is not selected')
            else:
                logger.info('Plotting the fairness metric is selected')
                filepath = os.path.join(self.output_folder, 'fairness_report_{}.png'.format(column))
                performance_plot = performance_metric.to_plot(tile='Performance metrics for feature:{}'.format(column))
                performance_plot.savefig(filepath)
