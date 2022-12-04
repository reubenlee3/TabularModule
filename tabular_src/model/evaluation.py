import numpy as np
import pandas as pd
import json
from copy import deepcopy
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix,
                             classification_report)

from ..utils import get_logger

logger = get_logger(__name__)


class EvaluateClassification(object):

    def __init__(self, labels, preds, multi_label=False):
        assert isinstance(labels, (pd.Series, np.ndarray))
        assert isinstance(preds, (pd.Series, np.ndarray))
        if isinstance(labels, pd.Series):
            labels = np.asarray(labels)
        if isinstance(preds, pd.Series):
            preds = np.asarray(preds)

        self.labels = labels
        self.preds = preds

        self.accuracy_score = accuracy_score(labels.flatten(), preds.flatten())
        self.f1_score = f1_score(labels.flatten(), preds.flatten(), average='macro')
        if not multi_label:
            self.confusion_matrix = confusion_matrix(labels, preds)
        else:
            self.confusion_matrix = multilabel_confusion_matrix(labels, preds)
        self.classification_report = classification_report(labels, preds, output_dict=True)

    def show(self):
        logger.info('Accuracy: {}'.format(self.accuracy_score))
        logger.info('F1 score: {}'.format(self.f1_score))
        logger.info('Confusion matrix:\n{}'.format(self.confusion_matrix))
        logger.info('Classification report:\n{}'.format(self.classification_report))
        print(classification_report(self.labels, self.preds))

    def to_dict(self):
        res = {
            'accuracy': self.accuracy_score,
            'f1_score': self.f1_score,
            'classification_report': self.classification_report,
            'confusion_matrix': self.confusion_matrix
        }
        return res

    def to_json(self):
        dic = self.to_dict()
        res = deepcopy(dic)
        res['confusion_matrix'] = res['confusion_matrix'].tolist()
        return res

    def save(self, filepath):
        res = self.to_json()
        with open(filepath, 'w') as fp:
            json.dump(res, fp)
            logger.info('Evaluation result saved to: {}'.format(filepath))


class EvaluateRegression(object):

    def __init__(self, labels, preds, multi_label=False):
        assert isinstance(labels, (pd.Series, np.ndarray))
        assert isinstance(preds, (pd.Series, np.ndarray))
        if isinstance(labels, pd.Series):
            labels = np.asarray(labels)
        if isinstance(preds, pd.Series):
            preds = np.asarray(preds)

        self.labels = labels
        self.preds = preds

        self.accuracy_score = accuracy_score(labels.flatten(), preds.flatten())
        self.f1_score = f1_score(labels.flatten(), preds.flatten(), average='macro')
        if not multi_label:
            self.confusion_matrix = confusion_matrix(labels, preds)
        else:
            self.confusion_matrix = multilabel_confusion_matrix(labels, preds)
        self.classification_report = classification_report(labels, preds, output_dict=True)

    def show(self):
        logger.info('Accuracy: {}'.format(self.accuracy_score))
        logger.info('F1 score: {}'.format(self.f1_score))
        logger.info('Confusion matrix:\n{}'.format(self.confusion_matrix))
        logger.info('Classification report:\n{}'.format(self.classification_report))
        print(classification_report(self.labels, self.preds))

    def to_dict(self):
        res = {
            'accuracy': self.accuracy_score,
            'f1_score': self.f1_score,
            'classification_report': self.classification_report,
            'confusion_matrix': self.confusion_matrix
        }
        return res

    def to_json(self):
        dic = self.to_dict()
        res = deepcopy(dic)
        res['confusion_matrix'] = res['confusion_matrix'].tolist()
        return res

    def save(self, filepath):
        res = self.to_json()
        with open(filepath, 'w') as fp:
            json.dump(res, fp)
            logger.info('Evaluation result saved to: {}'.format(filepath))