from sklearn.inspection import permutation_importance
import multiprocessing
import pandas as pd
import shap
from ..utils import get_logger

logger = get_logger(__name__)


def calculate_feature_importance(model, X_test, y_test, scoring_metric=None, n_repeats=30,
                                 random_state=42, n_jobs: int = -1) -> pd.DataFrame:
    """"""
    if (n_jobs == -1 and multiprocessing.cpu_count() > 60) or n_jobs > 60:
        # The Windows constant MAXIMUM_WAIT_OBJECTS (64) prevents monitoring more than 60 handles (see:
        # https://hg.python.org/cpython/file/80d1faa9735d/Modules/_winapi.c#l1339). Thus, we strictly enforce this
        # limit.
        n_jobs = 60

    pi = permutation_importance(estimator=model, X=X_test, y=y_test, scoring=scoring_metric, n_repeats=n_repeats,
                                random_state=random_state, n_jobs=n_jobs)

    feature_importances = pd.concat(
        [pd.Series(X_test.columns.to_list()), pd.Series(pi.importances_mean).round(5)], axis=1)
    feature_importances.columns = ['Variable', 'Value']

    return feature_importances


class ModelExplainer(object):
    """"""
    def __init__(self, estimator=None, train_data: pd.DataFrame = None, test_data: pd.DataFrame = None,
                 use_test: bool = False, data_points: int = 300000, seed: int = 33):
        """"""
        self.estimator = estimator
        self.train_data = train_data
        self.test_data = test_data
        if not use_test:
            self.intrepret_data = self.train_data.copy()
        else:
            self.intrepret_data = self.test_data.copy()
        # check if sampling required
        if self.intrepret_data.shape[0] > data_points:
            logger.info('Sampling data points to run shap values')
            self.intrepret_data = self.intrepret_data.sample(n=300000, random_state=seed)

    def build_shap_values(self, shap_explainer_method):
        """"""
        if shap_explainer_method == "probability":
            explainer = shap.TreeExplainer(
                model=self.estimator, model_output="probability", data=self.intrepret_data,
                feature_perturbation='interventional'
            )
        else:
            explainer = shap.TreeExplainer(self.estimator)

        return explainer(self.intrepret_data)

    def shap_summary(self):
        pass

    def shap_beesworm(self):
        pass

    def pdp(self):
        pass

    def intrepret_model(self, method: str = 'summary', plot: bool = False, path: str = None):
        pass

