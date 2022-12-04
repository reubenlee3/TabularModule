from sklearn.inspection import permutation_importance
import multiprocessing
import pandas as pd


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
